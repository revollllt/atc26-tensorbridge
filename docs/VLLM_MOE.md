# vLLM MoE Integration

TensorBridge supplies Hopper (SM90) NVFP4 linear and routed-expert execution through vLLM 0.32.0's
general plugin mechanism (`tensorbridge.vllm_plugin:register`, implemented in `tensorbridge/vllm/`).
vLLM source files are never patched: vLLM retains checkpoint parsing, sharding, routing, communication,
reduction, and shared-expert scheduling.

## Quick Start & Unified Presets

Install TensorBridge with the `vllm` extra and the CUDA major version matching PyTorch:

```bash
pip install --no-build-isolation '.[cu13,vllm]'

# 1. Existing NVFP4 checkpoint (ModelOpt or compressed-tensors; any offline quantizer)
vllm serve NVFP4_MODEL --additional-config '{"tensorbridge":{"moe":true,"exact":true,"moe_dispatch_dtype":"fp8","moe_gate_up_quant":"group_128","moe_down_quant":"group_128"}}'

# 2. Block-128 FP8 MoE checkpoint (e.g., GLM-5.3) with unified scale-up presets
vllm serve FP8_MODEL --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --additional-config '{"tensorbridge":{"preset":"tb_nvfp4_fp8","strict_moe":true}}'
```

| Preset | Precision (W × A) | Dispatch & Execution | Primary Role |
| --- | --- | --- | --- |
| `tb_nvfp4_fp8` | Exact NVFP4 × per-token FP8 | BF16 AG/RS, inherits `HummingGroupedExperts.apply` with TensorBridge NVFP4 GEMMs | Decode-optimized apples-to-apples NVFP4×FP8 |
| `tb_nvfp4_fp8_g128` | Exact NVFP4 × group-128 FP8 | FP8 AG/RS, adaptive fused SwiGLU+FP8 (`BN=256`) / multi-token Prefill (`BN=128` + interleaved SwiGLU) | Best Prefill TTFT, high-batch throughput & top accuracy |
| `tb_nvfp4_megamoe` | Exact NVFP4 × group-128 FP8 | Single-node `NVLinkSymmetricEP`: MegaMoE (`T ≤ 256`) + early `shared_experts` overlap + single-shot Contiguous Prefill (`T > 256`) | Best overall Decode TPOT, competitive Prefill TTFT, +3.25–4.18 GiB KV cache |

See [`docs/MEGAMOE.md`](MEGAMOE.md) for the MegaMoE & Prefill architecture and verified 8×H20 GLM-5.3 results.

When using a preset on a block-128 FP8 checkpoint, only routed experts are converted on-the-fly on GPU
to standard ModelOpt NVFP4; dense layers, attention, and shared experts remain in native FP8.

## Configuration Options

All options live under `additional_config.tensorbridge`:

| Option | Default | Values / Description |
| --- | --- | --- |
| `preset` | *(none)* | `tb_nvfp4_fp8`, `tb_nvfp4_fp8_g128`, `tb_nvfp4_megamoe` |
| `linear` | `false` | Enable TensorBridge for NVFP4 dense linear layers |
| `moe` | `false` | Enable TensorBridge for NVFP4 routed experts |
| `exact` | `true` | Exact E4M3 weight bridge; `false` requires FPMA-aware weights |
| `linear_quant` | `group_128` | `group_128`, `per_token` (NVFP4 dense linear activations) |
| `moe_dispatch_dtype` | `fp8` | `fp8`, `bf16` (EP dispatch dtype; `fp8` requires `group_128` gate/up) |
| `moe_gate_up_quant` | `group_128` | `group_128`, `per_token` |
| `moe_down_quant` | `group_128` | `group_128`, `per_token` |
| `moe_backend_mode` | `standard` | `standard`, `fused_swiglu`, `megamoe`, `humming_grouped` |
| `moe_nvlink_ep` | `false` | Enable single-node `NVLinkSymmetricEP` (requires active EP group) |
| `moe_shared_reserve_sms` | `0` | SMs withheld from persistent MegaMoE (`(total_sms - reserve_sms) * ctas_per_sm` CTAs) for concurrent local `shared_experts` |
| `moe_max_tokens_per_rank` | `256` | Chunk capacity per rank for `NVLinkSymmetricEP` MegaMoE (`T > cap` uses single-shot Contiguous Prefill) |
| `fp8_to_nvfp4` | `false` | On-the-fly GPU conversion of block-128 FP8 routed experts to ModelOpt NVFP4 |
| `nvfp4_quant_mode` | `mse_7ulp` | On-the-fly quantizer: `mse_7ulp`, `four_over_six`, `bridge_aware`, `rtn` |
| `strict_moe` | `false` | Raise instead of falling back to native vLLM MoE on unsupported layers |
| `fp8_moe_baseline` | `None` | Native FP4 baseline adapters for benchmarking: `marlin_nvfp4`, `humming_nvfp4`, `humming_mxfp4`, `humming_mxfp4_g128` |

## Checkpoint Compatibility vs. On-the-Fly Quantization

- **Pre-quantized NVFP4 checkpoints (`ModelOpt`, `ModelOptMixed`, `compressed-tensors`)**: TensorBridge
  accepts any valid NVFP4 checkpoint layout (`packed uint8 [N, K/2]`, `weight_scale fp8_e4m3 [N, K/16]`,
  `weight_scale_2` / `weight_global_scale fp32`) regardless of how the checkpoint was produced offline
  (RTN, AWQ, GPTQ, Four-Over-Six, MSE-opt, etc.). Compressed-tensors stores inverse global scales;
  ModelOpt stores multipliers. Fused shards with differing global scales are aligned via
  `serving.share_global_scale`.
- **On-the-fly FP8 → NVFP4 conversion (`tensorbridge/quant/modelopt.py`)**: Used only when quantizing
  a block-128 FP8 checkpoint during model load (`fp8_to_nvfp4: true` or `fp8_moe_baseline`). Produces
  100% standard ModelOpt NVFP4 tensors with zero inference/kernel overhead:
  - `mse_7ulp` *(default)*: Searches 7 adjacent E4M3 ULP scale bytes (`offsets -2..+4` around `amax/6`,
    covering both mild outlier clipping and Four-Over-Six) against the consuming backend's decoded levels.
  - `four_over_six`: Evaluates scales targeting E2M1 magnitudes `6` and `4` per 16-element group.
  - `bridge_aware`: Keeps RTN scales (`amax/6`) and rounds codes against the consuming bridge's actual
    E4M3 decoded levels.
  - `rtn`: Original ModelOpt round-to-nearest conversion.
  - TensorBridge scores candidates against its selected `exact` or `fpma` bridge decoder; native W4A16
    baselines (`marlin_nvfp4`, `humming_nvfp4`) score direct dequantization. All-zero groups retain exact
    zero scales and codes, and FP32 staging tiles are capped to keep conversion peak VRAM under 512 MiB.

## Execution Layouts & Scope

| Execution | Expert Layout | Activation Preparation & Reduction |
| --- | --- | --- |
| Local / TP | `indexed` | Native `moe_align_block_size` and weighted sum |
| AG/RS & DeepEP HT | `contiguous` | Fused scatter/quantize (or `HummingGroupedExperts` for `tb_nvfp4_fp8`); native unpermute/combine |
| DeepEP LL | `masked` | Native expert buffers and device counts; delegated weighted combine |
| Single-node NVLink EP | `masked` (`T ≤ cap`) / `contiguous` (Prefill) | `NVLinkSymmetricEP` MegaMoE (`4×` route-striped dispatch + direct NVLink Down epilogue) + early `shared_experts` overlap |

Supported scope: Hopper SM90, BF16 layer input, group-16 NVFP4 weights, grouped dimensions divisible by
128, no bias / LoRA / EPLB / PP, and `apply_router_weight_on_input=False`. Standard unclamped SiLU uses
the fused SwiGLU + FP8 epilogue (or adaptive `BN=128` + interleaved SwiGLU for large Prefill); other
gated/non-gated activations reuse vLLM's modular activations.
