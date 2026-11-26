# MegaMoE & Multi-Token Prefill

TensorBridge's `tb_nvfp4_megamoe` preset provides single-node Expert Parallelism (EP ≤ 8) over NVLink P2P
with **zero external communication dependencies** (no DeepEP, no NVSHMEM). Built on PyTorch's native
`torch.distributed._symmetric_memory`, `NVLinkSymmetricEP` executes both latency-sensitive Decode and
throughput-bound Prefill on **one shared HBM weight copy** (`w13` interleaved group-128 NVFP4 + `w2`
group-128 NVFP4).

---

## Architecture Overview

```text
Input BF16 [T_local, H] + topk_ids + topk_weights
  │
  ├─ T_eff <= max_tokens_per_rank (default 256):
  │    fp8.quantize_group_128(x)
  │    └─► sm90_w4a8_megamoe (ONE persistent kernel, zero cudaMemset, zero down_out HBM)
  │          1. 4× route-striped NVLink P2P Dispatch (128B vectorized FP8 + scale + slot atomic)
  │          2. Direct Ready-Ring Publisher (dynamic M-tiles per local expert)
  │          3. Unified Work Queue across all active SMs:
  │               • GateUp W4A8 WGMMA (BN=256) + fused SwiGLU + group-128 FP8 epilogue
  │               • Down   W4A8 WGMMA (BN=256) + direct 512B-warp NVLink P2P scatter epilogue
  │          4. Cooperative Weighted Combine Reduction → BF16 output
  │          5. Inline self-reset of touched counters/ring slots + 1-barrier NVLink sync
  │
  └─ T_eff > max_tokens_per_rank (large Prefill, eager):
       fp8.quantize_group_128(x) → NCCL AllGather(FP8, scales, routing)
       └─► Coalesced row-wise fp8.scatter_fp8 (early empty-token skip)
       └─► Adaptive GateUp:
             • Medium batches (selector BN=256): fused GateUp + SwiGLU + FP8
             • Large Prefill  (selector BN=128): unfused W4A8 GateUp (98.5% H20 peak)
                                                 + interleaved SwiGLU + FP8 quantization
       └─► Down W4A8 grouped GEMM (BN=128/256) → unpermute → NCCL ReduceScatter
```

### Key MegaMoE Design Choices

1. **Warp Specialization & Spill-Free Occupancy**:
   Each CTA runs 2 math warpgroups (`256` threads) and 1 producer/comm warpgroup (`128` threads) with
   explicit Hopper `setmaxnreg` budgets (`232` regs/thread for math, `40` for producer at `CTA=1`;
   `128`/`24` for spill-free `BM=8, CTA=2`).
2. **4× Route-Striped NVLink Dispatch**:
   All active CTAs cooperate on dispatching local tokens across NVLink P2P in 4× route-striped chunks
   with 128-byte vectorized copies (`uint4`) into each peer's symmetric `recv_fp8`/`recv_scale`/`recv_meta`
   slots, eliminating single-CTA dispatch bottlenecks.
3. **Direct 512B-Warp NVLink Down Epilogue (Zero `down_out` HBM Round-Trip)**:
   Instead of writing Down GEMM BF16 outputs to a local `[local_e * capacity, hidden]` HBM staging buffer
   and launching a separate NVLink Combine pass, each Down epilogue warp streams 512-byte BF16 slices
   directly over NVLink P2P into the owning rank's symmetric `combine_recv` buffer. On GLM-5.3 EP8
   (`32 × 2048 × 6144 × 2 B`), removing `down_out` also saves **805 MiB HBM per GPU**.
4. **Poisson M-Tile Selection & L2 Cache Hints**:
   `select_megamoe_config` sizes `BLOCK_M ∈ {8, 16, 24, 32, 48, 64}` from `avg_m + 2·√avg_m` so >95% of
   experts finish in one M-tile without re-reading NVFP4 weights. TMA loads apply `EVICT_FIRST` on
   one-shot weight tiles and `EVICT_LAST` on reused activation tiles.
5. **Early Multi-Stream `shared_experts` Overlap & Configurable `moe_shared_reserve_sms`**:
   For replicated/local `shared_experts` (`SP+EP` or `TP=1`), the vLLM plugin sets
   `mk_can_overlap_shared_experts = False` so vLLM launches `shared_experts` on `aux_stream` *before*
   `gate(x) + topk + fp8_quant + MegaMoE`. Users can optionally withhold `moe_shared_reserve_sms`
   (default `0`, configurable via `additional_config.tensorbridge.moe_shared_reserve_sms` or
   `--shared-reserve-sms`) so persistent MegaMoE launches `(total_sms - reserve_sms) * ctas_per_sm`
   CTAs without altering per-SM occupancy.

### Multi-Token Group-128 Prefill Optimizations

Both `tb_nvfp4_fp8_g128` and `tb_nvfp4_megamoe`'s Prefill path share two optimizations:
- **Coalesced Row-Wise `scatter_fp8`**: Reorders Triton tile layout so each warp copies contiguous
  128-byte slices per valid routed row with an early exit for invalid/empty routes (`480 µs → 100 µs`
  at 8K tokens on H20, **4.78×**).
- **Adaptive `BLOCK_N=128` + Interleaved SwiGLU+FP8**: Fused SwiGLU+FP8 requires `BLOCK_N=256` to hold
  both gate and up halves in one accumulator tile. At large Prefill M, `BLOCK_N=128` doubles wave
  parallelism and reaches 279.7 TFLOPS (**98.5%** of H20 FP8 peak). Because `w13` is already interleaved
  in 128-channel `[gate_128, up_128]` blocks, `serving.moe_forward_contiguous` and `NVLinkSymmetricEP`
  run unfused `BLOCK_N=128` directly on the same weight buffer followed by
  `fp8.quantize_moe(..., activation=True, interleaved_gate_up=True)` with zero weight duplication and
  bit-exact equivalence (`rel_diff = 0.00000`).

---

## Verified 8×H20 GLM-5.3 Performance

All numbers below are measured on an 8× NVIDIA H20 (SM90, 78 SMs, 96 GB HBM3, NVLink) node with CUDA 13.1,
PyTorch 2.13.0, and vLLM 0.32.0 on the real 78-layer **GLM-5.3** checkpoint (`E=256`, `top_k=8`, `H=6144`, `I=2048`).

### 1. Real Full-Model Decode (`DP=2, TP=4, EP=8, DCP=4`, ISL=80K, OSL=600, FP8 KV)

Measured via `benchmarks/bench_pd_prefill_decode.py --bench decode` (TPOT Mean in ms; lower is better):

| Concurrency | `humming_nvfp4` | `tb_nvfp4_fp8_g128` | `tb_nvfp4_fp8` | `humming_mxfp4_g128` | **`tb_nvfp4_megamoe`** | Peak Output (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|
| **C=1** | 23.58 | 23.48 | 22.68 | 25.36 | **21.72 (#1)** | 46.0 |
| **C=4** | 25.76 | 24.83 | 23.72 | 26.83 | **23.04 (#1)** | 173.6 |
| **C=8** | 26.40 | 25.77 | 24.44 | 25.14 | **23.55 (#1)** | 339.7 |
| **C=16** | 31.28 | 30.13 | 28.24 | 28.01 | **27.16 (#1)** | 589.1 |
| **C=32** | 38.91 | 35.67 | 34.33 | 33.68 | **33.40 (#1)** | 958.1 |
| **C=40** | 42.78 | 39.01 | 37.58 | 36.79 | **36.60 (#1)** | **1,010.0** |

`tb_nvfp4_megamoe` ranks **#1 at all 6 concurrencies** (`−0.5% to −14.4%` vs `humming_mxfp4_g128`,
`−2.6% to −4.2%` vs `tb_nvfp4_fp8`, `−7.9% to −14.4%` vs `humming_nvfp4`), breaking **1,010.0 tok/s**
aggregate output at `C=40` (`80K + 600`) while saving **+3.25–4.18 GiB** of KV cache capacity.

### 2. Real Full-Model Prefill (`DP=2, TP=4, EP=8`, `50K..320K × 0%/90%` Prefix Cache)

Measured via `benchmarks/bench_pd_prefill_decode.py --bench prefill` (TTFT Mean in ms; lower is better):

| ISL | Cache Hit | `humming_nvfp4` | `tb_nvfp4_fp8` | `humming_mxfp4_g128` | **`tb_nvfp4_fp8_g128`** | **`tb_nvfp4_megamoe`** |
|---:|---:|---:|---:|---:|---:|---:|
| 50K | 0% | 18,150 | 19,373 | 16,108 | **16,007 (#1)** | 16,090 |
| 50K | 90% | 2,110 | 1,969 | 1,889 | 1,869 | **1,840 (#1)** |
| 80K | 0% | 29,055 | 26,182 | 25,786 | **25,727 (#1)** | 25,949 |
| 80K | 90% | 3,354 | 3,072 | 3,015 | 2,998 | **2,975 (#1)** |
| 160K | 0% | 63,191 | 57,318 | **56,517** | 56,583 (+0.12%) | 56,960 |
| 160K | 90% | 7,491 | 6,888 | 6,802 | **6,749 (#1)** | 6,765 |
| 320K | 0% | 149,902 | 138,194 | **136,754** | 136,952 (+0.14%) | 137,693 |
| 320K | 90% | 20,342 | 19,166 | **18,943** | 18,990 (+0.25%) | 18,988 (+0.24%) |

TensorBridge's group-128 presets win **5/8** Prefill cells and sit within **0.12%–0.25%** of
`humming_mxfp4_g128` on the remaining 3, while beating `humming_nvfp4` by **6.7%–13.4%** across all 8/8 workloads.

### 3. Isolated 8-GPU EP8 MoE Layer vs External MegaMoE (`T=1..128`, Uniform)

Measured via `benchmarks/bench_moe_layer_ep8_compare.py` (CUDA-Graph median end-to-end latency in µs):

| `T / rank` | `sgl_pr53_sm90_fp8_megamoe` | `sgl_pr53_sm90_mxfp4_megamoe` | `humming_mxfp4_g128` | **`tb_nvfp4_megamoe`** | vs SGL FP8 | vs SGL MXFP4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 167.4 | 213.3 | 163.7 | **147.0 (#1)** | **1.14×** | **1.45×** |
| 2 | 226.5 | 288.0 | 191.7 | **173.4 (#1)** | **1.31×** | **1.66×** |
| 4 | 310.5 | 410.3 | 235.3 | **226.3 (#1)** | **1.37×** | **1.81×** |
| 8 | 407.3 | 531.1 | **293.4** | 295.7 | **1.38×** | **1.80×** |
| 16 | 421.5 | 543.0 | **315.6** | 322.4 | **1.31×** | **1.68×** |
| 32 | 446.9 | 553.9 | **361.4** | 379.6 | **1.18×** | **1.46×** |
| 64 | 541.7 | 646.5 | 453.2 | **426.1 (#1)** | **1.27×** | **1.52×** |
| 128 | 658.7 | 678.9 | 638.2 | **599.9 (#1)** | **1.10×** | **1.13×** |

`tb_nvfp4_megamoe` beats `sgl_pr53_sm90_mxfp4_megamoe` on all `T=1..128` by **1.13×–1.81×** and
`sgl_pr53_sm90_fp8_megamoe` by **1.10×–1.38×**, while achieving 27% lower relative RMS error (`0.1511` vs `0.2076`).

### 4. Real Full-Model Accuracy (WikiText-2 PPL & Official vLLM 5-Shot GSM8K)

Measured on the real 78-layer **GLM-5.3** checkpoint (`DP=8, TP=1, EP=8` for 4-bit modes with `mse_7ulp` and
`--strict-moe`; `DP=1, TP=8, EP=8` eager for `native_fp8`, BF16 KV cache):
- **WikiText-2 PPL**: `benchmarks/eval_wikitext_ppl.py` (`seqlen=2048`, 141 windows, BF16 head + `raw_logprobs`).
- **GSM8K**: unmodified upstream vLLM `tests/evals/gsm8k/gsm8k_eval.py` (`num_questions=1319`, `num_shots=5`,
  `max_tokens=256`, `temperature=0.0`, `seed=42`, `invalid_rate=0.00%` across all modes).

| Mode | WikiText-2 PPL | Δ PPL vs FP8 | GSM8K Accuracy (5-shot) | Δ GSM8K (pp) | GSM8K Throughput (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_fp8` (`TP=8, eager`) | 2.8027 | +0.0000 | 91.43% (1206/1319) | +0.00 | 166.4 |
| **`tb_nvfp4_megamoe`** | **2.8610** | **+0.0583** | **92.04% (1214/1319) (#1)** | **+0.61** | **1,089.5 (#1)** |
| **`tb_nvfp4_fp8_g128`** | **2.8596 (#1 4-bit)** | **+0.0568** | 91.13% (1202/1319) | −0.30 | 846.6 |
| `humming_nvfp4` (W4A16) | 2.8612 | +0.0584 | 91.89% (1212/1319) | +0.45 | 788.3 |
| `humming_mxfp4_g128` | 2.8973 | +0.0946 | 90.30% (1191/1319) | −1.14 | 781.9 |

`tb_nvfp4_megamoe` ranks **#1 on upstream vLLM GSM8K in both accuracy (`92.04%`, `+1.74 pp` over
`humming_mxfp4_g128`) and throughput (`1,089.5 tok/s`, `1.39×` vs `humming_mxfp4_g128`, `1.38×` vs
`humming_nvfp4`)**, while matching `tb_nvfp4_fp8_g128` and `humming_nvfp4` on WikiText-2 PPL (**38%–40% less
PPL degradation** than `humming_mxfp4_g128`).

---

## Code Map

| Component | Path |
| --- | --- |
| Persistent MegaMoE CUDA kernel | `tensorbridge/include/tensorbridge/impls/sm90_w4a8_megamoe.cuh` |
| MegaMoE host/device parameters | `tensorbridge/include/tensorbridge/common/megamoe_params.h` |
| NVLink P2P & atomic PTX helpers | `tensorbridge/include/tensorbridge/comm/nvlink_ptx.cuh` |
| C++ MegaMoE launcher | `tensorbridge/csrc/launcher/launcher_megamoe.hpp` |
| Python MegaMoE & fused SwiGLU API | `tensorbridge/megamoe.py` |
| Symmetric-memory NVLink EP executor | `tensorbridge/nvlink_ep.py` |
| vLLM 0.32 MoE integration & presets | `tensorbridge/vllm/moe.py`, `tensorbridge/vllm/presets.py` |
| 8-GPU verification & latency test | `tests/test_nvlink_ep_8gpu.py` |
