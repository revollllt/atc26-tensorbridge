# vLLM MoE integration

TensorBridge supplies NVFP4 expert computation through vLLM 0.32.0's modular MoE interface.
vLLM retains checkpoint parsing and sharding, routing, prepare/finalize, dispatch/combine,
TP reduction and shared-expert scheduling. The adapter is an explicit opt-in.

## Enable the adapter

Install TensorBridge with the vLLM extra and the CUDA major version used by PyTorch:

```bash
pip install --no-build-isolation '.[cu13,vllm]'
vllm serve MODEL --additional-config '{"tensorbridge":{"moe":true,"exact":true,"moe_dispatch_dtype":"fp8","moe_gate_up_quant":"group_128","moe_down_quant":"group_128"}}'
```

The checkpoint keeps its native quantization method, which vLLM detects; `"linear": true` also hands
its NVFP4 linear layers to TensorBridge, otherwise they keep their native method. The general plugin
entry point extends the compressed-tensors and ModelOpt configs; vLLM source files are not patched.

| Option under `additional_config.tensorbridge` | Default | Values |
| --- | --- | --- |
| `linear` | `false` | Enable NVFP4 linear layers |
| `moe` | `false` | Enable NVFP4 routed experts |
| `exact` | `true` | Exact bridge; `false` requires FPMA-aware weights |
| `linear_quant` | `per_token` | `per_token`, `group_128`; applies to NVFP4 linear layers |
| `moe_dispatch_dtype` | `fp8` | `fp8`, `bf16`; applies to EP dispatch |
| `moe_gate_up_quant` | `group_128` with FP8 EP dispatch; otherwise `per_token` | `per_token`, `group_128` |
| `moe_down_quant` | `per_token` | `per_token`, `group_128` |

Gate/up and down input quantization are independent and fixed at initialization. FP8 dispatch
requires gate/up group-128; an explicit incompatible setting fails at initialization. Experts
consume dispatched FP8 activations and scales directly. Combine remains BF16.

## Formats and execution

The adapter reuses native compressed-tensors, ModelOpt NVFP4 and ModelOpt mixed-recipe parameter
names, declarations and loaders. Ignored layers and other quantization recipes retain native
methods. Conversion runs once after TP/EP sharding. Prepared weights, scales and locks belong to
the layer; native MoE compute methods are not instantiated before selecting TensorBridge.

Compressed-tensors global scales are inverse scales; ModelOpt global scales are multipliers.
The kernels apply one global scale per expert (per dense weight), so fused shards with different global scales
(an expert's gate and up; q/k/v or gate/up of a dense layer) take the largest, and the other shards' E4M3 group
scales are multiplied by their own over the largest and rounded (`serving.share_global_scale`). vLLM's native
NVFP4 path takes the largest without changing the group scales. FPMA is checked with synthetic FPMA-aware weights.

| Execution | Expert layout | Activation preparation and reduction |
| --- | --- | --- |
| Local / TP | Indexed | Native alignment and weighted sum; reuse alignment when both GEMMs select the same BM |
| DeepEP HT | Contiguous | Fused BF16 scatter/quantize, or fused FP8 + group-128 scale scatter; native unpermute/combine |
| DeepEP LL | Masked | Native expert buffers, device counts and actual scale strides; delegated weighted combine |

Group-128 extends the existing group-scale mainloop to these three layouts. It applies each K128
activation scale to that group's partial accumulation. Per-token computation retains its existing
mainloop. Grouped Stream-K is not supported. Routes, offsets and counts stay on the GPU, and invalid
capacity is not read. Workspaces and permute scratch follow native per-ubatch/lane lifetimes and
support DBO as well as graph replay with changed routes.

The supported scope is Hopper SM90, BF16 layer input, group-16 NVFP4 weights and grouped dimensions
divisible by 128. Standard un-clamped SiLU runs the fused SiLU-mul quantizer; other gated and
non-gated MoE activations reuse vLLM's modular activation kernels. Layers outside this scope (bias,
router weights on input, LoRA, EPLB, PP, other EP backends or non-128-multiple local MoE shapes)
fall back to vLLM's native method with a single warning; prepared-weight snapshots remain outside this scope. Multi-node EP,
complete-model compressed-tensors/FPMA quality, multimodal quality and general task accuracy have
not been evaluated.
