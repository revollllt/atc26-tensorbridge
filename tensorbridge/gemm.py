"""`D = A @ B.T` with FP8 activations and NVFP4 weights, dense and grouped by expert."""

import dataclasses

import torch

from tensorbridge.jit import runtime
from tensorbridge.jit.codegen import GemmConfig, GroupedLayout
from tensorbridge.quant.nvfp4 import NVFP4Experts, NVFP4Weight


def fp8_nvfp4_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: NVFP4Weight,
    out: torch.Tensor | None = None,
    config: GemmConfig | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """BF16 `[M, N]` output of `(a * a_scale) @ dequant(weight).T`.

    `a` is FP8 E4M3 `[M, K]` with per-token FP32 scales `a_scale` `[M, 1]`
    (`tensorbridge.quant.fp8.quantize`), `weight` comes from
    `tensorbridge.quant.nvfp4.prepare`; `(N, K)` is `weight.shape`, whose tensors are padded to
    multiples of 128, while `a` and the output keep K and N columns. The kernel is selected from the shape
    alone and JIT-compiled on first use; pass `config` to pin it instead. The
    weight's bridge picks the kernel's table, not its config.

    `enable_pdl` launches the kernel with programmatic dependent launch: it may
    start before the previous kernel on the stream has finished and waits for it
    before reading any input, weights included, so the result is the same either
    way. It is faster only when that previous kernel lets its dependents launch
    early, as `tensorbridge.quant.fp8.quantize` does; after any other kernel it
    costs about half a microsecond.
    """
    n, k = weight.shape
    if config is None:
        config = runtime.select_config(a.size(0), n, k, device=a.get_device(), bridge=weight.bridge)
    kernel_id = runtime.get_kernel(n, k, config, bridge=weight.bridge, device=a.get_device())
    return torch.ops.tensorbridge.launch_kernel(
        kernel_id, a, a_scale, weight.weight, weight.scale, weight.global_scale, weight.locks, out,
        enable_pdl,
    )


@dataclasses.dataclass(frozen=True)
class IndexedRouting:
    """Expert-sorted route IDs, padded per expert to `block_m`; input row is `id // top_k`."""

    sorted_ids: torch.Tensor  # int32 [padded route capacity]
    expert_ids: torch.Tensor  # int32 [route capacity / block_m]
    num_tokens_padded: torch.Tensor  # int32 [1], actual padded routes on the device
    block_m: int
    top_k: int


@dataclasses.dataclass(frozen=True)
class ContiguousRouting:
    """The rows of `a` grouped by expert, unpadded: expert `e` owns rows
    `expert_offsets[e] .. expert_offsets[e + 1]`."""

    expert_offsets: torch.Tensor  # int64 [E + 1], first 0, last the row count R


@dataclasses.dataclass(frozen=True)
class MaskedRouting:
    """`E` blocks of `capacity = a.size(0) // E` rows: expert `e` owns the first
    `expert_counts[e]` rows of block `e`."""

    expert_counts: torch.Tensor  # int32 [E]


Routing = IndexedRouting | ContiguousRouting | MaskedRouting


def fp8_nvfp4_grouped_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    experts: NVFP4Experts,
    routing: Routing,
    m: int,
    out: torch.Tensor | None = None,
    config: GemmConfig | None = None,
    enable_pdl: bool = False,
    *, group_scales: bool = False,
) -> torch.Tensor:
    """BF16 output of each expert's `(a_e * a_scale_e) @ dequant(experts.expert(e)).T` in one launch.
    `a` is FP8 E4M3. FP32 scales are `[rows, 1]`, or `[rows, K/128]` with `group_scales=True`.
    With group-128 input, weight scales use `[E, K/128, N, 8]`, as `serving.prepare_moe` prepares them. Masked
    group scales can remain strided `[E, capacity, K/128]`. `m` only hints the routed row count to
    config selection; capacity follows the actual tensors. Route metadata stays on the device."""
    sorted_ids = expert_ids = num_tokens_padded = expert_layout = None
    top_k = 1
    layout: GroupedLayout
    if isinstance(routing, IndexedRouting):
        layout = "indexed"
        sorted_ids, expert_ids = routing.sorted_ids, routing.expert_ids
        num_tokens_padded, top_k = routing.num_tokens_padded, routing.top_k
    elif isinstance(routing, ContiguousRouting):
        layout, expert_layout = "contiguous", routing.expert_offsets
    else:
        layout, expert_layout = "masked", routing.expert_counts
    num_experts, n, k = experts.shape
    if config is None:
        config = runtime.select_config(m, n, k, num_experts, layout, device=a.get_device(),
                                       bridge=experts.bridge, top_k=top_k, group_scales=group_scales)
    if isinstance(routing, IndexedRouting) and routing.block_m != config.block_m:
        raise ValueError("indexed routing must be aligned to the selected block_m")
    kernel_id = runtime.get_grouped_kernel(n, k, num_experts, layout, config, bridge=experts.bridge,
                                           device=a.get_device(), group_scales=group_scales)
    return torch.ops.tensorbridge.launch_grouped_kernel(
        kernel_id, a, a_scale, experts.weight, experts.scale, experts.global_scale, experts.locks,
        sorted_ids, expert_ids, num_tokens_padded, expert_layout, top_k, out, enable_pdl,
    )
