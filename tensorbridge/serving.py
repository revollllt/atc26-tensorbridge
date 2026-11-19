"""The part of an inference-framework integration that does not depend on the framework.

A framework adapter (`vllm_plugin.py`) declares the checkpoint's parameters, picks the layers TensorBridge serves
and decides when to call this module:

    prepare_linear       once per linear layer, when its weights are loaded
    prepare_moe          once per MoE layer, after checkpoint sharding
    precompile_prepared  once all layers are prepared, to batch-compile kernels for `[1 .. max_tokens]`
    autotune_prepared    once the M a CUDA graph will capture are known, before capture
    linear               in the layer's forward, from the framework's custom op
    moe_forward_*        two grouped GEMMs around activation and quantization, into caller buffers

Autotune is opt-in and off by default: `TENSORBRIDGE_AUTOTUNE=1` turns it on; otherwise every M is served by the
selection rule. While a CUDA graph is being captured, `linear` and `moe_forward_*` count the M that autotune never
measured, since capture freezes whatever config they got.
"""

import dataclasses
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

import torch

from tensorbridge import experimental
from tensorbridge.gemm import ContiguousRouting, IndexedRouting, MaskedRouting, fp8_nvfp4_gemm, fp8_nvfp4_grouped_gemm
from tensorbridge.jit import autotune, codegen, runtime
from tensorbridge.jit.codegen import Bridge, GemmConfig
from tensorbridge.quant import fp8, nvfp4


@dataclasses.dataclass(frozen=True, order=True)
class DenseShape:
    """One prepared dense linear shape on `device`."""

    device: int
    n: int
    k: int
    bridge: Bridge
    input_quant: fp8.InputQuant = "per_token"


@dataclasses.dataclass(frozen=True, order=True)
class GroupedShape:
    """One prepared projection: layer routes/token and input gather top-k are distinct for down."""

    device: int
    n: int
    k: int
    num_experts: int
    top_k: int  # layer routes per token, for translating capture T to routed R
    input_top_k: int  # indexed input row = route_id // input_top_k; down uses 1
    layout: Literal["indexed", "contiguous", "masked"]
    bridge: Bridge
    dispatch_ranks: int
    num_global_experts: int
    input_quant: fp8.InputQuant = "per_token"


DenseTunedKey = tuple[int, int, int, int, int, Bridge] | tuple[int, int, int, int, int, Bridge, fp8.InputQuant]

# Every dense weight shape prepared in this process (`DenseShape`, or legacy 4-tuple from tests).
prepared_shapes: set[DenseShape | tuple[int, int, int, Bridge]] = set()
# Linear layers prepared in this process (a fused layer counts once).
prepared_layers = 0
# Dense `(device, available SMs, N, K, M, bridge[, input_quant])` that autotune serves.
tuned: set[DenseTunedKey] = set()
# Calls captured in a CUDA graph with an (N, K, M) outside `tuned`, and those already warned about.
captured_untuned = 0
warned_untuned: set[DenseTunedKey | tuple[GroupedShape, int, int]] = set()
prepared_grouped: set[GroupedShape] = set()
tuned_grouped: set[tuple[GroupedShape, int, int]] = set()  # shape, available SMs, routed rows
precompiled_dense: set[tuple[DenseShape, int, int]] = set()  # shape, available SMs, max_tokens
precompiled_grouped: set[tuple[GroupedShape, int, int]] = set()  # shape, available SMs, max_tokens
# Stream-isolated Stream-K lock arrays (`int32 [1024]`), keyed by `(device, cuda_stream)`.
stream_locks: dict[tuple[int, int], torch.Tensor] = {}


def autotune_enabled() -> bool:
    return os.environ.get("TENSORBRIDGE_AUTOTUNE") == "1"


def locks_for_current_stream(fallback: torch.Tensor) -> torch.Tensor:
    """Return a Stream-K lock array isolated per CUDA stream so concurrent microbatches (DBO) never
    share mutable lock words. During CUDA graph capture, reuse an already-allocated stream buffer or
    fall back to `fallback` without allocating inside the capture."""
    device = fallback.get_device()
    stream_handle = torch.cuda.current_stream(device).cuda_stream
    key = (device, stream_handle)
    if (existing := stream_locks.get(key)) is not None:
        return existing
    if torch.cuda.is_current_stream_capturing():
        return fallback
    allocated = torch.zeros(1024, dtype=torch.int32, device=fallback.device)
    stream_locks[key] = allocated
    return allocated


def normalize_dense_shape(entry: DenseShape | tuple[int, int, int, Bridge]) -> DenseShape:
    if isinstance(entry, DenseShape):
        return entry
    device, n, k, bridge = entry
    return DenseShape(device, n, k, bridge, "per_token")


def share_global_scale(
    weight_scale: torch.Tensor, global_scales: torch.Tensor, shard_sizes: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """One global scale for the shards of a fused weight, as the kernel applies one per weight (per expert).

    `weight_scale` is E4M3 `[..., N, K / 16]` with the shards concatenated along N, and `global_scales` the
    shards' multipliers `[..., S]`. The shared global scale is their largest; every other shard's group scales
    are multiplied by its own over the largest and rounded to the nearest E4M3, which keeps their weights up to
    that rounding. Shards that already share their global scale are returned unchanged. vLLM requantizes
    per-tensor FP8 fused weights to their largest scale the same way (`requantize_with_max_scale`), but keeps
    NVFP4 group scales as they are, which scales the other shards' weights by the largest over their own.
    """
    if sum(shard_sizes) != weight_scale.shape[-2] or len(shard_sizes) != global_scales.shape[-1]:
        raise ValueError(f"shards {list(shard_sizes)} do not match the {weight_scale.shape[-2]} channels and "
                         f"{global_scales.shape[-1]} global scales")
    shared = global_scales.amax(-1, keepdim=True)
    if torch.equal(global_scales, shared.expand_as(global_scales)):
        return weight_scale, shared.squeeze(-1)
    sizes = torch.tensor(shard_sizes, device=global_scales.device)
    ratios = (global_scales / shared).repeat_interleave(sizes, dim=-1)  # [..., N]
    weight_scale = (weight_scale.float() * ratios.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return weight_scale, shared.squeeze(-1)


def prepare_linear(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bridge: Bridge = "exact",
    scale_convention: Literal["inverse", "multiplier"] = "inverse",
    shard_sizes: Sequence[int] | None = None,
    *,
    input_quant: fp8.InputQuant = "per_token",
) -> nvfp4.NVFP4Weight:
    """The kernel layout of an NVFP4 linear layer, for the `bridge` table (the FPMA table for
    FPMA-aware checkpoints, the exact one for standard NVFP4 ones) and `input_quant` (`per_token`
    or `group_128`).

    `weight_packed` is uint8 `[N, K / 2]` and `weight_scale` E4M3 `[N, K / 16]`, the shards of a
    fused layer concatenated along N; `weight_global_scale` holds one FP32 value per shard:
    compressed-tensors stores the `inverse` of the global scale, ModelOpt the `multiplier`. Shards
    with different global scales are brought to a shared one (`share_global_scale`), which needs
    their output widths, `shard_sizes`.
    """
    global prepared_layers
    shard_scales = weight_global_scale.float()
    global_scales = 1.0 / shard_scales if scale_convention == "inverse" else shard_scales
    if shard_sizes is not None:
        weight_scale, global_scale = share_global_scale(weight_scale, global_scales, shard_sizes)
    elif torch.equal(global_scales, global_scales[:1].expand_as(global_scales)):
        global_scale = global_scales[0]
    else:
        raise ValueError("fused shards with different global scales need their shard_sizes")
    weight = (experimental.prepare(weight_packed, weight_scale, global_scale.view(1), bridge=bridge)
              if input_quant == "group_128"
              else nvfp4.prepare(weight_packed, weight_scale, global_scale.view(1), bridge=bridge))
    n, k = weight.shape
    device = weight.weight.get_device()
    if input_quant == "per_token":
        prepared_shapes.add((device, n, k, bridge))
    else:
        prepared_shapes.add(DenseShape(device, n, k, bridge, input_quant))
    prepared_layers += 1
    return weight


def precompile_prepared(max_tokens: int) -> int:
    """Batch-compile every unique kernel that `select_config` picks across `1 .. max_tokens` on the
    current device for all prepared dense and grouped shapes, eliminating online NVRTC stalls on
    variable-length prefill or chunked-prefill batches. Returns the number of newly compiled kernels."""
    if max_tokens < 1:
        return 0
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("precompile builds kernels: call it before CUDA graph capture, not during it")
    device = torch.cuda.current_device()
    num_sms = runtime.get_num_sms(device)
    # Sample every integer M up to 256 (fine decode/small-chunk steps) and every 8-token tile boundary above it.
    dense_ms = list(range(1, min(max_tokens, 256) + 1))
    if max_tokens > 256:
        dense_ms.extend(range(264, max_tokens + 1, 8))
        if dense_ms[-1] != max_tokens:
            dense_ms.append(max_tokens)
    sources: list[codegen.KernelSource] = []
    dense_targets: list[tuple[DenseShape, GemmConfig]] = []
    for raw_shape in sorted(prepared_shapes, key=normalize_dense_shape):
        shape = normalize_dense_shape(raw_shape)
        if shape.device != device or (shape, num_sms, max_tokens) in precompiled_dense:
            continue
        group_scales = shape.input_quant == "group_128"
        configs = dict.fromkeys(
            runtime.select_config(m, shape.n, shape.k, device=device, bridge=shape.bridge, group_scales=group_scales)
            for m in dense_ms
        )
        for config in configs:
            source = (codegen.generate_pgroup(shape.n, shape.k, config, True, shape.bridge)
                      if group_scales else codegen.generate(shape.n, shape.k, config, 0, "dense", shape.bridge))
            sources.append(source)
            dense_targets.append((shape, config))
        precompiled_dense.add((shape, num_sms, max_tokens))

    grouped_targets: list[tuple[GroupedShape, GemmConfig]] = []
    for grouped_shape in sorted(prepared_grouped):
        if grouped_shape.device != device or (grouped_shape, num_sms, max_tokens) in precompiled_grouped:
            continue
        group_scales = grouped_shape.input_quant == "group_128"
        rows = sorted({
            max(1, (grouped_shape.dispatch_ranks * m * grouped_shape.top_k * grouped_shape.num_experts
                    + grouped_shape.num_global_experts - 1) // grouped_shape.num_global_experts)
            for m in dense_ms
        })
        configs = dict.fromkeys(
            runtime.select_config(row, grouped_shape.n, grouped_shape.k, grouped_shape.num_experts,
                                  grouped_shape.layout, device=device, bridge=grouped_shape.bridge,
                                  top_k=grouped_shape.input_top_k, group_scales=group_scales)
            for row in rows
        )
        for config in configs:
            source = (codegen.generate_pgroup(grouped_shape.n, grouped_shape.k, config, False, grouped_shape.bridge,
                                              grouped_shape.num_experts, grouped_shape.layout)
                      if group_scales else codegen.generate(grouped_shape.n, grouped_shape.k, config,
                                                            grouped_shape.num_experts, grouped_shape.layout,
                                                            grouped_shape.bridge))
            sources.append(source)
            grouped_targets.append((grouped_shape, config))
        precompiled_grouped.add((grouped_shape, num_sms, max_tokens))

    if not sources:
        return 0
    compiled_before = autotune.stats.compiled
    errors = autotune.compile_kernels(sources)
    if errors:
        failed_source, error = next(iter(errors.items()))
        raise RuntimeError(f"failed to precompile {failed_source.name}") from error
    # Register into the launcher's module table now so the first eager launch is a pure lookup.
    from tensorbridge.jit import compiler

    for shape, config in dense_targets:
        if shape.input_quant == "group_128":
            runtime.register_device_kernel(
                device, codegen.generate_pgroup(shape.n, shape.k, config, True, shape.bridge),
                compiler.INCLUDE_DIR, shape.n, shape.k, config,
                activation_scale_groups=codegen.pad_to_unit(shape.k) // 128, scale_m_major=True, block_scales=True,
            )
        else:
            runtime.get_kernel(shape.n, shape.k, config, bridge=shape.bridge, device=device)
    for grouped_shape, config in grouped_targets:
        runtime.get_grouped_kernel(
            grouped_shape.n, grouped_shape.k, grouped_shape.num_experts, grouped_shape.layout, config,
            bridge=grouped_shape.bridge, device=device, group_scales=grouped_shape.input_quant == "group_128",
        )
    return autotune.stats.compiled - compiled_before


def dense_tuned_key(
    device: int, num_sms: int, n: int, k: int, m: int, bridge: Bridge, input_quant: fp8.InputQuant,
) -> DenseTunedKey:
    if input_quant == "per_token":
        return (device, num_sms, n, k, m, bridge)
    else:
        return (device, num_sms, n, k, m, bridge, input_quant)


def autotune_prepared(ms: Sequence[int]) -> None:
    """Autotune every `(N, K)` prepared on the current device for the M in `ms`, the sizes a CUDA
    graph will capture. Call it before capture; shapes already autotuned for all of `ms` are
    skipped, so an adapter may call it once after model loading or per layer."""
    if not autotune_enabled() or not ms:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("autotune times kernels: call it before CUDA graph capture, not during it")
    device = torch.cuda.current_device()
    num_sms = runtime.get_num_sms(device)
    pending_dense = [
        shape for shape in sorted({normalize_dense_shape(entry) for entry in prepared_shapes})
        if shape.device == device and any(
            dense_tuned_key(device, num_sms, shape.n, shape.k, m, shape.bridge, shape.input_quant) not in tuned
            for m in ms
        )
    ]
    pending_grouped: list[tuple[GroupedShape, list[int]]] = []
    for grouped_shape in sorted(prepared_grouped):
        if grouped_shape.device != device:
            continue
        rows = sorted({
            (grouped_shape.dispatch_ranks * m * grouped_shape.top_k * grouped_shape.num_experts
             + grouped_shape.num_global_experts - 1) // grouped_shape.num_global_experts
            for m in ms if m > 0
        })
        if rows and not all((grouped_shape, num_sms, row) in tuned_grouped for row in rows):
            pending_grouped.append((grouped_shape, rows))

    # Batch-compile all candidate kernels across all shapes in a single process pool before timing.
    batch_sources: list[codegen.KernelSource] = []
    for shape in pending_dense:
        group_scales = shape.input_quant == "group_128"
        unmeasured = [
            m for m in dict.fromkeys(ms)
            if (device, num_sms, shape.n, shape.k, m, shape.bridge, group_scales, autotune.CANDIDATES)
            not in autotune.selected
            and str(m) not in autotune.read_cache(
                autotune.dense_cache_path(device, shape.n, shape.k, shape.bridge, group_scales))
        ]
        for m in unmeasured:
            for candidate in runtime.autotune_candidates(m, shape.n, shape.k, autotune.CANDIDATES, device,
                                                         group_scales=group_scales):
                batch_sources.append(
                    codegen.generate_pgroup(shape.n, shape.k, candidate, True, shape.bridge) if group_scales
                    else codegen.generate(shape.n, shape.k, candidate, 0, "dense", shape.bridge)
                )
    for grouped_shape, rows in pending_grouped:
        group_scales = grouped_shape.input_quant == "group_128"
        cached = autotune.read_cache(
            autotune.grouped_cache_path(device, grouped_shape.n, grouped_shape.k, grouped_shape.num_experts,
                                        grouped_shape.input_top_k, grouped_shape.layout, grouped_shape.bridge,
                                        group_scales)
        )
        for row in rows:
            if str(row) in cached:
                continue
            for candidate in runtime.autotune_candidates(
                row, grouped_shape.n, grouped_shape.k, autotune.CANDIDATES, device,
                grouped_shape.num_experts, grouped_shape.layout, group_scales,
            ):
                batch_sources.append(
                    codegen.generate_pgroup(grouped_shape.n, grouped_shape.k, candidate, False, grouped_shape.bridge,
                                            grouped_shape.num_experts, grouped_shape.layout) if group_scales
                    else codegen.generate(grouped_shape.n, grouped_shape.k, candidate, grouped_shape.num_experts,
                                          grouped_shape.layout, grouped_shape.bridge)
                )
    if batch_sources:
        autotune.compile_kernels(batch_sources)

    for shape in pending_dense:
        group_scales = shape.input_quant == "group_128"
        if group_scales:
            autotune.autotune(ms, shape.n, shape.k, device, shape.bridge, group_scales=True)
        else:
            autotune.autotune(ms, shape.n, shape.k, device, shape.bridge)
        tuned.update(dense_tuned_key(device, num_sms, shape.n, shape.k, m, shape.bridge, shape.input_quant)
                     for m in ms)
    for grouped_shape, rows in pending_grouped:
        autotune.autotune_grouped(
            rows, grouped_shape.n, grouped_shape.k, grouped_shape.num_experts, grouped_shape.input_top_k,
            grouped_shape.layout, device, grouped_shape.bridge,
            group_scales=grouped_shape.input_quant == "group_128",
        )
        tuned_grouped.update((grouped_shape, num_sms, row) for row in rows)


def linear(
    x: torch.Tensor, weight: nvfp4.NVFP4Weight, output: torch.Tensor,
    *, input_quant: fp8.InputQuant = "per_token",
) -> None:
    """`output = x @ dequant(weight).T` for BF16 `x` `[M, K]` into BF16 `output` `[M, N]`, with
    `x` quantized to FP8 (`per_token` or `group_128`). Both quantizers release their dependent
    early, so the GEMM uses programmatic dependent launch."""
    global captured_untuned
    active_weight = dataclasses.replace(weight, locks=locks_for_current_stream(weight.locks))
    if torch.cuda.is_current_stream_capturing() and autotune_enabled():
        n, k = active_weight.shape
        device = x.get_device()
        key = dense_tuned_key(device, runtime.get_num_sms(device), n, k, x.size(0), active_weight.bridge, input_quant)
        if key not in tuned:
            captured_untuned += 1
            if key not in warned_untuned:
                warned_untuned.add(key)
                warnings.warn(
                    f"TensorBridge: captured M={x.size(0)} of (N, K) = ({n}, {k}) without autotune; "
                    "it keeps the selection rule's config",
                    stacklevel=2,
                )
    if input_quant == "group_128":
        activations, activation_scale = fp8.quantize_group_128(x)
        experimental.fp8_nvfp4_group_quant_gemm(activations, activation_scale, active_weight, out=output,
                                                enable_pdl=True)
    else:
        activations, activation_scale = fp8.quantize(x)
        # The quantization kernel right before lets the GEMM launch early.
        fp8_nvfp4_gemm(activations, activation_scale, active_weight, out=output, enable_pdl=True)


@dataclasses.dataclass(frozen=True)
class NVFP4MoE:
    """Prepared projections with rule/autotune selection or explicitly pinned configs."""

    w13: nvfp4.NVFP4Experts  # [E, 2I, H] (gated) or [E, I, H] (non-gated)
    w2: nvfp4.NVFP4Experts  # [E, H, I]
    top_k: int
    num_global_experts: int
    dispatch_ranks: int
    layout: Literal["indexed", "contiguous", "masked"]
    configs: tuple[GemmConfig, GemmConfig] | None
    gate_up_quant: fp8.InputQuant = "per_token"
    down_quant: fp8.InputQuant = "per_token"
    is_gated: bool = True


class MoEBuffers(TypedDict):
    """Caller-owned contiguous buffers, with shapes and dtypes from `moe_buffer_specs`."""

    input: torch.Tensor
    input_scale: torch.Tensor
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    intermediate_scale: torch.Tensor
    output: torch.Tensor


# Custom intermediate activation callback `(output_bf16 [..., I], gate_up_bf16 [..., w13_n]) -> None`.
ActivationFn = Callable[[torch.Tensor, torch.Tensor], None]


def prepare_moe(
    w13_packed: torch.Tensor, w13_scale: torch.Tensor, w13_global_scale: torch.Tensor,
    w2_packed: torch.Tensor, w2_scale: torch.Tensor, w2_global_scale: torch.Tensor,
    top_k: int, num_global_experts: int | None = None, dispatch_ranks: int = 1,
    bridge: Bridge = "exact", *, configs: tuple[GemmConfig, GemmConfig] | None = None,
    layout: Literal["indexed", "contiguous", "masked"] = "indexed",
    scale_convention: Literal["inverse", "multiplier"] = "inverse",
    gate_up_quant: fp8.InputQuant = "per_token", down_quant: fp8.InputQuant = "per_token",
    is_gated: bool = True, layer_name: str = "MoE",
) -> NVFP4MoE:
    """Prepare NVFP4 experts after sharding. Global scales are `[E, 2]` (gated) or `[E, 1]` / `[E]`
    (non-gated) for w13 and `[E]` for down: compressed-tensors uses `inverse`, ModelOpt uses
    `multiplier`; a gated expert's gate and up are brought to a shared global scale
    (`share_global_scale`). The adapter chooses the layout and each projection's input quantization.
    `num_global_experts` defaults to the local count without EP. Omit `configs` for rule/autotune selection."""
    num_experts, w13_channels, half_hidden = w13_packed.shape
    shards = 2 if is_gated else 1
    half_inter = w13_channels // (2 * shards)
    if w13_channels % (2 * shards) != 0 or w2_packed.shape != (num_experts, half_hidden * 2, half_inter):
        raise ValueError("MoE gate/up and down projection shapes do not match")
    if configs is not None and any(config.use_stream_k for config in configs):
        raise ValueError("grouped GEMM does not support Stream-K")
    if scale_convention == "inverse":
        gate_up_globals = 1.0 / w13_global_scale.float()
        down_global = 1.0 / w2_global_scale.float()
    else:
        gate_up_globals = w13_global_scale.float()
        down_global = w2_global_scale.float()
    if is_gated:
        # Experts whose gate and up global scales differ get a shared one per expert.
        w13_scale, gate_up_global = share_global_scale(w13_scale, gate_up_globals, [w13_channels // 2] * 2)
    else:
        gate_up_global = gate_up_globals.reshape(num_experts)
    w13 = nvfp4.prepare_experts(w13_packed, w13_scale, gate_up_global, bridge=bridge)
    w2 = nvfp4.prepare_experts(w2_packed, w2_scale, down_global, bridge=bridge)
    experts, channels, width = w13.shape
    w13 = dataclasses.replace(w13, scale=(w13.scale.view(experts, width // 128, 8, channels)
                              .permute(0, 1, 3, 2).contiguous() if gate_up_quant == "group_128" else w13.scale))
    experts, channels, width = w2.shape
    w2 = dataclasses.replace(w2, scale=(w2.scale.view(experts, width // 128, 8, channels)
                             .permute(0, 1, 3, 2).contiguous() if down_quant == "group_128" else w2.scale))
    moe = NVFP4MoE(
        w13, w2, top_k, num_experts if num_global_experts is None else num_global_experts,
        dispatch_ranks, layout, configs, gate_up_quant, down_quant, is_gated,
    )
    if configs is None:
        prepared_grouped.update(moe_grouped_shapes(moe))
    return moe


def moe_grouped_shapes(moe: NVFP4MoE) -> tuple[GroupedShape, GroupedShape]:
    """The two projections share routed R; only indexed gate/up gathers from T token rows."""
    device = moe.w13.weight.get_device()
    experts, n13, k13 = moe.w13.shape
    _, n2, k2 = moe.w2.shape
    return (
        GroupedShape(device, n13, k13, experts, moe.top_k, moe.top_k if moe.layout == "indexed" else 1,
                     moe.layout, moe.w13.bridge, moe.dispatch_ranks, moe.num_global_experts, moe.gate_up_quant),
        GroupedShape(device, n2, k2, experts, moe.top_k, 1, moe.layout, moe.w2.bridge,
                     moe.dispatch_ranks, moe.num_global_experts, moe.down_quant),
    )


def record_grouped_capture(moe: NVFP4MoE, rows: int) -> None:
    """Count unmeasured automatic projection selections during capture; pinned configs opt out."""
    global captured_untuned
    if moe.configs is not None or not autotune_enabled() or not torch.cuda.is_current_stream_capturing():
        return
    device = moe.w13.weight.get_device()
    num_sms = runtime.get_num_sms(device)
    for shape in moe_grouped_shapes(moe):
        key = (shape, num_sms, rows)
        if key in tuned_grouped:
            continue
        captured_untuned += 1
        if key in warned_untuned:
            continue
        warned_untuned.add(key)
        warnings.warn(f"TensorBridge: captured grouped R={rows} of (N, K, E) = "
                      f"({shape.n}, {shape.k}, {shape.num_experts}) without autotune; "
                      "it keeps the selection rule's config", stacklevel=3)


def moe_rows(moe: NVFP4MoE, num_routes: int) -> int:
    """Expected local routed rows from global pre-dispatch `num_routes`, used as a scheduling hint."""
    return (num_routes * moe.w13.shape[0] + moe.num_global_experts - 1) // moe.num_global_experts


def moe_post_dispatch_rows(moe: NVFP4MoE, received_tokens: int) -> int:
    """Expected local routed rows on a rank after DeepEP HT deduplicated token dispatch.

    Each global token sends at most one copy to this rank, with probability `1 - (1 - p)^top_k`
    where `p = E_local / E_global` under uniform routing, and carries `top_k * p / (1 - (1 - p)^top_k)`
    local routes in expectation (at least 1 local route per received token, at most `received_tokens * top_k`)."""
    if received_tokens <= 0:
        return 0
    local_experts = moe.w13.shape[0]
    if local_experts >= moe.num_global_experts or moe.top_k <= 1:
        return received_tokens * moe.top_k
    local_share = local_experts / moe.num_global_experts
    hit_prob = 1.0 - (1.0 - local_share) ** moe.top_k
    expected = round(received_tokens * moe.top_k * local_share / hit_prob)
    return min(received_tokens * moe.top_k, max(received_tokens, expected))


def moe_block_sizes(moe: NVFP4MoE, rows: int) -> tuple[int, int]:
    """Query after autotune, before aligning either projection's routes."""
    if moe.configs is not None:
        return moe.configs[0].block_m, moe.configs[1].block_m
    gate_up, down = (runtime.select_config(rows, shape.n, shape.k, shape.num_experts, shape.layout, shape.device,
                                           bridge=shape.bridge, top_k=shape.input_top_k,
                                           group_scales=shape.input_quant == "group_128")
                     for shape in moe_grouped_shapes(moe))
    return gate_up.block_m, down.block_m


def moe_buffer_specs(
    moe: NVFP4MoE, num_tokens: int, *, unfused_activation: bool = False,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Buffers for all tokens this rank sees after dispatch, with capacity for every route.
    The actual local count stays on the device and may exceed its uniform-routing estimate."""
    rows = num_tokens * (moe.w13.shape[0] if moe.layout == "masked" else moe.top_k)
    input_rows = num_tokens if moe.layout == "indexed" else rows
    _, w13_channels, _ = moe.w13.shape
    _, hidden, intermediate = moe.w2.shape
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "input": ((input_rows, hidden), torch.float8_e4m3fn),
        "input_scale": ((input_rows, hidden // 128 if moe.gate_up_quant == "group_128" else 1), torch.float32),
        "gate_up": ((rows, w13_channels), torch.bfloat16),
        "intermediate": ((rows, intermediate), torch.float8_e4m3fn),
        "intermediate_scale": ((rows, intermediate // 128 if moe.down_quant == "group_128" else 1), torch.float32),
        "output": ((rows, hidden), torch.bfloat16),
    }
    if unfused_activation:
        specs["activated"] = ((rows, intermediate), torch.bfloat16)
    return specs


def moe_forward_indexed(
    moe: NVFP4MoE, hidden_states: torch.Tensor, routing13: IndexedRouting, routing2: IndexedRouting,
    buffers: MoEBuffers, input_scale: torch.Tensor | None = None,
    *, local_rows_hint: int | None = None, activation_fn: ActivationFn | None = None,
    activated_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize -> gate/up GEMM -> activation and quantize -> down GEMM.
    Returns `[T * top_k, H]` in route order; the caller aligns routes and combines top-k outputs.
    Each projection can use a different aligned BM. All buffers are supplied by the caller."""
    if moe.layout != "indexed" or routing13.top_k != moe.top_k or routing2.top_k != 1:
        raise ValueError("indexed MoE needs gate/up top_k routes and down top_k=1 routes")
    if not moe.is_gated and activation_fn is None:
        raise ValueError("non-gated MoE requires an explicit activation_fn")
    global_num_routes = hidden_states.size(0) * moe.top_k
    local_rows = moe_rows(moe, global_num_routes) if local_rows_hint is None else local_rows_hint
    record_grouped_capture(moe, local_rows)
    if input_scale is not None:
        inputs, scales = hidden_states, input_scale
    elif moe.gate_up_quant == "group_128":
        inputs, scales = fp8.quantize_moe(hidden_states, moe.gate_up_quant,
                                          out=(buffers["input"], buffers["input_scale"]))
    else:
        inputs, scales = fp8.quantize(hidden_states, out=(buffers["input"], buffers["input_scale"]))
    fp8_nvfp4_grouped_gemm(
        inputs, scales, moe.w13, routing13, local_rows,
        out=buffers["gate_up"], config=None if moe.configs is None else moe.configs[0], enable_pdl=input_scale is None,
        group_scales=moe.gate_up_quant == "group_128",
    )
    if activation_fn is not None:
        activated = buffers["gate_up"][:, :moe.w2.shape[2]] if activated_buffer is None else activated_buffer
        activation_fn(activated, buffers["gate_up"])
        if moe.down_quant == "group_128":
            fp8.quantize_moe(activated, moe.down_quant, out=(buffers["intermediate"], buffers["intermediate_scale"]))
        else:
            fp8.quantize(activated, out=(buffers["intermediate"], buffers["intermediate_scale"]))
    elif moe.down_quant == "group_128":
        fp8.quantize_moe(buffers["gate_up"], moe.down_quant, activation=True,
                         out=(buffers["intermediate"], buffers["intermediate_scale"]))
    else:
        fp8.silu_mul_quantize(buffers["gate_up"],
                              out=(buffers["intermediate"], buffers["intermediate_scale"]))
    return fp8_nvfp4_grouped_gemm(
        buffers["intermediate"], buffers["intermediate_scale"], moe.w2, routing2, local_rows,
        out=buffers["output"], config=None if moe.configs is None else moe.configs[1], enable_pdl=True,
        group_scales=moe.down_quant == "group_128",
    )


def moe_forward_contiguous(
    moe: NVFP4MoE, hidden_states: torch.Tensor, scatter_idx: torch.Tensor, routing: ContiguousRouting,
    buffers: MoEBuffers, input_scale: torch.Tensor | None = None, num_routes: int | None = None,
    *, local_rows_hint: int | None = None, activation_fn: ActivationFn | None = None,
    activated_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter-quantize (or scatter pre-quantized FP8) -> gate/up GEMM -> activation quantize -> down GEMM.
    Only rows below the device-side `expert_offsets[-1]` are valid in the expert-ordered output;
    the caller unpermutes, weights and combines them. No local row count is read on the host."""
    if moe.layout != "contiguous":
        raise ValueError("contiguous MoE needs a layer prepared for contiguous dispatch")
    if not moe.is_gated and activation_fn is None:
        raise ValueError("non-gated MoE requires an explicit activation_fn")
    capacity = buffers["input"].size(0)
    rows = (local_rows_hint if local_rows_hint is not None
            else moe_rows(moe, hidden_states.size(0) * moe.top_k if num_routes is None else num_routes))
    record_grouped_capture(moe, rows)
    if input_scale is not None:
        if hidden_states.data_ptr() == buffers["input"].data_ptr():
            inputs, scales = hidden_states, input_scale
            pdl_first = False
        else:
            inputs, scales = fp8.scatter_fp8(hidden_states, input_scale, scatter_idx,
                                             out=(buffers["input"], buffers["input_scale"]))
            pdl_first = True
    elif moe.gate_up_quant == "group_128":
        inputs, scales = fp8.quantize_moe(hidden_states, moe.gate_up_quant, scatter_idx=scatter_idx,
                                          out=(buffers["input"], buffers["input_scale"]))
        pdl_first = True
    else:
        inputs, scales = fp8.scatter_quantize(hidden_states, scatter_idx, capacity,
                                              out=(buffers["input"], buffers["input_scale"]))
        pdl_first = True
    fp8_nvfp4_grouped_gemm(
        inputs, scales, moe.w13, routing, rows,
        out=buffers["gate_up"], config=None if moe.configs is None else moe.configs[0], enable_pdl=pdl_first,
        group_scales=moe.gate_up_quant == "group_128",
    )
    valid_count = routing.expert_offsets[-1:]
    if activation_fn is not None:
        activated = (torch.empty((buffers["gate_up"].size(0), moe.w2.shape[2]),
                                 dtype=torch.bfloat16, device=buffers["gate_up"].device)
                     if activated_buffer is None else activated_buffer)
        activation_fn(activated, buffers["gate_up"])
        fp8.quantize_moe(activated, moe.down_quant, num_valid_rows=valid_count,
                         out=(buffers["intermediate"], buffers["intermediate_scale"]))
    elif moe.down_quant == "group_128":
        fp8.quantize_moe(buffers["gate_up"], moe.down_quant, activation=True,
                         num_valid_rows=valid_count,
                         out=(buffers["intermediate"], buffers["intermediate_scale"]))
    else:
        fp8.silu_mul_quantize(buffers["gate_up"], num_valid_rows=valid_count,
                              out=(buffers["intermediate"], buffers["intermediate_scale"]))
    return fp8_nvfp4_grouped_gemm(
        buffers["intermediate"], buffers["intermediate_scale"], moe.w2, routing, rows,
        out=buffers["output"], config=None if moe.configs is None else moe.configs[1], enable_pdl=True,
        group_scales=moe.down_quant == "group_128",
    )


def moe_forward_masked(
    moe: NVFP4MoE, hidden_states: torch.Tensor, routing: MaskedRouting, buffers: MoEBuffers,
    input_scale: torch.Tensor | None = None, num_routes: int | None = None,
    *, activation_fn: ActivationFn | None = None, activated_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume flattened expert buffers; return [E * capacity, H] for native LL combine."""
    if moe.layout != "masked":
        raise ValueError("masked MoE needs a layer prepared for masked dispatch")
    if not moe.is_gated and activation_fn is None:
        raise ValueError("non-gated MoE requires an explicit activation_fn")
    rows = hidden_states.size(0) if num_routes is None else moe_rows(moe, num_routes)
    record_grouped_capture(moe, rows)
    if input_scale is None:
        inputs, scales = fp8.quantize_moe(hidden_states, moe.gate_up_quant, expert_counts=routing.expert_counts,
                                          out=(buffers["input"], buffers["input_scale"]))
    else:
        inputs, scales = hidden_states, input_scale
    fp8_nvfp4_grouped_gemm(
        inputs, scales, moe.w13, routing, rows, out=buffers["gate_up"],
        config=None if moe.configs is None else moe.configs[0], enable_pdl=input_scale is None,
        group_scales=moe.gate_up_quant == "group_128",
    )
    if activation_fn is not None:
        experts = moe.w13.shape[0]
        capacity = buffers["gate_up"].size(0) // experts
        activated = (torch.empty((buffers["gate_up"].size(0), moe.w2.shape[2]),
                                 dtype=torch.bfloat16, device=buffers["gate_up"].device)
                     if activated_buffer is None else activated_buffer)
        activation_fn(activated.view(experts, capacity, -1), buffers["gate_up"].view(experts, capacity, -1))
        fp8.quantize_moe(activated, moe.down_quant, expert_counts=routing.expert_counts,
                         out=(buffers["intermediate"], buffers["intermediate_scale"]))
    else:
        fp8.quantize_moe(buffers["gate_up"], moe.down_quant, activation=True, expert_counts=routing.expert_counts,
                         out=(buffers["intermediate"], buffers["intermediate_scale"]))
    return fp8_nvfp4_grouped_gemm(
        buffers["intermediate"], buffers["intermediate_scale"], moe.w2, routing, rows, out=buffers["output"],
        config=None if moe.configs is None else moe.configs[1], enable_pdl=True,
        group_scales=moe.down_quant == "group_128",
    )
