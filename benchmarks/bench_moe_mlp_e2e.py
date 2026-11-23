"""One rank's contribution to an EP8 expert MLP, from token-major BF16 to weighted token-major BF16.

All modes consume the same global top-k IDs. Native modular experts receive the rank-local expert map.
Masked paths group and gather BF16 rows on GPU. No collective or router GEMM.
Run --check-adapters for a small numerical test through the production loader and native kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from collections.abc import Callable
from pathlib import Path

import common
import torch
from bench_deepgemm import block_dequant, deep_gemm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.v1.worker.workspace import init_workspace_manager

from tensorbridge import megamoe, serving
from tensorbridge.gemm import MaskedRouting, fp8_nvfp4_grouped_gemm
from tensorbridge.quant import fp8, nvfp4
from tensorbridge.vllm.baselines import NativeFp4MoEMethod, convert_block_fp8
from tensorbridge.vllm.plugin import TensorBridgeFp8Config

ADAPTERS = ("marlin_nvfp4", "humming_nvfp4", "humming_mxfp4", "humming_mxfp4_g128")
MODES = (*ADAPTERS, "deepgemm", "tb_unfused", "tb_fused_swiglu")
Projection = tuple[torch.Tensor, torch.Tensor]


def checkpoint(experts: int, channels: int, width: int) -> Projection:
    from vllm.third_party.deep_gemm.utils import per_block_cast_to_fp8

    codes = torch.empty((experts, channels, width), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((experts, channels // 128, width // 128), device="cuda")
    for expert in range(experts):
        weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16) / math.sqrt(width)
        weight[: channels // 2].mul_(2)  # exercise unequal gate/up scales
        codes[expert], scales[expert] = per_block_cast_to_fp8(weight, False)
    return codes, scales


def reference(
    routes: common.Routes, hidden: torch.Tensor, scores: torch.Tensor, gate_up: Projection, down: Projection
) -> torch.Tensor:
    output = torch.zeros_like(hidden, dtype=torch.float32)
    local_routes = routes.sorted_routes.split(routes.counts.tolist())
    for expert, count in enumerate(routes.counts.tolist()):
        if not count:
            continue
        route_ids = local_routes[expert].cuda()
        token_ids = route_ids // scores.size(1)
        w13, w2 = (block_dequant(w[expert], s[expert], *w.shape[1:]) for w, s in (gate_up, down))
        gate, up = (hidden[token_ids].float() @ w13.T).chunk(2, dim=-1)
        values = (torch.nn.functional.silu(gate) * up) @ w2.T
        output.index_add_(0, token_ids, values * scores.flatten()[route_ids, None])
    return output


def native_layer(
    config: VllmConfig, mode: str, gate_up: Projection, down: Projection, top_k: int, ep: int
) -> RoutedExperts:
    experts, channels, hidden = gate_up[0].shape
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=config.parallel_config.data_parallel_size,
        ep_size=ep,
        sp_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=config.parallel_config.data_parallel_rank,
        ep_rank=config.parallel_config.data_parallel_rank,
        use_ep=ep > 1,
        all2all_backend=config.parallel_config.all2all_backend,
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=experts * ep,
        num_local_experts=experts,
        num_logical_experts=experts * ep,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=channels // 2,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        moe_backend="deep_gemm" if mode == "fp8_deepgemm_agrs" else "auto",
    )
    if mode in ADAPTERS:
        config.additional_config = {"tensorbridge": {"fp8_moe_baseline": mode}}
    elif mode in ("tb_nvfp4_fp8", "tb_nvfp4_fp8_g128"):
        config.additional_config = {"tensorbridge": {"preset": mode, "strict_moe": True}}
    elif mode == "fp8_deepgemm_agrs":
        config.additional_config = {}
    else:
        raise ValueError(f"Unknown modular MoE backend {mode}")
    with torch.device("cuda"):
        manager = ExpertMapManager(4096, top_k, experts * ep, 0, None, parallel, "linear", False)
        layer = RoutedExperts(
            "model.layers.0.mlp.experts",
            torch.bfloat16,
            moe,
            TensorBridgeFp8Config(weight_block_size=[128, 128]),
            manager,
        )
    if mode in ADAPTERS and (
        not isinstance(layer.quant_method, NativeFp4MoEMethod) or layer.w13_weight.is_cuda or layer.w2_weight.is_cuda
    ):
        raise AssertionError(f"{mode}: adapter selection or CPU staging failed")
    for dest, source in zip(
        (layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv),
        (*gate_up, *down),
        strict=True,
    ):
        dest.copy_(source)
    with device_loading_context(layer, torch.device("cuda")):
        layer.quant_method.process_weights_after_loading(layer)
    if not (layer.w13_weight.is_cuda and layer.w2_weight.is_cuda):
        raise AssertionError(f"{mode}: converted weights must stay on GPU after device_loading_context")
    return layer


def check_weights(mode: str, gate_up: Projection, down: Projection) -> list[float]:
    errors = []
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    for gated, (weight, block_scales) in zip((True, False), (gate_up, down)):
        packed, group_scale, multiplier = convert_block_fp8(weight, block_scales, mode, gated)
        nibbles = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2)
        decoded = levels[(nibbles & 7).long()] * torch.where(nibbles >= 8, -1, 1)
        if multiplier is None:
            scales = (group_scale.float() - 127).exp2().repeat_interleave(32, dim=-1)
        else:
            scales = (group_scale.float() * multiplier[:, None, None]).repeat_interleave(16, dim=-1)
        source = block_dequant(weight, block_scales, *weight.shape[1:])
        error = float(torch.linalg.vector_norm(decoded * scales - source) / torch.linalg.vector_norm(source))
        if not error < 0.17:
            raise AssertionError(f"{mode} weight relative RMS {error} exceeds 0.17")
        errors.append(error)
    return errors


def run_shape(config: VllmConfig, args: argparse.Namespace, tokens: int) -> None:
    small = args.check_adapters
    experts, hidden, intermediate, ep, top_k = (4, 256, 256, 1, 2) if small else (256, 6144, 2048, 8, 8)
    local_e = experts // ep
    torch.manual_seed(41)
    w13, w2 = checkpoint(local_e, intermediate * 2, hidden), checkpoint(local_e, hidden, intermediate)
    shape = common.MoeShape("gate_up", 1, intermediate * 2, hidden, experts, top_k, tokens, "zipf", 41, ep=ep)
    routes = common.route(shape)
    x = common.moe_inputs(shape)
    scores = torch.randn(tokens, top_k, device="cuda").softmax(-1)
    expected = reference(routes, x, scores, w13, w2)
    if small:
        for mode in ADAPTERS:
            layer = native_layer(config, mode, w13, w2, top_k, ep)
            if mode == "humming_mxfp4_g128":
                quant = layer.quant_method.moe_quant_config
                if (quant.quant_dtype != torch.float8_e4m3fn or quant.block_shape != [1, 128]
                        or quant.w1_humming_config.input_scale_group_size != 128
                        or quant.w2_humming_config.input_scale_group_size != 128):
                    raise AssertionError(f"{mode}: expected FP8 group-128 dispatch and both projections")
            actual = layer.quant_method.apply(layer, x, scores, routes.topk_ids, None, None)
            rel_rms = float(torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected))
            if not rel_rms < 0.32:
                raise AssertionError(f"{mode}: native top-{top_k} MLP output relative RMS {rel_rms}")
            print(
                json.dumps(
                    {"mode": mode, "weight_rel_rms": check_weights(mode, w13, w2), "native_forward_rel_rms": rel_rms}
                ),
                flush=True,
            )
        return
    capacity = routes.masked_capacity()
    flat_ids = routes.topk_ids.flatten()
    offsets_axis = torch.arange(local_e + 1, device="cuda", dtype=torch.int32)
    row_numbers = torch.arange(routes.rows, device="cuda")

    def prepare_mode(mode: str) -> tuple[dict[str, object], dict[str, Callable[[], torch.Tensor]]]:
        masked = mode not in ADAPTERS
        storage_rows = local_e * capacity if masked else routes.rows
        counts = routes.counts.cuda().int()
        masked_pos = routes.masked_positions()
        token_ids = routes.sorted_routes.cuda() // top_k
        scores_local = scores.flatten()[routes.sorted_routes.cuda(), None]
        gathered = torch.empty(storage_rows, hidden, device="cuda", dtype=torch.bfloat16) if masked else x
        final = torch.empty_like(x)
        if mode.startswith("tb_"):
            gate_codes, gate_scales, gate_global = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(
                *w13, is_gated_w13=True, bridge="exact"
            )
            down_codes, down_scales, down_global = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(*w2, bridge="exact")
            prepared = serving.prepare_moe(
                gate_codes,
                gate_scales,
                gate_global,
                down_codes,
                down_scales,
                down_global,
                8,
                experts,
                8,
                "exact",
                layout="masked",
                scale_convention="multiplier",
                gate_up_quant="group_128",
                down_quant="group_128",
                fuse_swiglu_fp8=mode != "tb_unfused",
            )
            buffers = {
                name: torch.empty(dims, dtype=dtype, device="cuda")
                for name, (dims, dtype) in serving.moe_buffer_specs(prepared, capacity).items()
            }
            inputs = buffers["input"], buffers["input_scale"]
            inter = buffers["intermediate"], buffers["intermediate_scale"]
            output = buffers["output"]
            routing = MaskedRouting(counts)

            def expert_mlp() -> torch.Tensor:
                fp8.quantize_moe(gathered, "group_128", expert_counts=counts, out=inputs)
                if mode == "tb_fused_swiglu":
                    megamoe.fp8_nvfp4_grouped_gemm_swiglu_fp8(
                        *inputs, prepared.w13, routes.rows, expert_counts=counts, out=inter
                    )
                else:
                    fp8_nvfp4_grouped_gemm(
                        *inputs, prepared.w13, routing, routes.rows, out=buffers["gate_up"], group_scales=True
                    )
                    fp8.quantize_moe(buffers["gate_up"], "group_128", activation=True, expert_counts=counts, out=inter)
                return fp8_nvfp4_grouped_gemm(*inter, prepared.w2, routing, routes.rows, out=output, group_scales=True)
        elif mode == "deepgemm":
            dg = deep_gemm()
            expected_m = math.ceil(routes.rows / local_e)
            gate_weight, down_weight = (
                (
                    codes,
                    dg.transform_sf_into_required_layout(
                        scales, *codes.shape[1:], (128, 128), num_groups=local_e, disable_ue8m0_cast=True
                    ),
                )
                for codes, scales in (w13, w2)
            )
            gate_output = torch.empty((local_e, capacity, 2 * intermediate), device="cuda", dtype=torch.bfloat16)
            output = torch.empty((local_e, capacity, hidden), device="cuda", dtype=torch.bfloat16)
            activation_pairs = [
                (
                    torch.empty((storage_rows, width), device="cuda", dtype=torch.float8_e4m3fn),
                    torch.empty((storage_rows, width // 128), device="cuda", dtype=torch.float32),
                )
                for width in (hidden, intermediate)
            ]

            def expert_mlp() -> torch.Tensor:
                for source, pair, weight, dest, gated in (
                    (gathered, activation_pairs[0], gate_weight, gate_output, False),
                    (gate_output.flatten(0, 1), activation_pairs[1], down_weight, output, True),
                ):
                    codes, scales = fp8.quantize_moe(
                        source, "group_128", expert_counts=counts, activation=gated, out=pair
                    )
                    width = codes.size(1)
                    scales = dg.transform_sf_into_required_layout(
                        scales.view(local_e, capacity, -1),
                        capacity,
                        width,
                        (1, 128),
                        num_groups=local_e,
                        disable_ue8m0_cast=True,
                    )
                    dg.m_grouped_fp8_gemm_nt_masked(
                        (codes.view(local_e, capacity, width), scales),
                        weight,
                        dest,
                        counts,
                        expected_m,
                        disable_ue8m0_cast=True,
                    )
                return output.flatten(0, 1)
        else:
            layer = native_layer(config, mode, w13, w2, top_k, ep)

            def expert_mlp() -> torch.Tensor:
                return layer.quant_method.apply(layer, x, scores, routes.topk_ids, None, None)

        def gather() -> torch.Tensor:
            # Rank zero owns [0, local_e). Fixed shape for capture, freshly group the actual ids each replay.
            sorted_ids, order = torch.sort(flat_ids, stable=True)
            offsets = torch.searchsorted(sorted_ids, offsets_axis)
            counts.copy_(offsets.diff())
            local_order = order[: routes.rows]
            expert_ids = sorted_ids[: routes.rows].long()
            masked_pos.copy_(expert_ids * capacity + row_numbers - offsets[expert_ids])
            token_ids.copy_(local_order // top_k)
            scores_local.copy_(scores.flatten()[local_order, None])
            gathered.index_copy_(0, masked_pos, x[token_ids])
            return gathered

        def reduce_rows(expert_output: torch.Tensor) -> torch.Tensor:
            reduced = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
            reduced.index_add_(0, token_ids, expert_output[masked_pos].float() * scores_local)
            return final.copy_(reduced)

        def forward() -> torch.Tensor:
            if masked:
                gather()
                return reduce_rows(expert_mlp())
            else:
                return expert_mlp()

        result = forward()
        rel_rms = float(torch.linalg.vector_norm(result.float() - expected) / torch.linalg.vector_norm(expected))
        if not rel_rms < (0.12 if mode == "deepgemm" else 0.32):
            raise AssertionError(f"{mode}: MLP output relative RMS {rel_rms}")
        if masked:
            layout = "masked capacity"
            gate_rows = down_rows = storage_rows
        else:
            kernels = layer.quant_method.moe_kernel.fused_experts
            layout = type(kernels).__name__
            if isinstance(kernels, HummingIndexedExperts):
                gate_args, down_args, _ = kernels.prepare_humming_moe_kwargs(routes.topk_ids, layer.expert_map, None)
                gate_rows = int(gate_args["num_tokens_padded"].item())
                down_rows = int(down_args["num_tokens_padded"].item())
            elif isinstance(kernels, MarlinExperts):
                # The v0.32 W4A16 Marlin launcher chooses its M tile from the expected local rows.
                estimate = math.ceil(tokens * local_e / experts) * top_k / local_e
                block_m = next((b for b in (8, 16, 32, 48, 64) if estimate / b < 0.9), 64)
                aligned = moe_align_block_size(
                    routes.topk_ids, block_m, experts, layer.expert_map, ignore_invalid_experts=True
                )
                gate_rows = down_rows = int(aligned[2].item())
            else:
                # Humming grouped-contiguous addresses compact local expert segments.
                gate_rows = down_rows = routes.rows
        row = dict(
            mode=mode,
            tokens=tokens,
            global_experts=experts,
            local_experts=local_e,
            top_k=8,
            valid_rows=routes.rows,
            padded_rows=gate_rows - routes.rows,
            down_padded_rows=down_rows - routes.rows,
            row_basis=layout,
            rel_rms=rel_rms,
        )
        if not masked:
            return row, {"total": forward}
        expert_output = expert_mlp().clone()
        return row, {
            "total": forward,
            "gather": gather,
            "expert_mlp": expert_mlp,
            "reduce": lambda: reduce_rows(expert_output),
        }

    # Keep closures (and thus graph inputs, weights and workspaces) alive through all replays.
    # Warm all experts before capture so vLLM has already sized its shared workspace.
    runs = {mode: prepare_mode(mode) for mode in args.modes}
    graphs = {}
    for mode, (row, calls) in runs.items():
        for phase, call in calls.items():
            for _ in range(3):
                call()
            graphs[mode, phase] = common.capture(lambda _: call(), args.iters, 1)
        if not args.kernel_breakdown and mode not in ADAPTERS:
            continue
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                calls["total"]()
                torch.cuda.synchronize()
            row["profile_kernel_us"] = {
                e.key: e.self_device_time_total
                for e in profile.key_averages()
                if e.device_type == torch.autograd.DeviceType.CUDA
            }
    keys = list(graphs)
    rounds = {key: [] for key in keys}
    for shift in range(args.rounds):
        for key in keys[shift % len(keys) :] + keys[: shift % len(keys)]:
            rounds[key].append(common.time_graph(graphs[key], args.iters))
    for mode, (row, _) in runs.items():
        row["round_us"] = rounds[mode, "total"]
        row["total_us"] = statistics.median(rounds[mode, "total"])
        row["breakdown_us"] = {
            phase: statistics.median(times)
            for (name, phase), times in rounds.items()
            if name == mode and phase != "total"
        }
        print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-adapters", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+", default=[128])
    parser.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--kernel-breakdown", action="store_true", help="also collect an eager CUDA kernel profile")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.tokens + [args.iters, args.rounds]) <= 0:
        parser.error("tokens, iters and rounds must be positive")
    torch.set_float32_matmul_precision("highest")
    config = VllmConfig()
    device = torch.device("cuda", 0)
    with tempfile.TemporaryDirectory() as temp, set_current_vllm_config(config):
        try:
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, distributed_init_method=(Path(temp) / "rank").as_uri()
            )
            initialize_model_parallel(1, 1)
            init_workspace_manager(device)
            with set_forward_context(None, config), torch.inference_mode():
                for tokens in [64] if args.check_adapters else args.tokens:
                    run_shape(config, args, tokens)
        finally:
            if torch.distributed.is_initialized():
                from vllm.distributed import destroy_model_parallel

                try:
                    destroy_model_parallel()
                finally:
                    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
