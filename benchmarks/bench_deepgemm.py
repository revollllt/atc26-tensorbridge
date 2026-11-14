#!/usr/bin/env python3
"""DeepGEMM (deepseek-ai/DeepGEMM, as vLLM vendors it): FP8 E4M3 weights with a scale per 128 x 128
block x FP8 E4M3 activations with a scale per 1 x 128 group, the only operand formats it has on SM90.

    python benchmarks/bench_deepgemm.py --shape_n 8192 --shape_k 8192
    python benchmarks/bench_deepgemm.py --shape_n 512 --shape_k 7168 --num_experts 256 --top_k 8

An MoE workload runs DeepGEMM's m-grouped GEMM in its contiguous layout (every expert's rows padded
to its alignment) and its masked layout.
"""

import json
import math
from types import ModuleType

import common
import torch
from common import DenseShape, Entry, MoeShape, PreparedCall, Shape, Unsupported

BYTES_PER_WEIGHT = 1 + 4 / (128 * 128)  # FP8 and an FP32 scale per 128 x 128 block
RECIPE = "w8a8_float8e4m3_block128"
LAYOUTS = ("contiguous", "masked")


def deep_gemm() -> ModuleType:
    """vLLM's DeepGEMM, set to the SMs of the current context (`runtime.get_num_sms`)."""
    from vllm.third_party import deep_gemm as dg

    from tensorbridge.jit import runtime

    dg.set_num_sms(runtime.get_num_sms(torch.cuda.current_device()))
    return dg


def version() -> str:
    from importlib.metadata import version as package_version

    from vllm.third_party import deep_gemm as dg

    return f"{dg.__version__} (vllm {package_version('vllm')})"


def block_dequant(weight: torch.Tensor, scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """fp32 `[..., N, K]` of FP8 `weight` and its `[..., N / 128, K / 128]` block scales."""
    return weight.float() * scale.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[..., :n, :k]


def dense_entry() -> Entry:
    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        from vllm.third_party.deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

        assert isinstance(shape, DenseShape)
        dg = deep_gemm()
        torch.manual_seed(0)
        weight_hp = torch.randn(shape.n, shape.k, device="cuda", dtype=torch.bfloat16)  # [N, K]
        inputs_hp = torch.randn(shape.m, shape.k, device="cuda", dtype=torch.bfloat16)  # [M, K]
        weight, weight_scale = per_block_cast_to_fp8(weight_hp, False)  # fp8 [N, K], fp32 [N / 128, K / 128]
        reference_weight = block_dequant(weight, weight_scale, shape.n, shape.k)
        weight_scale = dg.transform_sf_into_required_layout(weight_scale, shape.n, shape.k, (128, 128),
                                                            disable_ue8m0_cast=True)
        activation, activation_scale = per_token_cast_to_fp8(inputs_hp, False)  # fp8 [M, K], fp32 [M, K / 128]
        reference_input = activation.float() * activation_scale.repeat_interleave(128, dim=1)[:, :shape.k]
        activation_scale = dg.transform_sf_into_required_layout(activation_scale, shape.m, shape.k, (1, 128),
                                                                disable_ue8m0_cast=True)
        weight_copies = [(weight, weight_scale)] + [(weight.clone(), weight_scale.clone()) for _ in range(copies - 1)]
        output = torch.empty(shape.m, shape.n, device="cuda", dtype=torch.bfloat16)

        def run(copy_index: int) -> torch.Tensor:
            dg.fp8_gemm_nt((activation, activation_scale), weight_copies[copy_index], output, disable_ue8m0_cast=True)
            return output

        def check() -> tuple[bool, str]:
            return common.check_output(run(0), reference_input @ reference_weight.t())

        # Activations carry an FP32 scale per 128 of K.
        traffic = common.dense_traffic(shape, BYTES_PER_WEIGHT, 1) + shape.m * (shape.k // 128 * 4 - 4)
        return PreparedCall(run, check, 2.0 * shape.m * shape.n * shape.k, traffic, "")

    return Entry("deepgemm_fp8", "deepgemm", RECIPE, version(), "fp8_block128", BYTES_PER_WEIGHT, "dense", prepare)


def moe_entry(layout: str) -> Entry:
    """The m-grouped GEMM over all local experts, its inputs sorted by expert: contiguous pads every
    expert's rows to DeepGEMM's alignment, masked takes blocks of `common.MASKED_ALIGNMENT`-aligned
    capacity per expert. The gated input of a down call is vLLM's SiLU-mul and 1 x 128 group
    quantization (`per_token_group_quant_fp8`)."""

    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        from vllm.third_party.deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

        assert isinstance(shape, MoeShape)
        dg = deep_gemm()
        routes = common.route(shape)
        experts, n, k, rows = shape.local_experts, shape.n, shape.k, routes.rows
        if rows == 0:
            return Unsupported("no route reaches this rank")
        # Cast a chunk of experts at a time and dequantized per expert for the reference: whole-rank weights can
        # exceed 2^31 elements, whose one-piece cast ran out of memory.
        casts = [per_block_cast_to_fp8(chunk.reshape(-1, k), False) for chunk in common.moe_weight_chunks(shape)]
        weight = torch.cat([cast for cast, _ in casts]).view(experts, n, k)
        block_scale = torch.cat([scale for _, scale in casts]).view(experts, math.ceil(n / 128), math.ceil(k / 128))
        del casts
        inputs_hp = common.moe_inputs(shape)
        weight_scale = dg.transform_sf_into_required_layout(block_scale, n, k, (128, 128), num_groups=experts,
                                                            disable_ue8m0_cast=True)
        weight_copies = [(weight, weight_scale)] + [(weight.clone(), weight_scale.clone()) for _ in range(copies - 1)]
        sorted_inputs_hp = inputs_hp[(routes.sorted_routes // shape.source_top_k).cuda()]
        activations, activation_scales = per_token_cast_to_fp8(sorted_inputs_hp, False)  # [rows, K], [rows, K / 128]
        bounds = routes.bounds
        reference = torch.empty(rows, n, device="cuda")
        for expert in (routes.counts > 0).nonzero().flatten().tolist():
            first, last = int(bounds[expert]), int(bounds[expert + 1])
            inputs = activations[first:last].float() * activation_scales[first:last].repeat_interleave(
                128, dim=1)[:, :k]
            reference[first:last] = inputs @ block_dequant(weight[expert], block_scale[expert], n, k).t()

        sorted_experts = torch.repeat_interleave(torch.arange(experts), routes.counts)
        capacity = routes.masked_capacity()
        if layout == "contiguous":
            alignment = dg.get_mk_alignment_for_contiguous_layout()
            padded_counts = (routes.counts + alignment - 1) // alignment * alignment
            padded_bounds = torch.cat((torch.zeros(1, dtype=torch.int64), padded_counts.cumsum(0)))
            positions = (padded_bounds[sorted_experts] + torch.arange(rows) - bounds[sorted_experts]).cuda()
            storage_rows = int(padded_counts.sum())
        else:
            positions = routes.masked_positions()
            storage_rows = experts * capacity
        activation = torch.zeros(storage_rows, k, dtype=activations.dtype, device="cuda")
        activation_scale = torch.ones(storage_rows, k // 128, device="cuda")
        activation[positions], activation_scale[positions] = activations, activation_scales
        if layout == "contiguous":
            metadata = (padded_bounds[:-1] + routes.counts).cuda().int()  # each expert's end of valid rows
            activation_scale = dg.transform_sf_into_required_layout(activation_scale, storage_rows, k, (1, 128),
                                                                    disable_ue8m0_cast=True)
        else:
            activation = activation.view(experts, capacity, k)
            activation_scale = dg.transform_sf_into_required_layout(
                activation_scale.view(experts, capacity, k // 128), capacity, k, (1, 128), num_groups=experts,
                disable_ue8m0_cast=True)
            metadata = routes.counts.cuda().int()
        output = torch.empty((*activation.shape[:-1], n), dtype=torch.bfloat16, device="cuda")
        expected_m = math.ceil(rows / experts)

        def run(copy_index: int) -> torch.Tensor:
            if layout == "contiguous":
                dg.m_grouped_fp8_gemm_nt_contiguous((activation, activation_scale), weight_copies[copy_index], output,
                                                   metadata, disable_ue8m0_cast=True, use_psum_layout=True)
            else:
                dg.m_grouped_fp8_gemm_nt_masked((activation, activation_scale), weight_copies[copy_index], output,
                                               metadata, expected_m, disable_ue8m0_cast=True)
            return output

        def check() -> tuple[bool, str]:
            return common.check_output(run(0).view(-1, n)[positions], reference)

        traffic = (routes.active_experts * n * k * BYTES_PER_WEIGHT + rows * (k + k // 128 * 4) + rows * n * 2)
        return PreparedCall(run, check, 2.0 * rows * n * k, traffic, json.dumps({"storage_rows": storage_rows}))

    return Entry(f"deepgemm_{layout}", "deepgemm", f"{RECIPE}_{layout}", version(), "fp8_block128",
                 BYTES_PER_WEIGHT, "moe", prepare, layout)


def entries() -> list[Entry]:
    return [dense_entry(), *(moe_entry(layout) for layout in LAYOUTS)]


def main() -> None:
    args = common.bench_parser(__doc__).parse_args()
    common.run_bench(entries(), args, {})


if __name__ == "__main__":
    main()
