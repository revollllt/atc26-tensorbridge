#!/usr/bin/env python3
"""Marlin (as vLLM ships it): NVFP4 weights (E4M3 scale per 16) x BF16 activations, the weight-only
NVFP4 kernel vLLM runs on Hopper. Expert entries call its native MoE GEMM on prearranged
contiguous or masked BF16 rows; routing/alignment preparation is outside timing.

    python benchmarks/bench_marlin.py --shape_n 8192 --shape_k 8192
"""

import json

import common
import torch
from common import DenseShape, Entry, MoeShape, PreparedCall, Shape, Unsupported

BYTES_PER_WEIGHT = 0.5625


def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        rand_marlin_weight_nvfp4_like,
    )

    assert isinstance(shape, DenseShape)
    torch.manual_seed(0)
    activation = (torch.randn(shape.m, shape.k, device="cuda") * 0.1).bfloat16()  # [M, K]
    dense = (torch.randn(shape.n, shape.k, device="cuda") * 0.1).bfloat16()  # [N, K]
    # weight_reference: the dequantized weights transposed, [K, N] (vLLM's test_marlin_gemm).
    weight_reference, qweight, scales, global_scale = rand_marlin_weight_nvfp4_like(dense, 16, input_dtype=None)
    workspace = marlin_make_workspace_new(activation.device)
    weight_copies = [(qweight, scales)] + [(qweight.clone(), scales.clone()) for _ in range(copies - 1)]

    def run(copy_index: int) -> torch.Tensor:
        copy_qweight, copy_scales = weight_copies[copy_index]
        return apply_fp4_marlin_linear(activation, copy_qweight, copy_scales, global_scale, workspace, shape.n,
                                       shape.k, bias=None, input_dtype=None)

    def check() -> tuple[bool, str]:
        return common.check_output(run(0), activation.float() @ weight_reference.float())

    return PreparedCall(run, check, 2.0 * shape.m * shape.n * shape.k,
                        common.dense_traffic(shape, BYTES_PER_WEIGHT, 2), "")


def moe_entry(layout: str) -> Entry:
    """One native Marlin expert GEMM on prearranged contiguous/masked BF16 rows.

    Marlin consumes aligned row IDs; alignment is prepared outside timing, as for the other entries.
    No activation, router weighting, reduction or communication is included.
    """
    def prepare_moe(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_nvfp4_like
        from vllm.scalar_type import scalar_types

        assert isinstance(shape, MoeShape)
        routes = common.route(shape)
        if routes.rows == 0:
            return Unsupported("no route reaches this rank")
        inputs = common.moe_inputs(shape)[(routes.sorted_routes // shape.source_top_k).cuda()]
        positions = (routes.masked_positions() if layout == "masked"
                     else torch.arange(routes.rows, device="cuda"))
        input_rows = shape.local_experts * routes.masked_capacity() if layout == "masked" else routes.rows
        activation = torch.zeros(input_rows, shape.k, device="cuda", dtype=torch.bfloat16)
        activation[positions] = inputs
        reference = torch.empty(routes.rows, shape.n, device="cuda", dtype=torch.float32)
        packed_weights, weight_scales, global_scales = [], [], []
        bounds = routes.bounds
        expert = 0
        for chunk in common.moe_weight_chunks(shape):
            for weight in chunk:
                restored, packed, scales, global_scale = rand_marlin_weight_nvfp4_like(weight, 16, input_dtype=None)
                packed_weights.append(packed)
                weight_scales.append(scales)
                global_scales.append(global_scale)
                first, last = int(bounds[expert]), int(bounds[expert + 1])
                reference[first:last] = inputs[first:last].float() @ restored.float()
                expert += 1
        weight = torch.stack(packed_weights)
        scales = torch.stack(weight_scales)
        global_scale = torch.stack(global_scales).flatten()
        del packed_weights, weight_scales, global_scales, chunk, restored, packed
        weight_copies = [(weight, scales, global_scale)] + [
            (weight.clone(), scales.clone(), global_scale.clone()) for _ in range(copies - 1)
        ]
        # The same aligned row IDs address either compact rows or valid rows of expert-capacity blocks.
        rows_per_expert = routes.masked_capacity() if layout == "masked" else routes.rows / shape.local_experts
        block_m = 64
        for candidate in (8, 16, 32, 48, 64):
            if rows_per_expert / candidate < 0.9:
                block_m = candidate
                break
            else:
                continue
        padded_counts = (routes.counts + block_m - 1) // block_m * block_m
        sorted_ids = torch.full((int(padded_counts.sum()),), input_rows, dtype=torch.int32)
        expert_ids = torch.repeat_interleave(torch.arange(shape.local_experts, dtype=torch.int32),
                                            padded_counts // block_m).cuda()
        padded_bounds = torch.cat((torch.zeros(1, dtype=torch.int64), padded_counts.cumsum(0)))
        cpu_positions = positions.cpu().int()
        for expert in range(shape.local_experts):
            first, last = int(bounds[expert]), int(bounds[expert + 1])
            start = int(padded_bounds[expert])
            sorted_ids[start:start + last - first] = cpu_positions[first:last]
        sorted_ids = sorted_ids.cuda()
        padded = torch.tensor([sorted_ids.numel()], device="cuda", dtype=torch.int32)
        topk_weights = torch.ones(input_rows, 1, device="cuda", dtype=torch.float32)
        workspace = get_marlin_workspace(activation.device)
        output = torch.empty(input_rows, shape.n, device="cuda", dtype=torch.bfloat16)

        def run(copy_index: int) -> torch.Tensor:
            copy_weight, copy_scales, copy_global = weight_copies[copy_index]
            return ops.moe_wna16_marlin_gemm(
                activation, output, copy_weight, None, copy_scales, None, copy_global, None, workspace,
                sorted_ids, expert_ids, padded, topk_weights, block_m, 1, False, scalar_types.float4_e2m1f,
                input_rows, shape.n, shape.k, False, True, False,
            )

        def check() -> tuple[bool, str]:
            return common.check_output(run(0)[positions], reference)

        traffic = routes.active_experts * shape.n * shape.k * BYTES_PER_WEIGHT
        traffic += routes.rows * (shape.k + shape.n) * 2
        return PreparedCall(run, check, 2.0 * routes.rows * shape.n * shape.k, traffic,
                            json.dumps({"moe_block_size": block_m, "activation_dtype": "bfloat16",
                                        "row_alignment": "prepared outside timing"}))

    from importlib.metadata import version

    return Entry(f"marlin_nvfp4a16_{layout}", "marlin", "w4a16_nvfp4_bfloat16", f"vllm {version('vllm')}",
                 "nvfp4_g16", BYTES_PER_WEIGHT, "moe", prepare_moe, layout)


def entries() -> list[Entry]:
    from importlib.metadata import version

    return [Entry("marlin_nvfp4a16", "marlin", "w4a16_nvfp4_bfloat16", f"vllm {version('vllm')}", "nvfp4_g16",
                  BYTES_PER_WEIGHT, "dense", prepare),
            *(moe_entry(layout) for layout in ("contiguous", "masked"))]


def main() -> None:
    args = common.bench_parser(__doc__).parse_args()
    common.run_bench(entries(), args, {})


if __name__ == "__main__":
    main()
