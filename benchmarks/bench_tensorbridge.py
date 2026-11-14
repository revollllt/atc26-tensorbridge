#!/usr/bin/env python3
"""TensorBridge: NVFP4 weights (E4M3 scale per 16) x FP8 E4M3 activations with per-token scales.

    python benchmarks/bench_tensorbridge.py --shape_n 8192 --shape_k 8192
    python benchmarks/bench_tensorbridge.py --shape_n 512 --shape_k 7168 --num_experts 256 --top_k 8

A dense workload times `fp8_nvfp4_gemm` on the FPMA bridge. An MoE workload times the grouped kernel
in each expert layout (indexed, contiguous, masked). The entry times the kernel.
"""

import dataclasses
import json
from typing import Literal

import common
import torch
from common import DenseShape, Entry, MoeShape, PreparedCall, Shape, Unsupported

from tensorbridge.gemm import (
    ContiguousRouting,
    IndexedRouting,
    MaskedRouting,
    fp8_nvfp4_gemm,
    fp8_nvfp4_grouped_gemm,
)
from tensorbridge.jit import runtime
from tensorbridge.quant import fp8, nvfp4

BYTES_PER_WEIGHT = 0.5625  # E2M1 codes and an E4M3 scale per 16 weights
# The measured kernel. The library default bridge is "exact".
BRIDGE = "fpma"
LAYOUTS = ("indexed", "contiguous", "masked")
Layout = Literal["indexed", "contiguous", "masked"]


def dense_entry() -> Entry:
    """`tensorbridge`: the dense FPMA kernel."""

    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        assert isinstance(shape, DenseShape)
        torch.manual_seed(0)
        weight_hp = torch.randn(shape.n, shape.k, device="cuda", dtype=torch.bfloat16)  # [N, K]
        inputs_hp = torch.randn(shape.m, shape.k, device="cuda", dtype=torch.bfloat16)  # [M, K]
        packed, scale, global_scale = nvfp4.quantize(weight_hp)
        weight = nvfp4.prepare(packed, scale, global_scale, bridge=BRIDGE)
        # Per copy: int32 [N_pad, K_pad / 8] codes, fp8 [K_pad / 16, N_pad] scales, fp32 [1] global
        # scale, int32 [1024] Stream-K locks. N and K are rounded up to multiples of 128.
        weight_copies = [weight] + [
            dataclasses.replace(weight, weight=weight.weight.clone(), scale=weight.scale.clone(),
                                global_scale=weight.global_scale.clone(), locks=torch.zeros_like(weight.locks))
            for _ in range(copies - 1)
        ]
        activation, activation_scale = fp8.quantize(inputs_hp)  # fp8 [M, K], fp32 [M, 1]
        config = runtime.select_config(shape.m, shape.n, shape.k, bridge=BRIDGE)

        def run(copy_index: int) -> torch.Tensor:
            return fp8_nvfp4_gemm(activation, activation_scale, weight_copies[copy_index], config=config)

        return PreparedCall(run, None, 2.0 * shape.m * shape.n * shape.k,
                            common.dense_traffic(shape, BYTES_PER_WEIGHT, 1), json.dumps(dataclasses.asdict(config)))

    return Entry("tensorbridge", "tensorbridge", "w4a8_nvfp4_float8e4m3", common.git_commit(common.REPO_ROOT),
                 "nvfp4_g16", BYTES_PER_WEIGHT, "dense", prepare, role="candidate")


def moe_entry(layout: Layout) -> Entry:
    """`tensorbridge_<layout>`: the grouped FPMA kernel over the rank's local experts.

    Indexed routes are aligned as vLLM aligns them (`moe_align_block_size`, with an expert map that
    drops other EP ranks' routes) and gathered by the kernel. Contiguous and masked inputs come
    sorted by expert; masked ones sit in blocks of `common.MASKED_ALIGNMENT`-aligned capacity.
    """

    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        assert isinstance(shape, MoeShape)
        routes = common.route(shape)
        experts, n, k, rows = shape.local_experts, shape.n, shape.k, routes.rows
        if rows == 0:
            return Unsupported("no route reaches this rank")
        config = runtime.select_config(rows, n, k, experts, layout, bridge=BRIDGE,
                                       top_k=shape.source_top_k if layout == "indexed" else 1)
        # Quantized a chunk of experts at a time (each chunk with its own global scale): whole-rank
        # weights can exceed 2^31 elements.
        parts = []
        for chunk in common.moe_weight_chunks(shape):  # bf16 [experts in the chunk, N, K]
            packed, scales, global_scale = nvfp4.quantize(chunk.reshape(-1, k))
            parts.append(nvfp4.prepare_experts(packed.view(-1, n, k // 2), scales.view(-1, n, k // 16),
                                               global_scale.expand(chunk.size(0)), bridge=BRIDGE))
        weight = nvfp4.NVFP4Experts(torch.cat([part.weight for part in parts]),
                                    torch.cat([part.scale for part in parts]),
                                    torch.cat([part.global_scale for part in parts]), parts[0].locks, BRIDGE)
        del parts
        inputs_hp = common.moe_inputs(shape)
        weight_copies = [weight] + [
            nvfp4.NVFP4Experts(weight.weight.clone(), weight.scale.clone(), weight.global_scale.clone(),
                               torch.zeros_like(weight.locks), BRIDGE) for _ in range(copies - 1)
        ]
        # Sorted rows: route r of the expert-major order reads input row `sorted_routes[r] // source_top_k`.
        sorted_inputs_hp = inputs_hp[(routes.sorted_routes // shape.source_top_k).cuda()]
        activations, activation_scales = fp8.quantize(sorted_inputs_hp)  # fp8 [rows, K], fp32 [rows, 1]
        if layout == "indexed":
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

            if shape.ep > 1:  # only the routes of this rank's experts, in its local expert indices
                sorted_ids, expert_ids, padded = common.ep_alignment(shape, config.block_m)(routes.topk_ids)
            else:
                sorted_ids, expert_ids, padded = moe_align_block_size(routes.topk_ids, config.block_m, experts)[:3]
            activation, activation_scale = fp8.quantize(inputs_hp)
            routing = IndexedRouting(sorted_ids, expert_ids, padded, config.block_m, shape.source_top_k)
            output_rows, read_rows = shape.tokens * shape.top_k, shape.input_rows
        elif layout == "contiguous":
            activation, activation_scale = activations, activation_scales
            routing = ContiguousRouting(routes.bounds.cuda())  # int64 [E + 1]
            output_rows, read_rows = rows, rows
        else:
            capacity = routes.masked_capacity()
            positions = routes.masked_positions()
            activation = torch.zeros(experts * capacity, k, dtype=activations.dtype, device="cuda")
            activation_scale = torch.ones(experts * capacity, 1, device="cuda")
            activation[positions], activation_scale[positions] = activations, activation_scales
            routing = MaskedRouting(routes.counts.cuda().int())
            output_rows, read_rows = experts * capacity, rows
        output = torch.empty(output_rows, n, dtype=torch.bfloat16, device="cuda")

        def run(copy_index: int) -> torch.Tensor:
            return fp8_nvfp4_grouped_gemm(activation, activation_scale, weight_copies[copy_index], routing, rows,
                                          output, config)

        return PreparedCall(run, None, 2.0 * rows * n * k,
                            common.moe_traffic(shape, routes, BYTES_PER_WEIGHT, read_rows),
                            json.dumps(dataclasses.asdict(config)))

    return Entry(f"tensorbridge_{layout}", "tensorbridge", f"w4a8_nvfp4_float8e4m3_{layout}",
                 common.git_commit(common.REPO_ROOT), "nvfp4_g16", BYTES_PER_WEIGHT, "moe", prepare, layout,
                 "candidate")


def entries(layouts: tuple[Layout, ...] = LAYOUTS) -> list[Entry]:
    """The dense entry and one MoE entry per layout."""
    return [dense_entry(), *(moe_entry(layout) for layout in layouts)]


def main() -> None:
    parser = common.bench_parser(__doc__)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    args = parser.parse_args()
    runtime.load_launcher()
    common.run_bench(entries(tuple(args.layouts)), args, {})


if __name__ == "__main__":
    main()
