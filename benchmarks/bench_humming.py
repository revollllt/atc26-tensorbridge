#!/usr/bin/env python3
"""humming (vllm-project/humming): MXFP4 weights (E2M1, E8M0 scale per 32) x FP8 E4M3 activations
with per-token scales, the configuration of humming's heuristics.

    python benchmarks/bench_humming.py --shape_n 8192 --shape_k 8192
    python benchmarks/bench_humming.py --shape_n 512 --shape_k 7168 --num_experts 256 --top_k 8

An MoE workload runs humming's grouped GEMM in each expert layout it has (indexed, contiguous, masked).
"""

import copy
import json
from typing import TYPE_CHECKING

import common
import torch
from common import DenseShape, Entry, MoeShape, PreparedCall, Shape, Unsupported

from tensorbridge.quant import fp8

if TYPE_CHECKING:
    from humming.layer import HummingLayer

BYTES_PER_WEIGHT = 0.53125  # E2M1 codes and an E8M0 scale per 32 weights
RECIPE = "w4a8_mxfp4_float8e4m3"
LAYOUTS = ("indexed", "contiguous", "masked")


def humming_version() -> str:
    """humming's version and the commit it was built from."""
    import humming
    from humming import _version

    return f"{humming.__version__} ({(_version.commit_id or '').removeprefix('g')})"


def humming_layer(weight_hp: torch.Tensor, input_scale_group_size: int = 0) -> tuple["HummingLayer", torch.Tensor]:
    """A humming layer of MXFP4 weights and FP8 activations from `weight_hp` BF16 `[N, K]` or `[E, N, K]`,
    and its dequantized weights (fp32, same shape) as the reference, as humming's own tests build it
    (`humming/testing/runner.py`). Activation scales: per token, or per `input_scale_group_size` K when
    it is not 0."""
    from humming.layer import HummingLayer
    from humming.schema import HummingWeightSchema

    num_experts = weight_hp.size(0) if weight_hp.dim() == 3 else None
    layer = HummingLayer(
        shape_n=weight_hp.size(-2), shape_k=weight_hp.size(-1),
        weight_config={"dtype": "float4e2m1", "scale_dtype": "float8e8m0", "group_size": 32},
        input_config={"dtype": "float8e4m3", "group_size": input_scale_group_size},
        num_experts=num_experts, torch_dtype=torch.bfloat16,
    ).cuda()
    tensors = HummingWeightSchema.quant_tensor(weight_hp, layer.weight_schema, torch.bfloat16)
    weight_reference = layer.weight_schema.dequant_tensors(tensors).float()
    layer.load_from_tensors(tensors)
    layer.transform()
    return layer, weight_reference


def humming_tuning(layer: "HummingLayer", rows: int, gemm_type: str) -> dict[str, object]:
    """The config humming's heuristics select for `rows` GEMM rows: the entry of its M-interval table
    `(lo, hi]` that holds `rows`, as its launcher picks it. Passing only this one entry compiles one
    kernel instead of the whole table."""
    from humming.tune import get_heuristics_config

    intervals = get_heuristics_config(layer.humming_config, gemm_type=gemm_type)
    return next(config for lo, hi, config in intervals if lo < rows <= hi)


def dense_entry() -> Entry:
    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        from humming.forward import may_process_input

        assert isinstance(shape, DenseShape)
        torch.manual_seed(0)
        layer, weight_reference = humming_layer(torch.randn(shape.n, shape.k, device="cuda", dtype=torch.bfloat16))
        layers = [layer] + [copy.deepcopy(layer) for _ in range(copies - 1)]
        inputs_hp = torch.randn(shape.m, shape.k, device="cuda", dtype=torch.bfloat16)  # [M, K]
        activation, _, token_scale = may_process_input(layer.humming_config, inputs_hp)  # fp8 [M, K]
        activation_scale = token_scale.view(-1, 1)  # fp32 [M, 1]
        tuning = json.dumps(humming_tuning(layer, shape.m, "dense"))

        def run(copy_index: int) -> torch.Tensor:
            return layers[copy_index](inputs=activation, input_scale=activation_scale, tuning_config=tuning)

        def check() -> tuple[bool, str]:
            return common.check_output(run(0), (activation.float() * activation_scale) @ weight_reference.t())

        return PreparedCall(run, check, 2.0 * shape.m * shape.n * shape.k,
                            common.dense_traffic(shape, BYTES_PER_WEIGHT, 1), tuning)

    return Entry("humming_mxfp4_fp8", "humming", RECIPE, humming_version(), "mxfp4_g32", BYTES_PER_WEIGHT,
                 "dense", prepare)


def moe_entry(layout: str) -> Entry:
    """humming's grouped GEMM over all local experts of an MoE layer: indexed gathers each route's
    input row through `sorted_ids` (gate_up reads a token's row once per expert; an EP rank's routes
    are aligned with an expert map, as vLLM's humming experts align them); contiguous and masked take
    the inputs sorted by expert, masked in blocks of `common.MASKED_ALIGNMENT`-aligned capacity per
    expert. Every call writes one output buffer."""
    gemm_type = "indexed" if layout == "indexed" else f"grouped_{layout}"

    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        from humming.testing.data import generate_moe_tensors

        assert isinstance(shape, MoeShape)
        routes = common.route(shape)
        experts, n, k, rows = shape.local_experts, shape.n, shape.k, routes.rows
        if rows == 0:
            return Unsupported("no route reaches this rank")
        weights_hp, inputs_hp = common.moe_operands(shape)
        layer, weight_reference = humming_layer(weights_hp)  # fp32 [E, N, K]
        layers = [layer] + [copy.deepcopy(layer) for _ in range(copies - 1)]
        sorted_inputs_hp = inputs_hp[(routes.sorted_routes // shape.source_top_k).cuda()]
        activations, activation_scales = fp8.quantize(sorted_inputs_hp)  # fp8 [rows, K], fp32 [rows, 1]
        bounds = routes.bounds
        reference = torch.empty(rows, n, device="cuda")
        for expert in (routes.counts > 0).nonzero().flatten().tolist():
            first, last = int(bounds[expert]), int(bounds[expert + 1])
            reference[first:last] = (activations[first:last].float()
                                     * activation_scales[first:last]) @ weight_reference[expert].t()
        tuning = humming_tuning(layer, rows, gemm_type)

        sorted_ids = expert_ids = padded = expert_layout = None
        top_k, read_rows = 1, rows
        if layout == "indexed":
            block_shape = tuning["block_shape"]  # (M, N, K) of the tile
            assert isinstance(block_shape, (list, tuple))
            if shape.ep > 1:  # as vLLM's humming experts align an EP rank's routes
                sorted_ids, expert_ids, padded = common.ep_alignment(shape, block_shape[0])(routes.topk_ids)
            else:
                _, expert_layout, sorted_ids, expert_ids, padded = generate_moe_tensors(
                    routes.topk_ids, experts, "indexed", block_size_config=block_shape[0])
            activation, activation_scale = fp8.quantize(inputs_hp)
            top_k, read_rows = shape.source_top_k, shape.input_rows
            positions = routes.sorted_routes.cuda()
        elif layout == "contiguous":
            activation, activation_scale = activations, activation_scales
            expert_layout = bounds.cuda()
            positions = torch.arange(rows, device="cuda")
        else:
            capacity = routes.masked_capacity()
            positions = routes.masked_positions()
            activation = torch.zeros(experts * capacity, k, dtype=activations.dtype, device="cuda")
            activation_scale = torch.ones(experts * capacity, 1, device="cuda")
            activation[positions], activation_scale[positions] = activations, activation_scales
            expert_layout = routes.counts.cuda().int()
        compute_config, tuning_config = json.dumps({"gemm_type": gemm_type}), json.dumps(tuning)

        def call(copy_index: int, output: torch.Tensor | None) -> torch.Tensor:
            return layers[copy_index](
                inputs=activation, outputs=output, input_scale=activation_scale, sorted_ids=sorted_ids,
                expert_ids=expert_ids, num_tokens_padded=padded, expert_layout=expert_layout, top_k=top_k,
                compute_config=compute_config, tuning_config=tuning_config,
            )

        output = call(0, None)  # BF16, rows as humming lays them out for this layout; every call writes it

        def run(copy_index: int) -> torch.Tensor:
            return call(copy_index, output)

        def check() -> tuple[bool, str]:
            return common.check_output(run(0)[positions], reference)

        return PreparedCall(run, check, 2.0 * rows * n * k,
                            common.moe_traffic(shape, routes, BYTES_PER_WEIGHT, read_rows), tuning_config)

    return Entry(f"humming_{layout}", "humming", f"{RECIPE}_{layout}", humming_version(), "mxfp4_g32",
                 BYTES_PER_WEIGHT, "moe", prepare, layout)


def entries() -> list[Entry]:
    return [dense_entry(), *(moe_entry(layout) for layout in LAYOUTS)]


def main() -> None:
    args = common.bench_parser(__doc__).parse_args()
    common.run_bench(entries(), args, {})


if __name__ == "__main__":
    main()
