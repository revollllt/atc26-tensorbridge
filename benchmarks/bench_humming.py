#!/usr/bin/env python3
"""Humming NVFP4 group-16 weights with BF16 inputs, or MXFP4 group-32 weights with FP8 inputs.

    python benchmarks/bench_humming.py --shape_n 8192 --shape_k 8192
    python benchmarks/bench_humming.py --shape_n 512 --shape_k 7168 --num_experts 256 --top_k 8
"""

import copy
import json
from typing import TYPE_CHECKING

import common
import torch
from common import DenseShape, Entry, MoeShape, PreparedCall, Shape, Unsupported

from tensorbridge.quant import fp8, nvfp4

if TYPE_CHECKING:
    from humming.layer import HummingLayer


def recipe(nv: bool, group_128: bool) -> tuple[str, str, float]:
    if group_128:
        return "humming_mxfp4_fp8_g128", "mxfp4_g32", 0.53125
    return ("humming_nvfp4_fp16", "nvfp4_g16", 0.5625) if nv else ("humming_mxfp4_fp8", "mxfp4_g32", 0.53125)


LAYOUTS = ("indexed", "contiguous", "masked")


def humming_version() -> str:
    """humming's version and the commit it was built from."""
    import humming
    from humming import _version

    return f"{humming.__version__} ({(_version.commit_id or '').removeprefix('g')})"


def humming_layer(
    weight_hp: torch.Tensor, input_scale_group_size: int = 0, *, nv: bool = False,
) -> tuple["HummingLayer", torch.Tensor]:
    """Build a grouped or dense Humming layer and return its dequantized FP32 reference weights."""
    from humming.layer import HummingLayer
    from humming.schema import HummingWeightSchema

    num_experts = weight_hp.size(0) if weight_hp.dim() == 3 else None
    layer = HummingLayer(
        shape_n=weight_hp.size(-2), shape_k=weight_hp.size(-1),
        weight_config=({"dtype": "float4e2m1", "scale_dtype": "float8e4m3", "group_size": 16, "scale_2_type": "tensor"}
                       if nv else {"dtype": "float4e2m1", "scale_dtype": "float8e8m0", "group_size": 32}),
        input_config=None if nv else {"dtype": "float8e4m3", "group_size": input_scale_group_size,
                                     **({"scale_dtype": "float32"} if input_scale_group_size == 128 else {})},
        num_experts=num_experts, torch_dtype=torch.bfloat16,
    ).cuda()
    if nv:
        chunks = weight_hp.unbind() if num_experts is not None else (weight_hp,)
        quantized = [nvfp4.quantize_modelopt(expert) for expert in chunks]
        packed, scales, multiplier = (torch.stack(parts) if num_experts is not None else parts[0]
                                      for parts in zip(*quantized))
        tensors = {"weight": packed.view(torch.int32), "weight_scale": scales, "weight_scale_2": multiplier.flatten()}
    else:
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


def humming_input(inputs: torch.Tensor, nv: bool, group_128: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if nv:
        return inputs, None
    if group_128:
        codes, scales = fp8.quantize_group_128(inputs)
        return codes, scales.contiguous()  # Humming SM90 consumes row-major FP32 scales
    return fp8.quantize(inputs)


def dequant_input(inputs: torch.Tensor, scales: torch.Tensor | None) -> torch.Tensor:
    if scales is None:
        return inputs.float()
    return (inputs.float().view(inputs.size(0), scales.size(1), -1) * scales.unsqueeze(-1)).view_as(inputs)


def dense_entry(nv: bool = False, group_128: bool = False) -> Entry:
    name, weight_format, weight_bytes = recipe(nv, group_128)

    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        from humming.forward import may_process_input

        assert isinstance(shape, DenseShape)
        if group_128 and shape.k % 128:
            return Unsupported("group-128 requires K divisible by 128")
        torch.manual_seed(0)
        weights = torch.randn(shape.n, shape.k, device="cuda", dtype=torch.bfloat16)
        layer, weight_reference = humming_layer(weights, 128 if group_128 else 0, nv=nv)
        layers = [layer] + [copy.deepcopy(layer) for _ in range(copies - 1)]
        inputs_hp = torch.randn(shape.m, shape.k, device="cuda", dtype=torch.bfloat16)  # [M, K]
        if not nv and not group_128:
            activation, _, activation_scale = may_process_input(layer.humming_config, inputs_hp)
            activation_scale = activation_scale.view(-1, 1)
        else:
            activation, activation_scale = humming_input(inputs_hp, nv, group_128)
        reference_input = dequant_input(activation, activation_scale)
        tuning = json.dumps(humming_tuning(layer, shape.m, "dense"))

        def run(copy_index: int) -> torch.Tensor:
            return layers[copy_index](inputs=activation, input_scale=activation_scale, tuning_config=tuning)

        def check() -> tuple[bool, str]:
            return common.check_output(run(0), reference_input @ weight_reference.t())

        return PreparedCall(run, check, 2.0 * shape.m * shape.n * shape.k,
                            common.dense_traffic(shape, weight_bytes, 2 if nv else 1)
                            + (4 * shape.m * (shape.k // 128 - 1) if group_128 else 0), tuning)

    return Entry(name, "humming", name, humming_version(), weight_format, weight_bytes,
                 "dense", prepare)


def moe_entry(layout: str, nv: bool = False, group_128: bool = False) -> Entry:
    """humming's grouped GEMM over all local experts of an MoE layer: indexed gathers each route's
    input row through `sorted_ids` (gate_up reads a token's row once per expert; an EP rank's routes
    are aligned with an expert map, as vLLM's humming experts align them); contiguous and masked take
    the inputs sorted by expert, masked in blocks of `common.MASKED_ALIGNMENT`-aligned capacity per
    expert. Every call writes one output buffer."""
    name, weight_format, weight_bytes = recipe(nv, group_128)
    gemm_type = "indexed" if layout == "indexed" else f"grouped_{layout}"

    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        from humming.testing.data import generate_moe_tensors

        assert isinstance(shape, MoeShape)
        if group_128 and shape.k % 128:
            return Unsupported("group-128 requires K divisible by 128")
        routes = common.route(shape)
        experts, n, k, rows = shape.local_experts, shape.n, shape.k, routes.rows
        if rows == 0:
            return Unsupported("no route reaches this rank")
        weights_hp, inputs_hp = common.moe_operands(shape)
        layer, weight_reference = humming_layer(weights_hp, 128 if group_128 else 0, nv=nv)  # fp32 [E, N, K]
        layers = [layer] + [copy.deepcopy(layer) for _ in range(copies - 1)]
        sorted_inputs_hp = inputs_hp[(routes.sorted_routes // shape.source_top_k).cuda()]
        activations, activation_scales = humming_input(sorted_inputs_hp, nv, group_128)
        reference_input = dequant_input(activations, activation_scales)
        bounds = routes.bounds
        reference = torch.empty(rows, n, device="cuda")
        for expert in (routes.counts > 0).nonzero().flatten().tolist():
            first, last = int(bounds[expert]), int(bounds[expert + 1])
            reference[first:last] = reference_input[first:last] @ weight_reference[expert].t()
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
            activation, activation_scale = humming_input(inputs_hp, nv, group_128)
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
            activation[positions] = activations
            if nv:
                activation_scale = None
            else:
                activation_scale = torch.ones(experts * capacity, activation_scales.size(1), device="cuda")
                activation_scale[positions] = activation_scales
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

        traffic = common.moe_traffic(shape, routes, weight_bytes, read_rows)
        traffic += read_rows * (k - 4 if nv else 4 * (k // 128 - 1) if group_128 else 0)
        return PreparedCall(run, check, 2.0 * rows * n * k, traffic, tuning_config)

    return Entry(f"{name}_{layout}", "humming", f"{name}_{layout}", humming_version(), weight_format,
                 weight_bytes, "moe", prepare, layout)


def entries(*, only_group_128: bool = False) -> list[Entry]:
    recipes = [(False, True)] if only_group_128 else [(True, False), (False, False), (False, True)]
    return [entry for nv, group_128 in recipes
            for entry in (dense_entry(nv, group_128), *(moe_entry(layout, nv, group_128) for layout in LAYOUTS))]


def main() -> None:
    args = common.bench_parser(__doc__).parse_args()
    common.run_bench(entries(), args, {})


if __name__ == "__main__":
    main()
