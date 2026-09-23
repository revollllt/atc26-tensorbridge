#!/usr/bin/env python3
"""Figure 11: GEMM latency on the 40 paper shapes for the kernels in this environment.

Weights and activations are random and each kernel runs as an isolated GEMM
(activation quantization excluded). After prime and warm-up calls, each of
`--rounds` rounds times `--iters` calls; the kernel order alternates between
rounds and a cell is the median over rounds of the per-call time in
microseconds. `--timing` selects how a round is timed, identically for every
kernel: `graph` replays the calls as one CUDA graph (GPU time only, default),
`loop` brackets the back-to-back launches with one event pair, and `launch`
brackets each call with its own event pair and synchronization (host launch
overhead included). TensorRT-LLM and QServe need their own environments and are
not measured here.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

import tensorbridge
from tensorbridge.quant import fp8, nvfp4

SHAPES = Path(__file__).resolve().parent / "data/shapes.csv"


def torch_bf16(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return lambda: a @ b.t()


def torch_fp8(m, n, k):  # per-token and per-channel scales
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(m, 1, device="cuda") + 0.5
    sb = torch.rand(1, n, device="cuda") + 0.5
    return lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def cutlass_w4a8(m, n, k, group=128):  # vLLM's CUTLASS W4A8: int4 weights, FP8 activations
    import vllm._custom_ops  # noqa: F401  registers torch.ops._C

    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    codes = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
    packed = torch.zeros(k // 8, n, device="cuda", dtype=torch.int32)
    for i in range(8):
        packed |= codes[i::8] << (4 * i)
    b = torch.ops._C.cutlass_encode_and_reorder_int4b(packed.t().contiguous().t())
    group_scales = torch.ops._C.cutlass_pack_scale_fp8(
        (torch.rand(k // group, n, device="cuda") + 0.5).to(torch.float8_e4m3fn))
    channel_scales = torch.rand(n, device="cuda") + 0.5
    token_scales = torch.rand(m, device="cuda") + 0.5
    return lambda: torch.ops._C.cutlass_w4a8_mm(a, b, group_scales, group, channel_scales,
                                                token_scales, None, None)


def marlin_nvfp4a16(m, n, k):  # vLLM's NVFP4 Marlin: NVFP4 weights, BF16 activations
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear, rand_marlin_weight_nvfp4_like)

    x = (torch.randn(m, k, device="cuda") * 0.1).bfloat16()
    dense = (torch.randn(n, k, device="cuda") * 0.1).bfloat16()
    _, qweight, scales, global_scale = rand_marlin_weight_nvfp4_like(dense, 16, input_dtype=None)
    workspace = marlin_make_workspace_new(x.device)
    return lambda: apply_fp4_marlin_linear(x, qweight, scales, global_scale, workspace, n, k,
                                           bias=None, input_dtype=None)


def tensorbridge_nvfp4a8(m, n, k):
    a, sa = fp8.quantize(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    weight = nvfp4.prepare(*nvfp4.quantize(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)))
    return lambda: tensorbridge.fp8_nvfp4_gemm(a, sa, weight)


OURS = "NVFP4A8 (ours)"
METHODS = {OURS: tensorbridge_nvfp4a8, "NVFP4A16 Marlin": marlin_nvfp4a16,
           "vLLM CUTLASS W4A8": cutlass_w4a8, "Torch FP8": torch_fp8, "Torch BF16": torch_bf16}


def graph_time(fn, iters):
    """Per-call time of `iters` calls captured in one CUDA graph."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def loop_time(fn, iters):
    """Per-call time of `iters` back-to-back launches between one event pair."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def launch_time(fn, iters):
    """Median of `iters` separately timed and synchronized launches."""
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


TIMERS = {"graph": graph_time, "loop": loop_time, "launch": launch_time}


def measure(m, n, k, methods, args):
    fns = {name: METHODS[name](m, n, k) for name in methods}
    for fn in fns.values():
        for _ in range(args.prime + args.warmup):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in fns}
    for r in range(args.rounds):
        for name in (methods if r % 2 == 0 else methods[::-1]):
            samples[name].append(TIMERS[args.timing](fns[name], args.iters))
    return {name: statistics.median(v) for name, v in samples.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--prime", type=int, default=8)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timing", choices=TIMERS, default="graph")
    p.add_argument("--output", type=Path, default=Path("ae/figure11/results/measured.md"))
    args = p.parse_args()

    torch.manual_seed(0)
    header = ["shape_id", "model", "module", "M", "N", "K", *args.methods]
    lines = [f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}; "
             f"median us, {args.timing} timing",
             "", "## Latency Pivot (us median)", "",
             "| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    speedups = {name: [] for name in args.methods}
    for i, row in enumerate(csv.DictReader(SHAPES.open()), 1):
        m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
        lat = measure(m, n, k, args.methods, args)
        cells = [f"P{i:02d}", row["model"], row["module"], m, n, k,
                 *[f"{lat[name]:.2f}" for name in args.methods]]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
        print(lines[-1], flush=True)
        for name in args.methods:
            speedups[name].append(lat[name] / lat.get(OURS, math.nan))
        torch.cuda.empty_cache()
    if OURS in args.methods:  # geometric mean over shapes of baseline / ours latency
        lines += ["", f"## Geometric-mean speedup of {OURS}", "", "| baseline | speedup |",
                  "| --- | ---: |"]
        lines += [f"| {name} | {math.exp(statistics.fmean(map(math.log, v))):.2f} |"
                  for name, v in speedups.items() if name != OURS]
        print("\n".join(lines[lines.index(f"## Geometric-mean speedup of {OURS}"):]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
