"""Check NVFP4A8 against the exact reference and time paired CUDA graphs."""

import argparse
import csv
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "tests"))

import torch
from nvfp4_reference import nvfp4_w4a8_reference

import tensorbridge
from tensorbridge.quant import fp8, nvfp4


class GemmResult(TypedDict):
    m: int
    n: int
    k: int
    rel_rms: float
    max_abs: float
    finite: bool
    tensorbridge_us: float
    bf16_us: float
    speedup: float
    tensorbridge_min_us: float
    tensorbridge_max_us: float
    bf16_min_us: float
    bf16_max_us: float


def capture_gemm(operation: Callable[[], torch.Tensor], prime: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for iteration in range(prime):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    torch.cuda.synchronize()
    return graph


def run_shape(
    n: int, k: int, m: int, dtype: torch.dtype, seed: int,
    rounds: int, prime: int, warmup: int, iters: int,
) -> GemmResult:
    torch.manual_seed(seed)
    # Weight [N, K] and activation [M, K], BF16.
    weight_hp = torch.randn((n, k), dtype=dtype, device="cuda:0")
    inputs_hp = torch.randn((m, k), dtype=dtype, device="cuda:0")
    packed, scale, global_scale = nvfp4.quantize(weight_hp)
    layer = tensorbridge.NVFP4Linear(nvfp4.prepare(packed, scale, global_scale))
    kernel_output = layer(inputs_hp).float()
    reference_output = nvfp4_w4a8_reference(
        inputs_hp=inputs_hp, weight_packed=packed, weight_scale_e4m3=scale,
        global_scale=global_scale, group_size=16, out_dtype=dtype,
    ).float()
    error_rms = (kernel_output - reference_output).square().mean().sqrt().item()
    reference_rms = reference_output.square().mean().sqrt().item()
    relative_rms = error_rms / reference_rms
    max_abs = (kernel_output - reference_output).abs().max().item()
    finite = bool(torch.isfinite(kernel_output).all().item())

    # Activation quantization and weight preparation are outside GEMM timing.
    quantized_input, input_scale = fp8.quantize(inputs_hp)
    weight_transposed = weight_hp.t().contiguous()
    graphs = {
        "tensorbridge": capture_gemm(lambda: layer(quantized_input, input_scale), prime),
        "bf16": capture_gemm(lambda: torch.matmul(inputs_hp, weight_transposed), prime),
    }
    latency_us: dict[str, list[float]] = {"tensorbridge": [], "bf16": []}
    for round_index in range(rounds):
        order = ("tensorbridge", "bf16") if round_index % 2 == 0 else ("bf16", "tensorbridge")
        for method in order:
            graph = graphs[method]
            for iteration in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for iteration in range(iters):
                graph.replay()
            end.record()
            end.synchronize()
            latency_us[method].append(start.elapsed_time(end) * 1000 / iters)
    tensorbridge_us = statistics.median(latency_us["tensorbridge"])
    bf16_us = statistics.median(latency_us["bf16"])
    return GemmResult(
        m=m, n=n, k=k, rel_rms=relative_rms, max_abs=max_abs, finite=finite,
        tensorbridge_us=tensorbridge_us, bf16_us=bf16_us, speedup=bf16_us / tensorbridge_us,
        tensorbridge_min_us=min(latency_us["tensorbridge"]),
        tensorbridge_max_us=max(latency_us["tensorbridge"]),
        bf16_min_us=min(latency_us["bf16"]), bf16_max_us=max(latency_us["bf16"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rel-rms", type=float, default=0.06)
    parser.add_argument("--shapes", default="4096,4096", help="Semicolon-separated N,K pairs")
    parser.add_argument("--ms", default="16,128,512", help="Comma-separated token counts")
    parser.add_argument("--shape-file", type=Path, help="CSV with M,N,K columns; overrides shapes/ms")
    parser.add_argument("--output", type=Path, default=Path("results/gemm.csv"))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--prime", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()
    if min(args.rounds, args.prime, args.warmup, args.iters) < 1:
        raise ValueError("rounds, prime, warmup and iters must be positive")
    if args.shape_file is None:
        shapes = [(int(m), *map(int, pair.split(",")))
                  for pair in args.shapes.split(";") for m in args.ms.split(",")]
    else:
        with args.shape_file.open(newline="") as source:
            shapes = [(int(row["M"]), int(row["N"]), int(row["K"])) for row in csv.DictReader(source)]
    if not shapes or any(m <= 0 or n <= 0 or k <= 0 or n % 128 or k % 128 for m, n, k in shapes):
        raise ValueError("Expected positive M and positive N,K divisible by 128")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This artifact requires a Hopper GPU (compute capability 9.0)")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}", flush=True)
    print(f"seed={args.seed}; rounds={args.rounds}; prime={args.prime}; warmup={args.warmup}; iters={args.iters}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(GemmResult.__annotations__))
        writer.writeheader()
        for m, n, k in shapes:
            result = run_shape(n, k, m, dtype, args.seed, args.rounds, args.prime, args.warmup, args.iters)
            writer.writerow(result)
            output.flush()
            print(f"M={m} N={n} K={k}: relative RMS={result['rel_rms']:.4f}; "
                  f"TensorBridge={result['tensorbridge_us']:.3f} us; "
                  f"BF16={result['bf16_us']:.3f} us; speedup={result['speedup']:.3f}x", flush=True)
            if not result["finite"] or result["rel_rms"] > args.max_rel_rms:
                raise RuntimeError(f"NVFP4A8 reference check failed for {(m, n, k)}: {result}")
            else:
                continue
    print(f"Saved {len(shapes)} shapes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
