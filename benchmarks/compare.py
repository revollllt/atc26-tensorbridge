#!/usr/bin/env python3
"""Compare TensorBridge with humming, cuBLAS FP8, DeepGEMM and Marlin on fixed shape sets, in one process.

    python benchmarks/compare.py --suites dense8_ffn
    python benchmarks/compare.py --suites ep_glm53_gate_up ep_glm53_down --rounds 3

Per shape, each entry's calls — the i-th on weight copy `i mod copies` — are captured into one CUDA
graph, and `rounds` rounds replay the graphs in alternating entry order (`common.time_attempts`).
Activation quantization and MoE routing are not timed. Weights are
streamed from DRAM. A dense shape set runs the dense entries; an MoE set (`ep_*`) runs each library's
entry of the layout its rows name. The run writes `runs/<date>-<commit>/` (`results.csv`, `meta.json`)
and prints, per suite and layout, the geometric-mean speedup of TensorBridge over each baseline and
the worst shape.
"""

import argparse
import csv
import dataclasses
import datetime
import json
import math
import statistics
import sys
from pathlib import Path
from typing import TypedDict

import bench_deepgemm
import bench_humming
import bench_marlin
import bench_tensorbridge
import bench_torch
import common
import torch
from common import MoeShape, Shape

from tensorbridge.jit import runtime

SUITES = sorted(path.stem for path in common.SHAPES_DIR.glob("*.csv"))
RESULT_COLUMNS = [
    "suite", "kernel", "layout", "M", "N", "K", "projection", "tp", "ep", "rank", "num_experts",
    "top_k", "tokens", "distribution", "l2_mode", "copies", "median_us", "min_us", "max_us", "spread",
    "round_us", "tflops", "hbm_gbps", "pct_peak_flops", "pct_peak_bw", "status", "reason", "config",
]


class ResultRow(TypedDict):
    suite: str
    kernel: str
    layout: str
    M: int  # an MoE call's rows: one per local route
    N: int
    K: int
    projection: str
    tp: int | None
    ep: int | None
    rank: int | None
    num_experts: int | None
    top_k: int | None
    tokens: int | None
    distribution: str
    l2_mode: str
    copies: int
    median_us: float | None
    min_us: float | None
    max_us: float | None
    spread: float | None
    round_us: str
    tflops: float | None
    hbm_gbps: float | None
    pct_peak_flops: float | None
    pct_peak_bw: float | None
    status: str  # ok, unsupported or invalid
    reason: str
    config: str


def result_row(suite: str, shape: Shape, attempt: common.Attempt, round_us: list[float],
               device: common.DeviceProfile) -> ResultRow:
    prepared = attempt["prepared"]
    timed = attempt["status"] == "ok" and prepared is not None and len(round_us) > 0
    median_us = statistics.median(round_us) if timed else None
    moe = shape if isinstance(shape, MoeShape) else None
    tflops = prepared.flops / median_us / 1e6 if prepared is not None and median_us else None
    hbm_gbps = prepared.traffic_bytes / median_us / 1e3 if prepared is not None and median_us else None
    return ResultRow(
        suite=suite, kernel=attempt["entry"].name, layout=attempt["entry"].layout,
        M=common.route(moe).rows if moe else shape.m, N=shape.n, K=shape.k,
        projection=moe.projection if moe else "", tp=moe.tp if moe else None, ep=moe.ep if moe else None,
        rank=moe.rank if moe else None, num_experts=moe.num_experts if moe else None,
        top_k=moe.top_k if moe else None, tokens=moe.tokens if moe else None,
        distribution=moe.distribution if moe else "", l2_mode=attempt["l2_mode"], copies=attempt["copies"],
        median_us=median_us, min_us=min(round_us) if timed else None, max_us=max(round_us) if timed else None,
        spread=max(round_us) / min(round_us) - 1 if timed else None,
        round_us=";".join(f"{value:.3f}" for value in round_us) if timed else "",
        tflops=tflops, hbm_gbps=hbm_gbps,
        pct_peak_flops=100 * tflops / device["peak_fp8_tflops"] if tflops else None,
        pct_peak_bw=100 * hbm_gbps / device["peak_bw_gbps"] if hbm_gbps else None,
        status=attempt["status"], reason=attempt["reason"], config=prepared.config if prepared is not None else "",
    )


def print_summary(rows: list[ResultRow]) -> None:
    """Geometric-mean speedup of TensorBridge over each baseline, and the shape with the smallest one.

    Speedup is baseline time / TensorBridge time, over the shapes where both ran. A baseline that
    cannot run a shape is left out of that ratio.
    """

    def shape_label(row: ResultRow) -> str:
        if row["projection"]:
            rank = f" ep{row['ep']}r{row['rank']}" if (row["ep"] or 1) > 1 else ""
            return f"{row['projection']} tp{row['tp']}{rank} T={row['tokens']} {row['distribution']}"
        return f"{row['M']}x{row['N']}x{row['K']}"

    for suite in dict.fromkeys(row["suite"] for row in rows):
        suite_rows = [row for row in rows if row["suite"] == suite]
        print(f"{suite} ({len(set(map(shape_label, suite_rows)))} shapes)")
        for layout in dict.fromkeys(row["layout"] for row in suite_rows):
            layout_rows = [row for row in suite_rows if row["layout"] == layout]
            latency: dict[tuple[str, str], float] = {}
            for row in layout_rows:
                median_us = row["median_us"]
                if row["status"] == "ok" and median_us is not None:
                    latency[row["kernel"], shape_label(row)] = median_us
            kernels = list(dict.fromkeys(row["kernel"] for row in layout_rows))
            subject = next(name for name in kernels if name.startswith("tensorbridge"))
            shapes = list(dict.fromkeys(shape_label(row) for row in layout_rows))
            for baseline in kernels:
                if baseline == subject:
                    continue
                speedups = {
                    shape: latency[baseline, shape] / latency[subject, shape]
                    for shape in shapes if (baseline, shape) in latency and (subject, shape) in latency
                }
                if speedups:
                    worst = min(speedups, key=speedups.__getitem__)
                    mean = math.exp(statistics.fmean(math.log(value) for value in speedups.values()))
                    print(f"  {layout}: {subject} over {baseline}: {len(speedups)} shapes, "
                          f"geomean {mean:.3f}, worst {worst} ({speedups[worst]:.3f})")
                else:
                    print(f"  {layout}: {subject} over {baseline}: no shape where both ran")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", nargs="+", default=["dense8_ffn"],
                        help=f"shape sets: {', '.join(SUITES)}; or a PATH.csv in their format")
    common.Timing.add_arguments(parser)
    parser.add_argument("--limit", type=int, help="only the first N shapes of each suite")
    parser.add_argument("--shard", help="INDEX/COUNT: only the shapes of every COUNT-th distinct weight (N, K) of "
                        "each suite, in file order, starting at weight INDEX (0-based), to split a suite over jobs")
    parser.add_argument("--out", type=Path, help="output directory (default benchmarks/runs/DATE-COMMIT)")
    parser.add_argument("--bridge", choices=("exact", "fpma"), default="exact")
    parser.add_argument("--group-128", action="store_true", help="compare TensorBridge and Humming at FP8 group-128")
    args = parser.parse_args()
    unknown_suites = [name for name in args.suites if name not in SUITES and not name.endswith(".csv")]
    if unknown_suites:
        parser.error(f"unknown suites {unknown_suites}: the shape sets are {SUITES}, or PATH.csv")
    if args.shard is not None:
        index, _, count = args.shard.partition("/")
        if not (index.isdigit() and count.isdigit() and int(index) < int(count)):
            parser.error(f"--shard {args.shard!r} is not INDEX/COUNT with 0 <= INDEX < COUNT")
        args.shard = (int(index), int(count))
    return args


def main() -> int:
    args = parse_args()
    registered = [
        *bench_tensorbridge.entries(bridge=args.bridge, group_128=args.group_128),
        *bench_humming.entries(only_group_128=args.group_128),
    ]
    if not args.group_128:
        registered.extend([
            *[entry for entry in bench_torch.entries() if entry.name.startswith("torch_fp8")],
            *bench_deepgemm.entries(),
            *bench_marlin.entries(),
        ])
    timing = common.Timing.from_arguments(args)
    suites: dict[str, list[Shape]] = {name: common.load_suite(name, args.limit) for name in args.suites}
    if args.shard is not None:
        shard_index, shard_count = args.shard
        for name, shapes in suites.items():
            weights = list(dict.fromkeys((shape.n, shape.k) for shape in shapes))
            suites[name] = [shape for shape in shapes if weights.index((shape.n, shape.k)) % shard_count == shard_index]
    runtime.load_launcher()
    device = common.profile_device()
    print(f"device: {json.dumps(device)}", flush=True)

    commit = common.git_commit(common.REPO_ROOT)
    run_id = f"{datetime.date.today():%Y%m%d}-{commit[:7]}{'-dirty' if commit.endswith('-dirty') else ''}"
    runs_dir = common.BENCH_DIR / "runs"
    run_dir = args.out or runs_dir / run_id
    suffix = 1
    while args.out is None and (run_dir / "results.csv").exists():
        suffix += 1
        run_dir = runs_dir / f"{run_id}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "bridge": args.bridge,
        "group_128": args.group_128,
        "run_id": run_dir.name,
        "device": device,
        "kernels": [
            {"name": entry.name, "role": entry.role, "library": entry.library, "layout": entry.layout,
             "version": entry.version, "weight_format": entry.weight_format, "bytes_per_weight": entry.bytes_per_weight}
            for entry in registered
        ],
        "timing": dataclasses.asdict(timing),
        "suites": {name: len(shapes) for name, shapes in suites.items()},
        "shard": list(args.shard) if args.shard is not None else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    started = datetime.datetime.now()
    print(f"{started:%H:%M:%S} writing {run_dir}", flush=True)
    rows: list[ResultRow] = []
    with (run_dir / "results.csv").open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, RESULT_COLUMNS)
        writer.writeheader()
        for suite, shapes in suites.items():
            for index, shape in enumerate(shapes):
                if isinstance(shape, MoeShape):
                    entries = [entry for entry in registered if entry.shape_kind == "moe"
                               and (not shape.layout or entry.layout == shape.layout)]
                else:
                    entries = [entry for entry in registered if entry.shape_kind == "dense"]
                attempts = common.prepare_and_check(shape, entries, timing, device["l2_bytes"])
                round_us = common.time_attempts(attempts, timing)
                written = [result_row(suite, shape, attempt, round_us.get(attempt["entry"].name, []), device)
                           for attempt in attempts]
                rows += written
                writer.writerows(written)
                results_file.flush()
                cells = [f"{row['kernel']} {row['median_us']:.1f}us" if row["median_us"] is not None
                         else f"{row['kernel']} {row['status']}" for row in written]
                print(f"{datetime.datetime.now():%H:%M:%S} {suite} {index + 1}/{len(shapes)} {shape.label()}: "
                      f"{', '.join(cells)}", flush=True)
                del attempts, written
                torch.cuda.empty_cache()
    print_summary(rows)
    print(f"{datetime.datetime.now():%H:%M:%S} done in {datetime.datetime.now() - started}; {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
