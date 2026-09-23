"""Offline autotuning: measure configs per GEMM shape and store the fastest.

    python -m tensorbridge.autotune 4096x6144x4096 512x24576x4096 ...

For each `MxNxK`, a coordinate descent starts at the shape model's config
(`csrc/heuristics/sm90.hpp`) and tries every value of one knob at a time,
keeping any change that is faster, until a pass changes nothing. Timing is the
median over rounds of CUDA-graph replays, so host overhead does not count.
The winners are merged into `tuned/<GPU name>.json`, which `select_config`
reads first; shapes not in the table keep the shape model.
"""

import argparse
import dataclasses
import json
import statistics

import torch

from tensorbridge.gemm import fp8_nvfp4_gemm
from tensorbridge.jit import runtime
from tensorbridge.quant import fp8, nvfp4

CHOICES = {
    "block_m": [128, 176, 256],
    "num_stages": [3, 4, 5, 6],
    "multicast_b": [1, 2],
    "num_producer_regs": [40, 56],
    "use_stream_k": [False, True],
    "use_tma_store": [True, False],
    "m_fast_tile_order": [False, True],
    "prebroadcast_scale": [False, True],
}


def graph_time(fn, calls=20, rounds=3):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    times = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(times)


def allowed(config, m):
    if config.use_stream_k and (config.multicast_b > 1 or config.m_fast_tile_order):
        return False  # the kernel runs Stream-K data-parallel only
    return config.multicast_b == 1 or -(-m // config.block_m) % config.multicast_b == 0


def tune(m, n, k):
    a, sa = fp8.quantize(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    weight = nvfp4.prepare(*nvfp4.quantize(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)))
    times = {}

    def time(config):
        if config not in times:
            try:
                fn = lambda: fp8_nvfp4_gemm(a, sa, weight, config=config)  # noqa: E731
                for _ in range(3):
                    fn()
                times[config] = graph_time(fn)
            except Exception:  # does not compile or launch, e.g. out of shared memory
                times[config] = float("inf")
        return times[config]

    best = default = runtime.select_config(m, n, k, tuned=False)
    changed = True
    while changed:
        changed = False
        for field, values in CHOICES.items():
            for value in values:
                config = dataclasses.replace(best, **{field: value})
                if config != best and allowed(config, m) and time(config) < time(best):
                    best, changed = config, True
    return best, time(default), time(best)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shapes", nargs="+", help="MxNxK")
    p.add_argument("--min-gain", type=float, default=1.01, help="keep only configs at least this much faster")
    args = p.parse_args()
    path = runtime.tuned_table_path()
    table = json.loads(path.read_text()) if path.exists() else {}
    for shape in args.shapes:
        m, n, k = map(int, shape.split("x"))
        best, default_us, best_us = tune(m, n, k)
        gain = default_us / best_us
        print(f"{shape}: {default_us:.2f} -> {best_us:.2f} us ({gain:.3f}x)", flush=True)
        table.pop(shape, None)
        if gain >= args.min_gain:
            table[shape] = dataclasses.asdict(best)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
