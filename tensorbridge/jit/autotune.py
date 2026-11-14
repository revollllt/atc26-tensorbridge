"""Autotune of the calls a CUDA graph will capture. It is opt-in and off by default: nothing calls it unless
the caller does
(`tensorbridge.autotune`, `TENSORBRIDGE_AUTOTUNE=1` for serving).

A captured call's shape is fixed, so its config can be measured once before capture. For each M the caller will
capture, at most `CANDIDATES` configs are timed: the selection rule's config and its neighbours
(`runtime.autotune_candidates`); a neighbour replaces the rule's choice only if it is faster in every round. Every
other call is still served by the rule alone. Results are cached on this machine (under
`TENSORBRIDGE_CACHE_DIR`) by device, `(N, K)`, M, the weights' bridge and a hash of the kernel and selector
sources. The two bridges compile the same source into different kernels, so each is timed with its own kernel;
the runtime's overrides distinguish bridge and indexed input top-k.

Timing follows inference, where every layer's weights are new to the L2: consecutive calls read
different copies of the weight, together more than twice the L2, so each call loads its weights
from DRAM.
"""

import dataclasses
import datetime
import functools
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import time
import warnings
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypedDict, TypeVar

import torch
from filelock import FileLock

from tensorbridge.jit import codegen, compiler, runtime
from tensorbridge.jit.codegen import Bridge, GemmConfig, GroupedLayout
from tensorbridge.quant import fp8, nvfp4

Key = TypeVar("Key", bound=Hashable)

CANDIDATES = 16  # Every class and shape: the rule's config and up to 15 neighbours.
ROUNDS = 5  # a candidate must beat the rule's choice in every one of them
ROUND_US = 1500  # length of a round
PACKAGE_DIR = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class Stats:
    """What autotune cost this process."""

    compiled: int = 0  # candidate kernels compiled, not found in the JIT cache
    measurements: int = 0  # (N, K, M, candidate) timed
    seconds: float = 0.0  # wall time spent measuring, compiling included
    failures: int = 0  # M that fell back to the rule's choice


class CacheRecord(TypedDict):
    """One M of a cache file; configs as `GemmConfig.to_list()`."""

    selected: list[int]
    model_pick: list[int]
    candidates: list[list[int]]  # in the order of `autotune_candidates`, the rule's choice first
    rounds_us: list[list[float]]  # per candidate, its time in every round
    median_us: list[float]
    measured_at: str


stats = Stats()
# (device, available SMs, N, K, M, bridge, group_scales, candidate budget).
selected: dict[tuple[int, int, int, int, int, Bridge, bool, int], GemmConfig] = {}
# (device, available SMs, N, K, E, top_k, layout, group_scales, rows, bridge, candidate budget).
selected_grouped: dict[tuple[int, int, int, int, int, int, GroupedLayout, bool, int, Bridge, int], GemmConfig] = {}


def weight_copies(weight_bytes: float, l2_bytes: int) -> int:
    """Copies of a weight that together exceed twice the L2, so a call cycling through them never
    finds its weights there (also the streaming mode of `benchmarks/`)."""
    return math.ceil(2 * l2_bytes / weight_bytes) + 1


def time_configs(
    calls: dict[Key, Callable[[int], object]], copies: int, rounds: int, round_us: float
) -> dict[Key, list[float]]:
    """Per-call microseconds of each call in every round: CUDA graphs of whole cycles through the
    `copies` weight copies (call `i` on copy `i mod copies`, so a copy is never read twice within
    the L2's reach, also across replays), replayed in rounds of about `round_us`, alternating the
    order of the calls."""
    current_stream = torch.cuda.current_stream()
    capture_stream = None if current_stream == torch.cuda.default_stream() else current_stream
    graphs: dict[Key, tuple[torch.cuda.CUDAGraph, int, int]] = {}
    for key, call in calls.items():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for copy in range(copies):
            call(copy)
        start.record()
        for copy in range(copies):
            call(copy)
        end.record()
        end.synchronize()
        estimate_us = max(start.elapsed_time(end) * 1000 / copies, 1.0)
        calls_per_graph = copies * max(1, round(200 / (copies * estimate_us)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for call_index in range(calls_per_graph):
                call(call_index % copies)
        replays = max(2, round(round_us / (calls_per_graph * estimate_us)))
        graphs[key] = (graph, calls_per_graph, replays)

    samples: dict[Key, list[float]] = {key: [] for key in graphs}
    order = list(graphs)
    for round_index in range(rounds):
        for key in order if round_index % 2 == 0 else order[::-1]:
            graph, calls_per_graph, replays = graphs[key]
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) * 1000 / (replays * calls_per_graph))
    return samples


def choose_config(model_pick: GemmConfig, rounds_us: dict[GemmConfig, list[float]]) -> GemmConfig:
    """Among the candidates faster than the rule's choice in every round, the one with the lowest
    median; the rule's choice if there is none. Rounds are compared one by one,
    since the candidates of a round ran back to back."""
    reference = rounds_us[model_pick]
    faster = [
        config for config, rounds in rounds_us.items()
        if config != model_pick and all(us < reference_us for us, reference_us in zip(rounds, reference, strict=True))
    ]
    if not faster:
        return model_pick
    return min(faster, key=lambda config: statistics.median(rounds_us[config]))


@functools.cache
def source_hash() -> str:
    """Hash of the kernel and the selector sources: a result is only valid for the code it timed."""
    files = sorted(
        [*PACKAGE_DIR.glob("include/tensorbridge/**/*.cuh"), *PACKAGE_DIR.glob("csrc/**/*.hpp"),
         *PACKAGE_DIR.glob("csrc/**/*.cpp"), *PACKAGE_DIR.glob("csrc/**/*.h")]
    )
    hasher = hashlib.sha256()
    for path in files:
        hasher.update(str(path.relative_to(PACKAGE_DIR)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def cache_path(device: int, n: int, k: int, bridge: Bridge = "exact") -> Path:
    device_name = torch.cuda.get_device_properties(device).name.replace(" ", "_")
    name = f"{device_name}-{runtime.get_num_sms(device)}-{n}x{k}-{bridge}-k{CANDIDATES}.json"
    return compiler.get_cache_dir() / "autotune" / source_hash() / name


def dense_cache_path(device: int, n: int, k: int, bridge: Bridge = "exact", group_scales: bool = False) -> Path:
    path = cache_path(device, n, k, bridge)
    return path.with_name(f"{path.stem}-group128.json") if group_scales else path


def read_cache(path: Path) -> dict[str, CacheRecord]:
    """The results of a cache file by M; a missing or unreadable file has none."""
    try:
        return json.loads(path.read_text())["results"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}


def write_cache(path: Path, problem: dict[str, str | int], records: dict[str, CacheRecord]) -> None:
    """Merge `records` into the cache file at `path` under its lock, replacing the file atomically; `problem`
    describes the call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        content = {**problem, "candidate_budget": CANDIDATES, "source_hash": source_hash(),
                   "results": {**read_cache(path), **records}}
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(content, indent=1) + "\n")
        os.replace(temporary, path)


def report_failure(n: int, k: int, m: int, error: BaseException) -> None:
    """The rule's choice serves this M; the first failure of the process is reported."""
    if stats.failures == 0:
        warnings.warn(
            f"TensorBridge autotune of N={n}, K={k}, M={m} failed, so the selection rule's choice serves it "
            f"(further failures are not reported): {error}",
            stacklevel=3,
        )
    stats.failures += 1


def compile_kernels(sources: Sequence[codegen.KernelSource]) -> dict[codegen.KernelSource, BaseException]:
    """Compile unique cache misses across shapes/configs in one process-pool batch."""
    flags = compiler.get_flags()
    pending = list(dict.fromkeys(
        kernel for kernel in sources
        if not (compiler.get_cache_dir() / compiler.cache_key(kernel.source, kernel.name, flags, compiler.INCLUDE_DIR)
                / "meta.json").exists()
    ))
    stats.compiled += len(pending)
    # Only cache misses compile before timing. Daemonic workers compile in-process.
    build_errors: dict[codegen.KernelSource, BaseException] = {}
    if not pending:
        return build_errors
    if multiprocessing.current_process().daemon:
        for kernel in pending:
            try:
                compiler.build(kernel.source, kernel.name)
            except RuntimeError as error:
                build_errors[kernel] = error
    else:
        compile_workers = min(len(pending), max(1, len(os.sched_getaffinity(0)) - 1))
        with ProcessPoolExecutor(compile_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            builds = {kernel: pool.submit(compiler.build, kernel.source, kernel.name) for kernel in pending}
            for kernel, build in builds.items():
                if (error := build.exception()) is not None:
                    build_errors[kernel] = error
    return build_errors


def compile_candidates(
    n: int, k: int, configs: Sequence[GemmConfig], bridge: Bridge,
    *, num_experts: int = 0, layout: GroupedLayout = "dense", group_scales: bool = False,
    scale_m_major: bool = False,
) -> dict[GemmConfig, BaseException]:
    """Compile unique cache misses for all capture sizes before GPU timing."""
    kernels = {config: (codegen.generate_pgroup(n, k, config, scale_m_major, bridge, num_experts, layout)
                        if group_scales else codegen.generate(n, k, config, num_experts, layout, bridge))
               for config in configs}
    source_errors = compile_kernels(list(kernels.values()))
    return {config: source_errors[kernel] for config, kernel in kernels.items() if kernel in source_errors}


def measure(
    ms: Sequence[int], n: int, k: int, device: int | None = None, bridge: Bridge = "exact",
    *, group_scales: bool = False,
) -> dict[int, GemmConfig]:
    """The config autotune selects for each M of `ms` on the weight shape `(n, k)` of the `bridge`
    kernel, measured unless this process or the cache already has it. Unlike `autotune`, it leaves
    what `runtime.select_config` returns unchanged, so the bench can time both."""
    if not 1 <= CANDIDATES <= 16:
        raise ValueError("autotune candidate budget must be between 1 and 16")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("autotune times kernels: call it before CUDA graph capture, not during it")
    device = torch.cuda.current_device() if device is None else device
    num_sms = runtime.get_num_sms(device)
    ms = list(dict.fromkeys(ms))
    pending = [m for m in ms if (device, num_sms, n, k, m, bridge, group_scales, CANDIDATES) not in selected]
    if not pending:
        return {m: selected[device, num_sms, n, k, m, bridge, group_scales, CANDIDATES] for m in ms}

    path = dense_cache_path(device, n, k, bridge, group_scales)
    cached = read_cache(path)
    for m in [m for m in pending if str(m) in cached]:
        selected[device, num_sms, n, k, m, bridge, group_scales, CANDIDATES] = GemmConfig.from_list(
            cached[str(m)]["selected"])
        pending.remove(m)
    if not pending:
        return {m: selected[device, num_sms, n, k, m, bridge, group_scales, CANDIDATES] for m in ms}

    started = time.perf_counter()
    ranked = {m: runtime.autotune_candidates(m, n, k, CANDIDATES, device, group_scales=group_scales) for m in pending}
    configs = dict.fromkeys(config for candidates in ranked.values() for config in candidates)
    build_errors = compile_candidates(n, k, list(configs), bridge, group_scales=group_scales,
                                      scale_m_major=group_scales)
    kernel_ids: dict[GemmConfig, int] = {}
    for config, error in build_errors.items():
        warnings.warn(f"autotune candidate did not compile: {config}: {error}", stacklevel=2)
    with torch.cuda.device(device):
        for config in configs:
            if config in build_errors:
                continue
            elif group_scales:
                kernel_ids[config] = runtime.register_device_kernel(
                    device, codegen.generate_pgroup(n, k, config, True, bridge), compiler.INCLUDE_DIR, n, k, config,
                    activation_scale_groups=codegen.pad_to_unit(k) // 128, scale_m_major=True, block_scales=True,
                )
            else:
                kernel_ids[config] = runtime.get_kernel(n, k, config, bridge=bridge)

        # Weight copies that stream from DRAM as in inference, with random values.
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        copies = weight_copies(n * k * 9 / 16, l2_bytes)
        weight = nvfp4.prepare(
            *nvfp4.quantize(torch.randn(n, k, device=device, dtype=torch.bfloat16)), bridge=bridge
        )
        if group_scales:
            groups, padded_n = weight.scale.shape
            reordered = weight.scale.view(groups // 8, 8, padded_n).permute(0, 2, 1).contiguous()
            weight = dataclasses.replace(weight, scale=reordered)
        weights = [weight] + [
            dataclasses.replace(weight, weight=weight.weight.clone(), scale=weight.scale.clone(),
                                global_scale=weight.global_scale.clone(), locks=torch.zeros_like(weight.locks))
            for _ in range(copies - 1)
        ]
        records: dict[str, CacheRecord] = {}
        for m in pending:
            model_pick = ranked[m][0]
            if model_pick in build_errors:
                raise RuntimeError(f"the rule's choice could not compile: {model_pick}") from build_errors[model_pick]
            sample = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            activations, activation_scale = (fp8.quantize_group_128(sample) if group_scales else fp8.quantize(sample))
            output = torch.empty(m, n, device=device, dtype=torch.bfloat16)
            calls = {
                config: lambda copy, kernel_id=kernel_ids[config]: torch.ops.tensorbridge.launch_kernel(
                    kernel_id, activations, activation_scale, weights[copy].weight, weights[copy].scale,  # noqa: F821
                    weights[copy].global_scale, weights[copy].locks, output,  # noqa: F821
                )
                for config in ranked[m] if config in kernel_ids
            }
            try:
                rounds_us = time_configs(calls, copies, ROUNDS, ROUND_US)
            except (RuntimeError, torch.OutOfMemoryError) as error:  # a kernel fault or no room for its graph
                report_failure(n, k, m, error)
                selected[device, num_sms, n, k, m, bridge, group_scales, CANDIDATES] = model_pick
                continue
            stats.measurements += len(calls)
            choice = selected[device, num_sms, n, k, m, bridge, group_scales, CANDIDATES] = choose_config(
                model_pick, rounds_us)
            records[str(m)] = {
                "selected": choice.to_list(),
                "model_pick": model_pick.to_list(),
                "candidates": [config.to_list() for config in rounds_us],
                "rounds_us": [[round(us, 3) for us in rounds] for rounds in rounds_us.values()],
                "median_us": [round(statistics.median(rounds), 3) for rounds in rounds_us.values()],
                "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        del weights, weight
        torch.cuda.empty_cache()  # the weight copies are temporary; give their memory back

    if records:
        write_cache(path, {"device_name": torch.cuda.get_device_properties(device).name, "num_sms": num_sms,
                           "n": n, "k": k, "bridge": bridge, "group_scales": int(group_scales)}, records)
    stats.seconds += time.perf_counter() - started
    return {m: selected[device, num_sms, n, k, m, bridge, group_scales, CANDIDATES] for m in ms}


def autotune(
    ms: Sequence[int], n: int, k: int, device: int | None = None, bridge: Bridge = "exact",
    *, group_scales: bool = False,
) -> dict[int, GemmConfig]:
    """Measure the config of each M of `ms` on the weight shape `(n, k)` of the `bridge` kernel
    and serve those M with it from now on. Call it before CUDA graph
    capture with the M that will be captured; every other M keeps the selection rule's choice.
    Overrides distinguish the bridge and group scales. Candidates compile in a spawn process
    pool, so a script that calls it needs the `if __name__ == "__main__":` guard that
    multiprocessing requires; without it every M not in the cache falls back to the rule's choice."""
    device = torch.cuda.current_device() if device is None else device
    results = measure(ms, n, k, device, bridge, group_scales=group_scales)
    for m, config in results.items():
        runtime.override_config(m, n, k, config, device, bridge=bridge, group_scales=group_scales)
    return results


def grouped_cache_path(device: int, n: int, k: int, num_experts: int, top_k: int, layout: GroupedLayout,
                       bridge: Bridge = "exact", group_scales: bool = False) -> Path:
    """Reuse the existing source/resource cache directory, with the grouped problem in the filename."""
    path = cache_path(device, n, k, bridge)
    return path.with_name(f"{path.stem}-e{num_experts}-topk{top_k}-{layout}"
                          f"{'-group128' if group_scales else ''}.json")


# Weights per quantization chunk of `grouped_timing_calls`: 2^28 BF16 weights, with temporaries of a few GiB.
WEIGHT_CHUNK_ELEMENTS = 1 << 28


def grouped_timing_calls(
    n: int, k: int, num_experts: int, top_k: int, layout: GroupedLayout, rows: int,
    configs: Sequence[GemmConfig], device: int, bridge: Bridge, counts: torch.Tensor | None = None,
    *, group_scales: bool = False,
) -> tuple[dict[GemmConfig, Callable[[int], torch.Tensor]], int]:
    """Routes prepared outside timing: each expert's row count from `counts` (int64 [num_experts],
    summing to `rows`), else uniform sampled routes. Rotate
    disjoint occupied expert windows and weight copies so more than twice L2 of weights separate reads of
    the same expert."""
    generator = torch.Generator().manual_seed(0)
    if counts is None:
        counts = torch.bincount(torch.randint(num_experts, (rows,), generator=generator), minlength=num_experts)
    elif counts.numel() != num_experts or int(counts.sum()) != rows:
        raise ValueError(f"counts must give {num_experts} experts' rows summing to {rows}")
    counts = counts[counts > 0]
    active = counts.numel()
    windows = num_experts // active
    expert_bytes = n * k * 9 / 16
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    copies = math.ceil((2 * l2_bytes + active * expert_bytes) / (windows * active * expert_bytes))
    # Quantized a chunk of whole experts at a time, each chunk with its own global scale: the quantization's
    # temporaries of a whole rank's experts in one piece do not fit the device memory.
    experts_per_chunk = max(1, WEIGHT_CHUNK_ELEMENTS // (n * k))
    parts = []
    for first in range(0, num_experts, experts_per_chunk):
        chunk = min(experts_per_chunk, num_experts - first)
        packed, scale, global_scale = nvfp4.quantize(torch.randn(chunk * n, k, dtype=torch.bfloat16, device=device))
        parts.append(nvfp4.prepare_experts(packed.view(chunk, n, k // 2), scale.view(chunk, n, k // 16),
                                           global_scale.expand(chunk), bridge=bridge))
    weight = nvfp4.NVFP4Experts(torch.cat([part.weight for part in parts]), torch.cat([part.scale for part in parts]),
                                torch.cat([part.global_scale for part in parts]), parts[0].locks, bridge)
    del parts
    if group_scales:
        weight = dataclasses.replace(weight, scale=weight.scale.view(num_experts, k // 128, 8, n)
                                     .permute(0, 1, 3, 2).contiguous())
    weights = [weight] + [nvfp4.NVFP4Experts(weight.weight.clone(), weight.scale.clone(), weight.global_scale.clone(),
                                           torch.zeros_like(weight.locks), bridge) for _ in range(copies - 1)]
    input_rows = rows // top_k if layout == "indexed" else rows
    if layout == "masked":
        input_rows = num_experts * int(counts.max())
    inputs = torch.randn(input_rows, k, dtype=torch.bfloat16, device=device)
    if group_scales:
        activations = torch.empty_like(inputs, dtype=torch.float8_e4m3fn)
        activation_scale = torch.empty(input_rows, k // 128, dtype=torch.float32, device=device)
        fp8.quantize_moe(inputs, "group_128", out=(activations, activation_scale))
    else:
        activations, activation_scale = fp8.quantize(inputs)
    output = torch.empty(input_rows * top_k if layout == "indexed" else input_rows, n,
                         dtype=torch.bfloat16, device=device)
    route_experts = torch.repeat_interleave(torch.arange(active), counts)[torch.randperm(rows, generator=generator)]
    calls: dict[GemmConfig, Callable[[int], torch.Tensor]] = {}
    for config in configs:
        kernel_id = runtime.get_grouped_kernel(n, k, num_experts, layout, config, bridge=bridge,
                                               group_scales=group_scales)
        # (sorted IDs, expert IDs, padded count, offsets/counts), one fixed route table per window.
        tables: list[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]] = []
        for window in range(windows):
            base = window * active
            if layout == "indexed":
                ids, experts = [], []
                for expert, count in enumerate(counts.tolist()):
                    padded = math.ceil(count / config.block_m) * config.block_m
                    ids.append(torch.cat(((route_experts == expert).nonzero().flatten(),
                                          torch.full((padded - count,), rows))))
                    experts.append(torch.full((padded // config.block_m,), base + expert))
                sorted_ids = torch.cat(ids).to(device=device, dtype=torch.int32)
                expert_ids = torch.cat(experts).to(device=device, dtype=torch.int32)
                padded_rows = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device=device)
                tables.append((sorted_ids, expert_ids, padded_rows, None))
            else:
                expert_counts = torch.zeros(num_experts, dtype=torch.int32)
                expert_counts[base:base + active] = counts.to(torch.int32)
                metadata = (torch.cat((torch.zeros(1, dtype=torch.int64), expert_counts.cumsum(0)))
                            if layout == "contiguous" else expert_counts)
                tables.append((None, None, None, metadata.to(device=device)))

        def launch(copy: int, kernel: int = kernel_id,
                   metadata: list[tuple[torch.Tensor | None, torch.Tensor | None,
                                        torch.Tensor | None, torch.Tensor | None]] = tables) -> torch.Tensor:
            expert_weight = weights[copy // windows]
            sorted_ids, expert_ids, padded_rows, expert_layout = metadata[copy % windows]
            return torch.ops.tensorbridge.launch_grouped_kernel(
                kernel, activations, activation_scale, expert_weight.weight, expert_weight.scale,
                expert_weight.global_scale, expert_weight.locks, sorted_ids, expert_ids, padded_rows,
                expert_layout, top_k if layout == "indexed" else 1, output,
            )
        calls[config] = launch
    return calls, copies * windows


def measure_grouped(
    ms: Sequence[int], n: int, k: int, num_experts: int, top_k: int, layout: GroupedLayout,
    device: int | None = None, bridge: Bridge = "exact", *, group_scales: bool = False,
) -> dict[int, GemmConfig]:
    """The config autotune selects for each routed row count of `ms` of a grouped call, as `measure` for dense
    calls, without changing runtime selection. Compile and launch errors propagate; an unmeasured configuration is
    never recorded as an autotune result."""
    if not 1 <= CANDIDATES <= 16:
        raise ValueError("autotune candidate budget must be between 1 and 16")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("autotune times kernels: call it before CUDA graph capture, not during it")
    if layout == "dense" or num_experts < 1 or top_k < 1:
        raise ValueError("grouped autotune needs a grouped layout, experts and a positive top_k")
    if any(row < 1 or (layout == "indexed" and row % top_k) for row in ms):
        raise ValueError("grouped capture rows must be positive and indexed rows must divide by top_k")
    device = torch.cuda.current_device() if device is None else device
    resources = (device, runtime.get_num_sms(device), n, k, num_experts, top_k, layout, group_scales)
    requested = list(dict.fromkeys(ms))
    path = grouped_cache_path(device, n, k, num_experts, top_k, layout, bridge, group_scales)
    cached = read_cache(path)
    pending: list[int] = []
    for row in requested:
        key = (*resources, row, bridge, CANDIDATES)
        if key in selected_grouped:
            continue
        if str(row) in cached:
            selected_grouped[key] = GemmConfig.from_list(cached[str(row)]["selected"])
        else:
            pending.append(row)
    if not pending:
        return {row: selected_grouped[*resources, row, bridge, CANDIDATES] for row in requested}
    started = time.perf_counter()
    ranked_rows = {row: runtime.autotune_candidates(row, n, k, CANDIDATES, device, num_experts, layout, group_scales)
                   for row in pending}
    configs = list(dict.fromkeys(config for ranked in ranked_rows.values() for config in ranked))
    build_errors = compile_candidates(n, k, configs, bridge, num_experts=num_experts, layout=layout,
                                      group_scales=group_scales)
    if build_errors:
        config, error = next(iter(build_errors.items()))
        raise RuntimeError(f"grouped autotune candidate could not compile: {config}") from error
    records: dict[str, CacheRecord] = {}
    with torch.cuda.device(device):
        for row in pending:
            ranked = ranked_rows[row]
            calls, copies = grouped_timing_calls(n, k, num_experts, top_k, layout, row, ranked, device, bridge,
                                                group_scales=group_scales)
            samples = time_configs(calls, copies, ROUNDS, ROUND_US)
            choice = choose_config(ranked[0], samples)
            stats.measurements += len(samples)
            selected_grouped[*resources, row, bridge, CANDIDATES] = choice
            records[str(row)] = {
                "selected": choice.to_list(), "model_pick": ranked[0].to_list(),
                "candidates": [config.to_list() for config in samples],
                "rounds_us": [[round(value, 3) for value in sample] for sample in samples.values()],
                "median_us": [round(statistics.median(sample), 3) for sample in samples.values()],
                "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            del calls
    write_cache(path, {"device_name": torch.cuda.get_device_properties(device).name, "num_sms": resources[1],
                       "n": n, "k": k, "num_experts": num_experts, "top_k": top_k, "layout": layout, "bridge": bridge},
                records)
    stats.seconds += time.perf_counter() - started
    return {row: selected_grouped[*resources, row, bridge, CANDIDATES] for row in requested}


def autotune_grouped(
    ms: Sequence[int], n: int, k: int, num_experts: int, top_k: int, layout: GroupedLayout,
    device: int | None = None, bridge: Bridge = "exact", *, group_scales: bool = False,
) -> dict[int, GemmConfig]:
    """Measure grouped capture sizes (`ms`, routed rows) and serve them with the results, as `autotune` does for
    dense calls."""
    device = torch.cuda.current_device() if device is None else device
    results = measure_grouped(ms, n, k, num_experts, top_k, layout, device, bridge, group_scales=group_scales)
    for row, config in results.items():
        runtime.override_config(row, n, k, config, device, num_experts, layout, bridge,
                                top_k if layout == "indexed" else 1, group_scales)
    return results
