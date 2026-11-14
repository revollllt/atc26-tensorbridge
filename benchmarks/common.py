"""What every benchmark entry shares: the workloads, the call protocol, the timing and the results.

A library's entry (`bench_<library>.py`) builds the call of one workload shape.
`Entry.prepare(shape, copies)` makes `copies` copies of the weights, which the timing cycles through
so that every call reads its weights from HBM, and the inputs, and returns a `PreparedCall` (the run,
FLOPs, bytes, and a check of an output of the library's own format when that entry compares one), or
`Unsupported` with the reason. `prepare_and_check` prepares every entry of a shape and runs its check;
`time_attempts` then captures each ready entry's calls into one CUDA graph and replays the graphs in
alternating order, round after round. Activation quantization and routing are not timed.

`bench_<library>.py` runs one library over a range of M and writes
`results/<gpu>/<workload>/<library>_<recipe>.json`: per M, the time of a call (ms), its memory
bandwidth (GB/s) and compute throughput (TFLOPS). `compare.py` runs the libraries on fixed shape sets
in one process.
"""

import argparse
import csv
import dataclasses
import datetime
import functools
import json
import math
import statistics
import subprocess
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, TypedDict

import torch

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
SHAPES_DIR = BENCH_DIR / "shapes"
RESULTS_DIR = BENCH_DIR / "results"
# A correct output is within this relative RMS error of the entry's own-format reference.
MAX_REL_RMS = 1e-2
# Rows per expert block of a masked layout: a multiple of every library's token tile.
MASKED_ALIGNMENT = 128


@dataclasses.dataclass(frozen=True)
class DenseShape:
    """`D[M, N] = A[M, K] @ B[N, K]^T`."""

    m: int
    n: int
    k: int

    def label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


@dataclasses.dataclass(frozen=True)
class MoeShape:
    """One MoE projection call: each of `tokens` tokens is routed to `top_k` of `num_experts`
    experts, whose weights are `[N, K]`; with expert parallelism, the work of rank `rank` of `ep`,
    which holds `num_experts / ep` of them (all-to-all and combine are not part of it)."""

    projection: str  # "gate_up" (a token's input row feeds its top_k routes) or "down" (one row per route)
    tp: int
    n: int
    k: int
    num_experts: int
    top_k: int
    tokens: int
    distribution: str  # "balanced", "zipf", "equal" or "single"
    seed: int
    ep: int = 1
    rank: int = 0
    layout: str = ""  # the one expert layout the call runs in (indexed, contiguous, masked); "" for every layout
    capacity: int = 0  # masked rows per expert; 0: the most rows of an expert rounded up to MASKED_ALIGNMENT

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.ep

    @property
    def input_rows(self) -> int:
        """Rows of the unsorted input: one per token for gate_up, one per route for down."""
        return self.tokens if self.projection == "gate_up" else self.tokens * self.top_k

    @property
    def source_top_k(self) -> int:
        """Routes per input row."""
        return self.top_k if self.projection == "gate_up" else 1

    def label(self) -> str:
        rank = f" ep{self.ep}r{self.rank}" if self.ep > 1 else ""
        return f"{self.projection} tp{self.tp}{rank} T={self.tokens} {self.distribution}"


Shape = DenseShape | MoeShape


@dataclasses.dataclass(frozen=True)
class Routes:
    """The routes of an MoE call that reach this rank's experts, as every library takes them."""

    topk_ids: torch.Tensor  # int32 [tokens, top_k] (cuda): global expert of every route
    counts: torch.Tensor  # int64 [local experts] (cpu): routes of each local expert
    sorted_routes: torch.Tensor  # int64 [rows] (cpu): route ids `token * top_k + slot`, grouped by local expert
    capacity: int  # rows per expert of the masked layout

    @property
    def rows(self) -> int:
        return int(self.counts.sum())

    @property
    def bounds(self) -> torch.Tensor:
        """int64 [local experts + 1] (cpu): the first sorted row of each expert, then `rows`."""
        return torch.cat((torch.zeros(1, dtype=torch.int64), self.counts.cumsum(0)))

    @property
    def active_experts(self) -> int:
        return int((self.counts > 0).sum())

    def masked_positions(self) -> torch.Tensor:
        """int64 [rows] (cuda): where each sorted row lies in a masked layout of `masked_capacity` rows
        per expert."""
        experts = torch.repeat_interleave(torch.arange(self.counts.numel()), self.counts)
        return (experts * self.masked_capacity() + torch.arange(self.rows) - self.bounds[experts]).cuda()

    def masked_capacity(self) -> int:
        return self.capacity


def route_tokens(shape: MoeShape) -> torch.Tensor:
    """Global expert ids, int32 `[tokens, top_k]` (cuda): `shape.distribution` from `shape.seed`."""
    generator = torch.Generator().manual_seed(shape.seed)
    num_routes = shape.tokens * shape.top_k
    if shape.distribution == "balanced":
        # Round robin over the (token, slot) pairs: every expert gets routes / num_experts routes (+-1).
        expert_ids = torch.arange(num_routes).view(shape.tokens, shape.top_k) % shape.num_experts
    elif shape.distribution == "zipf":
        # P(expert e) proportional to 1 / (e + 1); top_k distinct experts per token.
        probabilities = 1.0 / torch.arange(1, shape.num_experts + 1, dtype=torch.float64)
        expert_ids = torch.multinomial(
            probabilities.repeat(shape.tokens, 1), shape.top_k, replacement=False, generator=generator
        )
    elif shape.distribution == "equal":
        # DeepEP-style dispatch to rank `shape.rank`: all routes of the tokens go to its local experts, route j
        # (token j // top_k, slot j % top_k) to local expert j mod local_experts, so the experts get the floor or
        # ceil of the mean rows, a token's experts are distinct, and with fewer routes than experts some get none.
        if shape.top_k > shape.local_experts:
            raise ValueError(f"equal routing needs top_k {shape.top_k} <= {shape.local_experts} local experts")
        local_ids = torch.arange(num_routes).view(shape.tokens, shape.top_k) % shape.local_experts
        expert_ids = shape.rank * shape.local_experts + local_ids
    elif shape.distribution == "single":
        expert_ids = torch.zeros(shape.tokens, shape.top_k, dtype=torch.int64)
    else:
        raise ValueError(f"unknown distribution {shape.distribution!r}")
    return expert_ids.to(device="cuda", dtype=torch.int32)


def sort_local_routes(topk_ids: torch.Tensor, first: int, local_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Of the routes `topk_ids` (`[tokens, top_k]` global experts), those of experts `first` to
    `first + local_experts - 1`: int64 routes per local expert, and int64 route ids
    `token * top_k + slot` grouped by local expert in a stable order (both cpu)."""
    global_experts = topk_ids.flatten().cpu().long()
    local_routes = ((global_experts >= first) & (global_experts < first + local_experts)).nonzero().flatten()
    local_experts_of_routes = global_experts[local_routes] - first
    counts = torch.bincount(local_experts_of_routes, minlength=local_experts)
    return counts, local_routes[torch.argsort(local_experts_of_routes, stable=True)]


@functools.cache
def route(shape: MoeShape) -> Routes:
    """The routes of `shape` that reach rank `shape.rank`'s experts, sorted by expert (stable)."""
    topk_ids = route_tokens(shape)
    counts, sorted_routes = sort_local_routes(topk_ids, shape.rank * shape.local_experts, shape.local_experts)
    most_rows = int(counts.max())
    capacity = shape.capacity or math.ceil(most_rows / MASKED_ALIGNMENT) * MASKED_ALIGNMENT
    if capacity < most_rows:
        raise ValueError(f"masked capacity {capacity} is below the {most_rows} rows of an expert")
    return Routes(topk_ids, counts, sorted_routes, capacity)


def expert_map(shape: MoeShape) -> torch.Tensor:
    """int32 `[num_experts + 1]` (cuda): the local index of each global expert, -1 for the experts of
    other ranks and, in the last slot, for out-of-range ids (`invalid_to_last`), as vLLM's
    `moe_align_block_size` takes it with `ignore_invalid_experts`."""
    mapping = torch.full((shape.num_experts + 1,), -1, dtype=torch.int32)
    first = shape.rank * shape.local_experts
    mapping[first:first + shape.local_experts] = torch.arange(shape.local_experts, dtype=torch.int32)
    return mapping.cuda()


def invalid_to_last(shape: MoeShape, topk_ids: torch.Tensor) -> torch.Tensor:
    """`topk_ids` with every id outside `[0, num_experts)` replaced by `num_experts`, the slot of
    `expert_map` that maps to no local expert."""
    return torch.where((topk_ids < 0) | (topk_ids >= shape.num_experts), shape.num_experts, topk_ids).int()


# Elements of a chunk of `moe_weight_chunks`: a 2^28-element BF16 chunk is 512 MiB, and its quantization's
# temporaries stay a few GiB (whole-rank weights above 2^31 elements ran out of memory in prepare).
WEIGHT_CHUNK_ELEMENTS = 1 << 28
WEIGHT_SEED_BASE = 1 << 40  # above any seed `moe_inputs` takes


def moe_weight_chunks(shape: MoeShape) -> Iterator[torch.Tensor]:
    """BF16 weights of the local experts, whole experts at a time (`[experts in the chunk, N, K]`, at most
    `WEIGHT_CHUNK_ELEMENTS` unless one expert is larger): expert e is drawn from its own seed, so the
    weights do not depend on the chunking. The seeds stay clear of `moe_inputs`'s: an expert seeded like
    the inputs would draw its weights from the inputs' random stream (expert 0 of seed 0)."""
    per_chunk = max(1, WEIGHT_CHUNK_ELEMENTS // (shape.n * shape.k))
    generator = torch.Generator(device="cuda")
    for first in range(0, shape.local_experts, per_chunk):
        chunk = torch.empty(min(per_chunk, shape.local_experts - first), shape.n, shape.k, device="cuda",
                            dtype=torch.bfloat16)
        for index in range(chunk.size(0)):
            generator.manual_seed(WEIGHT_SEED_BASE + shape.seed * shape.local_experts + first + index)
            chunk[index].normal_(generator=generator)
        yield chunk


def moe_inputs(shape: MoeShape) -> torch.Tensor:
    """BF16 inputs `[input rows, K]` of an MoE call, from its seed."""
    torch.manual_seed(shape.seed)
    return torch.randn(shape.input_rows, shape.k, device="cuda", dtype=torch.bfloat16)


def moe_operands(shape: MoeShape) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 weights `[local experts, N, K]` (`moe_weight_chunks` joined) and inputs of an MoE call."""
    return torch.cat(list(moe_weight_chunks(shape))), moe_inputs(shape)


# An indexed call's alignment, as vLLM's `moe_align_block_size` returns it: int32 sorted route ids padded
# per expert to the token tile, int32 local expert per tile, int32 [1] padded route count (all cuda).
Alignment = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def ep_alignment(shape: MoeShape, block_m: int) -> Callable[[torch.Tensor], Alignment]:
    """The alignment of an EP rank's routes `topk_ids` (int32 `[tokens, top_k]` global experts) to
    `block_m`: vLLM's `moe_align_block_size` with the rank's expert map and `ignore_invalid_experts`,
    so routes to other ranks' experts and padding rows are left out, as vLLM's humming experts
    align them."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    mapping = expert_map(shape)

    def align(topk_ids: torch.Tensor) -> Alignment:
        sorted_ids, expert_ids, padded = moe_align_block_size(
            invalid_to_last(shape, topk_ids), block_m, shape.num_experts + 1, mapping, ignore_invalid_experts=True)[:3]
        return sorted_ids, expert_ids, padded

    return align


@dataclasses.dataclass(frozen=True)
class PreparedCall:
    run: Callable[[int], torch.Tensor]  # weight copy index -> output
    check: Callable[[], tuple[bool, str]] | None  # (correct, the measured error); None times without a comparison
    flops: float
    traffic_bytes: float  # what one call must move at least: weights with scales, inputs, output
    config: str  # the kernel configuration the implementation selected, "" if it exposes none


@dataclasses.dataclass(frozen=True)
class Unsupported:
    reason: str


@dataclasses.dataclass(frozen=True)
class Entry:
    """One library's implementation of one recipe: `name` in comparisons, and
    `<library>_<recipe>.json` among the results."""

    name: str
    library: str
    recipe: str
    version: str
    weight_format: str
    bytes_per_weight: float  # including the weight scales
    shape_kind: Literal["dense", "moe"]
    prepare: Callable[[Shape, int], PreparedCall | Unsupported]  # (shape, weight copies)
    layout: str = "dense"  # an MoE entry's expert layout: indexed, contiguous or masked
    role: Literal["candidate", "baseline"] = "baseline"


def check_output(output: torch.Tensor, reference: torch.Tensor) -> tuple[bool, str]:
    """Correct: all finite, and within `MAX_REL_RMS` relative RMS error of the reference."""
    output, reference = output.float(), reference.float()
    if not bool(torch.isfinite(output).all()):
        return False, "non-finite output"
    error_rms = (output - reference).square().mean().sqrt()
    rel_rms = (error_rms / reference.square().mean().sqrt().clamp_min(1e-30)).item()
    return rel_rms <= MAX_REL_RMS, f"rel_rms {rel_rms:.2e}"


def dense_traffic(shape: DenseShape, bytes_per_weight: float, bytes_per_activation: float) -> float:
    """Weights with their scales, activations (with a 4-byte per-token scale when quantized), and
    the BF16 output."""
    activation_scale_bytes = 4 if bytes_per_activation < 2 else 0
    return (
        shape.n * shape.k * bytes_per_weight
        + shape.m * (shape.k * bytes_per_activation + activation_scale_bytes)
        + shape.m * shape.n * 2
    )


def moe_traffic(shape: MoeShape, routes: Routes, bytes_per_weight: float, input_rows: int) -> float:
    """The weights of the experts that get routes, `input_rows` FP8 input rows with a 4-byte scale,
    and the BF16 output of every route."""
    return (routes.active_experts * shape.n * shape.k * bytes_per_weight
            + input_rows * (shape.k + 4) + routes.rows * shape.n * 2)


def git_commit(path: Path) -> str:
    """HEAD of the checkout at `path`, with `-dirty` when tracked files are modified."""
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()

    return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


# ---- Timing ---------------------------------------------------------------------------------------


class DeviceProfile(TypedDict):
    gpu_name: str
    num_sms: int
    l2_bytes: int
    smem_per_block_optin: int
    driver_version: int
    cuda_runtime: str
    torch_version: str
    peak_fp8_tflops: float  # best of two large per-tensor FP8 GEMMs
    copy_gbps: float  # read + write of a 4 GiB device copy
    read_gbps: float  # a read-only reduction over 4 GiB
    peak_bw_gbps: float  # the larger of the two


@dataclasses.dataclass(frozen=True)
class Timing:
    l2: Literal["stream", "warm"] = "stream"  # stream: every call reads its weights from HBM
    memory_budget_gib: int = 8  # weight copies of one entry
    iters: int = 40  # minimum calls per CUDA graph; timing rounds up to whole weight-copy cycles
    rounds: int = 6
    prime: int = 8  # calls before capture
    warmup: int = 4

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--l2", choices=["stream", "warm"], default="stream")
        parser.add_argument("--memory-budget-gib", type=int, default=8)
        parser.add_argument("--iters", type=int, default=40)
        parser.add_argument("--rounds", type=int, default=6)
        parser.add_argument("--prime", type=int, default=8)
        parser.add_argument("--warmup", type=int, default=4)

    @staticmethod
    def from_arguments(args: argparse.Namespace) -> "Timing":
        return Timing(args.l2, args.memory_budget_gib, args.iters, args.rounds, args.prime, args.warmup)


def capture(call: Callable[[int], torch.Tensor], iters: int, copies: int) -> torch.cuda.CUDAGraph:
    """One CUDA graph of `iters` calls, the i-th on weight copy `i mod copies`, replayed once."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call_index in range(iters):
            call(call_index % copies)
    graph.replay()
    return graph


def time_graph(graph: torch.cuda.CUDAGraph, iters: int) -> float:
    """Per-call time in microseconds of one replay of a graph of `iters` calls."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def profile_device() -> DeviceProfile:
    """Device properties and measured peaks."""
    from cuda.bindings import driver

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    peak_fp8_tflops = 0.0
    for size in (8192, 16384):
        a = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")

        def matmul(copy_index: int) -> torch.Tensor:
            return torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: F821

        matmul(0)
        graph = capture(matmul, 10, 1)
        seconds = min(time_graph(graph, 10) for _ in range(5)) * 1e-6
        peak_fp8_tflops = max(peak_fp8_tflops, 2 * size**3 / seconds / 1e12)
        del a, b, graph

    # HBM bandwidth: the larger of a device copy and a read-only reduction. The copy kernel is
    # SM-bound where SMs are few: 1.7 TB/s copying against 3.5 TB/s reading on H20 (78 SMs).
    source = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
    destination = torch.empty_like(source)
    values = source.view(torch.float32).uniform_()  # fp32 [1 Gi]

    def device_copy(copy_index: int) -> torch.Tensor:
        return destination.copy_(source)  # noqa: F821

    def device_read(copy_index: int) -> torch.Tensor:
        return values.sum()  # noqa: F821

    bandwidth_gbps = []
    for traffic_bytes, operation in ((2 * (4 << 30), device_copy), (4 << 30, device_read)):
        operation(0)
        graph = capture(operation, 5, 1)
        bandwidth_gbps.append(traffic_bytes / (min(time_graph(graph, 5) for _ in range(5)) * 1e-6) / 1e9)
    copy_gbps, read_gbps = bandwidth_gbps
    del source, destination, values, graph
    torch.cuda.empty_cache()
    _, driver_version = driver.cuDriverGetVersion()
    return DeviceProfile(
        gpu_name=properties.name, num_sms=properties.multi_processor_count,
        l2_bytes=properties.L2_cache_size, smem_per_block_optin=properties.shared_memory_per_block_optin,
        driver_version=driver_version, cuda_runtime=str(torch.version.cuda),
        torch_version=torch.__version__, peak_fp8_tflops=peak_fp8_tflops,
        copy_gbps=copy_gbps, read_gbps=read_gbps, peak_bw_gbps=max(copy_gbps, read_gbps),
    )


def stream_copies(stream_bytes: float, allocated_bytes: float, l2_bytes: int, budget_bytes: int) -> tuple[int, bool]:
    """Weight copies whose `stream_bytes` (the weights a call reads) together exceed twice the L2
    (`autotune.weight_copies`), each of `allocated_bytes`, and whether the memory budget limited them."""
    from tensorbridge.jit import autotune

    copies = autotune.weight_copies(stream_bytes, l2_bytes)
    if copies * allocated_bytes <= budget_bytes:
        return copies, False
    return max(1, int(budget_bytes // allocated_bytes)), True


def weight_bytes(entry: Entry, shape: Shape) -> tuple[float, float]:
    """Bytes of the weights one call reads, and of one copy of them: an MoE call reads the experts
    that get routes, and a copy holds every local expert."""
    matrix_bytes = entry.bytes_per_weight * shape.n * shape.k
    if isinstance(shape, MoeShape):
        return route(shape).active_experts * matrix_bytes, shape.local_experts * matrix_bytes
    return matrix_bytes, matrix_bytes


class Attempt(TypedDict):
    """An entry on one shape, between its check and its timing."""

    entry: Entry
    copies: int
    l2_mode: str  # stream, stream_limited or warm
    prepared: PreparedCall | None
    status: str  # ok, unsupported or invalid
    reason: str


def describe(error: Exception) -> str:
    """The error and where it was raised, for the `reason` column."""
    frame = traceback.extract_tb(error.__traceback__)[-1]
    return f"{type(error).__name__}: {error} ({Path(frame.filename).name}:{frame.lineno})"


def prepare_and_check(shape: Shape, entries: list[Entry], timing: Timing, l2_bytes: int) -> list[Attempt]:
    """Prepare every entry on `shape` and run its check. An entry that cannot run gets its status and
    reason; the others are left `ok` and ready to time."""
    attempts: list[Attempt] = []
    for entry in entries:
        if timing.l2 == "stream":
            copies, limited = stream_copies(*weight_bytes(entry, shape), l2_bytes, timing.memory_budget_gib << 30)
        else:
            copies, limited = 1, False
        attempt = Attempt(entry=entry, copies=copies, l2_mode="stream_limited" if limited else timing.l2,
                          prepared=None, status="ok", reason="")
        attempts.append(attempt)
        # The implementations are third-party or JIT-compiled code: a failure is recorded against this
        # entry and shape, and the run goes on.
        try:
            outcome = entry.prepare(shape, copies)
        except Exception as error:
            attempt.update(status="unsupported", reason="prepare: " + describe(error))
            continue
        if isinstance(outcome, Unsupported):
            attempt.update(status="unsupported", reason=outcome.reason)
            continue
        if outcome.check is None:
            attempt.update(prepared=outcome)
            continue
        try:
            correct, detail = outcome.check()
        except Exception as error:
            attempt.update(status="invalid", reason="check: " + describe(error))
            continue
        attempt.update(prepared=outcome, status="ok" if correct else "invalid", reason=detail)
    return attempts


def time_attempts(attempts: list[Attempt], timing: Timing) -> dict[str, list[float]]:
    """Per-call times in microseconds of every ready entry, one per round, the graphs replayed in
    alternating order."""
    graphs: dict[str, tuple[torch.cuda.CUDAGraph, int]] = {}
    for attempt in attempts:
        prepared = attempt["prepared"]
        if attempt["status"] != "ok" or prepared is None:
            continue
        try:
            for call_index in range(timing.prime + timing.warmup):
                prepared.run(call_index % attempt["copies"])
            torch.cuda.synchronize()
            copies = attempt["copies"]
            calls_per_graph = math.ceil(timing.iters / copies) * copies
            graphs[attempt["entry"].name] = (capture(prepared.run, calls_per_graph, copies), calls_per_graph)
        except Exception as error:
            attempt.update(status="unsupported", reason="timing: " + describe(error))

    names = list(graphs)
    round_us: dict[str, list[float]] = {name: [] for name in names}
    for round_index in range(timing.rounds):
        for name in names if round_index % 2 == 0 else names[::-1]:
            graph, calls_per_graph = graphs[name]
            round_us[name].append(time_graph(graph, calls_per_graph))
    return round_us


# ---- Shape sets -----------------------------------------------------------------------------------


def load_suite(name: str, limit: int | None = None) -> list[Shape]:
    """The shapes of `shapes/<name>.csv`, or of the file `name` when it ends in `.csv`: dense sets have
    `M, N, K`, MoE sets one projection call per row (optional columns `layout`, `capacity`: `MoeShape`'s)."""
    path = Path(name) if name.endswith(".csv") else SHAPES_DIR / f"{name}.csv"
    rows = list(csv.DictReader(path.open()))[:limit]
    if rows and "projection" not in rows[0]:
        return [DenseShape(int(row["M"]), int(row["N"]), int(row["K"])) for row in rows]
    return [
        MoeShape(row["projection"], int(row.get("tp") or 1), int(row["N"]), int(row["K"]), int(row["num_experts"]),
                 int(row["top_k"]), int(row["tokens"]), row["distribution"], int(row["seed"]),
                 int(row.get("ep") or 1), int(row.get("rank") or 0), row.get("layout") or "",
                 int(row.get("capacity") or 0))
        for row in rows
    ]


# ---- Results of one workload (bench_<library>.py) -------------------------------------------------

DEFAULT_M_LIST = [2**i for i in range(15)]  # 1 to 16384, as humming's benchmarks


def workload_name(shape: Shape) -> str:
    """`dense_n<N>_k<K>`, or `moe_<projection>_e<E>_k<top_k>_n<N>_k<K>_<distribution>[_ep<ep>r<rank>]`."""
    if isinstance(shape, DenseShape):
        return f"dense_n{shape.n}_k{shape.k}"
    rank = f"_ep{shape.ep}r{shape.rank}" if shape.ep > 1 else ""
    return (f"moe_{shape.projection}_e{shape.num_experts}_k{shape.top_k}_n{shape.n}_k{shape.k}"
            f"_{shape.distribution}{rank}")


def bench_parser(description: str) -> argparse.ArgumentParser:
    """The arguments of every `bench_<library>.py`: one workload, its M, and the timing."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument("--shape_m_list", type=int, nargs="+", default=DEFAULT_M_LIST,
                        help="M of a dense workload, tokens of an MoE one (default 1 to 16384)")
    parser.add_argument("--num_experts", type=int, default=0, help="an MoE workload's experts (0: dense)")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--projection", choices=["gate_up", "down"], default="gate_up")
    parser.add_argument("--distribution", choices=["balanced", "zipf"], default="balanced")
    parser.add_argument("--ep", type=int, default=1, help="expert parallelism: measure the local work of one rank")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path,
                        help=f"default {RESULTS_DIR.relative_to(REPO_ROOT)}/<gpu>/<workload>")
    parser.add_argument("--no_save", action="store_true", help="print the table only")
    Timing.add_arguments(parser)
    return parser


def workload_shapes(args: argparse.Namespace) -> list[Shape]:
    if args.num_experts == 0:
        return [DenseShape(m, args.shape_n, args.shape_k) for m in args.shape_m_list]
    return [MoeShape(args.projection, 1, args.shape_n, args.shape_k, args.num_experts, args.top_k, tokens,
                     args.distribution, args.seed, args.ep, args.rank) for tokens in args.shape_m_list]


def run_bench(entries: list[Entry], args: argparse.Namespace, packages: dict[str, str]) -> None:
    """Measure `entries` on the workload of `args` (`bench_parser`), one M after the other, all
    entries of an M in one set of alternating rounds; print a table and write each entry's
    `<library>_<recipe>.json`."""
    timing = Timing.from_arguments(args)
    shapes = workload_shapes(args)
    entries = [entry for entry in entries if entry.shape_kind == ("dense" if args.num_experts == 0 else "moe")]
    device = profile_device()
    gpu_dirs = {"NVIDIA H100 80GB HBM3": "h100", "NVIDIA H20": "h20"}
    gpu = gpu_dirs.get(device["gpu_name"], device["gpu_name"].lower().replace(" ", "_"))
    output_dir = args.output_dir or RESULTS_DIR / gpu / workload_name(shapes[0])
    results: dict[str, dict[str, dict[str, float | str]]] = {entry.name: {} for entry in entries}
    print(f"{device['gpu_name']}: {workload_name(shapes[0])}, {', '.join(entry.name for entry in entries)}", flush=True)
    for shape in shapes:
        m = shape.m if isinstance(shape, DenseShape) else shape.tokens
        attempts = prepare_and_check(shape, entries, timing, device["l2_bytes"])
        round_us = time_attempts(attempts, timing)
        cells = []
        for attempt in attempts:
            prepared, rounds = attempt["prepared"], round_us.get(attempt["entry"].name, [])
            if attempt["status"] != "ok" or prepared is None or not rounds:
                cells.append(f"{attempt['entry'].name} {attempt['status']}: {attempt['reason']}")
                continue
            median_us = statistics.median(rounds)
            results[attempt["entry"].name][str(m)] = {
                "time": median_us / 1e3, "memory_gbps": prepared.traffic_bytes / median_us / 1e3,
                "compute_tops": prepared.flops / median_us / 1e6, "config": prepared.config,
            }
            cells.append(f"{attempt['entry'].name} {median_us:8.2f} us "
                         f"{prepared.traffic_bytes / median_us / 1e3:7.0f} GB/s "
                         f"{prepared.flops / median_us / 1e6:6.1f} TFLOPS")
        print(f"M={m:<6} " + " | ".join(cells), flush=True)
        del attempts
        torch.cuda.empty_cache()

    if args.no_save:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_shape = {"shape_n": args.shape_n, "shape_k": args.shape_k}
    if args.num_experts:
        problem_shape.update(num_experts=args.num_experts, top_k=args.top_k, projection=args.projection,
                             distribution=args.distribution, ep=args.ep, rank=args.rank, seed=args.seed)
    for entry in entries:
        path = output_dir / f"{entry.library}_{entry.recipe}.json"
        path.write_text(json.dumps({
            "problem": {**problem_shape, "recipe": entry.recipe, "weight_format": entry.weight_format,
                        "bytes_per_weight": entry.bytes_per_weight,
                        **({"layout": entry.layout} if args.num_experts else {})},
            "device": {"device_name": device["gpu_name"], "memory_gbps": device["peak_bw_gbps"],
                       "compute_tops": device["peak_fp8_tflops"]},
            "packages": {"torch": torch.__version__, entry.library: entry.version, **packages},
            "timing": {**dataclasses.asdict(timing), "date": f"{datetime.date.today()}"},
            "result": results[entry.name],
        }, indent=4) + "\n")
        print(f"wrote {path}", flush=True)
