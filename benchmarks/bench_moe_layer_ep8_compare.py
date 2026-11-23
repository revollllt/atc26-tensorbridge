"""Single-node EP8 expert layer comparison. One backend per torchrun. Summarization runs on CPU."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import multiprocessing
import os
import statistics
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

if TYPE_CHECKING:
    from tensorbridge.nvlink_ep import NVLinkSymmetricEP

FP4_ADAPTERS = ("marlin_nvfp4", "humming_nvfp4", "humming_mxfp4", "humming_mxfp4_g128")
EXTERNAL = ("dg_upstream_sm90_fp8_megamoe", "sgl_pr53_sm90_fp8_megamoe", "sgl_pr53_sm90_mxfp4_megamoe")
BASELINES = (*FP4_ADAPTERS, EXTERNAL[0], EXTERNAL[2])
MODULAR_TB = ("tb_nvfp4_fp8", "tb_nvfp4_fp8_g128")
MODULAR = (*FP4_ADAPTERS, "fp8_deepgemm_agrs", *MODULAR_TB)
NVLINK = ("tb_nvfp4_megamoe",)
BACKENDS = (*MODULAR, *EXTERNAL, *NVLINK)
COLLECTIVES = ("allgather_reducescatter", "flashinfer_nvlink_one_sided")
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 191, 192, 193, 255, 256, 257, 512)
Projection = tuple[torch.Tensor, torch.Tensor]
Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, order=True)
class Case:
    tokens: int
    routing: str
    seed: int
    world: int
    experts: int
    hidden: int
    intermediate: int
    top_k: int
    shared_width: int
    rounds: int
    iters: int
    request_graph: bool
    gpu: str
    node: str
    torch_version: str


@dataclass
class Result:
    backend: str
    all2all: str
    capacity: int | None = None
    implementation: str = ""
    dispatch: str = ""
    e2e_us: float | None = None
    e2e_mode: str = ""
    e2e_round_us: list[float] | None = None
    kernel_only_us: float | None = None
    kernel_only_mode: str = ""
    kernel_only_round_us: list[float] | None = None
    graph_error: str = ""
    kernel_only_graph_error: str = ""
    tokens_by_rank: list[int] | None = None
    rel_rms: float | None = None
    cos_sim: float | None = None
    status: str = "ok"
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--backend", choices=BACKENDS)
    action.add_argument("--summarize", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="run JSONL, or report prefix for .md and .csv")
    parser.add_argument("--routing", choices=("uniform", "zipf", "mixed_empty"), default="uniform")
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--with-shared-mlp", action="store_true")
    parser.add_argument("--shared-intermediate", type=int, default=2048)
    parser.add_argument("--shared-reserve-sms", type=int, default=0)
    parser.add_argument("--max-tokens-per-rank", type=int, default=256, help="TensorBridge NVLinkEP chunk capacity")
    parser.add_argument("--all2all-backend", choices=COLLECTIVES, default=COLLECTIVES[0])
    parser.add_argument("--profile", action="store_true", help="capture an extra lockstep batch in Nsight")
    parser.add_argument("--tb-rule-only", action="store_true", help="disable saved TB tile profiles for this process")
    parser.add_argument("--deepgemm-path", type=Path)
    parser.add_argument("--point-timeout", type=int, default=900, help="external worker timeout per T in seconds")
    args = parser.parse_args(argv)
    if args.summarize:
        return args
    if (
        min(
            *args.tokens,
            args.experts,
            args.hidden,
            args.intermediate,
            args.top_k,
            args.shared_intermediate,
            args.iters,
            args.rounds,
            args.max_tokens_per_rank,
        )
        <= 0
    ):
        parser.error("shapes, tokens, iterations, rounds and capacity must be positive")
    if args.shared_reserve_sms < 0:
        parser.error("--shared-reserve-sms must be non-negative")
    if args.experts % 8 or args.top_k > args.experts or args.hidden % 128 or args.intermediate % 128:
        parser.error("EP8 needs E divisible by 8, top_k <= E, and block-128 H/I")
    if args.profile and (len(args.tokens) != 1 or args.backend not in MODULAR):
        parser.error("--profile takes one modular backend and one token point per Nsight capture")
    if args.tb_rule_only and (args.backend not in MODULAR_TB or os.environ.get("TENSORBRIDGE_AUTOTUNE") == "1"):
        parser.error("--tb-rule-only requires a modular TB preset with autotune disabled")
    if args.backend in MODULAR_TB and args.all2all_backend != COLLECTIVES[0]:
        parser.error(f"{args.backend} requires allgather_reducescatter")
    if args.point_timeout <= 0:
        parser.error("--point-timeout must be positive")
    if args.backend in EXTERNAL and args.deepgemm_path is None:
        parser.error("external MegaMoE requires --deepgemm-path pointing to its import root")
    return args


def rank_sizes(tokens: int, routing: str, world: int) -> list[int]:
    return [0 if routing == "mixed_empty" and rank % 2 else tokens for rank in range(world)]


def make_inputs(case: Case, rank: int, device: torch.device) -> Inputs:
    # A private CPU generator keeps routing and activations independent of kernel setup and CUDA RNG state.
    generator = torch.Generator().manual_seed(case.seed + 65537 * rank + 1000003 * case.tokens)
    rows = rank_sizes(case.tokens, case.routing, case.world)[rank]
    x = torch.randn(rows, case.hidden, generator=generator, dtype=torch.bfloat16, device="cpu")
    if rows:
        probabilities = (
            torch.arange(1, case.experts + 1, dtype=torch.float64, device="cpu").reciprocal()
            if case.routing == "zipf"
            else torch.ones(case.experts, dtype=torch.float64, device="cpu")
        )
        ids = torch.multinomial(probabilities.expand(rows, -1), case.top_k, replacement=False, generator=generator)
    else:
        ids = torch.empty(0, case.top_k, dtype=torch.long, device="cpu")
    scores = torch.randn(rows, case.top_k, generator=generator, dtype=torch.float32, device="cpu").softmax(-1)
    return x.to(device), ids.to(device=device, dtype=torch.int32), scores.to(device)


def fp8_shard(experts: int, channels: int, width: int, generator: torch.Generator, device: torch.device) -> Projection:
    codes = torch.empty((experts, channels, width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty((experts, channels // 128, width // 128), dtype=torch.float32, device=device)
    for expert in range(experts):
        weight = torch.randn(channels, width, dtype=torch.bfloat16, device=device, generator=generator) / math.sqrt(
            width
        )
        blocks = weight.float().view(channels // 128, 128, width // 128, 128)
        scale = blocks.abs().amax((1, 3)).clamp_min(1e-12) / 448
        codes[expert] = (blocks / scale[:, None, :, None]).reshape(channels, width).to(codes.dtype)
        scales[expert] = scale
    return codes, scales


def dequant_expert(projection: Projection, expert: int) -> torch.Tensor:
    codes, scales = projection
    channels, width = codes.shape[1:]
    return (
        codes[expert].float().view(channels // 128, 128, width // 128, 128) * scales[expert, :, None, :, None]
    ).reshape(channels, width)


def reference(
    inputs: Inputs,
    sizes: list[int],
    rank: int,
    gate_up: Projection,
    down: Projection,
    shared: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    # Gather equal-sized buffers, then remove padding. Even empty owners still compute for remote tokens.
    gathered = []
    for local in inputs:
        padded = F.pad(local, (0, 0, 0, max(sizes) - len(local)))
        peers = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(peers, padded)
        gathered.append(torch.cat([tensor[:length] for tensor, length in zip(peers, sizes, strict=True)]))
    x, ids, scores = gathered
    x = x.float()
    partial = torch.zeros_like(x)
    local_experts = gate_up[0].size(0)
    for expert in range(local_experts):
        tokens, slots = (ids == rank * local_experts + expert).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        gate, up = (x[tokens] @ dequant_expert(gate_up, expert).T).chunk(2, -1)
        output = (F.silu(gate) * up) @ dequant_expert(down, expert).T
        partial.index_add_(0, tokens, output * scores[tokens, slots, None])
    dist.all_reduce(partial)
    local_ref = partial.narrow(0, sum(sizes[:rank]), sizes[rank])
    if shared is None:
        return local_ref
    gate, up = (inputs[0].float() @ shared[0].float().T).chunk(2, -1)
    return local_ref + (F.silu(gate) * up) @ shared[1].float().T


def error_metrics(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    if y.shape != ref.shape or y.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 {tuple(ref.shape)}, got {y.dtype} {tuple(y.shape)}")
    actual, expected = y.double().flatten(), ref.double().flatten()
    norms = torch.stack(
        ((actual - expected).square().sum(), expected.dot(expected), actual.dot(actual), actual.dot(expected))
    )
    dist.all_reduce(norms)
    diff, target, output, dot = norms.tolist()
    return math.sqrt(diff / target), dot / math.sqrt(output * target) if output > 0 else 0.0


def rank_errors(error: str, control: dist.ProcessGroup) -> str:
    errors = [""] * dist.get_world_size(control)
    dist.all_gather_object(errors, error, group=control)
    return " | ".join(f"rank {rank}: {msg}" for rank, msg in enumerate(errors) if msg)


def measure(
    call: Callable[[], torch.Tensor],
    case: Case,
    control: dist.ProcessGroup,
    *,
    recv_check: tuple[torch.Tensor, torch.Tensor] | None = None,
    profile: bool = False,
) -> tuple[list[float], str, str, torch.Tensor]:
    # Capture only after JIT/allocation and all collective communicators are warm.
    for _ in range(3):
        output = call()
    torch.cuda.synchronize()
    dist.barrier(group=control)
    capture_error = ""
    graph = None
    if case.request_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = call()
        except (RuntimeError, ValueError, AssertionError, NotImplementedError) as error:
            capture_error = f"{type(error).__name__}: {error}"
        capture_error = rank_errors(capture_error, control)
        if capture_error:
            graph = None
        else:
            # Lazy graph instantiation may reject a capture on its first replay.
            try:
                graph.replay()
                torch.cuda.synchronize()
            except RuntimeError as error:
                capture_error = f"first replay: {type(error).__name__}: {error}"
            capture_error = rank_errors(capture_error, control)
            if capture_error:
                graph = None
                output = call()
            else:
                output = captured
    else:
        graph = None
    times = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    warmup_iters = 3
    initial_counts = recv_check[0].clone() if recv_check is not None else None
    for _ in range(case.rounds):
        # Enqueue a WORLD NCCL rendezvous on the caller stream without waking the host at completion.
        dist.barrier(group=dist.group.WORLD, async_op=True).block_current_stream()
        for _ in range(warmup_iters):
            if graph is None:
                output = call()
            else:
                graph.replay()
        start.record()
        for _ in range(case.iters):
            if graph is None:
                output = call()
            else:
                graph.replay()
        stop.record()
        stop.synchronize()
        latency = torch.tensor(start.elapsed_time(stop) * 1000 / case.iters, dtype=torch.float64, device="cpu")
        dist.all_reduce(latency, op=dist.ReduceOp.MAX, group=control)
        times.append(latency.item())
    if recv_check is not None:
        counters, routes_per_call = recv_check
        launches = case.rounds * (warmup_iters + case.iters)
        observed = counters - initial_counts
        expected = routes_per_call * launches
        if not torch.equal(observed, expected):
            raise ArithmeticError(
                f"DeepGEMM receive counts over {launches} launches: "
                f"observed {observed.tolist()}, expected {expected.tolist()}"
            )
    if profile:
        # Nsight subscribes before rendezvous. Match the normal round and label only the measured replays.
        with torch.cuda.profiler.profile():
            dist.barrier(group=dist.group.WORLD, async_op=True).block_current_stream()
            launch = call if graph is None else graph.replay
            for _ in range(warmup_iters):
                launch()
            with torch.cuda.nvtx.range("ep8_layer"):
                for _ in range(case.iters):
                    launch()
                torch.cuda.synchronize()
    return times, "eager" if graph is None else "graph", capture_error, output


def external_weights(
    dg: ModuleType, backend: str, gate_up: Projection, down: Projection
) -> tuple[Projection, Projection]:
    if backend != "sgl_pr53_sm90_mxfp4_megamoe":
        return dg.transform_weights_for_mega_moe_sm90(gate_up, down)
    projections = []
    for projection in (gate_up, down):
        quantized = [
            dg.utils.per_token_cast_to_fp4(dequant_expert(projection, expert).bfloat16(), use_ue8m0=True, gran_k=32)
            for expert in range(projection[0].size(0))
        ]
        packed, scales = zip(*quantized, strict=True)
        projections.append((torch.stack(packed), torch.stack(scales)))
    return dg.transform_weights_for_mega_moe_sm90_fp4(*projections)


def compatible_kwargs(function: Callable[..., object], options: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(function)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def run(args: argparse.Namespace, stack: ExitStack, point_dir: Path | None = None, point_index: int = 0) -> None:
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise ValueError("launch with torchrun --standalone --nproc-per-node=8")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        "nccl", init_method=(point_dir / "store").as_uri() if point_dir else "env://", rank=rank, world_size=8
    )
    control = dist.new_group(backend="gloo")
    local_e = args.experts // 8
    generator = torch.Generator(device=device).manual_seed(args.seed + 2000003 + rank)
    gate_up = fp8_shard(local_e, 2 * args.intermediate, args.hidden, generator, device)
    down = fp8_shard(local_e, args.hidden, args.intermediate, generator, device)
    if args.with_shared_mlp:
        generator.manual_seed(args.seed + 3000017)  # replicated across all eight ranks
        shared = tuple(
            torch.randn(n, k, dtype=torch.bfloat16, device=device, generator=generator) / math.sqrt(k)
            for n, k in ((2 * args.shared_intermediate, args.hidden), (args.hidden, args.shared_intermediate))
        )
    else:
        shared = None
    ep: NVLinkSymmetricEP | None = None
    capacity = None
    if args.backend in MODULAR:
        # Set before importing the plugin. Tile profiles load once in each worker.
        os.environ["TENSORBRIDGE_DISABLE_PROFILES"] = "1" if args.tb_rule_only else "0"
        from bench_moe_mlp_e2e import native_layer
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import destroy_model_parallel, init_distributed_environment, initialize_model_parallel
        from vllm.forward_context import get_forward_context, set_forward_context
        from vllm.v1.worker.workspace import init_workspace_manager

        config = VllmConfig(
            parallel_config=ParallelConfig(
                data_parallel_size=8,
                data_parallel_size_local=8,
                enable_expert_parallel=True,
                is_moe_model=True,
                distributed_executor_backend="external_launcher",
                all2all_backend=args.all2all_backend,
            )
        )
        config.scheduler_config.max_num_batched_tokens = max(
            config.scheduler_config.max_num_batched_tokens, max(args.tokens)
        )
        stack.enter_context(set_current_vllm_config(config))
        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank)
        stack.callback(destroy_model_parallel)
        initialize_model_parallel(1, 1)
        init_workspace_manager(device)
        layer = native_layer(config, args.backend, gate_up, down, args.top_k, 8)
        kernel = layer.quant_method.moe_kernel
        implementation = type(kernel.fused_experts).__name__
        dispatch = type(kernel.prepare_finalize).__name__
    elif args.backend in NVLINK:
        from tensorbridge import serving
        from tensorbridge.nvlink_ep import NVLinkSymmetricEP
        from tensorbridge.quant import nvfp4

        w13, s13, g13 = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(*gate_up, is_gated_w13=True, bridge="exact")
        w2, s2, g2 = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(*down, is_gated_w13=False, bridge="exact")
        prepared = serving.prepare_moe(
            w13,
            s13,
            g13,
            w2,
            s2,
            g2,
            args.top_k,
            args.experts,
            8,
            "exact",
            layout="masked",
            scale_convention="multiplier",
            gate_up_quant="group_128",
            down_quant="group_128",
            fuse_swiglu_fp8=True,
        )
        capacity = args.max_tokens_per_rank
        ep = NVLinkSymmetricEP(
            dist.group.WORLD,
            local_e,
            args.hidden,
            args.intermediate,
            args.top_k,
            max_tokens_per_rank=capacity,
            device=device,
        )
        stack.callback(ep.close)
        ep.precompile(prepared.w13, prepared.w2, capture_sizes=args.tokens)
        implementation, dispatch = "megamoe", "NVLinkSymmetricEP"
    else:
        root = args.deepgemm_path.resolve()
        sys.path.insert(0, str(root))
        import deep_gemm as dg

        if not Path(dg.__file__).absolute().is_relative_to(root):
            raise ImportError(f"Requested {root}, imported {dg.__file__}")
        transformed = external_weights(dg, args.backend, gate_up, down)
        upstream = args.backend == "dg_upstream_sm90_fp8_megamoe"
        mx = args.backend == "sgl_pr53_sm90_mxfp4_megamoe"
        factory = dg.get_symm_buffer_for_sm90_mega_moe if upstream else dg.get_symm_buffer_for_mega_moe
        function = dg.fp8_fp4_mega_moe if mx else dg.fp8_mega_moe
        if upstream:
            pre_dispatch = None
        else:
            try:
                from deep_gemm import mega_moe_pre_dispatch_sm90 as pre_dispatch
            except ImportError as error:
                if error.name != "deep_gemm":
                    raise
                pre_dispatch = None
        implementation = f"{dg.__file__}:{function.__name__}"
        dispatch = "cast+copy" if pre_dispatch is None else pre_dispatch.__name__
        buffer = None
        cum_stats = torch.zeros(local_e, device=device, dtype=torch.int32)
    output_file = args.output or Path(f"{args.backend}-{args.routing}.jsonl")
    if rank == 0:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        writer = stack.enter_context(output_file.open("x"))
    for index, tokens in enumerate(args.tokens):
        if point_dir is not None and index != point_index:
            continue
        case = Case(
            tokens,
            args.routing,
            args.seed,
            8,
            args.experts,
            args.hidden,
            args.intermediate,
            args.top_k,
            args.shared_intermediate if shared is not None else 0,
            args.rounds,
            args.iters,
            not args.eager,
            torch.cuda.get_device_name(device),
            os.uname().nodename,
            str(torch.__version__),
        )
        result = Result(
            args.backend + ("_rule" if args.tb_rule_only else ""),
            args.all2all_backend if args.backend in MODULAR else "",
            capacity,
            implementation,
            dispatch,
        )
        sizes = rank_sizes(tokens, case.routing, case.world)
        result.tokens_by_rank = sizes
        if point_dir is not None and rank == 0:
            (point_dir / "case.json").write_text(json.dumps({"case": asdict(case), **asdict(result)}))
        inputs = make_inputs(case, rank, device)
        x, ids, scores = inputs
        check = index == 0 or args.check_correctness
        expected = reference(inputs, sizes, rank, gate_up, down, shared) if check else None
        y = torch.empty_like(x)
        shared_out = torch.empty_like(x) if shared is not None else None

        def shared_hook() -> None:
            gate, up = (x @ shared[0].T).chunk(2, -1)
            torch.mm(F.silu(gate) * up, shared[1].T, out=shared_out)

        with ExitStack() as point:
            if args.backend in MODULAR:
                point.enter_context(
                    set_forward_context(
                        None,
                        config,
                        num_tokens=len(x),
                        num_tokens_across_dp=torch.tensor(sizes, dtype=torch.int32, device="cpu"),
                    )
                )
                point.enter_context(get_forward_context().dp_metadata.sp_local_sizes(1))

                def routed() -> torch.Tensor:
                    return layer.quant_method.apply(layer, x, scores, ids, None, None)
            elif ep is not None:

                def routed() -> torch.Tensor:
                    return ep.forward(
                        x,
                        ids,
                        scores,
                        prepared.w13,
                        prepared.w2,
                        shared_experts_hook=shared_hook if shared is not None else None,
                        global_num_tokens=max(sizes),
                        reserve_sms=args.shared_reserve_sms,
                        out=y,
                    )
            else:
                # Prefer one common capacity throughout the sweep. If the build rejects it, try the point's capacity.
                attempts = []
                for requested in dict.fromkeys((max(256, max(args.tokens)), max(256, tokens))):
                    local_error = ""
                    try:
                        buffer = factory(
                            dist.group.WORLD,
                            args.experts,
                            requested,
                            args.top_k,
                            args.hidden,
                            args.intermediate,
                            **compatible_kwargs(factory, dict(use_fp8_dispatch=True, activation="swiglu")),
                        )
                        stack.callback(buffer.destroy)
                    except (ValueError, RuntimeError, AssertionError) as error:
                        local_error = f"capacity={requested}, {type(error).__name__}: {error}"
                        # Keep the exact cause if another rank is already in symmetric rendezvous.
                        (point_dir / f"allocation-{rank}").write_text(local_error)
                    rejection = rank_errors(local_error, control)
                    if rejection:
                        attempts.append(rejection)
                    else:
                        break
                else:
                    result.status, result.error = "unsupported", " | ".join(attempts)
                if result.status == "ok":
                    result.capacity = buffer.num_max_tokens_per_rank
                    result.dispatch = f"{dispatch}; ids={ids.dtype}->{buffer.topk_idx.dtype}"

                options = compatible_kwargs(
                    function,
                    dict(
                        cumulative_local_expert_recv_stats=cum_stats,
                        recipe=(1, 1, 32) if mx else (128, 128, 128),
                        activation="swiglu",
                        fast_math=True,
                        num_tokens_bound=max(sizes),
                    ),
                )

                def kernel_only() -> torch.Tensor:
                    function(y, *transformed, buffer, **options)
                    return y

                def routed() -> torch.Tensor:
                    if pre_dispatch is None:
                        if not len(x):
                            return kernel_only()
                        codes, scales = dg.utils.per_token_cast_to_fp8(
                            x, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
                        )
                        buffer.x[: len(x)].copy_(codes)
                        buffer.x_sf[: len(x)].copy_(scales)
                        buffer.topk_idx[: len(x)].copy_(ids)
                        buffer.topk_weights[: len(x)].copy_(scores)
                    else:
                        pre_dispatch(
                            x,
                            ids,
                            scores,
                            buffer.x,
                            buffer.x_sf,
                            buffer.topk_idx,
                            buffer.topk_weights,
                            len(x),
                            group_size=128,
                            routed_scaling_factor=1.0,
                        )
                    return kernel_only()

            def forward() -> torch.Tensor:
                routed_out = routed()
                if shared is None:
                    return routed_out
                if ep is None:
                    shared_hook()
                return routed_out + shared_out

            if result.status == "ok":
                times, mode, error, actual = measure(forward, case, control, profile=args.profile)
                result.e2e_round_us, result.e2e_mode, result.graph_error = times, mode, error
                result.e2e_us = statistics.median(times)
                if expected is not None:
                    result.rel_rms, result.cos_sim = error_metrics(actual, expected)
                    if not all(math.isfinite(value) for value in (result.cos_sim, result.rel_rms)):
                        result.status, result.error = "invalid", "nonfinite global output error"
                        result.rel_rms = result.cos_sim = None
                if args.profile:
                    counts = torch.bincount(ids.flatten().long(), minlength=case.experts)
                    dist.all_reduce(counts)
                    histogram = counts.narrow(0, rank * local_e, local_e).tolist()
                    diagnostics = dict(rank=rank, backend=result.backend, tokens=tokens, expert_rows=histogram)
                    if args.backend in MODULAR_TB and args.all2all_backend == COLLECTIVES[0]:
                        from tensorbridge import megamoe
                        from tensorbridge.jit import runtime

                        rows = math.ceil(sum(sizes) * case.top_k / case.world)
                        grouped = args.backend == "tb_nvfp4_fp8"
                        for projection, n, k in (("gate_up", 2 * case.intermediate, case.hidden),
                                                 ("down", case.hidden, case.intermediate)):
                            if (
                                projection == "gate_up"
                                and args.backend == "tb_nvfp4_fp8_g128"
                                and megamoe.should_fuse_swiglu_contiguous(rows, n, k, local_e, device=local_rank)
                            ):
                                tile = megamoe.select_swiglu_config(
                                    rows, n, k, local_e, "contiguous", device=local_rank
                                )
                            else:
                                tile = runtime.select_config(rows, n, k, local_e, "contiguous",
                                                             device=local_rank, group_scales=not grouped)
                            diagnostics[projection] = dict(tile=asdict(tile), row_hint=rows, tile_rows=sum(
                                math.ceil(count / tile.block_m) * tile.block_m for count in histogram
                            ))
                    print(json.dumps(diagnostics), flush=True)
                if args.backend in EXTERNAL:
                    # Symmetric input buffers still contain this point's staged tokens and routes.
                    if check and "cumulative_local_expert_recv_stats" in options:
                        counts = torch.bincount(ids.flatten().long(), minlength=case.experts).int()
                        dist.all_reduce(counts)
                        recv_check = (cum_stats, counts.narrow(0, rank * local_e, local_e))
                    else:
                        recv_check = None
                    times, mode, error, _ = measure(kernel_only, case, control, recv_check=recv_check)
                    result.kernel_only_round_us, result.kernel_only_mode = times, mode
                    result.kernel_only_us = statistics.median(times)
                    result.kernel_only_graph_error = error
        reasons = rank_errors(result.error, control)
        result.error = reasons
        if rank == 0:
            line = json.dumps({"case": asdict(case), **asdict(result)}, allow_nan=False)
            print(line, file=writer, flush=True)
            if point_dir is None:
                print(line, flush=True)


def summarize(paths: Sequence[Path], prefix: Path) -> None:
    groups: dict[Case, dict[tuple[str, str], Result]] = {}
    for path in paths:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                record = json.loads(line)
                case = Case(**record.pop("case"))
                result = Result(**record)
                key = result.backend, result.all2all
                peers = groups.setdefault(case, {})
                if key in peers:
                    raise ValueError(f"{path}:{number}: duplicate {key} at {case}; select one run")
                peers[key] = result

    rows = []
    for case, peers in sorted(groups.items()):
        # Direct kernels can be compared against each modular transport; never silently cross transports.
        for collective in sorted({transport for _, transport in peers if transport} or {""}):
            panel = {name: result for (name, transport), result in peers.items() if transport in ("", collective)}
            for name in sorted(panel):
                result = panel[name]
                row = {
                    **asdict(case),
                    "comparison_all2all": collective,
                    "backend": name,
                    "all2all": result.all2all,
                    "status": result.status,
                    "e2e_mode": result.e2e_mode,
                    "e2e_us": result.e2e_us,
                    "kernel_only_mode": result.kernel_only_mode,
                    "kernel_only_us": result.kernel_only_us,
                    "global_tokens": sum(rank_sizes(case.tokens, case.routing, case.world)),
                    "rel_rms": result.rel_rms,
                    "cos_sim": result.cos_sim,
                    "capacity": result.capacity,
                    "error": result.error,
                    "graph_error": result.graph_error,
                    "kernel_only_graph_error": result.kernel_only_graph_error,
                }
                for baseline in BASELINES:
                    base = panel.get(baseline)
                    row[f"vs_{baseline}"] = (
                        base.e2e_us / result.e2e_us
                        if base
                        and base.status == result.status == "ok"
                        and base.e2e_mode == result.e2e_mode
                        and base.e2e_us
                        and result.e2e_us
                        else None
                    )
                rows.append(row)
    if not rows:
        raise ValueError("No EP8 measurements in input files")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = (
        "tokens",
        "routing",
        "backend",
        "status",
        "e2e_mode",
        "e2e_us",
        "kernel_only_us",
        *(f"vs_{name}" for name in BASELINES),
        "rel_rms",
        "cos_sim",
    )
    scope_keys = [field.name for field in fields(Case) if field.name not in ("tokens", "routing")] + [
        "comparison_all2all"
    ]
    rows.sort(key=lambda row: tuple(row[key] for key in (*scope_keys, "routing", "tokens", "backend")))
    lines = []
    previous = None
    for row in rows:
        scope = tuple((key, row[key]) for key in scope_keys)
        if previous != scope:
            lines.extend(
                (
                    "",
                    " ".join(f"{k}={v}" for k, v in scope),
                    "",
                    "| " + " | ".join(columns) + " |",
                    "| " + " | ".join("---" for _ in columns) + " |",
                )
            )
            previous = scope
        values = [row[key] for key in columns]
        lines.append(
            "| "
            + " | ".join("—" if v is None else f"{v:.5g}" if isinstance(v, float) else str(v) for v in values)
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines).strip() + "\n")


def external_point(args: argparse.Namespace, directory: Path, index: int) -> None:
    args.output = directory / "result.jsonl"
    rank = int(os.environ["RANK"])
    try:
        with torch.no_grad(), ExitStack() as stack:
            try:
                run(args, stack, directory, index)
            except Exception as error:
                # Publish before releasing symmetric memory or destroying a partially launched NCCL world.
                failure = {
                    "rank": rank,
                    "trace": traceback.format_exc(),
                    "status": "unsupported"
                    if isinstance(error, (ValueError, NotImplementedError, AssertionError))
                    else "error",
                }
                message = directory / f"error-{rank}.tmp"
                message.write_text(json.dumps(failure))
                message.replace(directory / f"error-{rank}.json")
                raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def external_sweep(args: argparse.Namespace) -> None:
    """Keep the CPU control group alive when a rank-local kernel rejection strands GPU peers."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise ValueError("launch with torchrun --standalone --nproc-per-node=8")
    output = args.output or Path(f"{args.backend}-{args.routing}.jsonl")
    with ExitStack() as stack:
        if rank == 0:
            root = stack.enter_context(tempfile.TemporaryDirectory(prefix="tb-ep8-"))
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = stack.enter_context(output.open("x"))
        else:
            root = ""
        roots = [root]
        dist.broadcast_object_list(roots, src=0)
        for index, tokens in enumerate(args.tokens):
            directory = Path(roots[0]) / str(index)
            directory.mkdir(exist_ok=True)
            child = multiprocessing.get_context("spawn").Process(target=external_point, args=(args, directory, index))
            child.start()
            deadline = time.monotonic() + args.point_timeout
            try:
                while True:
                    child.join(timeout=0.5)
                    failed = child.exitcode not in (0, None) or (directory / f"error-{rank}.json").exists()
                    state = torch.tensor([failed, child.is_alive(), time.monotonic() >= deadline], dtype=torch.int32)
                    dist.all_reduce(state, op=dist.ReduceOp.MAX)
                    failed, running, timeout = state.tolist()
                    if failed or timeout or not running:
                        break
            finally:
                if child.is_alive():
                    child.kill()
                child.join()
            dist.barrier()
            if not failed and not timeout:
                record = json.loads((directory / "result.jsonl").read_text())
            else:
                failures = [json.loads(path.read_text()) for path in sorted(directory.glob("error-*.json"))]
                errors = [f"rank {report['rank']}: {report['trace']}" for report in failures]
                errors.extend(path.read_text() for path in sorted(directory.glob("allocation-*")))
                if timeout:
                    errors.append(f"timed out after {args.point_timeout}s")
                reason = "\n".join(errors) or "GPU worker exited abnormally"
                if not (directory / "case.json").exists():
                    raise RuntimeError(f"External backend setup failed before T={tokens}: {reason}")
                record = json.loads((directory / "case.json").read_text())
                status = (
                    "unsupported"
                    if failures and all(r["status"] == "unsupported" for r in failures) and not timeout
                    else "error"
                )
                record.update(status=status, error=reason)
            if rank == 0:
                line = json.dumps(record, allow_nan=False)
                print(line, file=writer, flush=True)
                print(line, flush=True)
            dist.barrier()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args.summarize, args.output or Path("ep8-summary"))
        return
    try:
        if args.backend in EXTERNAL:
            external_sweep(args)
        else:
            with ExitStack() as stack, torch.no_grad():
                run(args, stack)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
