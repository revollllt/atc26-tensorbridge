"""Route B (GC-MoE): CUDA Green Context SM partitioning and bidirectional streaming overlap for MoE EP.

Splits a Hopper GPU into a dedicated communication partition (`comm_stream`) and a remaining compute
partition (`compute_stream`) via `cuDevSmResourceSplitByCount`. Compute kernels launched on
`compute_stream` keep independent per-projection Route-A configs (`gate_up` and `down` stay separate
kernels, automatically selecting profiles for `compute_sms`), while overlapping with EP dispatch and
combine through a lock-free ready ring (`StreamingMaskedScheduler`) and epilogue completion signals.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from tensorbridge.jit import runtime
from tensorbridge.jit.codegen import GemmConfig
from tensorbridge.quant.nvfp4 import NVFP4Experts


class GreenContextPair:
    """A hardware-partitioned `(comm, compute)` pair of CUDA Green Contexts and non-blocking streams."""

    def __init__(self, min_comm_sms: int = 8, device: int | None = None) -> None:
        runtime.load_launcher()
        self.device = torch.cuda.current_device() if device is None else device
        pid, comm_ptr, compute_ptr, comm_sms, compute_sms = (
            torch.ops.tensorbridge.create_green_context_partition(self.device, int(min_comm_sms))
        )
        self._partition_id: int | None = int(pid)
        self.comm_sms: int = int(comm_sms)
        self.compute_sms: int = int(compute_sms)
        self.comm_stream = torch.cuda.ExternalStream(int(comm_ptr), device=torch.device("cuda", self.device))
        self.compute_stream = torch.cuda.ExternalStream(int(compute_ptr), device=torch.device("cuda", self.device))

    def close(self) -> None:
        if self._partition_id is not None:
            pid = self._partition_id
            self._partition_id = None
            torch.ops.tensorbridge.destroy_green_context_partition(pid)

    def __enter__(self) -> GreenContextPair:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclasses.dataclass
class StreamingOverlapWorkspace:
    """Device-side handshake buffers shared between Comm GC (Dispatch/Combine) and Compute GC (GateUp/Down).

    All control words are 32-bit CUDA tensors so `reset()` can run inside a captured CUDA Graph without
    host synchronization.
    """

    num_experts: int
    ring_capacity: int
    ready_ring: torch.Tensor       # int32 [ring_capacity, 4]: (expert_id, row_begin, row_end, m_tile_idx)
    write_head: torch.Tensor       # int32 [1]: monotonic published M-tile count (release by producer)
    dispatch_done: torch.Tensor    # int32 [1]: set to 1 with release semantics when dispatch finishes
    work_cursor: torch.Tensor      # int32 [1]: global atomic cursor stolen by Compute CTAs
    tile_done_counts: torch.Tensor # int32 [E]: per-expert finished tile counter incremented by Down epilogue
    combine_ready: torch.Tensor    # int32 [E]: set to 1 with release semantics when expert e finishes Down
    _ctrl_buf: torch.Tensor | None = None  # single cache-line-aligned backing buffer for 1-call reset()

    @classmethod
    def allocate(
        cls, num_experts: int, ring_capacity: int, device: torch.device | int | None = None
    ) -> StreamingOverlapWorkspace:
        dev = torch.device("cuda", torch.cuda.current_device() if device is None else device) if not isinstance(
            device, torch.device
        ) else device
        # Pack all 5 control regions into one 128-byte-cache-line-padded buffer so `reset()` executes as a
        # single cuMemsetD32Async (~1.5us) with zero false sharing between Comm and Compute SMs.
        line = 32  # 32 * 4B = 128B L2 cache line
        e_stride = ((num_experts + line - 1) // line) * line
        ctrl = torch.zeros((3 * line + 2 * e_stride,), dtype=torch.int32, device=dev)
        return cls(
            num_experts=num_experts,
            ring_capacity=ring_capacity,
            ready_ring=torch.zeros((ring_capacity, 4), dtype=torch.int32, device=dev),
            write_head=ctrl[0:1],
            dispatch_done=ctrl[line:line + 1],
            work_cursor=ctrl[2 * line:2 * line + 1],
            tile_done_counts=ctrl[3 * line:3 * line + num_experts],
            combine_ready=ctrl[3 * line + e_stride:3 * line + e_stride + num_experts],
            _ctrl_buf=ctrl,
        )

    def reset(self) -> None:
        """Zero all control counters in-place (safe inside CUDA Graph capture)."""
        if self._ctrl_buf is not None:
            self._ctrl_buf.zero_()
        else:
            self.write_head.zero_()
            self.dispatch_done.zero_()
            self.work_cursor.zero_()
            self.tile_done_counts.zero_()
            self.combine_ready.zero_()


def fp8_nvfp4_grouped_gemm_streaming(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    experts: NVFP4Experts,
    workspace: StreamingOverlapWorkspace,
    m_hint: int,
    out: torch.Tensor | None = None,
    config: GemmConfig | None = None,
    enable_pdl: bool = False,
    *,
    group_scales: bool = False,
) -> torch.Tensor:
    """Launch a masked grouped GEMM that consumes ready M-tiles dynamically from `workspace.ready_ring`.

    Must be launched on `compute_stream` concurrently with a Dispatch producer on `comm_stream` that
    writes valid FP8 rows + scales into `a` / `a_scale`, appends frozen `(expert_id, row_begin, row_end, m_tile_idx)`
    entries to `workspace.ready_ring`, increments `workspace.write_head` with release semantics, and
    sets `workspace.dispatch_done = 1` once all experts have arrived.
    """
    num_experts, n, k = experts.shape
    if config is None:
        config = runtime.select_config(
            m_hint, n, k, num_experts, "masked", device=a.get_device(),
            bridge=experts.bridge, group_scales=group_scales,
        )
    kernel_id = runtime.get_grouped_kernel(
        n, k, num_experts, "masked", config, bridge=experts.bridge, device=a.get_device(),
        group_scales=group_scales, streaming=True, signal_done=False,
    )
    return torch.ops.tensorbridge.launch_streaming_grouped_kernel(
        kernel_id, a, a_scale, experts.weight, experts.scale, experts.global_scale, experts.locks,
        None, workspace.ready_ring, workspace.write_head, workspace.dispatch_done, workspace.work_cursor,
        None, None, out, enable_pdl,
    )


def fp8_nvfp4_grouped_gemm_signaling(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    experts: NVFP4Experts,
    expert_counts: torch.Tensor,
    workspace: StreamingOverlapWorkspace,
    m_hint: int,
    out: torch.Tensor | None = None,
    config: GemmConfig | None = None,
    enable_pdl: bool = False,
    *,
    group_scales: bool = False,
) -> torch.Tensor:
    """Launch a masked grouped GEMM (`down`) that signals `workspace.combine_ready[e] = 1` per expert.

    As soon as all `(m_tiles * n_tiles)` of expert `e` have reached global memory, the last finishing
    CTA release-stores `workspace.combine_ready[e] = 1` so a concurrent Combine kernel on `comm_stream`
    can begin sending expert `e`'s rows back immediately without waiting for straggler experts.
    """
    num_experts, n, k = experts.shape
    if config is None:
        config = runtime.select_config(
            m_hint, n, k, num_experts, "masked", device=a.get_device(),
            bridge=experts.bridge, group_scales=group_scales,
        )
    kernel_id = runtime.get_grouped_kernel(
        n, k, num_experts, "masked", config, bridge=experts.bridge, device=a.get_device(),
        group_scales=group_scales, streaming=False, signal_done=True,
    )
    return torch.ops.tensorbridge.launch_streaming_grouped_kernel(
        kernel_id, a, a_scale, experts.weight, experts.scale, experts.global_scale, experts.locks,
        expert_counts, None, None, None, None,
        workspace.tile_done_counts, workspace.combine_ready, out, enable_pdl,
    )


class GCMoEOrchestrator:
    """Multi-stream CUDA-Graph-safe orchestrator for GC-MoE bidirectional overlap.

    Execution DAG on `(comm_stream, compute_stream)`:
        fork_event (on caller stream)
         ├── comm_stream:    dispatch_fn(ws) ──────────────────────────► combine_fn(ws) ──┐
         └── compute_stream: [shared_expert_fn()] -> GateUp(streaming) -> Act+Quant -> Down(signaling) ─┴─► join
    """

    def __init__(self, partition: GreenContextPair, workspace: StreamingOverlapWorkspace) -> None:
        self.partition = partition
        self.workspace = workspace
        self._fork_ev = torch.cuda.Event()
        self._comm_done_ev = torch.cuda.Event()
        self._compute_done_ev = torch.cuda.Event()

    def prewarm_kernels(
        self,
        w13: NVFP4Experts,
        w2: NVFP4Experts,
        m_hint: int,
        *,
        gate_up_group_scales: bool = True,
        down_group_scales: bool = False,
        gate_up_config: GemmConfig | None = None,
        down_config: GemmConfig | None = None,
    ) -> tuple[GemmConfig, GemmConfig]:
        """Pre-compile & load both streaming/signaling CUDA modules before launching concurrent polling streams.

        CUDA's `cuModuleLoad` implicitly synchronizes all active streams on the device, so loading a
        new compute module while a polling Combine kernel is already spinning on `comm_stream` would deadlock.
        """
        e13, n13, k13 = w13.shape
        e2, n2, k2 = w2.shape
        with torch.cuda.stream(self.partition.compute_stream):
            cfg13 = gate_up_config or runtime.select_config(
                m_hint, n13, k13, e13, "masked", device=self.partition.device,
                bridge=w13.bridge, group_scales=gate_up_group_scales,
            )
            cfg2 = down_config or runtime.select_config(
                m_hint, n2, k2, e2, "masked", device=self.partition.device,
                bridge=w2.bridge, group_scales=down_group_scales,
            )
        runtime.get_grouped_kernel(
            n13, k13, e13, "masked", cfg13, bridge=w13.bridge, device=self.partition.device,
            group_scales=gate_up_group_scales, streaming=True, signal_done=False,
        )
        runtime.get_grouped_kernel(
            n2, k2, e2, "masked", cfg2, bridge=w2.bridge, device=self.partition.device,
            group_scales=down_group_scales, streaming=False, signal_done=True,
        )
        return cfg13, cfg2

    def run_layer(
        self,
        *,
        dispatch_fn: Callable[[StreamingOverlapWorkspace], None],
        gate_up_fn: Callable[[StreamingOverlapWorkspace], None],
        act_quant_fn: Callable[[], None],
        down_fn: Callable[[StreamingOverlapWorkspace], None],
        combine_fn: Callable[[StreamingOverlapWorkspace], None],
        shared_expert_fn: Callable[[], None] | None = None,
    ) -> None:
        """Launch one overlapped MoE layer across `(comm_stream, compute_stream)` (graph-capturable)."""
        caller = torch.cuda.current_stream(self.partition.device)
        # Reset handshake counters on caller stream before forking.
        self.workspace.reset()
        self._fork_ev.record(caller)

        with torch.cuda.stream(self.partition.comm_stream):
            self.partition.comm_stream.wait_event(self._fork_ev)
            dispatch_fn(self.workspace)
            combine_fn(self.workspace)
            self._comm_done_ev.record(self.partition.comm_stream)

        with torch.cuda.stream(self.partition.compute_stream):
            self.partition.compute_stream.wait_event(self._fork_ev)
            if shared_expert_fn is not None:
                shared_expert_fn()
            gate_up_fn(self.workspace)
            act_quant_fn()
            down_fn(self.workspace)
            self._compute_done_ev.record(self.partition.compute_stream)

        caller.wait_event(self._comm_done_ev)
        caller.wait_event(self._compute_done_ev)


class PrefillChunkOrchestrator:
    """Chunk-pipelined Prefill EP (`contiguous` layout) overlap over `(comm_stream, compute_stream)`.

    Splits a Prefill batch into `num_chunks` (typically 2) micro-batches and pipelines:
        comm_stream:    Dispatch(c=0) -> Dispatch(c=1) -> Combine(c=0) -> ... -> Combine(last)
        compute_stream:                 [wait D0] Compute(c=0) -> [wait D1] Compute(c=1) -> ...
    so Dispatch(c+1) and Combine(c-1) execute on `comm_stream` concurrently with `GateUp -> Act+Quant -> Down`
    on `compute_stream` without interfering with `contiguous` Route-A `compute_sms` GEMM configs.
    """

    def __init__(self, partition: GreenContextPair, num_chunks: int = 2) -> None:
        if num_chunks < 2:
            raise ValueError(f"num_chunks must be >= 2 for chunk pipelining, got {num_chunks}")
        self.partition = partition
        self.num_chunks = num_chunks
        self._fork_ev = torch.cuda.Event()
        self._disp_evs = [torch.cuda.Event() for _ in range(num_chunks)]
        self._comp_evs = [torch.cuda.Event() for _ in range(num_chunks)]
        self._comm_done_ev = torch.cuda.Event()
        self._compute_done_ev = torch.cuda.Event()

    def run_chunks(
        self,
        *,
        dispatch_chunk_fn: Callable[[int], None],
        compute_chunk_fn: Callable[[int], None],
        combine_chunk_fn: Callable[[int], None],
    ) -> None:
        """Execute `num_chunks` overlapped Prefill EP chunks (CUDA-Graph-capturable)."""
        caller = torch.cuda.current_stream(self.partition.device)
        self._fork_ev.record(caller)

        with torch.cuda.stream(self.partition.comm_stream):
            self.partition.comm_stream.wait_event(self._fork_ev)
            dispatch_chunk_fn(0)
            self._disp_evs[0].record(self.partition.comm_stream)

        for c in range(self.num_chunks):
            if c + 1 < self.num_chunks:
                with torch.cuda.stream(self.partition.comm_stream):
                    dispatch_chunk_fn(c + 1)
                    self._disp_evs[c + 1].record(self.partition.comm_stream)

            with torch.cuda.stream(self.partition.compute_stream):
                self.partition.compute_stream.wait_event(self._disp_evs[c])
                compute_chunk_fn(c)
                self._comp_evs[c].record(self.partition.compute_stream)

            with torch.cuda.stream(self.partition.comm_stream):
                self.partition.comm_stream.wait_event(self._comp_evs[c])
                combine_chunk_fn(c)

        with torch.cuda.stream(self.partition.comm_stream):
            self._comm_done_ev.record(self.partition.comm_stream)
        with torch.cuda.stream(self.partition.compute_stream):
            self._compute_done_ev.record(self.partition.compute_stream)

        caller.wait_event(self._comm_done_ev)
        caller.wait_event(self._compute_done_ev)


class DeepEPv2LowLatencyBridge:
    """Adapter connecting a `deep_ep.Buffer` (DeepEP v2 Low-Latency) to `GCMoEOrchestrator`.

    Keeps `deep_ep` an optional runtime dependency: instantiating this bridge wraps an already-created
    `deep_ep.Buffer` and provides `make_dispatch_fn` / `make_combine_fn` hooks that run on `comm_stream`
    using DeepEP v2's `low_latency_dispatch` and `low_latency_combine` (`return_recv_hook=True`).
    """

    def __init__(self, buffer: object, num_max_dispatch_tokens_per_rank: int, num_experts: int) -> None:
        self.buffer = buffer
        self.num_max_dispatch_tokens_per_rank = int(num_max_dispatch_tokens_per_rank)
        self.num_experts = int(num_experts)

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        *,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = True,
    ):
        """Invoke `buffer.low_latency_dispatch` on the active stream (intended for `comm_stream`)."""
        return self.buffer.low_latency_dispatch(  # type: ignore[attr-defined]
            x,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            round_scale=round_scale,
            use_ue8m0=use_ue8m0,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
        )

    def combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: object,
        *,
        async_finish: bool = False,
        return_recv_hook: bool = True,
        out: torch.Tensor | None = None,
    ):
        """Invoke `buffer.low_latency_combine` on the active stream (intended for `comm_stream`)."""
        return self.buffer.low_latency_combine(  # type: ignore[attr-defined]
            x,
            topk_idx,
            topk_weights,
            handle,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            out=out,
        )
