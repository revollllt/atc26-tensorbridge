"""Fused SwiGLU+FP8 grouped GEMM and MegaMoE Python interface."""

from __future__ import annotations

import dataclasses
import math

import torch

from tensorbridge.jit import runtime
from tensorbridge.jit.codegen import GemmConfig
from tensorbridge.quant.nvfp4 import NVFP4Experts


def should_fuse_swiglu_contiguous(
    m_hint: int,
    n: int,
    k: int,
    num_experts: int,
    *,
    device: int | None = None,
    bridge: str = "exact",
) -> bool:
    """Return True when fused SwiGLU+FP8 (`BLOCK_N == 256`) outperforms unfused `BLOCK_N == 128` + interleaved SwiGLU.

    For small/medium routed batches where the SM90 selector chooses `BLOCK_N == 256`, fusing SwiGLU+FP8 eliminates
    the `[M, 2I]` BF16 round-trip and activation launch overhead. For large multi-token Prefill batches where the
    selector chooses `BLOCK_N == 128`, running unfused `BLOCK_N == 128` directly on the same interleaved `w13`
    weights followed by interleaved `quantize_moe` achieves near-peak tensor throughput.
    """
    dev = torch.cuda.current_device() if device is None else device
    cfg = runtime.select_config(
        m_hint, n, k, num_experts, "contiguous", device=dev, bridge=bridge, group_scales=True,
    )
    return cfg.block_n == 256


def select_swiglu_config(
    m_hint: int,
    n: int,
    k: int,
    num_experts: int,
    layout: str,
    *,
    device: int | None = None,
    bridge: str = "exact",
) -> GemmConfig:
    """Select the fastest spill-free `BLOCK_N == 256` `pgroup` config for fused SwiGLU+FP8 `gate_up`."""
    if layout not in ("masked", "contiguous"):
        raise ValueError(f"Fused SwiGLU+FP8 requires masked or contiguous layout, got {layout!r}")
    dev = torch.cuda.current_device() if device is None else device
    cfg = runtime.select_config(
        m_hint, n, k, num_experts, layout, device=dev, bridge=bridge, group_scales=True,
    )

    def _is_spill_risk(c: GemmConfig) -> bool:
        if c.ctas_per_sm >= 2 and c.block_m >= 16:
            return True
        if c.num_stages == 4 and (k // 128) % 4 == 0 and c.block_m > 32:
            return True
        if c.num_stages >= 7:
            return True
        return False

    if cfg.block_n == 256 and not _is_spill_risk(cfg):
        return cfg

    candidates = [
        c for c in runtime.all_legal_configs(
            m_hint, n, k, num_experts=num_experts, layout=layout, device=dev, group_scales=True,
        )
        if c.block_n == 256 and not c.use_stream_k
    ]
    if not candidates:
        raise ValueError(f"No clean BLOCK_N=256 pgroup config for {(m_hint, n, k, num_experts, layout)}")
    avg_m = m_hint / max(1, num_experts)
    target_bm = 32 if avg_m >= 64 else min(cfg.block_m, 48)
    stage_rank = {4: 0, 5: 1, 6: 2, 3: 3} if avg_m >= 64 else {5: 0, 6: 1, 4: 2, 3: 3}
    candidates.sort(key=lambda c: (
        _is_spill_risk(c),
        abs(c.block_m - target_bm),
        -c.ctas_per_sm if (avg_m <= 8 and c.block_m == 8) else c.ctas_per_sm,
        stage_rank.get(c.num_stages, 9),
        c.m_fast_tile_order,
    ))
    return candidates[0]


def fp8_nvfp4_grouped_gemm_swiglu_fp8(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    interleaved_experts: NVFP4Experts,
    m_hint: int,
    *,
    expert_counts: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    config: GemmConfig | None = None,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused `GateUp` W4A8 grouped GEMM + in-epilogue `SwiGLU` + dynamic `group-128` FP8 E4M3 quantization.

    Takes `interleaved_experts` produced by `tensorbridge.quant.nvfp4.interleave_gate_up_experts(w13)` and
    writes `(fp8_out [M, I], fp8_scale [M, I/128])` directly, eliminating the BF16 `[M, 2I]` intermediate buffer
    and standalone activation kernel.
    """
    num_experts, n, k = interleaved_experts.shape
    if n % 256 != 0:
        raise ValueError(f"Fused SwiGLU+FP8 requires N % 256 == 0, got {n}")
    if (expert_counts is None) == (expert_offsets is None):
        raise ValueError("Specify exactly one of `expert_counts` (masked) or `expert_offsets` (contiguous)")
    layout = "masked" if expert_counts is not None else "contiguous"
    expert_layout = expert_counts if expert_counts is not None else expert_offsets
    if config is None:
        config = select_swiglu_config(
            m_hint, n, k, num_experts, layout, device=a.get_device(), bridge=interleaved_experts.bridge,
        )
    if config.block_n != 256:
        raise ValueError(f"Fused SwiGLU+FP8 requires block_n=256, got {config}")
    kernel_id = runtime.get_grouped_kernel(
        n, k, num_experts, layout, config, bridge=interleaved_experts.bridge, device=a.get_device(),
        group_scales=True, fuse_swiglu_fp8=True,
    )
    rows = a.size(0)
    if out is None:
        d = torch.empty((rows, n // 2), dtype=torch.float8_e4m3fn, device=a.device)
        d_scale = torch.empty((rows, n // 256), dtype=torch.float32, device=a.device)
    else:
        d, d_scale = out
    torch.ops.tensorbridge.launch_fused_swiglu_grouped_kernel(
        kernel_id, a, a_scale, interleaved_experts.weight, interleaved_experts.scale,
        interleaved_experts.global_scale, interleaved_experts.locks, d, d_scale,
        expert_layout, enable_pdl,
    )
    return d, d_scale


@dataclasses.dataclass
class MegaMoEWorkspace:
    """Device-side lock-free unified task ring & synchronization state for `sm90_w4a8_megamoe`.

    Touched slots are self-reset inline by the kernel's last CTA before the NVLink barrier (zero `cudaMemset`).
    """

    num_experts: int
    ring_capacity: int
    ready_ring: torch.Tensor         # int32 [ring_capacity, 4]: (expert_id, row_begin, row_end, m_tile_idx)
    write_head: torch.Tensor         # int32 [1]
    dispatch_done: torch.Tensor      # int32 [1]
    gu_work_cursor: torch.Tensor     # int32 [1]: unified monotonic ticket cursor
    gu_tile_done: torch.Tensor       # int32 [ring_capacity]: GateUp N-tiles finished per M-tile
    down_ready_ring: torch.Tensor    # int32 [task_ring_len]: packed (op<<31 | n_blk<<20 | ready_idx), init -1
    down_write_cursor: torch.Tensor  # int32 [1]: unified task ring write tail
    _ctrl_buf: torch.Tensor          # single cache-line-aligned backing buffer

    @classmethod
    def allocate(
        cls,
        num_experts: int,
        ring_capacity: int,
        device: torch.device | int | None = None,
        *,
        max_tasks_per_m_tile: int = 64,
        sentinel_pad: int = 1024,
    ) -> MegaMoEWorkspace:
        dev = (
            torch.device("cuda", torch.cuda.current_device() if device is None else device)
            if not isinstance(device, torch.device)
            else device
        )
        line = 32  # 128B L2 cache line in int32 words
        r_stride = ((ring_capacity + line - 1) // line) * line
        ctrl = torch.zeros((4 * line + r_stride,), dtype=torch.int32, device=dev)
        off = 0
        write_head = ctrl[off:off + 1]
        off += line
        dispatch_done = ctrl[off:off + 1]
        off += line
        gu_work_cursor = ctrl[off:off + 1]
        off += line
        down_write_cursor = ctrl[off:off + 1]
        off += line
        gu_tile_done = ctrl[off:off + ring_capacity]
        task_ring_len = ring_capacity * max_tasks_per_m_tile + max(1024, sentinel_pad)
        return cls(
            num_experts=num_experts,
            ring_capacity=ring_capacity,
            ready_ring=torch.zeros((ring_capacity, 4), dtype=torch.int32, device=dev),
            write_head=write_head,
            dispatch_done=dispatch_done,
            gu_work_cursor=gu_work_cursor,
            gu_tile_done=gu_tile_done,
            down_ready_ring=torch.full((task_ring_len,), -1, dtype=torch.int32, device=dev),
            down_write_cursor=down_write_cursor,
            _ctrl_buf=ctrl,
        )

    def reset(self) -> None:
        """Reset all control counters & task ring sentinels (CUDA-Graph safe)."""
        self._ctrl_buf.zero_()
        self.down_ready_ring.fill_(-1)


def select_megamoe_config(m_hint: int, num_experts: int) -> tuple[int, int, int, int]:
    """Select `(block_m, num_stages, ctas_per_sm, math_warpgroups)` for `sm90_w4a8_megamoe`.

    Uses Poisson routing headroom (`avg_m + 2*sqrt(avg_m)`) so >95% of experts finish in a single
    M-tile without 2x weight re-reads, while keeping `CTA=2` only for spill-free `BM=8` and `CTA=1`
    (232 regs/thread) for `BM=16..64`.
    """
    avg_m = m_hint / max(1, num_experts)
    target_m = avg_m + 2.0 * math.sqrt(max(avg_m, 0.0))
    if target_m <= 8.5:
        return (8, 4, 2, 2)
    if target_m <= 16.0:
        return (16, 6, 1, 2)
    if target_m <= 24.0:
        return (24, 6, 1, 2)
    if target_m <= 32.0:
        return (32, 6, 1, 2)
    if target_m <= 48.0:
        return (48, 5, 1, 2)
    return (64, 5, 1, 2)


def fp8_nvfp4_megamoe_layer(
    peer_ptrs: list[int],
    symm_offsets: list[int],
    local_x_fp8: torch.Tensor,
    local_x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_final_out: torch.Tensor,
    local_gen: torch.Tensor,
    local_phase_done: torch.Tensor,
    a_gu: torch.Tensor,
    sfa_gu: torch.Tensor,
    w13_interleaved: NVFP4Experts,
    w2_g128: NVFP4Experts,
    expert_counts: torch.Tensor,
    workspace: MegaMoEWorkspace,
    inter_fp8: torch.Tensor,
    inter_scale: torch.Tensor,
    *,
    ep_rank: int,
    local_e: int,
    capacity: int,
    block_m: int = 8,
    num_stages: int = 4,
    ctas_per_sm: int = 2,
    math_warpgroups: int = 2,
    enable_pdl: bool = False,
    reserve_sms: int = 0,
) -> torch.Tensor:
    """Launch the persistent MegaMoE kernel (zero cudaMemset, zero down_out HBM round-trip)."""
    num_e, n_gu, k_gu = w13_interleaved.shape
    num_e2, n_dn, k_dn = w2_g128.shape
    if num_e != num_e2 or num_e != local_e:
        raise ValueError(f"Expert count mismatch: w13={num_e}, w2={num_e2}, local_e={local_e}")
    if w13_interleaved.bridge != w2_g128.bridge:
        raise ValueError(f"Bridge mismatch between w13 ({w13_interleaved.bridge}) and w2 ({w2_g128.bridge})")
    if w13_interleaved.scale.dim() != 4 or w2_g128.scale.dim() != 4:
        raise ValueError("MegaMoE requires group-128 prepared experts (4D scale tensors) for both w13 and w2")
    kid = runtime.get_megamoe_kernel(
        n_gu, k_gu, n_dn, k_dn, num_e, block_m, num_stages, ctas_per_sm, math_warpgroups,
        bridge=w13_interleaved.bridge, device=a_gu.get_device(),
    )
    return torch.ops.tensorbridge.launch_megamoe_kernel(
        kid, peer_ptrs, symm_offsets,
        local_x_fp8, local_x_scale, topk_ids, topk_weights, local_final_out, local_gen, local_phase_done,
        a_gu, sfa_gu, w13_interleaved.weight, w13_interleaved.scale, w13_interleaved.global_scale,
        inter_fp8, inter_scale, w2_g128.weight, w2_g128.scale, w2_g128.global_scale,
        expert_counts, workspace.ready_ring, workspace.write_head, workspace.dispatch_done,
        workspace.gu_work_cursor, workspace.gu_tile_done, workspace.down_ready_ring,
        workspace.down_write_cursor, ep_rank, local_e, capacity, enable_pdl, reserve_sms,
    )
