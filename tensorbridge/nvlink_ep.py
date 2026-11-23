"""Single-node NVLink Symmetric Memory Expert Parallelism (`NVLinkSymmetricEP`).

Zero external communication dependencies (no DeepEP, no NVSHMEM): uses PyTorch's native
`torch.distributed._symmetric_memory` over NVLink P2P coupled with TensorBridge's MegaMoE
kernel (`sm90_w4a8_megamoe`) for Decode / small-to-medium batches (`T <= max_tokens_per_rank`)
and single-shot full-SM Contiguous Prefill (`BN=128` + interleaved SwiGLU+FP8 or fused `BN=256`) for
large Prefill batches (`T > max_tokens_per_rank`). Both paths share ONE HBM weight copy (`w13` interleaved +
`w2` group-128).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from tensorbridge.gemm import ContiguousRouting, fp8_nvfp4_grouped_gemm
from tensorbridge.jit import runtime
from tensorbridge.megamoe import (
    MegaMoEWorkspace,
    fp8_nvfp4_grouped_gemm_swiglu_fp8,
    fp8_nvfp4_megamoe_layer,
    select_megamoe_config,
    select_swiglu_config,
    should_fuse_swiglu_contiguous,
)
from tensorbridge.quant import fp8
from tensorbridge.quant.nvfp4 import NVFP4Experts

_ALIGN = 128


def _align_up(n: int, align: int = _ALIGN) -> int:
    return ((n + align - 1) // align) * align


class NVLinkSymmetricEP:
    """Persistent symmetric-memory NVLink EP communicator and MegaMoE / Prefill executor."""

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        local_e: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        *,
        max_tokens_per_rank: int = 256,
        device: int | torch.device | None = None,
    ) -> None:
        if hidden % 128 != 0 or intermediate % 128 != 0:
            raise ValueError(f"hidden ({hidden}) and intermediate ({intermediate}) must be multiples of 128")
        if not (1 <= local_e <= 128):
            raise ValueError(f"local_e must be in [1, 128], got {local_e}")

        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)
        if self.ep_size > 16:
            raise ValueError(f"NVLinkSymmetricEP supports single-node ep_size <= 16, got {self.ep_size}")

        if device is None:
            self.device_idx = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            self.device_idx = device.index if device.index is not None else torch.cuda.current_device()
        else:
            self.device_idx = int(device)
        self.device = torch.device("cuda", self.device_idx)

        self.local_e = int(local_e)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.top_k = int(top_k)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.capacity = self.ep_size * self.max_tokens_per_rank
        self.total_rows = self.local_e * self.capacity

        # Compute 128-byte-aligned offsets inside ONE contiguous symmetric NVLink allocation.
        off = 0
        self.off_recv_fp8 = off
        recv_fp8_bytes = self.total_rows * self.hidden
        off = _align_up(off + recv_fp8_bytes)

        self.off_recv_scale = off
        recv_scale_bytes = self.total_rows * (self.hidden // 128) * 4
        off = _align_up(off + recv_scale_bytes)

        self.off_recv_meta = off
        recv_meta_bytes = self.total_rows * 4
        off = _align_up(off + recv_meta_bytes)

        self.off_combine_recv = off
        combine_recv_bytes = self.max_tokens_per_rank * self.top_k * self.hidden * 2
        off = _align_up(off + combine_recv_bytes)

        self.off_recv_slot_count = off
        off = _align_up(off + self.local_e * 4)

        self.off_expert_arrived = off
        off = _align_up(off + self.local_e * 4)

        self.off_combine_arrived = off
        off = _align_up(off + 4)

        self.off_barrier_flags = off
        off = _align_up(off + 16 * 4)

        self.symm_offsets = [
            self.off_recv_fp8,
            self.off_recv_scale,
            self.off_recv_meta,
            self.off_combine_recv,
            self.off_recv_slot_count,
            self.off_expert_arrived,
            self.off_combine_arrived,
            self.off_barrier_flags,
        ]

        import torch.distributed._symmetric_memory as symm_mem

        with torch.cuda.device(self.device_idx):
            self.symm_buf = symm_mem.empty(off, dtype=torch.uint8, device=self.device)
            self.symm_buf.zero_()
            torch.cuda.synchronize(self.device_idx)
            self.symm_hdl = symm_mem.rendezvous(self.symm_buf, group=ep_group)
            self.peer_ptrs = [int(p) for p in self.symm_hdl.buffer_ptrs]
            if len(self.peer_ptrs) != self.ep_size:
                raise RuntimeError(
                    f"Symmetric memory rendezvous returned {len(self.peer_ptrs)} pointers for ep_size={self.ep_size}"
                )

            # Zero-copy local views into the symmetric buffer for MegaMoE TMA reads.
            self.recv_fp8 = (
                self.symm_buf[self.off_recv_fp8 : self.off_recv_fp8 + recv_fp8_bytes]
                .view(torch.float8_e4m3fn)
                .view(self.total_rows, self.hidden)
            )
            self.recv_scale = (
                self.symm_buf[self.off_recv_scale : self.off_recv_scale + recv_scale_bytes]
                .view(torch.float32)
                .view(self.total_rows, self.hidden // 128)
            )

            self._ring_capacity = self.local_e * ((self.capacity + 7) // 8)
            tasks_per_m_tile = (2 * self.intermediate) // 256 + self.hidden // 256
            self.megamoe_ws = MegaMoEWorkspace.allocate(
                self.local_e, self._ring_capacity, device=self.device, max_tasks_per_m_tile=tasks_per_m_tile,
            )

            self.expert_counts = torch.zeros((self.local_e,), dtype=torch.int32, device=self.device)
            self.x_fp8_buf = torch.empty(
                (self.max_tokens_per_rank, self.hidden), dtype=torch.float8_e4m3fn, device=self.device,
            )
            self.x_scale_buf = torch.empty(
                (self.max_tokens_per_rank, self.hidden // 128), dtype=torch.float32, device=self.device,
            )
            self.inter_fp8 = torch.empty(
                (self.total_rows, self.intermediate), dtype=torch.float8_e4m3fn, device=self.device,
            )
            self.inter_scale = torch.empty(
                (self.total_rows, self.intermediate // 128), dtype=torch.float32, device=self.device,
            )
            self.local_gen = torch.zeros((1,), dtype=torch.int32, device=self.device)
            self.local_phase_done = torch.zeros((2,), dtype=torch.int32, device=self.device)

    def close(self) -> None:
        """Release any resources (no-op; retained for context-manager symmetry)."""

    def _prepare_megamoe_kernel(
        self,
        w13: NVFP4Experts,
        w2: NVFP4Experts,
        m_hint: int,
    ) -> tuple[int, int, int, int]:
        """Resolve and register the MegaMoE kernel."""
        bm, stages, ctas, wg = select_megamoe_config(m_hint, self.local_e)
        _, n_gu, k_gu = w13.shape
        _, n_dn, k_dn = w2.shape
        runtime.get_megamoe_kernel(
            n_gu, k_gu, n_dn, k_dn, self.local_e, bm, stages, ctas, wg,
            bridge=w13.bridge, device=self.device_idx,
        )
        return bm, stages, ctas, wg

    def precompile(
        self,
        w13: NVFP4Experts,
        w2: NVFP4Experts,
        m_hints: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048),
        *, capture_sizes: Sequence[int] = (),
    ) -> None:
        """Pre-compile MegaMoE and Contiguous Prefill kernels before CUDA Graph capture."""
        e13, n13, k13 = w13.shape
        e2, n2, k2 = w2.shape
        chunk_sizes = {self.max_tokens_per_rank, *(size % self.max_tokens_per_rank for size in capture_sizes)}
        for m in sorted(set(m_hints) | {max(1, size * self.top_k) for size in chunk_sizes}):
            self._prepare_megamoe_kernel(w13, w2, m)

        for m_prefill in (1024, 2048, 4096, 8192, 16384):
            fuse_gu = should_fuse_swiglu_contiguous(m_prefill, n13, k13, e13, device=self.device_idx, bridge=w13.bridge)
            cfg_gu_p = (
                select_swiglu_config(m_prefill, n13, k13, e13, "contiguous", device=self.device_idx, bridge=w13.bridge)
                if fuse_gu
                else runtime.select_config(
                    m_prefill, n13, k13, e13, "contiguous", device=self.device_idx, bridge=w13.bridge, group_scales=True,
                )
            )
            cfg_dn_p = runtime.select_config(
                m_prefill, n2, k2, e2, "contiguous", device=self.device_idx, bridge=w2.bridge, group_scales=True,
            )
            runtime.get_grouped_kernel(
                n13, k13, e13, "contiguous", cfg_gu_p, bridge=w13.bridge, device=self.device_idx,
                group_scales=True, fuse_swiglu_fp8=fuse_gu,
            )
            runtime.get_grouped_kernel(
                n2, k2, e2, "contiguous", cfg_dn_p, bridge=w2.bridge, device=self.device_idx,
                group_scales=True,
            )

    def _forward_prefill_contiguous(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: NVFP4Experts,
        w2: NVFP4Experts,
        out: torch.Tensor,
        *,
        eff_tokens: int,
        shared_experts_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        shared_experts_hook: Callable[[], None] | None = None,
        prefill_permute_fns: tuple[
            Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
        ] | None = None,
    ) -> torch.Tensor:
        """Single-shot full-SM Contiguous Prefill fast path for `eff_tokens > max_tokens_per_rank`."""
        num_tokens = x.size(0)
        max_t = int(eff_tokens)
        total_m = self.ep_size * max_t

        x_fp8 = torch.empty((max_t, self.hidden), dtype=torch.float8_e4m3fn, device=self.device)
        x_scale = torch.empty((max_t, self.hidden // 128), dtype=torch.float32, device=self.device)
        if num_tokens > 0:
            fp8.quantize_group_128(x, out=(x_fp8[:num_tokens], x_scale[:num_tokens]))

        if num_tokens == max_t and topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
            ids_local = topk_ids
        else:
            ids_local = torch.full((max_t, self.top_k), -1, dtype=torch.int32, device=self.device)
            if num_tokens > 0:
                ids_local[:num_tokens].copy_(topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32))

        if num_tokens == max_t and topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
            w_local = topk_weights
        else:
            w_local = torch.zeros((max_t, self.top_k), dtype=torch.float32, device=self.device)
            if num_tokens > 0:
                w_local[:num_tokens].copy_(
                    topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32)
                )

        all_gather_fn = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
        reduce_scatter_fn = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)

        if self.ep_size > 1:
            all_fp8 = torch.empty((total_m, self.hidden), dtype=torch.float8_e4m3fn, device=self.device)
            all_scale = torch.empty((total_m, self.hidden // 128), dtype=torch.float32, device=self.device)
            all_ids = torch.empty((total_m, self.top_k), dtype=torch.int32, device=self.device)
            all_w = torch.empty((total_m, self.top_k), dtype=torch.float32, device=self.device)
            all_gather_fn(all_fp8, x_fp8, group=self.ep_group)
            all_gather_fn(all_scale, x_scale, group=self.ep_group)
            all_gather_fn(all_ids, ids_local, group=self.ep_group)
            all_gather_fn(all_w, w_local, group=self.ep_group)
            del x_fp8, x_scale
        else:
            all_fp8, all_scale, all_ids, all_w = x_fp8, x_scale, ids_local, w_local

        shared_out: torch.Tensor | None = None
        if shared_experts_hook is not None:
            shared_experts_hook()
        elif shared_experts_fn is not None and num_tokens > 0:
            shared_out = shared_experts_fn(x)

        if prefill_permute_fns is not None:
            prepare_scatter_fn, unpermute_fn = prefill_permute_fns
            expert_offsets, scatter = prepare_scatter_fn(all_ids)
            del all_ids
            local_rows_hint = max(1, (total_m * self.top_k + self.ep_size - 1) // self.ep_size)
            cap_rows = total_m * self.top_k
            a_perm = torch.empty((cap_rows, self.hidden), dtype=torch.float8_e4m3fn, device=self.device)
            s_perm = torch.empty((cap_rows, self.hidden // 128), dtype=torch.float32, device=self.device)
            fp8.scatter_fp8(all_fp8, all_scale, scatter, num_valid_rows=expert_offsets[-1:], out=(a_perm, s_perm))
            del all_fp8, all_scale
            e13, n13, k13 = w13.shape
            if should_fuse_swiglu_contiguous(
                local_rows_hint, n13, k13, e13, device=self.device_idx, bridge=w13.bridge,
            ):
                inter_fp8, inter_scale = fp8_nvfp4_grouped_gemm_swiglu_fp8(
                    a_perm, s_perm, w13, local_rows_hint, expert_offsets=expert_offsets, enable_pdl=True,
                )
                del a_perm, s_perm
            else:
                gate_up_out = fp8_nvfp4_grouped_gemm(
                    a_perm, s_perm, w13, ContiguousRouting(expert_offsets), local_rows_hint,
                    enable_pdl=True, group_scales=True,
                )
                del a_perm, s_perm
                inter_fp8 = torch.empty((cap_rows, self.intermediate), dtype=torch.float8_e4m3fn, device=self.device)
                inter_scale = torch.empty(
                    (cap_rows, self.intermediate // 128), dtype=torch.float32, device=self.device,
                )
                fp8.quantize_moe(
                    gate_up_out, "group_128", activation=True, num_valid_rows=expert_offsets[-1:],
                    out=(inter_fp8, inter_scale), interleaved_gate_up=True,
                )
                del gate_up_out
            down_out = fp8_nvfp4_grouped_gemm(
                inter_fp8, inter_scale, w2, ContiguousRouting(expert_offsets), local_rows_hint,
                enable_pdl=True, group_scales=True,
            )
            del inter_fp8, inter_scale
            gathered_out = torch.empty((total_m, self.hidden), dtype=torch.bfloat16, device=self.device)
            unpermute_fn(
                gathered_out, down_out.view(total_m, self.top_k, self.hidden),
                all_w, scatter.flatten(), expert_offsets,
            )
            del down_out, all_w, scatter, expert_offsets
        else:
            e_start = self.ep_rank * self.local_e
            e_end = e_start + self.local_e
            flat_ids = all_ids.view(-1)
            valid_pos = torch.nonzero((flat_ids >= e_start) & (flat_ids < e_end), as_tuple=False).squeeze(-1)
            num_valid = int(valid_pos.numel())

            gathered_out = torch.zeros((total_m, self.hidden), dtype=torch.bfloat16, device=self.device)
            if num_valid > 0:
                local_eids = flat_ids.index_select(0, valid_pos) - e_start
                order = torch.argsort(local_eids, stable=False)
                sorted_pos = valid_pos.index_select(0, order)
                sorted_eids = local_eids.index_select(0, order)
                src_tok = torch.div(sorted_pos, self.top_k, rounding_mode="floor")
                w_perm = all_w.view(-1).index_select(0, sorted_pos)

                counts = torch.bincount(sorted_eids, minlength=self.local_e)
                expert_offsets = torch.zeros((self.local_e + 1,), dtype=torch.int64, device=self.device)
                torch.cumsum(counts, dim=0, out=expert_offsets[1:])

                a_perm = all_fp8.index_select(0, src_tok)
                s_perm = all_scale.index_select(0, src_tok)
                del all_fp8, all_scale, all_ids, all_w, valid_pos, local_eids, order, sorted_pos, sorted_eids, counts
                e13, n13, k13 = w13.shape
                if should_fuse_swiglu_contiguous(
                    num_valid, n13, k13, e13, device=self.device_idx, bridge=w13.bridge,
                ):
                    inter_fp8, inter_scale = fp8_nvfp4_grouped_gemm_swiglu_fp8(
                        a_perm, s_perm, w13, num_valid, expert_offsets=expert_offsets,
                    )
                    del a_perm, s_perm
                else:
                    gate_up_out = fp8_nvfp4_grouped_gemm(
                        a_perm, s_perm, w13, ContiguousRouting(expert_offsets), num_valid, group_scales=True,
                    )
                    del a_perm, s_perm
                    inter_fp8 = torch.empty(
                        (num_valid, self.intermediate), dtype=torch.float8_e4m3fn, device=self.device,
                    )
                    inter_scale = torch.empty(
                        (num_valid, self.intermediate // 128), dtype=torch.float32, device=self.device,
                    )
                    fp8.quantize_moe(
                        gate_up_out, "group_128", activation=True,
                        out=(inter_fp8, inter_scale), interleaved_gate_up=True,
                    )
                    del gate_up_out
                down_out = fp8_nvfp4_grouped_gemm(
                    inter_fp8, inter_scale, w2, ContiguousRouting(expert_offsets), num_valid, group_scales=True,
                )
                del inter_fp8, inter_scale, expert_offsets
                down_out.mul_(w_perm.unsqueeze(-1))
                del w_perm
                gathered_out.index_add_(0, src_tok, down_out)
                del down_out, src_tok
            else:
                del all_fp8, all_scale, all_ids, all_w, valid_pos

        if self.ep_size > 1:
            out_padded = torch.empty((max_t, self.hidden), dtype=torch.bfloat16, device=self.device)
            reduce_scatter_fn(out_padded, gathered_out, group=self.ep_group)
            del gathered_out
            if num_tokens > 0:
                out.copy_(out_padded[:num_tokens])
            del out_padded
        else:
            if num_tokens > 0:
                out.copy_(gathered_out[:num_tokens])
            del gathered_out

        if shared_out is not None:
            out.add_(shared_out)
        return out

    def _run_megamoe_chunk(
        self,
        x_chunk: torch.Tensor,
        topk_ids_chunk: torch.Tensor,
        topk_weights_chunk: torch.Tensor,
        w13: NVFP4Experts,
        w2: NVFP4Experts,
        out_chunk: torch.Tensor,
        *,
        m_hint: int,
        shared_experts_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        shared_experts_hook: Callable[[], None] | None,
        reserve_sms: int = 0,
    ) -> None:
        t = x_chunk.size(0)
        x_fp8 = self.x_fp8_buf[:t]
        x_scale = self.x_scale_buf[:t]
        if t > 0:
            fp8.quantize_group_128(x_chunk, out=(x_fp8, x_scale))

        topk_ids_i32 = (
            topk_ids_chunk if (topk_ids_chunk.dtype == torch.int32 and topk_ids_chunk.is_contiguous())
            else topk_ids_chunk.to(torch.int32).contiguous()
        )
        topk_w_f32 = (
            topk_weights_chunk if (topk_weights_chunk.dtype == torch.float32 and topk_weights_chunk.is_contiguous())
            else topk_weights_chunk.to(torch.float32).contiguous()
        )

        # Warp-specialized MegaMoE: 4x route-striped NVLink Dispatch + Direct Publisher +
        # GateUp+SwiGLU+FP8 + Down with direct 512B-warp NVLink P2P scatter epilogue + Weighted Reduction +
        # Inline Self-Reset + 1-Barrier in ONE persistent kernel (zero cudaMemset, zero down_out HBM round-trip).
        bm, stages, ctas, wg = self._prepare_megamoe_kernel(w13, w2, m_hint)
        fp8_nvfp4_megamoe_layer(
            self.peer_ptrs, self.symm_offsets,
            x_fp8, x_scale, topk_ids_i32, topk_w_f32, out_chunk,
            self.local_gen, self.local_phase_done,
            self.recv_fp8, self.recv_scale, w13, w2, self.expert_counts, self.megamoe_ws,
            self.inter_fp8, self.inter_scale,
            ep_rank=self.ep_rank, local_e=self.local_e, capacity=self.capacity,
            block_m=bm, num_stages=stages, ctas_per_sm=ctas, math_warpgroups=wg,
            reserve_sms=reserve_sms,
        )
        if shared_experts_hook is not None:
            shared_experts_hook()
        elif shared_experts_fn is not None and t > 0:
            out_chunk.add_(shared_experts_fn(x_chunk))

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: NVFP4Experts,
        w2: NVFP4Experts,
        *,
        shared_experts_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        shared_experts_hook: Callable[[], None] | None = None,
        prefill_permute_fns: tuple[
            Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
        ] | None = None,
        global_num_tokens: int | None = None,
        reserve_sms: int = 0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one full EP MoE layer (Dispatch + Compute + Combine + optional Shared Experts).

        Automatically routes large Prefill (`eff_tokens > max_tokens_per_rank`) through single-shot full-SM
        Contiguous Prefill and Decode / small-to-medium batches (`eff_tokens <= max_tokens_per_rank`) through
        MegaMoE.
        """
        num_tokens = x.size(0)
        if out is None:
            out = torch.empty((num_tokens, self.hidden), dtype=torch.bfloat16, device=x.device)

        eff_tokens = global_num_tokens if global_num_tokens is not None else num_tokens
        if self.ep_size == 1 and num_tokens == 0:
            if shared_experts_hook is not None:
                shared_experts_hook()
            return out

        if eff_tokens > self.max_tokens_per_rank and not torch.cuda.is_current_stream_capturing():
            return self._forward_prefill_contiguous(
                x, topk_ids, topk_weights, w13, w2, out,
                eff_tokens=eff_tokens,
                shared_experts_fn=shared_experts_fn,
                shared_experts_hook=shared_experts_hook,
                prefill_permute_fns=prefill_permute_fns,
            )

        num_chunks = max(1, math.ceil(eff_tokens / self.max_tokens_per_rank))
        for c in range(num_chunks):
            beg = min(c * self.max_tokens_per_rank, num_tokens)
            end = min((c + 1) * self.max_tokens_per_rank, num_tokens)
            chunk_eff_t = min(self.max_tokens_per_rank, max(0, eff_tokens - c * self.max_tokens_per_rank))
            m_hint = max(1, chunk_eff_t * self.top_k)
            self._run_megamoe_chunk(
                x[beg:end], topk_ids[beg:end], topk_weights[beg:end], w13, w2, out[beg:end],
                m_hint=m_hint, shared_experts_fn=shared_experts_fn,
                shared_experts_hook=shared_experts_hook if c == 0 else None,
                reserve_sms=reserve_sms,
            )
        return out
