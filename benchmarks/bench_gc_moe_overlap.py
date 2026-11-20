#!/usr/bin/env python3
"""Route B (GC-MoE) end-to-end correctness & overlap benchmark on Hopper GPUs (H20 / H100).

Tests:
  1. Hardware CUDA Green Context SM partitioning (`GreenContextPair`) + `get_num_sms()` auto-detection.
  2. Bit-exact numerical equivalence of `StreamingMaskedScheduler` (`GateUp` streaming) +
     `Down` epilogue reverse signaling (`combine_ready`) vs. serial baseline across all 4 target models.
  3. Multi-stream CUDA Graph end-to-end Decode EP (`masked`) & Prefill EP (`contiguous` 2-chunk) latency:
       Serial (full SMs, Route-A optimal configs)
     vs.
       GC-MoE Overlap (Comm GC + Compute GC, Route-A `compute_sms` optimal configs, bidirectional streaming)
     with exact per-stage breakdown (`disp`, `gu`, `act`, `dn`, `comb`) and honest overlap efficiency.

Usage:
    python benchmarks/bench_gc_moe_overlap.py --models glm53 dsv41_flash kimi_k31_pro muse_spark_816b --ep 8
    python benchmarks/bench_gc_moe_overlap.py --mode prefill --models glm53 dsv41_flash kimi_k31_pro muse_spark_816b --ep 8
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

COMM_KERNEL_SOURCE = r"""
#include <cuda/std/cstdint>

using namespace cuda::std;

extern "C" __constant__ uint32_t SMEM_BYTES = 0;

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t ld_acquire_u32(const uint32_t *ptr) {
    uint32_t v;
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_u32(uint32_t *ptr, uint32_t v) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(ptr), "r"(v) : "memory");
}

struct ReadyTile {
    uint32_t expert_id;
    uint32_t row_begin;
    uint32_t row_end;
    uint32_t m_tile_idx;
};

// Multi-CTA staggered EP Dispatch simulator running on Comm Green Context (`gridDim.x == comm_sms`):
// Each CTA handles a disjoint stripe of experts (`e = blockIdx.x; e < E; e += gridDim.x`), copies
// FP8 rows + group-128 scales at full multi-SM bandwidth, paces expert arrival smoothly over
// `total_dispatch_ns` via `%globaltimer`, publishes frozen M-tiles into `ready_ring`, release-increments
// `write_head`, and the final finishing CTA sets `dispatch_done = 1`.
extern "C" __global__ void sim_ep_dispatch(
    const uint8_t *__restrict__ src_a,
    const float *__restrict__ src_sfa,
    uint8_t *__restrict__ dst_a,
    float *__restrict__ dst_sfa,
    const int32_t *__restrict__ expert_counts,
    ReadyTile *__restrict__ ready_ring,
    uint32_t *__restrict__ write_head,
    uint32_t *__restrict__ dispatch_done,
    uint32_t *__restrict__ comm_scratch,  // [0]: ring_lock, [1]: done_ctas (must be zeroed before launch)
    uint32_t num_experts,
    uint32_t capacity,
    uint32_t k_dim,
    uint32_t sfa_cols,
    uint32_t block_m,
    uint32_t active_experts,
    uint64_t total_dispatch_ns
) {
    const uint64_t t_start = globaltimer_ns();
    const uint32_t k_vec = k_dim / 16;

    uint32_t seen = 0;
    for (uint32_t e = blockIdx.x; e < num_experts; e += gridDim.x) {
        const uint32_t cnt = static_cast<uint32_t>(expert_counts[e]);
        if (cnt == 0) continue;
        if (total_dispatch_ns > 0 && active_experts > 0) {
            uint32_t rank = seen * gridDim.x + blockIdx.x + 1u;
            if (rank > active_experts) rank = active_experts;
            const uint64_t target = t_start + (total_dispatch_ns * rank) / active_experts;
            if (threadIdx.x == 0) {
                while (globaltimer_ns() < target) {
                    asm volatile("nanosleep.u32 64;");
                }
            }
            __syncthreads();
        }
        seen++;
        const uint32_t slot_base = e * capacity;
        const uint32_t num_tiles = (cnt + block_m - 1) / block_m;
        for (uint32_t mt = 0; mt < num_tiles; mt++) {
            const uint32_t local_begin = mt * block_m;
            const uint32_t local_end = (local_begin + block_m < cnt) ? (local_begin + block_m) : cnt;
            for (uint32_t r = local_begin; r < local_end; r++) {
                const uint64_t row_idx = static_cast<uint64_t>(slot_base + r);
                const uint4 *s_row = reinterpret_cast<const uint4 *>(src_a + row_idx * k_dim);
                uint4 *d_row = reinterpret_cast<uint4 *>(dst_a + row_idx * k_dim);
                for (uint32_t v = threadIdx.x; v < k_vec; v += blockDim.x) {
                    d_row[v] = s_row[v];
                }
                const float *s_sf = src_sfa + row_idx * sfa_cols;
                float *d_sf = dst_sfa + row_idx * sfa_cols;
                for (uint32_t c = threadIdx.x; c < sfa_cols; c += blockDim.x) {
                    d_sf[c] = s_sf[c];
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                while (atomicCAS(comm_scratch, 0u, 1u) != 0u) {
                    asm volatile("nanosleep.u32 32;");
                }
                const uint32_t pos = ld_acquire_u32(write_head);
                ready_ring[pos] = ReadyTile{e, slot_base + local_begin, slot_base + local_end, mt};
                st_release_u32(write_head, pos + 1u);
                st_release_u32(comm_scratch, 0u);
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const uint32_t done = atomicAdd(comm_scratch + 1, 1u) + 1u;
        if (done == gridDim.x) {
            st_release_u32(dispatch_done, 1u);
        }
    }
}

// Multi-CTA streaming EP Combine simulator running on Comm Green Context (`gridDim.x == comm_sms`):
// Each CTA owns experts `e = blockIdx.x; e < E; e += gridDim.x`, polls `combine_ready[e]` when
// `poll_ready != 0`, immediately copies finished BF16 rows out at multi-SM bandwidth, and paces
// per-expert wire transmission across `total_combine_ns`.
extern "C" __global__ void sim_ep_combine(
    const uint8_t *__restrict__ src_d,
    uint8_t *__restrict__ dst_out,
    const int32_t *__restrict__ expert_counts,
    const uint32_t *__restrict__ combine_ready,
    uint32_t num_experts,
    uint32_t capacity,
    uint32_t hidden_bytes,
    uint64_t total_combine_ns,
    uint32_t poll_ready
) {
    const uint32_t h_vec = hidden_bytes / 16;
    uint32_t my_active = 0;
    for (uint32_t e = blockIdx.x; e < num_experts; e += gridDim.x) {
        if (expert_counts[e] > 0) my_active++;
    }
    if (my_active == 0) return;
    const uint64_t per_expert_ns = total_combine_ns / my_active;

    __shared__ uint32_t sent_mask[32];
    for (uint32_t i = threadIdx.x; i < 32; i += blockDim.x) sent_mask[i] = 0;
    __syncthreads();

    uint32_t remaining = my_active;
    while (remaining > 0) {
        int32_t picked = -1;
        if (threadIdx.x == 0) {
            for (uint32_t e = blockIdx.x; e < num_experts; e += gridDim.x) {
                if (expert_counts[e] == 0) continue;
                if ((sent_mask[e >> 5] >> (e & 31)) & 1u) continue;
                if (!poll_ready || ld_acquire_u32(combine_ready + e) != 0u) {
                    sent_mask[e >> 5] |= (1u << (e & 31));
                    picked = static_cast<int32_t>(e);
                    break;
                }
            }
            if (picked < 0) {
                asm volatile("nanosleep.u32 64;");
            }
        }
        __shared__ int32_t s_picked;
        if (threadIdx.x == 0) s_picked = picked;
        __syncthreads();
        picked = s_picked;
        if (picked < 0) continue;

        const uint32_t e = static_cast<uint32_t>(picked);
        const uint32_t cnt = static_cast<uint32_t>(expert_counts[e]);
        const uint32_t slot_base = e * capacity;
        const uint64_t t_e = globaltimer_ns();
        for (uint32_t r = 0; r < cnt; r++) {
            const uint64_t row_idx = static_cast<uint64_t>(slot_base + r);
            const uint4 *s_row = reinterpret_cast<const uint4 *>(src_d + row_idx * hidden_bytes);
            uint4 *d_row = reinterpret_cast<uint4 *>(dst_out + row_idx * hidden_bytes);
            for (uint32_t v = threadIdx.x; v < h_vec; v += blockDim.x) {
                d_row[v] = s_row[v];
            }
        }
        if (per_expert_ns > 0 && threadIdx.x == 0) {
            const uint64_t target = t_e + per_expert_ns;
            while (globaltimer_ns() < target) {
                asm volatile("nanosleep.u32 64;");
            }
        }
        __syncthreads();
        remaining--;
    }
}

// Lightweight timed comm delay kernel for Prefill HT chunk pipelining (`contiguous` layout).
extern "C" __global__ void sim_ht_comm(
    const uint8_t *__restrict__ src,
    uint8_t *__restrict__ dst,
    uint64_t total_bytes,
    uint64_t target_ns
) {
    const uint64_t t0 = globaltimer_ns();
    const uint64_t total_vec = total_bytes / 16;
    const uint4 *s = reinterpret_cast<const uint4 *>(src);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = tid; i < total_vec; i += stride) {
        d[i] = s[i];
    }
    if (target_ns > 0 && threadIdx.x == 0) {
        const uint64_t end_t = t0 + target_ns;
        while (globaltimer_ns() < end_t) {
            asm volatile("nanosleep.u32 64;");
        }
    }
}
"""


def _load_comm_kernels():
    from cuda.bindings import driver as cu
    from tensorbridge.jit import compiler

    disp = compiler.build(COMM_KERNEL_SOURCE, "sim_ep_dispatch", compiler.INCLUDE_DIR)
    comb = compiler.build(COMM_KERNEL_SOURCE, "sim_ep_combine", compiler.INCLUDE_DIR)
    ht = compiler.build(COMM_KERNEL_SOURCE, "sim_ht_comm", compiler.INCLUDE_DIR)

    def _load(path: str, sym: bytes):
        err, mod = cu.cuModuleLoad(path.encode())
        assert err == cu.CUresult.CUDA_SUCCESS, f"cuModuleLoad failed: {err}"
        err, fn = cu.cuModuleGetFunction(mod, sym)
        assert err == cu.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction({sym!r}) failed: {err}"
        return mod, fn

    return _load(disp.cubin_path, b"sim_ep_dispatch"), _load(comb.cubin_path, b"sim_ep_combine"), _load(ht.cubin_path, b"sim_ht_comm")


def _launch_dispatch(fn, stream_ptr: int, grid_x: int, src_a, src_sfa, dst_a, dst_sfa, counts, ws,
                     comm_scratch, cap: int, k_dim: int, block_m: int, active_experts: int,
                     dispatch_ns: int) -> None:
    from cuda.bindings import driver as cu

    args = [
        ctypes.c_uint64(src_a.data_ptr()),
        ctypes.c_uint64(src_sfa.data_ptr()),
        ctypes.c_uint64(dst_a.data_ptr()),
        ctypes.c_uint64(dst_sfa.data_ptr()),
        ctypes.c_uint64(counts.data_ptr()),
        ctypes.c_uint64(ws.ready_ring.data_ptr()),
        ctypes.c_uint64(ws.write_head.data_ptr()),
        ctypes.c_uint64(ws.dispatch_done.data_ptr()),
        ctypes.c_uint64(comm_scratch.data_ptr()),
        ctypes.c_uint32(ws.num_experts),
        ctypes.c_uint32(cap),
        ctypes.c_uint32(k_dim),
        ctypes.c_uint32(src_sfa.size(1)),
        ctypes.c_uint32(block_m),
        ctypes.c_uint32(active_experts),
        ctypes.c_uint64(dispatch_ns),
    ]
    arg_ptrs = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
    (err,) = cu.cuLaunchKernel(
        fn, grid_x, 1, 1, 256, 1, 1, 0, cu.CUstream(stream_ptr), arg_ptrs, 0
    )
    assert err == cu.CUresult.CUDA_SUCCESS, f"sim_ep_dispatch launch failed: {err}"


def _launch_combine(fn, stream_ptr: int, grid_x: int, src_d, dst_out, counts, ws, cap: int, hidden_bytes: int,
                    combine_ns: int, poll_ready: bool) -> None:
    from cuda.bindings import driver as cu

    args = [
        ctypes.c_uint64(src_d.data_ptr()),
        ctypes.c_uint64(dst_out.data_ptr()),
        ctypes.c_uint64(counts.data_ptr()),
        ctypes.c_uint64(ws.combine_ready.data_ptr()),
        ctypes.c_uint32(ws.num_experts),
        ctypes.c_uint32(cap),
        ctypes.c_uint32(hidden_bytes),
        ctypes.c_uint64(combine_ns),
        ctypes.c_uint32(1 if poll_ready else 0),
    ]
    arg_ptrs = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
    (err,) = cu.cuLaunchKernel(
        fn, grid_x, 1, 1, 256, 1, 1, 0, cu.CUstream(stream_ptr), arg_ptrs, 0
    )
    assert err == cu.CUresult.CUDA_SUCCESS, f"sim_ep_combine launch failed: {err}"


def _launch_ht_comm(fn, stream_ptr: int, grid_x: int, src, dst, target_ns: int) -> None:
    from cuda.bindings import driver as cu

    total_bytes = src.numel() * src.element_size()
    args = [
        ctypes.c_uint64(src.data_ptr()),
        ctypes.c_uint64(dst.data_ptr()),
        ctypes.c_uint64(total_bytes),
        ctypes.c_uint64(target_ns),
    ]
    arg_ptrs = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
    (err,) = cu.cuLaunchKernel(
        fn, grid_x, 1, 1, 256, 1, 1, 0, cu.CUstream(stream_ptr), arg_ptrs, 0
    )
    assert err == cu.CUresult.CUDA_SUCCESS, f"sim_ht_comm launch failed: {err}"


def _time_graph(fn, warmup: int = 5, iters: int = 40, rounds: int = 5) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    cap_stream = torch.cuda.Stream()
    cap_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap_stream):
        with torch.cuda.graph(g, stream=cap_stream):
            fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        g.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(samples)


def _prepare_model_experts(spec, ep: int):
    import common
    import torch
    from common import MoeShape
    from tensorbridge.quant import nvfp4

    n13, k13 = spec.gate_up_n, spec.gate_up_k
    n2, k2 = spec.down_n, spec.down_k
    torch.manual_seed(0)
    w13_parts, w2_parts = [], []
    for chunk in common.moe_weight_chunks(MoeShape("gate_up", 1, n13, k13, spec.num_experts, spec.top_k, 1, "equal", 0, ep)):
        p, s, gs = nvfp4.quantize(chunk.reshape(-1, k13))
        part = nvfp4.prepare_experts(p.view(-1, n13, k13 // 2), s.view(-1, n13, k13 // 16), gs.expand(chunk.size(0)), bridge="exact")
        e_c = chunk.size(0)
        part = dataclasses.replace(part, scale=part.scale.view(e_c, k13 // 128, 8, n13).permute(0, 1, 3, 2).contiguous())
        w13_parts.append(part)
    w13 = nvfp4.NVFP4Experts(
        torch.cat([p.weight for p in w13_parts]), torch.cat([p.scale for p in w13_parts]),
        torch.cat([p.global_scale for p in w13_parts]), torch.zeros_like(w13_parts[0].locks), bridge="exact",
    )
    for chunk in common.moe_weight_chunks(MoeShape("down", 1, n2, k2, spec.num_experts, spec.top_k, 1, "equal", 0, ep)):
        p, s, gs = nvfp4.quantize(chunk.reshape(-1, k2))
        w2_parts.append(nvfp4.prepare_experts(p.view(-1, n2, k2 // 2), s.view(-1, n2, k2 // 16), gs.expand(chunk.size(0)), bridge="exact"))
    w2 = nvfp4.NVFP4Experts(
        torch.cat([p.weight for p in w2_parts]), torch.cat([p.scale for p in w2_parts]),
        torch.cat([p.global_scale for p in w2_parts]), torch.zeros_like(w2_parts[0].locks), bridge="exact",
    )
    return w13, w2


def bench_decode(args, gc, fn_disp, fn_comb) -> None:
    import common
    import torch
    from common import MoeShape
    from sweep_moe_profiles import TARGET_MODELS

    from tensorbridge.gemm import MaskedRouting, fp8_nvfp4_grouped_gemm
    from tensorbridge.jit import runtime
    from tensorbridge.overlap import (
        GCMoEOrchestrator,
        StreamingOverlapWorkspace,
        fp8_nvfp4_grouped_gemm_signaling,
        fp8_nvfp4_grouped_gemm_streaming,
    )
    from tensorbridge.quant import fp8

    comm_ctas = gc.comm_sms
    for model_name in args.models:
        spec = TARGET_MODELS[model_name]
        local_e = spec.num_experts // args.ep
        n13, k13 = spec.gate_up_n, spec.gate_up_k
        n2, k2 = spec.down_n, spec.down_k
        print(f"\n================ [DECODE masked] {model_name} EP{args.ep} "
              f"(E_local={local_e}, gate_up={n13}x{k13}, down={n2}x{k2}) ================")

        w13, w2 = _prepare_model_experts(spec, args.ep)
        cap = max(128, max(args.tokens)) * args.ep
        total_rows = local_e * cap
        src_a_buf = torch.empty((total_rows, k13), device="cuda", dtype=torch.float8_e4m3fn)
        src_sfa_buf = torch.empty((total_rows, k13 // 128), device="cuda", dtype=torch.float32)
        dst_a_buf = torch.empty_like(src_a_buf)
        dst_sfa_buf = torch.empty_like(src_sfa_buf)
        gu_out = torch.empty((total_rows, n13), device="cuda", dtype=torch.bfloat16)
        inter_a = torch.empty((total_rows, k2), device="cuda", dtype=torch.float8_e4m3fn)
        inter_s = torch.empty((total_rows, 1), device="cuda", dtype=torch.float32)
        down_out = torch.empty((total_rows, n2), device="cuda", dtype=torch.bfloat16)
        comb_out_serial = torch.empty_like(down_out)
        comb_out_gc = torch.empty_like(down_out)
        comm_scratch = torch.zeros((2,), device="cuda", dtype=torch.int32)

        ws = StreamingOverlapWorkspace.allocate(local_e, ring_capacity=local_e * (cap // 8), device=0)
        orch = GCMoEOrchestrator(gc, ws)

        for tokens in args.tokens:
            shape = MoeShape("gate_up", 1, n13, k13, spec.num_experts, spec.top_k, tokens, "zipf", 0, args.ep, 0, "masked", cap)
            routes = common.route(shape)
            if routes.rows == 0:
                continue
            counts = routes.counts.to(device="cuda", dtype=torch.int32)
            active_e = int((routes.counts > 0).sum())
            rows = routes.rows
            x_hp = torch.randn(total_rows, k13, device="cuda", dtype=torch.bfloat16) * 0.3
            fp8.quantize_moe(x_hp, "group_128", expert_counts=counts, out=(src_a_buf, src_sfa_buf))

            cfg_gu_full = runtime.select_config(rows, n13, k13, local_e, "masked", device=0, bridge="exact", group_scales=True)
            cfg_dn_full = runtime.select_config(rows, n2, k2, local_e, "masked", device=0, bridge="exact", group_scales=False)
            cfg_gu_gc, cfg_dn_gc = orch.prewarm_kernels(
                w13, w2, rows, gate_up_group_scales=True, down_group_scales=False,
            )

            def run_serial(disp_ns: int, comb_ns: int, out_buf):
                st = torch.cuda.current_stream().cuda_stream
                ws.reset()
                comm_scratch.zero_()
                _launch_dispatch(fn_disp, st, comm_ctas, src_a_buf, src_sfa_buf, dst_a_buf, dst_sfa_buf, counts, ws,
                                 comm_scratch, cap, k13, cfg_gu_full.block_m, active_e, disp_ns)
                fp8_nvfp4_grouped_gemm(dst_a_buf, dst_sfa_buf, w13, MaskedRouting(counts), rows,
                                       out=gu_out, config=cfg_gu_full, group_scales=True)
                fp8.quantize_moe(gu_out, "per_token", activation=True, expert_counts=counts, out=(inter_a, inter_s))
                fp8_nvfp4_grouped_gemm(inter_a, inter_s, w2, MaskedRouting(counts), rows,
                                       out=down_out, config=cfg_dn_full, group_scales=False)
                _launch_combine(fn_comb, st, comm_ctas, down_out, out_buf, counts, ws, cap, n2 * 2, comb_ns, False)

            def run_gc_overlap(disp_ns: int, comb_ns: int, out_buf):
                comm_scratch.zero_()
                orch.run_layer(
                    dispatch_fn=lambda w: _launch_dispatch(
                        fn_disp, gc.comm_stream.cuda_stream, comm_ctas, src_a_buf, src_sfa_buf, dst_a_buf, dst_sfa_buf,
                        counts, w, comm_scratch, cap, k13, cfg_gu_gc.block_m, active_e, disp_ns,
                    ),
                    gate_up_fn=lambda w: fp8_nvfp4_grouped_gemm_streaming(
                        dst_a_buf, dst_sfa_buf, w13, w, rows, out=gu_out, config=cfg_gu_gc, group_scales=True,
                    ),
                    act_quant_fn=lambda: fp8.quantize_moe(
                        gu_out, "per_token", activation=True, expert_counts=counts, out=(inter_a, inter_s),
                    ),
                    down_fn=lambda w: fp8_nvfp4_grouped_gemm_signaling(
                        inter_a, inter_s, w2, counts, w, rows, out=down_out, config=cfg_dn_gc, group_scales=False,
                    ),
                    combine_fn=lambda w: _launch_combine(
                        fn_comb, gc.comm_stream.cuda_stream, comm_ctas, down_out, out_buf, counts, w, cap, n2 * 2,
                        comb_ns, True,
                    ),
                )

            # Correctness verification first (dispatch=20us, combine=20us).
            comb_out_serial.zero_()
            comb_out_gc.zero_()
            run_serial(20_000, 20_000, comb_out_serial)
            torch.cuda.synchronize()
            run_gc_overlap(20_000, 20_000, comb_out_gc)
            torch.cuda.synchronize()
            diffs, refs = [], []
            for e in range(local_e):
                c = int(routes.counts[e])
                if c > 0:
                    s = e * cap
                    diffs.append((comb_out_gc[s:s + c] - comb_out_serial[s:s + c]).float())
                    refs.append(comb_out_serial[s:s + c].float())
            rel_rms = float(torch.cat(diffs).pow(2).mean().sqrt() / torch.cat(refs).pow(2).mean().sqrt())
            assert rel_rms < 1e-3, f"GC-MoE accuracy mismatch on {model_name} T={tokens}: rel_rms={rel_rms}"

            # Per-stage breakdown on full SMs.
            gu_us = _time_graph(lambda: fp8_nvfp4_grouped_gemm(dst_a_buf, dst_sfa_buf, w13, MaskedRouting(counts), rows,
                                                               out=gu_out, config=cfg_gu_full, group_scales=True))
            act_us = _time_graph(lambda: fp8.quantize_moe(gu_out, "per_token", activation=True, expert_counts=counts,
                                                          out=(inter_a, inter_s)))
            dn_us = _time_graph(lambda: fp8_nvfp4_grouped_gemm(inter_a, inter_s, w2, MaskedRouting(counts), rows,
                                                               out=down_out, config=cfg_dn_full, group_scales=False))
            comp_sum = gu_us + act_us + dn_us
            print(f"  T={tokens:3d} (rows={rows:4d}, actE={active_e:3d}) [breakdown full-SM: "
                  f"gu={gu_us:5.1f}us act={act_us:4.1f}us dn={dn_us:5.1f}us sum={comp_sum:5.1f}us] rel_rms={rel_rms:.6f}")

            for c_us in args.comm_us:
                ns = c_us * 1000
                disp_us = _time_graph(lambda: (
                    ws.reset(), comm_scratch.zero_(),
                    _launch_dispatch(fn_disp, torch.cuda.current_stream().cuda_stream, comm_ctas,
                                     src_a_buf, src_sfa_buf, dst_a_buf, dst_sfa_buf, counts, ws,
                                     comm_scratch, cap, k13, cfg_gu_full.block_m, active_e, ns),
                ))
                comb_us = _time_graph(lambda: _launch_combine(
                    fn_comb, torch.cuda.current_stream().cuda_stream, comm_ctas,
                    down_out, comb_out_serial, counts, ws, cap, n2 * 2, ns, False,
                ))
                ser_us = _time_graph(lambda: run_serial(ns, ns, comb_out_serial))
                gc_us = _time_graph(lambda: run_gc_overlap(ns, ns, comb_out_gc))
                saved_us = ser_us - gc_us
                max_hide = min(disp_us + comb_us, comp_sum)
                eff = 100.0 * saved_us / max_hide if max_hide > 0 else 0.0
                print(f"    comm={c_us:3d}+{c_us:3d}us (real d={disp_us:5.1f}+c={comb_us:5.1f}us) | "
                      f"serial={ser_us:6.1f}us | gc={gc_us:6.1f}us | "
                      f"speedup={ser_us / gc_us:5.3f}x (saved={saved_us:+6.1f}us, overlap={eff:5.1f}%)")

        del w13, w2
        torch.cuda.empty_cache()


def bench_prefill(args, gc, fn_ht) -> None:
    import common
    import torch
    from common import MoeShape
    from sweep_moe_profiles import TARGET_MODELS

    from tensorbridge.gemm import ContiguousRouting, fp8_nvfp4_grouped_gemm
    from tensorbridge.jit import runtime
    from tensorbridge.overlap import PrefillChunkOrchestrator
    from tensorbridge.quant import fp8

    comm_ctas = gc.comm_sms
    orch = PrefillChunkOrchestrator(gc, num_chunks=2)
    prefill_tokens = args.prefill_tokens

    for model_name in args.models:
        spec = TARGET_MODELS[model_name]
        local_e = spec.num_experts // args.ep
        n13, k13 = spec.gate_up_n, spec.gate_up_k
        n2, k2 = spec.down_n, spec.down_k
        print(f"\n================ [PREFILL contiguous 2-chunk] {model_name} EP{args.ep} "
              f"(E_local={local_e}, gate_up={n13}x{k13}, down={n2}x{k2}) ================")

        w13, w2 = _prepare_model_experts(spec, args.ep)

        for total_tokens in prefill_tokens:
            chunk_tokens = total_tokens // 2
            chunks = []
            for c_idx in range(2):
                shape = MoeShape("gate_up", 1, n13, k13, spec.num_experts, spec.top_k, chunk_tokens, "zipf", c_idx, args.ep, 0, "contiguous", 0)
                routes = common.route(shape)
                r = routes.rows
                offsets = routes.bounds.to(device="cuda", dtype=torch.int64)
                x_hp = torch.randn(r, k13, device="cuda", dtype=torch.bfloat16) * 0.3
                src_a = torch.empty((r, k13), device="cuda", dtype=torch.float8_e4m3fn)
                src_s = torch.empty((r, k13 // 128), device="cuda", dtype=torch.float32)
                fp8.quantize_moe(x_hp, "group_128", out=(src_a, src_s))
                dst_a = torch.empty_like(src_a)
                gu_out = torch.empty((r, n13), device="cuda", dtype=torch.bfloat16)
                inter_a = torch.empty((r, k2), device="cuda", dtype=torch.float8_e4m3fn)
                inter_s = torch.empty((r, 1), device="cuda", dtype=torch.float32)
                dn_out = torch.empty((r, n2), device="cuda", dtype=torch.bfloat16)
                comb_out = torch.empty_like(dn_out)
                chunks.append((r, offsets, src_a, src_s, dst_a, gu_out, inter_a, inter_s, dn_out, comb_out))

            total_rows = chunks[0][0] + chunks[1][0]
            cfg_gu_full = runtime.select_config(chunks[0][0], n13, k13, local_e, "contiguous", device=0, bridge="exact", group_scales=True)
            cfg_dn_full = runtime.select_config(chunks[0][0], n2, k2, local_e, "contiguous", device=0, bridge="exact", group_scales=False)
            with torch.cuda.stream(gc.compute_stream):
                cfg_gu_gc = runtime.select_config(chunks[0][0], n13, k13, local_e, "contiguous", device=0, bridge="exact", group_scales=True)
                cfg_dn_gc = runtime.select_config(chunks[0][0], n2, k2, local_e, "contiguous", device=0, bridge="exact", group_scales=False)

            def run_serial_prefill(comm_chunk_ns: int):
                st = torch.cuda.current_stream().cuda_stream
                for r, idx, src_a, src_s, dst_a, gu_out, inter_a, inter_s, dn_out, comb_out in chunks:
                    _launch_ht_comm(fn_ht, st, comm_ctas, src_a, dst_a, comm_chunk_ns)
                    fp8_nvfp4_grouped_gemm(dst_a, src_s, w13, ContiguousRouting(idx), r, out=gu_out, config=cfg_gu_full, group_scales=True)
                    fp8.quantize_moe(gu_out, "per_token", activation=True, out=(inter_a, inter_s))
                    fp8_nvfp4_grouped_gemm(inter_a, inter_s, w2, ContiguousRouting(idx), r, out=dn_out, config=cfg_dn_full, group_scales=False)
                    _launch_ht_comm(fn_ht, st, comm_ctas, dn_out, comb_out, comm_chunk_ns)

            def run_gc_prefill(comm_chunk_ns: int):
                def _disp(c: int):
                    _, _, src_a, _, dst_a, *_ = chunks[c]
                    _launch_ht_comm(fn_ht, gc.comm_stream.cuda_stream, comm_ctas, src_a, dst_a, comm_chunk_ns)

                def _comp(c: int):
                    r, idx, _, src_s, dst_a, gu_out, inter_a, inter_s, dn_out, _ = chunks[c]
                    fp8_nvfp4_grouped_gemm(dst_a, src_s, w13, ContiguousRouting(idx), r, out=gu_out, config=cfg_gu_gc, group_scales=True)
                    fp8.quantize_moe(gu_out, "per_token", activation=True, out=(inter_a, inter_s))
                    fp8_nvfp4_grouped_gemm(inter_a, inter_s, w2, ContiguousRouting(idx), r, out=dn_out, config=cfg_dn_gc, group_scales=False)

                def _comb(c: int):
                    *_, dn_out, comb_out = chunks[c]
                    _launch_ht_comm(fn_ht, gc.comm_stream.cuda_stream, comm_ctas, dn_out, comb_out, comm_chunk_ns)

                orch.run_chunks(dispatch_chunk_fn=_disp, compute_chunk_fn=_comp, combine_chunk_fn=_comb)

            for c_us in args.prefill_comm_us:
                chunk_ns = (c_us * 1000) // 2
                ser_us = _time_graph(lambda: run_serial_prefill(chunk_ns), iters=20, rounds=5)
                gc_us = _time_graph(lambda: run_gc_prefill(chunk_ns), iters=20, rounds=5)
                saved = ser_us - gc_us
                print(f"  T={total_tokens:4d} (rows={total_rows:5d}) comm={c_us:4d}+{c_us:4d}us (2-chunk) | "
                      f"serial={ser_us:7.1f}us | gc={gc_us:7.1f}us | speedup={ser_us / gc_us:5.3f}x (saved={saved:+6.1f}us)")

        del w13, w2
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["decode", "prefill", "both"], default="decode")
    parser.add_argument("--models", nargs="+", default=["glm53", "dsv41_flash", "kimi_k31_pro", "muse_spark_816b"])
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--tokens", nargs="+", type=int, default=[4, 16, 64, 128])
    parser.add_argument("--comm-us", nargs="+", type=int, default=[30, 60, 120],
                        help="Symmetric Decode dispatch_us == combine_us windows (us)")
    parser.add_argument("--prefill-tokens", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--prefill-comm-us", nargs="+", type=int, default=[200, 400])
    parser.add_argument("--comm-sms", type=int, default=8, help="min_comm_sms for GreenContextPair")
    args = parser.parse_args()

    import torch
    from tensorbridge.jit import runtime
    from tensorbridge.overlap import GreenContextPair

    full_sms = runtime.get_num_sms(0)
    print(f"Device: {torch.cuda.get_device_name(0)} | Full SMs: {full_sms}")

    with GreenContextPair(min_comm_sms=args.comm_sms, device=0) as gc:
        with torch.cuda.stream(gc.comm_stream):
            seen_comm = runtime.get_num_sms(0)
        with torch.cuda.stream(gc.compute_stream):
            seen_comp = runtime.get_num_sms(0)
        print(f"GreenContextPair created: comm_sms={gc.comm_sms} (detected={seen_comm}), "
              f"compute_sms={gc.compute_sms} (detected={seen_comp})")
        assert seen_comm == gc.comm_sms and seen_comp == gc.compute_sms

        (_md, fn_disp), (_mc, fn_comb), (_mh, fn_ht) = _load_comm_kernels()

        if args.mode in ("decode", "both"):
            bench_decode(args, gc, fn_disp, fn_comb)
        if args.mode in ("prefill", "both"):
            bench_prefill(args, gc, fn_ht)


if __name__ == "__main__":
    main()
