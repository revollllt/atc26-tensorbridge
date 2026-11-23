#pragma once

#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <tensorbridge/comm/nvlink_ptx.cuh>
#include <tensorbridge/common/megamoe_params.h>
#include <tensorbridge/common/ptx.cuh>
#include <tensorbridge/common/utils.cuh>
#include <tensorbridge/dequant/fpma.cuh>
#include <tensorbridge/epilogue/sm90.cuh>
#include <tensorbridge/layout/sm90.cuh>
#include <tensorbridge/mma/sm90.cuh>

namespace tensorbridge {

namespace layout {

// 4-slot CTA-local handoff queue for `sm90_w4a8_megamoe`.
struct MegaMoeQueue {
    static constexpr uint32_t kQueueSlots = 4;
    uint32_t op[kQueueSlots];          // 0 = GateUp+SwiGLU+FP8g128, 1 = Down(g128)
    uint32_t expert_id[kQueueSlots];
    uint32_t row_begin[kQueueSlots];
    uint32_t row_end[kQueueSlots];
    uint32_t n_block_idx[kQueueSlots];
    uint32_t ready_idx[kQueueSlots];
    uint32_t has_work[kQueueSlots];
    alignas(8) BarrierWord tile_ready[kQueueSlots];
    BarrierWord tile_consumed[kQueueSlots];
};

// Scratch shared memory used by the MegaMoE comm warps (`tma_warp_idx == 1, 2, 3`)
// and post-compute reduction/self-reset in `sm90_w4a8_megamoe`.
struct MegaMoeCommSmem {
    static constexpr uint32_t kMaxLocalE = 128;
    static constexpr uint32_t kMaxRoutes = 2048;  // supports up to 256 tokens/rank * top_k=8
    uint32_t counts[kMaxLocalE];
    uint32_t offsets[kMaxLocalE + 1];
    uint32_t write_pos[kMaxLocalE];
    uint32_t is_last_cta;
    uint32_t next_gen;
    uint32_t routes[kMaxRoutes];
};

struct MegaMoeDownSmem {
    uint32_t meta[64];                         // max BLOCK_M = 64
    uint32_t peer_cnt[kMaxNvlinkEpRanks];      // max 16 ranks
};

template <uint32_t BLOCK_M, uint32_t kNumStages>
struct MegaMoeStorage : PgroupRing<BLOCK_M, 256, kNumStages>, MegaMoeQueue {
    MegaMoeCommSmem comm;
    MegaMoeDownSmem dn_comm;
};

}  // namespace layout

// Single SM90 persistent W4A8 MegaMoE kernel fusing:
//   - 4x route-striped parallel NVLink P2P Dispatch + direct M-tile publishing (Warps 9..11)
//   - GateUp (NVFP4 interleaved x FP8 g128) + in-epilogue SwiGLU + dynamic group-128 FP8 quant (`op == 0`)
//   - Down   (NVFP4 x FP8 g128) + direct 512B-warp NVLink P2P scatter epilogue (`op == 1`)
//   - Weighted BF16 reduction + inline ring self-reset + single NVLink barrier
// Uses a unified `BLOCK_N = 256, BLOCK_K = 128` `pgroup` WGMMA mainloop with TMA L2 hints (`EVICT_FIRST` weights,
// `EVICT_LAST` activations) and a zero-CAS lock-free unified task ring (`atomicAdd` ticket stealing + warp enqueue).
template <uint32_t N_GU, uint32_t K_GU, uint32_t N_DN, uint32_t K_DN, uint32_t BLOCK_M, uint32_t kNumStages,
          uint32_t kNumCtasPerSm, uint32_t kNumMathWarpgroups, bool kExactBridge, uint32_t kNumExperts>
__global__ __launch_bounds__(128 * (kNumMathWarpgroups + 1), kNumCtasPerSm) void sm90_w4a8_megamoe(
    const __grid_constant__ MegaMoeParams params) {
    using Storage = layout::MegaMoeStorage<BLOCK_M, kNumStages>;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto &smem = *reinterpret_cast<Storage *>(smem_buffer);

    constexpr uint32_t BLOCK_N = 256;
    constexpr uint32_t BLOCK_K = 128;
    static_assert(N_GU % 256 == 0 && K_GU % 128 == 0 && N_DN % 256 == 0 && K_DN % 128 == 0,
                  "MegaMoE requires N_GU, N_DN multiples of 256 and K_GU, K_DN multiples of 128");
    static_assert(K_DN == N_GU / 2, "Down K must equal GateUp N / 2");
    static_assert(BLOCK_M % 8 == 0 && BLOCK_M >= 8 && BLOCK_M <= 64, "Invalid MegaMoE BLOCK_M");
    static_assert(kNumMathWarpgroups == 1 || kNumMathWarpgroups == 2, "Invalid math warp groups");
    static_assert(kNumCtasPerSm >= 1 && kNumCtasPerSm <= (kNumMathWarpgroups == 2 ? 2 : 4), "Invalid CTAs per SM");
    static_assert(kNumMathWarpgroups == 2 || kNumCtasPerSm >= 2, "1-WG requires >= 2 CTAs/SM");
    static_assert(kNumStages >= 3 && kNumStages <= 8, "Invalid pipeline stages");

    using WGMMA = typename mma::sm90::FP8RSMMASelector<BLOCK_M>::type;

    constexpr uint32_t kNumNBlocksGu = N_GU / BLOCK_N;
    constexpr uint32_t kNumNBlocksDn = N_DN / BLOCK_N;
    constexpr uint32_t kNumKBlocksGu = K_GU / BLOCK_K;
    constexpr uint32_t kNumKBlocksDn = K_DN / BLOCK_K;
    constexpr uint32_t kNumMathThreads = 128 * kNumMathWarpgroups;
    constexpr uint32_t kNumThreads = kNumMathThreads + 128;
    constexpr uint32_t kTilesN = 2;
    constexpr uint32_t kHalvesPerTile = 2 / kNumMathWarpgroups;
    constexpr uint32_t kNumSubTiles = kTilesN * kHalvesPerTile;
    constexpr uint32_t kOutputRows = Storage::kOutputRows;
    constexpr uint32_t kNumKIters = BLOCK_K / WGMMA::K;
    constexpr bool kBatchMMA = BLOCK_M * kNumSubTiles > 128;
    constexpr uint32_t kTmaBytesPerStage = BLOCK_M * 128 + kTilesN * (Storage::kTileBytesB + Storage::kTileBytesSFB);
    constexpr uint32_t kMathBarrier = 1;
    constexpr uint32_t kCommBarrier = 2;
    constexpr uint32_t kTasksPerMTile = kNumNBlocksGu + kNumNBlocksDn;

    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kLaunchRegs = 65536 / (kNumThreads * kNumCtasPerSm) / 8 * 8;
    constexpr uint32_t kNumMathRegisters = (kLaunchRegs * kNumThreads - kNumTMARegisters * 128) / kNumMathThreads / 8 * 8;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    if (warp_idx == kNumMathThreads / 32 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_a_gu));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_b_gu));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_sfb_gu));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_a_dn));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_b_dn));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_sfb_dn));

        for (uint32_t s = 0; s < kNumStages; s++) {
            smem.full_barriers[s]->init(1 + 32);
            smem.empty_barriers[s]->init(kNumMathThreads / 32);
        }
        for (uint32_t q = 0; q < Storage::kQueueSlots; q++) {
            smem.tile_ready[q]->init(1);
            smem.tile_consumed[q]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();
    cudaGridDependencySynchronize();

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    struct Task {
        uint32_t op, expert_id, row_begin, row_end, n_block_idx, ready_idx;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        const uint32_t tma_warp_idx = warp_idx - kNumMathThreads / 32;
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (tma_warp_idx >= 1u) {
            // Warp-specialized NVLink P2P Dispatch + Direct Publisher on Warps 9..11 (96 threads).
            const uint32_t comm_tid = (tma_warp_idx - 1u) * 32u + lane_idx;
            const uint32_t bid = blockIdx.x;
            const MegaMoeParams &p = params;

                // Phase 1: Multi-CTA Parallel NVLink P2P Dispatch (`n_stripes` CTAs per peer, route-striped `r % n_stripes`).
                // Splitting by route index gives 4x NVLink bandwidth even when 100% of tokens hit a single hot expert (Zipf).
                const uint32_t n_stripes = (gridDim.x >= p.ep_size * 4u + 1u) ? 4u : 1u;
                const uint32_t n_disp_ctas = p.ep_size * n_stripes;
                if (bid < n_disp_ctas) {
                    const uint32_t dst_r = bid % p.ep_size;
                    const uint32_t stripe = bid / p.ep_size;
                    const uint32_t e_start = dst_r * p.local_e;
                    const uint32_t e_end = e_start + p.local_e;
                    const uint32_t total_routes = p.num_tokens * p.top_k;
                    const uint32_t h_vecs = p.hidden / 16u;
                    const uint32_t s_groups = p.hidden / 128u;

                    for (uint32_t le = comm_tid; le < p.local_e; le += 96u) {
                        smem.comm.counts[le] = 0u;
                    }
                    cutlass::arch::NamedBarrier::sync(96, kCommBarrier);

                    for (uint32_t r = stripe + comm_tid * n_stripes; r < total_routes; r += 96u * n_stripes) {
                        const int32_t eid = p.topk_ids[r];
                        if (eid >= static_cast<int32_t>(e_start) && eid < static_cast<int32_t>(e_end)) {
                            atomicAdd(&smem.comm.counts[static_cast<uint32_t>(eid) - e_start], 1u);
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(96, kCommBarrier);

                    if (comm_tid == 0u) {
                        uint32_t acc = 0u;
                        for (uint32_t le = 0u; le < p.local_e; le++) {
                            smem.comm.offsets[le] = acc;
                            smem.comm.write_pos[le] = acc;
                            acc += smem.comm.counts[le];
                        }
                        smem.comm.offsets[p.local_e] = acc;
                    }
                    cutlass::arch::NamedBarrier::sync(96, kCommBarrier);

                    for (uint32_t r = stripe + comm_tid * n_stripes; r < total_routes; r += 96u * n_stripes) {
                        const int32_t eid = p.topk_ids[r];
                        if (eid >= static_cast<int32_t>(e_start) && eid < static_cast<int32_t>(e_end)) {
                            const uint32_t le = static_cast<uint32_t>(eid) - e_start;
                            const uint32_t pos = atomicAdd(&smem.comm.write_pos[le], 1u);
                            const uint32_t t = r / p.top_k;
                            const uint32_t k = r % p.top_k;
                            smem.comm.routes[pos] = (t << 8) | k;
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(96, kCommBarrier);

                    uint8_t *peer_fp8 = nvlink_ep::symm_ptr<uint8_t>(p, dst_r, p.off_recv_fp8);
                    float *peer_scale = nvlink_ep::symm_ptr<float>(p, dst_r, p.off_recv_scale);
                    uint32_t *peer_meta = nvlink_ep::symm_ptr<uint32_t>(p, dst_r, p.off_recv_meta);
                    uint32_t *peer_slot_cnt = nvlink_ep::symm_ptr<uint32_t>(p, dst_r, p.off_recv_slot_count);
                    uint32_t *peer_arrived = nvlink_ep::symm_ptr<uint32_t>(p, dst_r, p.off_expert_arrived);

                    const uint32_t total_peer_routes = smem.comm.offsets[p.local_e];
                    if (total_peer_routes > 0u) {
                        // Reserve destination row offsets for ALL active local experts in parallel (1 NVLink RTT!).
                        for (uint32_t le = comm_tid; le < p.local_e; le += 96u) {
                            const uint32_t cnt = smem.comm.counts[le];
                            smem.comm.write_pos[le] =
                                (cnt > 0u) ? nvlink_ep::atomic_add_sys_u32(peer_slot_cnt + le, cnt) : 0u;
                        }
                        cutlass::arch::NamedBarrier::sync(96, kCommBarrier);

                        for (uint32_t le = 0u; le < p.local_e; le++) {
                            const uint32_t cnt = smem.comm.counts[le];
                            if (cnt == 0u) continue;
                            const uint32_t base_row = smem.comm.write_pos[le];
                            const uint32_t start_pos = smem.comm.offsets[le];
                            for (uint32_t i = 0u; i < cnt; i++) {
                                const uint32_t dst_row = base_row + i;
                                if (dst_row >= p.capacity) continue;
                                const uint32_t packed_rk = smem.comm.routes[start_pos + i];
                                const uint32_t src_t = packed_rk >> 8;
                                const uint32_t k_slot = packed_rk & 0xffu;
                                const uint64_t dst_global_row = static_cast<uint64_t>(le) * p.capacity + dst_row;

                                const auto *src_v = reinterpret_cast<const int4 *>(
                                    p.local_x_fp8 + static_cast<uint64_t>(src_t) * p.hidden);
                                auto *dst_v = reinterpret_cast<int4 *>(peer_fp8 + dst_global_row * p.hidden);
                                for (uint32_t v = comm_tid; v < h_vecs; v += 96u) {
                                    dst_v[v] = src_v[v];
                                }
                                const float *src_s = p.local_x_scale + static_cast<uint64_t>(src_t) * s_groups;
                                float *dst_s = peer_scale + dst_global_row * s_groups;
                                for (uint32_t g = comm_tid; g < s_groups; g += 96u) {
                                    dst_s[g] = src_s[g];
                                }
                                if (comm_tid == 0u) {
                                    peer_meta[dst_global_row] =
                                        (p.ep_rank << 24) | (k_slot << 16) | (src_t & 0xffffu);
                                }
                            }
                        }
                        cutlass::arch::NamedBarrier::sync(96, kCommBarrier);
                    }
                    if (comm_tid == 0u) {
                        if (total_peer_routes > 0u) nvlink_ep::fence_acq_rel_sys();
                        nvlink_ep::atomic_add_sys_u32(peer_arrived, 1u);
                    }
                } else if (bid == n_disp_ctas && tma_warp_idx == 1u) {
                    // Phase 2: Direct Single-Step Local Publisher (writes ready_ring + down_ready_ring directly!).
                    uint32_t n_mtiles = 0u;
                    if (lane_idx == 0u) {
                        const uint32_t *arrived = nvlink_ep::symm_ptr<const uint32_t>(p, p.ep_rank, p.off_expert_arrived);
                        const uint32_t *slot_cnt = nvlink_ep::symm_ptr<const uint32_t>(p, p.ep_rank, p.off_recv_slot_count);
                        while (nvlink_ep::ld_acquire_sys_u32(arrived) < n_disp_ctas) {
                            ptx::nanosleep_ns(64u);
                        }
                        uint32_t head = 0u;
                        for (uint32_t le = 0u; le < p.local_e; le++) {
                            const uint32_t raw_cnt = nvlink_ep::ld_acquire_sys_u32(slot_cnt + le);
                            const uint32_t cnt = raw_cnt < p.capacity ? raw_cnt : p.capacity;
                            p.expert_counts[le] = static_cast<int32_t>(cnt);
                            if (cnt > 0u) {
                                const uint32_t n_tiles = (cnt + BLOCK_M - 1u) / BLOCK_M;
                                const uint32_t e_base = le * p.capacity;
                                for (uint32_t mt = 0u; mt < n_tiles; mt++) {
                                    const uint32_t rb = e_base + mt * BLOCK_M;
                                    const uint32_t re = e_base + ((mt + 1u) * BLOCK_M < cnt ? (mt + 1u) * BLOCK_M : cnt);
                                    p.ready_ring[head] = MegaMoeTile{le, rb, re, mt};
                                    head++;
                                }
                            }
                        }
                        n_mtiles = head;
                        *p.write_head = n_mtiles;
                        *params.down_write_cursor = n_mtiles * kNumNBlocksGu;
                    }
                    n_mtiles = __shfl_sync(0xffffffff, n_mtiles, 0);
                    __syncwarp();
                    ptx::fence_acq_rel_gpu();
                    const uint32_t total_gu = n_mtiles * kNumNBlocksGu;
                    for (uint32_t idx = lane_idx; idx < total_gu; idx += 32u) {
                        const uint32_t m = idx / kNumNBlocksGu;
                        const uint32_t nb = idx % kNumNBlocksGu;
                        ptx::st_relaxed_gpu_u32(params.down_ready_ring + idx, (nb << 20) | m);
                    }
                    ptx::fence_acq_rel_gpu();
                    if (lane_idx == 0u) {
                        ptx::gmem_signal_release(p.dispatch_done, 1u);
                    }
                }
        }
        if (tma_warp_idx == 0) {
            uint32_t queue_seq = 0;
            auto steal_next_task = [&](Task &task) -> bool {
                const uint32_t slot = queue_seq % Storage::kQueueSlots;
                const uint32_t q_phase = (queue_seq / Storage::kQueueSlots) & 1;
                queue_seq++;
                if (cute::elect_one_sync()) {
                    smem.tile_consumed[slot]->wait(q_phase ^ 1);
                    // Single hardware-coalesced `atomicAdd` per tile (ZERO `atomicCAS` contention!).
                    const uint32_t ticket = ptx::gmem_atomic_add_u32(params.gu_work_cursor, 1u);
                    uint32_t has_work = 0, op = 0, expert_id = 0, row_begin = 0, row_end = 0, n_blk = 0, r_idx = 0;
                    while (true) {
                        const uint32_t packed_task = ptx::ld_acquire_gpu_u32(params.down_ready_ring + ticket);
                        if (packed_task != 0xffffffffu) {
                            ptx::async_proxy_fence_global();
                            op = packed_task >> 31;
                            n_blk = (packed_task >> 20) & 0x7ffu;
                            r_idx = packed_task & 0xfffffu;
                            const MegaMoeTile st = params.ready_ring[r_idx];
                            expert_id = st.expert_id;
                            row_begin = st.row_begin;
                            row_end = st.row_end;
                            has_work = 1u;
                            break;
                        }
                        if (ptx::ld_acquire_gpu_u32(params.dispatch_done) != 0u) {
                            const uint32_t total_tasks = ptx::ld_acquire_gpu_u32(params.write_head) * kTasksPerMTile;
                            if (ticket >= total_tasks) {
                                has_work = 0u;
                                break;
                            }
                        }
                        ptx::nanosleep_ns(64u);
                    }
                    smem.op[slot] = op;
                    smem.expert_id[slot] = expert_id;
                    smem.row_begin[slot] = row_begin;
                    smem.row_end[slot] = row_end;
                    smem.n_block_idx[slot] = n_blk;
                    smem.ready_idx[slot] = r_idx;
                    smem.has_work[slot] = has_work;
                    smem.tile_ready[slot]->arrive();
                }
                __syncwarp();
                if (!smem.has_work[slot]) return false;
                task.op = smem.op[slot];
                task.expert_id = smem.expert_id[slot];
                task.row_begin = smem.row_begin[slot];
                task.row_end = smem.row_end[slot];
                task.n_block_idx = smem.n_block_idx[slot];
                task.ready_idx = smem.ready_idx[slot];
                return true;
            };

            Task task;
            while (steal_next_task(task)) {
                const bool is_dn = task.op != 0u;
                const uint32_t m_idx = task.row_begin;
                const uint32_t tile_valid_rows = task.row_end - m_idx;
                const uint32_t num_k_blocks = is_dn ? kNumKBlocksDn : kNumKBlocksGu;
                const float *sfa_ptr = is_dn ? params.inter_scale : params.sfa_gu;
                const uint32_t sfa_stride_m = is_dn ? params.sfa_dn_stride_m : params.sfa_gu_stride_m;
                const uint32_t sfa_stride_k = is_dn ? params.sfa_dn_stride_k : params.sfa_gu_stride_k;
                const uint64_t sfa_stride_e = is_dn ? params.sfa_dn_stride_expert : params.sfa_gu_stride_expert;
                const uint64_t tile_sfa_base =
                    task.expert_id * sfa_stride_e +
                    static_cast<uint64_t>(m_idx - task.expert_id * params.capacity) * sfa_stride_m;

                #pragma unroll 2
                for (uint32_t k_block = 0; k_block < num_k_blocks; k_block++) {
                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    smem.empty_barriers[next_stage]->wait(phase ^ (next_stage == 0) ^ 1);

                    if (cute::elect_one_sync()) {
                        auto *barrier = reinterpret_cast<uint64_t *>(static_cast<layout::Barrier *>(smem.full_barriers[stage_idx]));
                        if (!is_dn) {
                            cute::SM90_TMA_LOAD_2D::copy(
                                &params.tensor_map_a_gu, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST),
                                smem.a[stage_idx], k_block * 128, m_idx);
                            for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
                                const uint32_t n_idx = task.n_block_idx * BLOCK_N + tile_n * 128;
                                cute::SM90_TMA_LOAD_2D::copy(
                                    &params.tensor_map_b_gu, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST),
                                    smem.b[stage_idx] + tile_n * Storage::kTileBytesB, k_block * 16,
                                    task.expert_id * N_GU + n_idx);
                                cute::SM90_TMA_LOAD_2D::copy(
                                    &params.tensor_map_sfb_gu, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST),
                                    smem.sfb[stage_idx] + tile_n * Storage::kTileBytesSFB, n_idx,
                                    task.expert_id * kNumKBlocksGu + k_block);
                            }
                        } else {
                            cute::SM90_TMA_LOAD_2D::copy(
                                &params.tensor_map_a_dn, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST),
                                smem.a[stage_idx], k_block * 128, m_idx);
                            for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
                                const uint32_t n_idx = task.n_block_idx * BLOCK_N + tile_n * 128;
                                cute::SM90_TMA_LOAD_2D::copy(
                                    &params.tensor_map_b_dn, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST),
                                    smem.b[stage_idx] + tile_n * Storage::kTileBytesB, k_block * 16,
                                    task.expert_id * N_DN + n_idx);
                                cute::SM90_TMA_LOAD_2D::copy(
                                    &params.tensor_map_sfb_dn, barrier, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST),
                                    smem.sfb[stage_idx] + tile_n * Storage::kTileBytesSFB, n_idx,
                                    task.expert_id * kNumKBlocksDn + k_block);
                            }
                        }
                        smem.full_barriers[stage_idx]->arrive_and_expect_tx(kTmaBytesPerStage);
                    }

                    const uint64_t k_sfa_base = tile_sfa_base + static_cast<uint64_t>(k_block) * sfa_stride_k;
                    for (uint32_t i = 0; i < math::ceil_div(BLOCK_M, 32); i++) {
                        const uint32_t token = i * 32 + lane_idx;
                        if (token >= BLOCK_M) continue;
                        const uint64_t offset = k_sfa_base + static_cast<uint64_t>(token) * sfa_stride_m;
                        ptx::cp_async_load_pred(sfa_ptr + offset, smem.sfa[stage_idx] + token, token < tile_valid_rows);
                    }
                    ptx::cp_async_arrive(smem.full_barriers[stage_idx]);
                    advance_pipeline();
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const uint32_t r_0 = lane_idx / 4 + (warp_idx % 4) * 16 + (warp_idx / 4) * 64;

        uint32_t packed[kNumSubTiles][kNumKIters][2];
        uint32_t scale_rows[kNumSubTiles][4];
        uint32_t fragment[kNumSubTiles][kNumKIters][4];
        float accum[kNumSubTiles][WGMMA::kNumAccum];
        float final_accum[kNumSubTiles][WGMMA::kNumAccum];

        auto load_weights_pair = [&](uint32_t stage, uint32_t k) {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                const uint32_t half = i % kHalvesPerTile, tile_n = i / kHalvesPerTile;
                fpma::load_weights_pair(smem.b[stage] + tile_n * Storage::kTileBytesB, k, threadIdx.x + half * 128,
                                        packed[i][k], packed[i][k + 1]);
                if (k == 0) {
                    fpma::load_scale_rows(smem.sfb[stage] + tile_n * Storage::kTileBytesSFB, r_0 + half * 64,
                                          scale_rows[i]);
                }
            }
        };

        auto dequant = [&](uint32_t k) {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                uint32_t scales[4];
                fpma::scale_bytes(scale_rows[i], k, scales);
                fpma::dequant<kExactBridge>(packed[i][k], scales, fragment[i][k]);
            }
        };

        auto fence_weights = [&]() {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < 4; j++) cute::warpgroup_fence_operand(scale_rows[i][j]);
                for (uint32_t k = 0; k < kNumKIters; k++) {
                    for (uint32_t j = 0; j < 2; j++) cute::warpgroup_fence_operand(packed[i][k][j]);
                }
            }
        };

        auto make_a_desc = [&](uint32_t stage, uint32_t k) -> uint64_t {
            const uint32_t offset = k * WGMMA::K;
            const cute::GmmaDescriptor ring_desc = mma::sm90::make_smem_desc(smem.a[0]);
            const uint32_t start = ring_desc.reg32_[0] + (stage * Storage::kKBlockBytesA + offset) / 16;
            return static_cast<uint64_t>(ring_desc.reg32_[1]) << 32 | start;
        };

        uint32_t full_phases = 0;
        auto wait_full = [&](uint32_t stage) {
            smem.full_barriers[stage]->wait((full_phases >> stage) & 1);
            full_phases ^= 1u << stage;
        };

        auto fence_accum = [&]() {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < WGMMA::kNumAccum; j++) cute::warpgroup_fence_operand(accum[i][j]);
            }
        };

        uint32_t queue_seq = 0;
        auto pop_next_task = [&](Task &task) -> bool {
            const uint32_t slot = queue_seq % Storage::kQueueSlots;
            const uint32_t q_phase = (queue_seq / Storage::kQueueSlots) & 1;
            queue_seq++;
            smem.tile_ready[slot]->wait(q_phase);
            if (!smem.has_work[slot]) {
                smem.tile_consumed[slot]->arrive();
                return false;
            }
            task.op = smem.op[slot];
            task.expert_id = smem.expert_id[slot];
            task.row_begin = smem.row_begin[slot];
            task.row_end = smem.row_end[slot];
            task.n_block_idx = smem.n_block_idx[slot];
            task.ready_idx = smem.ready_idx[slot];
            smem.tile_consumed[slot]->arrive();
            return true;
        };

        Task task;
        while (pop_next_task(task)) {
            const bool is_dn = task.op != 0u;
            const uint32_t num_k_blocks = is_dn ? kNumKBlocksDn : kNumKBlocksGu;
            const float global_scale = is_dn ? params.global_scale_dn[task.expert_id]
                                             : params.global_scale_gu[task.expert_id];

            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < WGMMA::kNumAccum; j++) final_accum[i][j] = 0.0f;
            }

            // First K block weights
            wait_full(stage_idx);
            load_weights_pair(stage_idx, 0);
            if constexpr (kBatchMMA) {
                load_weights_pair(stage_idx, 2);
                fence_weights();
                for (uint32_t k = 0; k < kNumKIters; k++) dequant(k);
            } else {
                dequant(0);
            }

            // Unified pgroup WGMMA mainloop for both GateUp and Down!
            stage_idx = for_each_ring_slot<kNumStages>(stage_idx, num_k_blocks, [&](uint32_t stage,
                                                                                    uint32_t k_blocks_left) {
                const bool has_next_k_block = k_blocks_left > 1;
                const uint32_t next_stage = (stage + 1) % kNumStages;
                fence_accum();

                if constexpr (kBatchMMA) {
                    cute::warpgroup_arrive();
                    for (uint32_t k = 0; k < kNumKIters; k++) {
                        const uint64_t a_desc = make_a_desc(stage, k);
                        for (uint32_t i = 0; i < kNumSubTiles; i++) WGMMA::wgmma(fragment[i][k], a_desc, accum[i], k != 0);
                    }
                    cute::warpgroup_commit_batch();
                    if (has_next_k_block) {
                        wait_full(next_stage);
                        load_weights_pair(next_stage, 0);
                        load_weights_pair(next_stage, 2);
                    }
                } else {
                    for (uint32_t k = 0; k < kNumKIters; k++) {
                        const uint64_t a_desc = make_a_desc(stage, k);
                        cute::warpgroup_arrive();
                        for (uint32_t i = 0; i < kNumSubTiles; i++) WGMMA::wgmma(fragment[i][k], a_desc, accum[i], k != 0);
                        if (k == 0) {
                            load_weights_pair(stage, 2);
                            dequant(1);
                        } else if (k + 1 < kNumKIters) {
                            dequant(k + 1);
                        }
                    }
                    cute::warpgroup_commit_batch();
                    if (has_next_k_block) {
                        wait_full(next_stage);
                        load_weights_pair(next_stage, 0);
                    }
                }
                cute::warpgroup_wait<0>();
                fence_accum();

                for (uint32_t group = 0; group < BLOCK_M / 8; group++) {
                    const float2 block_scales =
                        *reinterpret_cast<const float2 *>(&smem.sfa[stage][group * 8 + (lane_idx % 4) * 2]);
                    for (uint32_t i = 0; i < kNumSubTiles; i++) {
                        float *sum = final_accum[i] + group * 4;
                        const float *product = accum[i] + group * 4;
                        sum[0] = fmaf(product[0], block_scales.x, sum[0]);
                        sum[1] = fmaf(product[1], block_scales.y, sum[1]);
                        sum[2] = fmaf(product[2], block_scales.x, sum[2]);
                        sum[3] = fmaf(product[3], block_scales.y, sum[3]);
                    }
                }

                if constexpr (kBatchMMA) {
                    fence_weights();
                    for (uint32_t k = 0; k < kNumKIters; k++) dequant(k);
                }

                if (cute::elect_one_sync()) smem.empty_barriers[stage]->arrive();
                __syncwarp();
                if constexpr (!kBatchMMA) {
                    if (has_next_k_block) dequant(0);
                }
            });

            const uint32_t m_idx = task.row_begin;
            const uint32_t num_tokens = min(task.row_end - task.row_begin, BLOCK_M);

            if (!is_dn) {
                // ---- GateUp Epilogue: Fused SwiGLU + group-128 FP8 quant + push ready M-tile to task_ring ----
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

                epilogue::store_swiglu_fp8_g128_impl<N_GU, BLOCK_M, kNumMathThreads, kMathBarrier, kHalvesPerTile>(
                    params.inter_fp8, params.inter_scale, smem, m_idx, num_tokens, task.n_block_idx, global_scale,
                    final_accum, r_0);

                // Count finished GateUp N-tiles for this M-tile; last N-tile pushes all `kNumNBlocksDn`
                // Down tasks in ONE coalesced 32-lane warp store after a single `fence.acq_rel.gpu`.
                if (warp_idx == 0) {
                    uint32_t is_last = 0, base = 0;
                    const auto [is_leader, leader] = cute::elect_one_leader_sync();
                    if (is_leader) {
                        const uint32_t done = ptx::gmem_atomic_add_u32(params.gu_tile_done + task.ready_idx, 1u) + 1u;
                        if (done == kNumNBlocksGu) {
                            base = ptx::gmem_atomic_add_u32(params.down_write_cursor, kNumNBlocksDn);
                            is_last = 1u;
                        }
                    }
                    is_last = __shfl_sync(0xffffffff, is_last, leader);
                    base = __shfl_sync(0xffffffff, base, leader);
                    __syncwarp();  // Formal PTX happens-before from leader's atom.acq_rel to lanes 1..31
                    if (is_last) {
                        ptx::fence_acq_rel_gpu();
                        for (uint32_t nb = lane_idx; nb < kNumNBlocksDn; nb += 32) {
                            const uint32_t packed_task = (1u << 31) | (nb << 20) | task.ready_idx;  // op=1 (Down)
                            ptx::st_relaxed_gpu_u32(params.down_ready_ring + base + nb, packed_task);
                        }
                    }
                }
            } else {
                // ---- Down Epilogue: Direct Coalesced 512B Warp NVLink Scatter (ZERO down_out HBM round-trip!) ----
                const uint32_t n_idx = task.n_block_idx * BLOCK_N;
                const uint32_t tid = threadIdx.x;
                const MegaMoeParams &p = params;
                const uint32_t *local_meta = nvlink_ep::symm_ptr<const uint32_t>(p, p.ep_rank, p.off_recv_meta);

                float2 token_scales[kOutputRows / 8];
                for (uint32_t g = 0; g < kOutputRows / 8; g++) token_scales[g] = make_float2(global_scale, global_scale);

                if (tid < p.ep_size) smem.dn_comm.peer_cnt[tid] = 0u;
                for (uint32_t i = 0; i < kNumSubTiles; i++) {
                    auto *box = reinterpret_cast<nv_bfloat16 *>(smem.d[i / kHalvesPerTile]) +
                                (i % kHalvesPerTile) * kOutputRows * 64;
                    epilogue::write_stsm<kOutputRows>(box, final_accum[i], token_scales, 0);
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

                if (tid < num_tokens) {
                    const uint32_t meta = local_meta[m_idx + tid];
                    smem.dn_comm.meta[tid] = meta;
                    atomicAdd(&smem.dn_comm.peer_cnt[meta >> 24], 1u);
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

                // Latch per-peer tile counts in registers BEFORE the post-scatter sync so the next tile can
                // safely overwrite `smem.dn_comm.peer_cnt` without racing Warp 0's remote atomic adds.
                const uint32_t my_peer_cnt = (warp_idx == 0u && lane_idx < p.ep_size) ? smem.dn_comm.peer_cnt[lane_idx] : 0u;

                // 1 warp (32 lanes) -> 1 row (256 BF16 = 512B = 32 x int4 = 4 contiguous 128B NVLink bursts).
                const auto *smem_d = reinterpret_cast<const int4 *>(smem.d);
                const uint32_t swizzle_base = cute::cast_smem_ptr_to_uint(smem_d) / 128u;
                const uint32_t box = lane_idx >> 3;
                const uint32_t col = lane_idx & 7u;
                constexpr uint32_t kBoxVecs = kOutputRows * 8u;
                for (uint32_t row = warp_idx; row < num_tokens; row += kNumMathThreads / 32u) {
                    const uint32_t meta = smem.dn_comm.meta[row];
                    const uint32_t src_r = meta >> 24;
                    const uint32_t k_slot = (meta >> 16) & 0xffu;
                    const uint32_t src_t = meta & 0xffffu;
                    const int4 val = smem_d[box * kBoxVecs + row * 8u + (col ^ ((swizzle_base + row) & 7u))];
                    auto *peer_comb = nvlink_ep::symm_ptr<nv_bfloat16>(p, src_r, p.off_combine_recv);
                    auto *dst_row = reinterpret_cast<int4 *>(
                        peer_comb + (static_cast<uint64_t>(src_t) * p.top_k + k_slot) * N_DN + n_idx);
                    dst_row[lane_idx] = val;
                }

                cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);
                if (warp_idx == 0u) {
                    if (lane_idx == 0u) nvlink_ep::fence_acq_rel_sys();
                    __syncwarp();
                    if (my_peer_cnt > 0u) {
                        uint32_t *peer_arr = nvlink_ep::symm_ptr<uint32_t>(p, lane_idx, p.off_combine_arrived);
                        nvlink_ep::atomic_add_sys_u32(peer_arr, my_peer_cnt);
                    }
                }
            }
        }
    }

    // Join Math (Warps 0..7), TMA (Warp 8), and Comm (Warps 9..11) on this CTA.
    __syncthreads();
    if (warp_idx >= kNumMathThreads / 32) return;

    // Weighted BF16 Reduction + Inline Self-Reset + 1 NVLink Barrier (Math warps 0..7 = 256 threads).
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n_ctas = gridDim.x;
    const MegaMoeParams &p = params;

    const uint32_t expected_combine_tiles = p.num_tokens * p.top_k * kNumNBlocksDn;
    if (expected_combine_tiles > 0u) {
        const uint32_t *local_comb_arr = nvlink_ep::symm_ptr<const uint32_t>(p, p.ep_rank, p.off_combine_arrived);
        if (tid == 0u) {
            while (nvlink_ep::ld_acquire_sys_u32(local_comb_arr) < expected_combine_tiles) {
                ptx::nanosleep_ns(64u);
            }
        }
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        const auto *comb_bf16 = nvlink_ep::symm_ptr<const nv_bfloat16>(p, p.ep_rank, p.off_combine_recv);
        auto *out_bf16 = static_cast<nv_bfloat16 *>(p.local_final_out);
        const uint32_t h_pairs = p.hidden / 2u;
        for (uint32_t t = bid; t < p.num_tokens; t += n_ctas) {
            const float *w_row = p.topk_weights + static_cast<uint64_t>(t) * p.top_k;
            const auto *in_base = reinterpret_cast<const nv_bfloat162 *>(
                comb_bf16 + static_cast<uint64_t>(t) * p.top_k * p.hidden);
            auto *out_row = reinterpret_cast<nv_bfloat162 *>(out_bf16 + static_cast<uint64_t>(t) * p.hidden);
            for (uint32_t j = tid; j < h_pairs; j += kNumMathThreads) {
                float2 acc = make_float2(0.0f, 0.0f);
                #pragma unroll 4
                for (uint32_t k = 0u; k < p.top_k; k++) {
                    const float w = w_row[k];
                    const float2 val = __bfloat1622float2(in_base[static_cast<uint64_t>(k) * h_pairs + j]);
                    acc.x = fmaf(val.x, w, acc.x);
                    acc.y = fmaf(val.y, w, acc.y);
                }
                out_row[j] = __float22bfloat162_rn(acc);
            }
        }
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);
    }

    if (tid == 0u) {
        ptx::fence_acq_rel_gpu();
        const uint32_t done = ptx::gmem_atomic_add_u32(p.local_phase_done, 1u) + 1u;
        smem.comm.is_last_cta = (done == n_ctas) ? 1u : 0u;
    }
    cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

    if (smem.comm.is_last_cta != 0u) {
        // Inline self-reset of ONLY the touched task_ring slots + remote/local control words (ZERO cudaMemset!).
        const uint32_t n_mtiles = *p.write_head;
        const uint32_t total_reset = n_mtiles * kTasksPerMTile + n_ctas;
        for (uint32_t i = tid; i < total_reset; i += kNumMathThreads) {
            params.down_ready_ring[i] = 0xffffffffu;
        }
        for (uint32_t i = tid; i < n_mtiles; i += kNumMathThreads) {
            params.gu_tile_done[i] = 0u;
        }
        uint32_t *slot_cnt = nvlink_ep::symm_ptr<uint32_t>(p, p.ep_rank, p.off_recv_slot_count);
        uint32_t *arrived = nvlink_ep::symm_ptr<uint32_t>(p, p.ep_rank, p.off_expert_arrived);
        for (uint32_t le = tid; le < p.local_e; le += kNumMathThreads) {
            slot_cnt[le] = 0u;
            arrived[le] = 0u;
        }
        if (tid == 0u) {
            *nvlink_ep::symm_ptr<uint32_t>(p, p.ep_rank, p.off_combine_arrived) = 0u;
            *p.write_head = 0u;
            *p.dispatch_done = 0u;
            *params.gu_work_cursor = 0u;
            *params.down_write_cursor = 0u;
            *p.local_phase_done = 0u;
            const uint32_t next_gen = *p.local_gen + 1u;
            *p.local_gen = next_gen;
            smem.comm.next_gen = next_gen;
        }
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        const uint32_t gen = smem.comm.next_gen;
        if (tid < 32u) {
            if (tid == 0u) nvlink_ep::fence_acq_rel_sys();
            __syncwarp();
            if (tid < p.ep_size) {
                uint32_t *peer_bar = nvlink_ep::symm_ptr<uint32_t>(p, tid, p.off_barrier_flags);
                nvlink_ep::st_release_sys_u32(peer_bar + p.ep_rank, gen);
            }
            __syncwarp();
            if (tid < p.ep_size) {
                const uint32_t *my_bar = nvlink_ep::symm_ptr<const uint32_t>(p, p.ep_rank, p.off_barrier_flags);
                while (nvlink_ep::ld_acquire_sys_u32(my_bar + tid) < gen) {
                    ptx::nanosleep_ns(64u);
                }
            }
        }
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);
        if (tid == 0u) cudaTriggerProgrammaticLaunchCompletion();
    }
}

}  // namespace tensorbridge
