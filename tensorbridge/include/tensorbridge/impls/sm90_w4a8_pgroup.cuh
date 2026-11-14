#pragma once

#include <cuda_device_runtime_api.h>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <tensorbridge/common/params.h>
#include <tensorbridge/common/ptx.cuh>
#include <tensorbridge/common/utils.cuh>
#include <tensorbridge/dequant/fpma.cuh>
#include <tensorbridge/epilogue/sm90.cuh>
#include <tensorbridge/layout/sm90.cuh>
#include <tensorbridge/memory/g2s.cuh>
#include <tensorbridge/mma/sm90.cuh>
#include <tensorbridge/scheduler/sm90.cuh>

namespace tensorbridge {

// NVFP4 weights x FP8 activations with `[1, 128]` activation scales, dense and grouped.
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t OUTPUT_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumStages, bool kUseStreamK, bool kMFastTileOrder, bool kScaleMMajor, bool kExactBridge,
          GroupedLayout kLayout = GroupedLayout::kDense, uint32_t kNumExperts = 0>
__global__ __launch_bounds__(384, 1) void sm90_w4a8_pgroup(const __grid_constant__ GemmParams params) {
    // Shape checks
    static_assert(BLOCK_M % 8 == 0 && BLOCK_M >= 16 && BLOCK_M <= 192, "Invalid token tile");
    static_assert(BLOCK_N == 128 || BLOCK_N == 256, "A CTA computes one or two channel tiles");
    static_assert(SHAPE_N % BLOCK_N == 0 && SHAPE_K % 128 == 0, "N and K must be multiples of the tile");
    static_assert(OUTPUT_N % 8 == 0 && OUTPUT_N <= SHAPE_N && SHAPE_N - OUTPUT_N < 128, "D has the weight's N columns");
    static_assert(kNumStages >= 3 && kNumStages <= 32, "Too few stages to overlap loads, or too many barriers");

    // Types
    constexpr uint32_t BLOCK_K = 128;
    using WGMMA = typename mma::sm90::FP8RSMMASelector<BLOCK_M>::type;
    using Storage = layout::PgroupStorage<BLOCK_M, BLOCK_N, kNumStages, kLayout, kNumExperts>;
    constexpr bool kIndexed = kLayout == GroupedLayout::kIndexed;
    constexpr bool kGrouped = kLayout != GroupedLayout::kDense;
    static_assert(!kGrouped || !kUseStreamK, "Grouped kernels do not split K");

    // Configs
    constexpr uint32_t kNumMathThreads = 256;
    constexpr uint32_t kNumThreads = 384;
    constexpr uint32_t kTilesN = Storage::kTilesN;
    constexpr uint32_t kOutputRows = Storage::kOutputRows;
    constexpr uint32_t kNumKIters = BLOCK_K / WGMMA::K;
    // NOTES: on the largest token tiles the promotion outlasts a K block's WGMMAs, which are then committed as one
    // batch, so that the next K block's dequantization follows the promotion and the compiler schedules the two
    // together; on smaller tiles the dequantization runs between the WGMMAs
    constexpr bool kBatchMMA = BLOCK_M * kTilesN > 128;
    constexpr uint32_t kTmaBytesPerStage = BLOCK_M * 128 + kTilesN * (Storage::kTileBytesB + Storage::kTileBytesSFB);
    constexpr uint32_t kMathBarrier = 1;
    constexpr uint32_t kGatherBarrier = 2, kNumGatherThreads = 96;

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = kIndexed ? 64 : 40;
    constexpr uint32_t kLaunchRegs = 65536 / kNumThreads / 8 * 8;
    constexpr uint32_t kNumMathRegisters = (kLaunchRegs * kNumThreads - kNumTMARegisters * 128) / kNumMathThreads / 8 * 8;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Data on shared memory
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto &smem = *reinterpret_cast<Storage *>(smem_buffer);

    if (threadIdx.x == kNumMathThreads) {
        // Prefetch TMA descriptors at the very beginning
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_sfb));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_d));

        // Initialize barriers
        for (uint32_t s = 0; s < kNumStages; s++) {
            smem.full_barriers[s]->init(kIndexed ? 1 + kNumGatherThreads : 1 + 32);
            smem.empty_barriers[s]->init(kNumMathThreads / 32);
        }

        if constexpr (kGrouped) smem.routing_ready->init(kIndexed ? 1 : 32);
        else {}
        if constexpr (kIndexed) {
            smem.rows_consumed[0]->init(kNumMathThreads);
            smem.rows_consumed[1]->init(kNumMathThreads);
        } else {}

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
        cutlass::arch::fence_view_async_shared();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    auto scheduler = sched::make_scheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder, kUseStreamK, false,
                                           kLayout, kNumExperts>(params, smem);
    typename decltype(scheduler)::Tile tile;

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        if constexpr (kGrouped) {
            if (warp_idx == kNumMathThreads / 32 + 1) {
                scheduler.load_routing(params.expert_offsets, params.expert_counts, params.num_tokens_padded);
                scheduler.prepare();
            } else {}
        } else {}
        scheduler.wait_ready();
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32) {
            // TMA warp
            while (scheduler.get_next_block(tile)) {
                const uint32_t m_idx = tile.row_begin;
                #pragma unroll 2
                for (uint32_t k_block = 0; k_block < tile.num_k_blocks; k_block++) {
                    // Wait consumer release
                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    smem.empty_barriers[next_stage]->wait(phase ^ (next_stage == 0) ^ 1);
                    const uint32_t k_block_idx = tile.k_block_idx + k_block;

                    // Issue TMA
                    if (lane_idx == 0) {
                        if constexpr (!kIndexed) g2s::load_activations(params, smem, stage_idx, k_block_idx, m_idx);
                        else {}
                        g2s::load_weights<SHAPE_N, SHAPE_K>(params, smem, stage_idx, k_block_idx, tile.n_block_idx, tile.expert_id);
                        smem.full_barriers[stage_idx]->arrive_and_expect_tx(
                            kTmaBytesPerStage - (kIndexed ? BLOCK_M * 128 : 0));
                    }

                    if constexpr (!kIndexed) {
                        // Read only this expert's valid rows; masked scales may be transposed.
                        for (uint32_t i = 0; i < math::ceil_div(BLOCK_M, 32); i++) {
                            const uint32_t token = i * 32 + lane_idx;
                            if (token >= BLOCK_M) continue;
                            const uint32_t row = m_idx + token;
                            uint64_t row_offset;
                            if constexpr (kLayout == GroupedLayout::kMasked) {
                                const uint32_t capacity = params.shape_m / kNumExperts;
                                row_offset = tile.expert_id * params.sfa_stride_expert +
                                             static_cast<uint64_t>(row % capacity) * params.sfa_stride_m;
                            } else {
                                row_offset = static_cast<uint64_t>(row) * params.sfa_stride_m;
                            }
                            const uint64_t offset = row_offset + static_cast<uint64_t>(k_block_idx) * params.sfa_stride_k;
                            ptx::cp_async_load_pred(params.sfa + offset, smem.sfa[stage_idx] + token, row < tile.row_end);
                        }
                        ptx::cp_async_arrive(smem.full_barriers[stage_idx]);
                    } else {}
                    advance_pipeline();
                }
            }

            // Let the next kernel launch
            cudaTriggerProgrammaticLaunchCompletion();
        } else if constexpr (kIndexed) {
            g2s::Gather<SHAPE_K, kNumGatherThreads, kGatherBarrier> gather(threadIdx.x - kNumMathThreads - 32);
            uint32_t tile_count = 0;
            while (scheduler.get_next_block(tile)) {
                const uint32_t buffer = tile_count % 2, buffer_phase = (tile_count / 2) & 1;
                smem.rows_consumed[buffer]->wait(buffer_phase ^ 1);
                gather.load_routes(params, smem, tile.row_begin, buffer);
                #pragma unroll 2
                for (uint32_t k_block = 0; k_block < tile.num_k_blocks; k_block++) {
                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    smem.empty_barriers[next_stage]->wait(phase ^ (next_stage == 0) ^ 1);
                    const uint32_t k_index = tile.k_block_idx + k_block;
                    gather.load_activations(params, smem, stage_idx, k_index, buffer);
                    const uint32_t gather_thread = threadIdx.x - kNumMathThreads - 32;
                    for (uint32_t row = gather_thread; row < BLOCK_M; row += kNumGatherThreads) {
                        const uint32_t input_row = smem.input_rows[buffer][row];
                        const uint64_t offset = static_cast<uint64_t>(input_row) * params.sfa_stride_m +
                                                static_cast<uint64_t>(k_index) * params.sfa_stride_k;
                        ptx::cp_async_load_pred(params.sfa + offset, smem.sfa[stage_idx] + row,
                                               input_row < params.shape_m);
                    }
                    ptx::cp_async_arrive(smem.full_barriers[stage_idx]);
                    advance_pipeline();
                }
                tile_count++;
            }
        } else if constexpr (kUseStreamK) {
            // Zero the tiles whose last K slice is this CTA's
            if (warp_idx == kNumMathThreads / 32 + 2) {
                epilogue::zero_stream_k_tiles<OUTPUT_N, BLOCK_M, BLOCK_N>(scheduler, static_cast<nv_bfloat16 *>(params.d),
                                                                         params.locks);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const uint32_t r_0 = lane_idx / 4 + (warp_idx % 4) * 16 + (warp_idx / 4) * 64;

        // Weights before and after the FPMA bridge, and accumulation for WGMMA
        uint32_t packed[kTilesN][kNumKIters][2];
        uint32_t scale_rows[kTilesN][4];
        uint32_t fragment[kTilesN][kNumKIters][4];
        float accum[kTilesN][WGMMA::kNumAccum];
        float final_accum[kTilesN][WGMMA::kNumAccum];

        // Read two k32 iterations' weights from shared memory, with the K block's scales before the first two
        auto load_weights_pair = [&](uint32_t stage, uint32_t k) {
            for (uint32_t t = 0; t < kTilesN; t++) {
                fpma::load_weights_pair(smem.b[stage] + t * Storage::kTileBytesB, k, threadIdx.x, packed[t][k],
                                        packed[t][k + 1]);
                if (k == 0) fpma::load_scale_rows(smem.sfb[stage] + t * Storage::kTileBytesSFB, r_0, scale_rows[t]);
            }
        };

        // Run one k32 iteration's weights through the FPMA bridge
        auto dequant = [&](uint32_t k) {
            for (uint32_t t = 0; t < kTilesN; t++) {
                uint32_t scales[4];
                fpma::scale_bytes(scale_rows[t], k, scales);
                fpma::dequant<kExactBridge>(packed[t][k], scales, fragment[t][k]);
            }
        };

        // NOTES: without the fences the compiler builds a batch's lookup tables before its WGMMAs are waited, in
        // registers the largest token tiles do not have
        auto fence_weights = [&]() {
            for (uint32_t t = 0; t < kTilesN; t++) {
                for (uint32_t i = 0; i < 4; i++) cute::warpgroup_fence_operand(scale_rows[t][i]);
                for (uint32_t k = 0; k < kNumKIters; k++) {
                    for (uint32_t i = 0; i < 2; i++) cute::warpgroup_fence_operand(packed[t][k][i]);
                }
            }
        };

        // Wait TMA arrivals, with one phase bit per stage
        // NOTES: a single phase register spills on the largest token tiles of this kernel
        uint32_t full_phases = 0;
        auto wait_full = [&](uint32_t stage) {
            smem.full_barriers[stage]->wait((full_phases >> stage) & 1);
            full_phases ^= 1u << stage;
        };

        auto fence_accum = [&]() {
            for (uint32_t t = 0; t < kTilesN; t++) {
                for (uint32_t i = 0; i < WGMMA::kNumAccum; i++) cute::warpgroup_fence_operand(accum[t][i]);
            }
        };

        // Persistently schedule over blocks
        scheduler.wait_ready();
        uint32_t tile_count = 0;
        while (scheduler.get_next_block(tile)) {
            for (uint32_t t = 0; t < kTilesN; t++) {
                for (uint32_t i = 0; i < WGMMA::kNumAccum; i++) final_accum[t][i] = 0;
            }
            const float global_scale = params.global_scale[tile.expert_id];

            // Wait TMA arrivals
            wait_full(stage_idx);

            // Prepare the first k32 iteration, or the first K block
            load_weights_pair(stage_idx, 0);
            dequant(0);
            if constexpr (kBatchMMA) {
                load_weights_pair(stage_idx, 2);
                for (uint32_t k = 1; k < kNumKIters; k++) dequant(k);
            }

            stage_idx = for_each_ring_slot<kNumStages>(stage_idx, tile.num_k_blocks, [&](uint32_t stage,
                                                                                        uint32_t k_blocks_left) {
                const bool has_next_k_block = k_blocks_left > 1;
                const uint32_t next_stage = (stage + 1) % kNumStages;
                fence_accum();

                if constexpr (kBatchMMA) {
                    // Commit WGMMA instructions
                    cute::warpgroup_arrive();
                    for (uint32_t k = 0; k < kNumKIters; k++) {
                        const uint64_t a_desc = mma::sm90::make_smem_desc(smem.a[stage] + k * WGMMA::K).desc_;
                        for (uint32_t t = 0; t < kTilesN; t++) WGMMA::wgmma(fragment[t][k], a_desc, accum[t], k != 0);
                    }
                    cute::warpgroup_commit_batch();

                    // Read the next K block's weights, then wait the WGMMAs
                    if (has_next_k_block) {
                        wait_full(next_stage);
                        load_weights_pair(next_stage, 0);
                        load_weights_pair(next_stage, 2);
                    }
                } else {
                    // Commit WGMMA instructions, dequantizing each next k32 iteration while the previous ones run
                    for (uint32_t k = 0; k < kNumKIters; k++) {
                        const uint64_t a_desc = mma::sm90::make_smem_desc(smem.a[stage] + k * WGMMA::K).desc_;
                        cute::warpgroup_arrive();
                        for (uint32_t t = 0; t < kTilesN; t++) WGMMA::wgmma(fragment[t][k], a_desc, accum[t], k != 0);
                        if (k == 0) {
                            load_weights_pair(stage, 2);
                            dequant(1);
                        } else if (k + 1 < kNumKIters) {
                            dequant(k + 1);
                        }
                    }
                    cute::warpgroup_commit_batch();

                    // Read the next K block's first weights, then wait the WGMMAs
                    if (has_next_k_block) {
                        wait_full(next_stage);
                        load_weights_pair(next_stage, 0);
                    }
                }
                cute::warpgroup_wait<0>();
                fence_accum();

                // Promote with scales
                for (uint32_t group = 0; group < BLOCK_M / 8; group++) {
                    const float2 block_scales =
                        *reinterpret_cast<const float2 *>(&smem.sfa[stage][group * 8 + (lane_idx % 4) * 2]);
                    for (uint32_t t = 0; t < kTilesN; t++) {
                        float *sum = final_accum[t] + group * 4;
                        const float *product = accum[t] + group * 4;
                        sum[0] = fmaf(product[0], block_scales.x, sum[0]);
                        sum[1] = fmaf(product[1], block_scales.y, sum[1]);
                        sum[2] = fmaf(product[2], block_scales.x, sum[2]);
                        sum[3] = fmaf(product[3], block_scales.y, sum[3]);
                    }
                }

                if constexpr (kBatchMMA) {
                    // Dequantize the next K block
                    // NOTES: after the last K block too: behind a branch the compiler would not schedule it with
                    // the promotion
                    fence_weights();
                    for (uint32_t k = 0; k < kNumKIters; k++) dequant(k);
                }

                // Release this stage
                if (lane_idx == 0) smem.empty_barriers[stage]->arrive();
                __syncwarp();
                if constexpr (!kBatchMMA) {
                    if (has_next_k_block) dequant(0);
                }
            });

            // Scale by the global scale, round to BF16 and store the tile
            float2 token_scales[kOutputRows / 8];
            for (uint32_t group = 0; group < kOutputRows / 8; group++) token_scales[group] = make_float2(global_scale, global_scale);
            const uint32_t m_idx = tile.row_begin, n_idx = tile.n_block_idx * BLOCK_N;
            const auto *smem_d = reinterpret_cast<const int4 *>(smem.d);
            for (uint32_t chunk = 0; chunk < BLOCK_M / kOutputRows; chunk++) {
                // Wait last TMA store to be finished
                cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

                // Stream-K: wait for the tile to be zeroed
                if (chunk == 0 && !tile.is_full_depth()) {
                    ptx::gmem_lock_acquire<kNumMathThreads, kMathBarrier>(&params.locks[tile.lock_idx], -1);
                }

                // Write to shared memory
                for (uint32_t t = 0; t < kTilesN; t++) {
                    epilogue::write_stsm<kOutputRows>(reinterpret_cast<nv_bfloat16 *>(smem.d[t]), final_accum[t], token_scales,
                                                      chunk);
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

                // Store to global memory
                if constexpr (kGrouped) {
                    const int32_t *output_rows;
                    if constexpr (kIndexed) output_rows = smem.row_ids[tile_count % 2];
                    else output_rows = nullptr;
                    epilogue::store_rows<OUTPUT_N, BLOCK_N, kOutputRows, kNumMathThreads>(
                        smem_d, static_cast<nv_bfloat16 *>(params.d), output_rows, m_idx, n_idx,
                        kIndexed ? params.num_routes : min(tile.row_end - tile.row_begin, BLOCK_M), chunk);
                } else {
                    cute::tma_store_fence();
                    epilogue::store_tma<BLOCK_N, kOutputRows>(smem_d, &params.tensor_map_d, m_idx, n_idx,
                                                            !tile.is_full_depth(), chunk);
                }
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);
            if (!tile.is_full_depth() && threadIdx.x == 0) ptx::gmem_lock_increment(&params.locks[tile.lock_idx]);
            if constexpr (kIndexed) smem.rows_consumed[tile_count % 2]->arrive();
            else {}
            tile_count++;
        }

        // Wait for the last TMA stores before the shared memory goes
        cute::tma_store_wait<0>();
    }

    __syncthreads();
}

}  // namespace tensorbridge
