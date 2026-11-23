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
          uint32_t kNumStages, uint32_t kNumCtasPerSm, uint32_t kNumMathWarpgroups, bool kUseStreamK,
          bool kMFastTileOrder, bool kScaleMMajor, bool kExactBridge,
          GroupedLayout kLayout = GroupedLayout::kDense, uint32_t kNumExperts = 0,
          bool kFuseSwiGluFp8 = false>
__global__ __launch_bounds__(128 * (kNumMathWarpgroups + 1), kNumCtasPerSm) void sm90_w4a8_pgroup(
    const __grid_constant__ GemmParams params) {
    // Shape checks
    static_assert(BLOCK_M % 8 == 0 && BLOCK_M >= 8 && BLOCK_M <= 192, "Invalid token tile");
    static_assert(BLOCK_N == 128 || BLOCK_N == 256, "A CTA computes one or two channel tiles");
    static_assert(SHAPE_N % BLOCK_N == 0 && SHAPE_K % 128 == 0, "N and K must be multiples of the tile");
    static_assert(OUTPUT_N % 8 == 0 && OUTPUT_N <= SHAPE_N && SHAPE_N - OUTPUT_N < 128, "D has the weight's N columns");
    static_assert(kLayout == GroupedLayout::kDense || OUTPUT_N == SHAPE_N, "Grouped weights are multiples of 128");
    static_assert(kNumMathWarpgroups == 1 || kNumMathWarpgroups == 2, "One or two math warp groups");
    static_assert(kNumCtasPerSm >= 1 && kNumCtasPerSm <= (kNumMathWarpgroups == 2 ? 2 : 4), "Invalid CTAs per SM");
    static_assert(kNumMathWarpgroups == 2 || kNumCtasPerSm >= 2, "One math warp group exceeds 256 registers per thread");
    static_assert(kNumStages >= 3 && kNumStages <= 32, "Too few stages to overlap loads, or too many barriers");
    static_assert(!kFuseSwiGluFp8 || (BLOCK_N == 256 && SHAPE_N % 256 == 0 && OUTPUT_N % 256 == 0 &&
                                      kLayout != GroupedLayout::kIndexed && !kUseStreamK),
                  "Fused SwiGLU+FP8 epilogue requires BLOCK_N==256, OUTPUT_N%256==0, non-indexed, non-Stream-K");

    // Types
    constexpr uint32_t BLOCK_K = 128;
    using WGMMA = typename mma::sm90::FP8RSMMASelector<BLOCK_M>::type;
    using Storage = layout::PgroupStorage<BLOCK_M, BLOCK_N, kNumStages, kLayout, kNumExperts>;
    constexpr bool kIndexed = kLayout == GroupedLayout::kIndexed;
    constexpr bool kGrouped = kLayout != GroupedLayout::kDense;
    static_assert(!kGrouped || !kUseStreamK, "Grouped kernels do not split K");

    // Configs
    constexpr uint32_t kNumMathThreads = 128 * kNumMathWarpgroups;
    constexpr uint32_t kNumThreads = kNumMathThreads + 128;
    constexpr uint32_t kTilesN = Storage::kTilesN;
    constexpr uint32_t kHalvesPerTile = 2 / kNumMathWarpgroups;
    constexpr uint32_t kNumSubTiles = kTilesN * kHalvesPerTile;
    constexpr uint32_t kOutputRows = Storage::kOutputRows;
    constexpr uint32_t kNumKIters = BLOCK_K / WGMMA::K;
    // NOTES: on the largest token tiles the promotion outlasts a K block's WGMMAs, which are then committed as one
    // batch, so that the next K block's dequantization follows the promotion and the compiler schedules the two
    // together; on smaller tiles the dequantization runs between the WGMMAs
    constexpr bool kBatchMMA = BLOCK_M * kNumSubTiles > 128;
    constexpr bool kLookAhead = kGrouped && !kIndexed;
    constexpr bool kUseRingDescs = kGrouped && !kIndexed && (kNumCtasPerSm >= 2 || BLOCK_M < 64);
    constexpr uint32_t kTmaBytesPerStage = BLOCK_M * 128 + kTilesN * (Storage::kTileBytesB + Storage::kTileBytesSFB);
    constexpr uint32_t kMathBarrier = 1;
    constexpr uint32_t kGatherBarrier = 2, kNumGatherThreads = 96;

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = kIndexed ? 64 : 40;
    constexpr uint32_t kLaunchRegs = 65536 / (kNumThreads * kNumCtasPerSm) / 8 * 8;
    constexpr uint32_t kNumMathRegisters = (kLaunchRegs * kNumThreads - kNumTMARegisters * 128) / kNumMathThreads / 8 * 8;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Data on shared memory
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto &smem = *reinterpret_cast<Storage *>(smem_buffer);

    // Block scheduler
    auto scheduler = sched::make_scheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder, kUseStreamK,
                                           false, kLayout, kNumExperts>(params, smem);
    using Tile = typename decltype(scheduler)::Tile;
    Tile tile;

    // Load the routing, overlapped with the barrier initialization
    if constexpr (kGrouped) {
        if (warp_idx == kNumMathThreads / 32 + 1) {
            cudaGridDependencySynchronize();
            scheduler.load_routing(params.expert_offsets, params.expert_counts, params.num_tokens_padded);
        }
    }

    if (warp_idx == kNumMathThreads / 32 && cute::elect_one_sync()) {
        // Prefetch TMA descriptors at the very beginning
        if constexpr (!kIndexed) {
            cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_a));
        }
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_sfb));
        if constexpr (!kFuseSwiGluFp8) {
            cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_d));
        }

        // Initialize barriers
        for (uint32_t s = 0; s < kNumStages; s++) {
            smem.full_barriers[s]->init(kIndexed ? 1 + kNumGatherThreads : 1 + 32);
            smem.empty_barriers[s]->init(kNumMathThreads / 32);
        }

        if constexpr (kGrouped) {
            smem.routing_ready->init(kIndexed ? 1 : 32);
        } else {}
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

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        const uint32_t tma_warp_idx = warp_idx - kNumMathThreads / 32;
        if constexpr (kGrouped) {
            if (tma_warp_idx == 1) scheduler.prepare();
        }
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (tma_warp_idx == 0) {
            // TMA warp
            scheduler.wait_ready();
            uint32_t capacity = 0;
            if constexpr (kLayout == GroupedLayout::kMasked) capacity = params.shape_m / kNumExperts;
            while (scheduler.get_next_block(tile)) {
                const uint32_t m_idx = tile.row_begin;
                uint64_t tile_sfa_base = 0;
                if constexpr (!kIndexed) {
                    if constexpr (kLayout == GroupedLayout::kMasked) {
                        tile_sfa_base = tile.expert_id * params.sfa_stride_expert +
                                        static_cast<uint64_t>(m_idx - tile.expert_id * capacity) * params.sfa_stride_m;
                    } else {
                        tile_sfa_base = static_cast<uint64_t>(m_idx) * params.sfa_stride_m;
                    }
                }
                const uint32_t tile_valid_rows = tile.row_end - m_idx;
                #pragma unroll 2
                for (uint32_t k_block = 0; k_block < tile.num_k_blocks; k_block++) {
                    // Wait consumer release
                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    smem.empty_barriers[next_stage]->wait(phase ^ (next_stage == 0) ^ 1);
                    const uint32_t k_block_idx = tile.k_block_idx + k_block;

                    // Issue TMA
                    if (cute::elect_one_sync()) {
                        if constexpr (!kIndexed) g2s::load_activations(params, smem, stage_idx, k_block_idx, m_idx);
                        else {}
                        g2s::load_weights<SHAPE_N, SHAPE_K>(params, smem, stage_idx, k_block_idx, tile.n_block_idx, tile.expert_id);
                        smem.full_barriers[stage_idx]->arrive_and_expect_tx(
                            kTmaBytesPerStage - (kIndexed ? BLOCK_M * 128 : 0));
                    }

                    if constexpr (!kIndexed) {
                        // Read only this expert's valid rows; masked scales may be transposed.
                        const uint64_t k_sfa_base =
                            tile_sfa_base + static_cast<uint64_t>(k_block_idx) * params.sfa_stride_k;
                        for (uint32_t i = 0; i < math::ceil_div(BLOCK_M, 32); i++) {
                            const uint32_t token = i * 32 + lane_idx;
                            if (token >= BLOCK_M) continue;
                            const uint64_t offset = k_sfa_base + static_cast<uint64_t>(token) * params.sfa_stride_m;
                            ptx::cp_async_load_pred(params.sfa + offset, smem.sfa[stage_idx] + token,
                                                    token < tile_valid_rows);
                        }
                        ptx::cp_async_arrive(smem.full_barriers[stage_idx]);
                    } else {}
                    advance_pipeline();
                }
            }

            // Let the next kernel launch
            cudaTriggerProgrammaticLaunchCompletion();
        } else if constexpr (kIndexed) {
            scheduler.wait_ready();
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
            if (tma_warp_idx == 2) {
                epilogue::zero_stream_k_tiles<OUTPUT_N, BLOCK_M, BLOCK_N>(scheduler, static_cast<nv_bfloat16 *>(params.d),
                                                                         params.locks);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const uint32_t r_0 = lane_idx / 4 + (warp_idx % 4) * 16 + (warp_idx / 4) * 64;

        // Weights before and after the FPMA bridge, and accumulation for WGMMA
        uint32_t packed[kNumSubTiles][kNumKIters][2];
        uint32_t scale_rows[kNumSubTiles][4];
        uint32_t fragment[kNumSubTiles][kNumKIters][4];
        float accum[kNumSubTiles][WGMMA::kNumAccum];
        float final_accum[kNumSubTiles][WGMMA::kNumAccum];

        // Read two k32 iterations' weights from shared memory, with the K block's scales before the first two
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

        // Run one k32 iteration's weights through the FPMA bridge
        auto dequant = [&](uint32_t k) {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                uint32_t scales[4];
                fpma::scale_bytes(scale_rows[i], k, scales);
                fpma::dequant<kExactBridge>(packed[i][k], scales, fragment[i][k]);
            }
        };

        // NOTES: without the fences the compiler builds a batch's lookup tables before its WGMMAs are waited, in
        // registers the largest token tiles do not have
        auto fence_weights = [&]() {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < 4; j++) cute::warpgroup_fence_operand(scale_rows[i][j]);
                for (uint32_t k = 0; k < kNumKIters; k++) {
                    for (uint32_t j = 0; j < 2; j++) cute::warpgroup_fence_operand(packed[i][k][j]);
                }
            }
        };

        // WGMMA descriptor of one k32 iteration's activations
        auto make_a_desc = [&](uint32_t stage, uint32_t k) -> uint64_t {
            const uint32_t offset = k * WGMMA::K;
            if constexpr (kUseRingDescs) {
                const cute::GmmaDescriptor ring_desc = mma::sm90::make_smem_desc(smem.a[0]);
                const uint32_t start = ring_desc.reg32_[0] + (stage * Storage::kKBlockBytesA + offset) / 16;
                return static_cast<uint64_t>(ring_desc.reg32_[1]) << 32 | start;
            } else {
                return mma::sm90::make_smem_desc(smem.a[stage] + offset).desc_;
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
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < WGMMA::kNumAccum; j++) cute::warpgroup_fence_operand(accum[i][j]);
            }
        };

        // Persistently schedule over blocks
        scheduler.wait_ready();
        Tile next_tile;
        bool has_next = scheduler.get_next_block(next_tile);
        while (has_next) {
            tile = next_tile;
            if constexpr (kLookAhead) has_next = scheduler.get_next_block(next_tile);

            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < WGMMA::kNumAccum; j++) final_accum[i][j] = 0;
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
                        const uint64_t a_desc = make_a_desc(stage, k);
                        for (uint32_t i = 0; i < kNumSubTiles; i++) WGMMA::wgmma(fragment[i][k], a_desc, accum[i], k != 0);
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
                    // Dequantize the next K block
                    // NOTES: after the last K block too: behind a branch the compiler would not schedule it with
                    // the promotion
                    fence_weights();
                    for (uint32_t k = 0; k < kNumKIters; k++) dequant(k);
                }

                // Release this stage
                if (cute::elect_one_sync()) smem.empty_barriers[stage]->arrive();
                __syncwarp();
                if constexpr (!kBatchMMA) {
                    if (has_next_k_block) dequant(0);
                }
            });

            // Scale by the global scale, round to BF16 and store the tile (or fuse SwiGLU + group-128 FP8 quant).
            if constexpr (kFuseSwiGluFp8) {
                epilogue::store_swiglu_fp8_g128<OUTPUT_N, BLOCK_M, kNumMathThreads, kMathBarrier, kHalvesPerTile>(
                    params, smem, tile, global_scale, final_accum, r_0);
            } else {
                const uint32_t m_idx = tile.row_begin, n_idx = tile.n_block_idx * BLOCK_N;
                const uint32_t num_tokens = min(tile.row_end - tile.row_begin, BLOCK_M);
                constexpr bool kStorePartialFromRegs = kGrouped && !kIndexed && BLOCK_M == 8;
                if (kStorePartialFromRegs && num_tokens < BLOCK_M) {
                    auto *output = static_cast<nv_bfloat16 *>(params.d);
                    const uint32_t token = 2 * (lane_idx % 4);
                    for (uint32_t i = 0; i < kNumSubTiles; i++) {
                        for (uint32_t r = 0; r < 2; r++) {
                            const nv_bfloat162 rounded = __float22bfloat162_rn(make_float2(
                                final_accum[i][2 * r] * global_scale, final_accum[i][2 * r + 1] * global_scale));
                            nv_bfloat16 *column = output + static_cast<uint64_t>(m_idx + token) * OUTPUT_N + n_idx +
                                                  (i / kHalvesPerTile) * 128 + (i % kHalvesPerTile) * 64 + r_0 + 8 * r;
                            if (token < num_tokens) column[0] = rounded.x;
                            if (token + 1 < num_tokens) column[OUTPUT_N] = rounded.y;
                        }
                    }
                } else {
                    float2 token_scales[kOutputRows / 8];
                    for (uint32_t group = 0; group < kOutputRows / 8; group++) {
                        token_scales[group] = make_float2(global_scale, global_scale);
                    }
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
                        for (uint32_t i = 0; i < kNumSubTiles; i++) {
                            auto *box = reinterpret_cast<nv_bfloat16 *>(smem.d[i / kHalvesPerTile]) +
                                        (i % kHalvesPerTile) * kOutputRows * 64;
                            epilogue::write_stsm<kOutputRows>(box, final_accum[i], token_scales, chunk);
                        }
                        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

                        // Store to global memory
                        if constexpr (kIndexed) {
                            epilogue::store_rows<OUTPUT_N, BLOCK_N, kOutputRows, kNumMathThreads>(
                                smem_d, static_cast<nv_bfloat16 *>(params.d), smem.row_ids[tile.route_buffer], 0, n_idx,
                                params.num_routes, chunk);
                        } else if constexpr (kGrouped) {
                            if (num_tokens == BLOCK_M) {
                                cute::tma_store_fence();
                                epilogue::store_tma<BLOCK_N, kOutputRows>(smem_d, &params.tensor_map_d, m_idx, n_idx, false,
                                                                          chunk);
                            } else {
                                epilogue::store_rows<OUTPUT_N, BLOCK_N, kOutputRows, kNumMathThreads>(
                                    smem_d, static_cast<nv_bfloat16 *>(params.d), nullptr, m_idx, n_idx, num_tokens, chunk);
                            }
                        } else {
                            cute::tma_store_fence();
                            epilogue::store_tma<BLOCK_N, kOutputRows>(smem_d, &params.tensor_map_d, m_idx, n_idx,
                                                                      !tile.is_full_depth(), chunk);
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);
                    if (!tile.is_full_depth() && warp_idx == 0 && cute::elect_one_sync()) {
                        ptx::gmem_lock_increment(&params.locks[tile.lock_idx]);
                    }
                    if constexpr (kIndexed) smem.rows_consumed[tile.route_buffer]->arrive();
                    else {}
                }
            }
            if constexpr (!kLookAhead) has_next = scheduler.get_next_block(next_tile);
        }

        // Wait for the last TMA stores before the shared memory goes
        cute::tma_store_wait<0>();
    }

    __syncthreads();
}

}  // namespace tensorbridge
