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

// NVFP4 weights x FP8 activations with per-token scales, dense or grouped by expert.
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t OUTPUT_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t BLOCK_K, uint32_t kNumStages, uint32_t kNumCtasPerSm, uint32_t kNumMathWarpgroups,
          bool kUseStreamK, bool kAlignedSlices, bool kMFastTileOrder, GroupedLayout kLayout, uint32_t kNumExperts,
          bool kExactBridge>
__global__ __launch_bounds__(128 * (kNumMathWarpgroups + 1), kNumCtasPerSm) void sm90_w4a8_ptoken(
    const __grid_constant__ GemmParams params) {
    // Shape checks
    static_assert(BLOCK_M % 8 == 0 && BLOCK_M <= 256, "Invalid token tile");
    static_assert(SHAPE_N % BLOCK_N == 0 && SHAPE_K % BLOCK_K == 0, "N and K must be multiples of the tile");
    static_assert(OUTPUT_N % 8 == 0 && OUTPUT_N <= SHAPE_N && SHAPE_N - OUTPUT_N < 128, "D has the weight's N columns");
    static_assert(kLayout == GroupedLayout::kDense || OUTPUT_N == SHAPE_N, "Grouped weights are multiples of 128");
    static_assert(BLOCK_K == 128 || BLOCK_K == 256, "A stage holds one or two K blocks");
    static_assert(BLOCK_N == 128 || BLOCK_N == 256, "A CTA computes one or two channel tiles");
    static_assert(kNumMathWarpgroups == 1 || kNumMathWarpgroups == 2, "One or two math warp groups");
    static_assert(kNumCtasPerSm >= 1 && kNumCtasPerSm <= (kNumMathWarpgroups == 2 ? 2 : 4), "Invalid CTAs per SM");
    static_assert(kNumMathWarpgroups == 2 || kNumCtasPerSm >= 2, "One math warp group exceeds 256 registers per thread");
    static_assert(kNumStages >= 3 && kNumStages <= 30, "Too few stages to overlap loads, or too many barriers");
    static_assert(!kUseStreamK || kLayout == GroupedLayout::kDense, "Grouped GEMM does not split K");

    // Types
    using WGMMA = typename mma::sm90::FP8RSMMASelector<BLOCK_M>::type;
    using Storage = layout::PtokenStorage<BLOCK_M, BLOCK_N, BLOCK_K, kNumStages, kLayout, kNumExperts>;

    // Configs
    constexpr bool kGrouped = kLayout != GroupedLayout::kDense;
    constexpr bool kIndexed = kLayout == GroupedLayout::kIndexed;
    constexpr uint32_t kNumMathThreads = 128 * kNumMathWarpgroups;
    constexpr uint32_t kNumThreads = kNumMathThreads + 128;
    constexpr uint32_t kNumGatherThreads = 96;
    constexpr uint32_t kNumKIters = BLOCK_K / WGMMA::K;
    constexpr uint32_t kNumBuffers = 2;
    constexpr uint32_t kHalvesPerTile = 2 / kNumMathWarpgroups;
    constexpr uint32_t kNumSubTiles = Storage::kTilesN * kHalvesPerTile;
    constexpr uint32_t kTmaBytesPerStage =
        (kIndexed ? 0 : Storage::SMEM_A_SIZE_PER_STAGE) + Storage::SMEM_B_SIZE_PER_STAGE + Storage::SMEM_SFB_SIZE_PER_STAGE;
    constexpr uint32_t kMathBarrier = 1, kGatherBarrier = 2;
    constexpr bool kOneKBlockTiles = !kUseStreamK && SHAPE_K == BLOCK_K;
    constexpr bool kLookAhead = kGrouped && !kIndexed;
    // NOTES: fewer uniform instructions per QGMMA than rebuilding the descriptor, on these kernels only
    constexpr bool kUseRingDescs = kGrouped && !kIndexed && (kNumCtasPerSm >= 2 || BLOCK_M < 64);

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = kIndexed ? 64 : 40;
    constexpr uint32_t kLaunchRegs = 65536 / (kNumThreads * kNumCtasPerSm) / 8 * 8;
    constexpr uint32_t kNumMathRegisters = (kLaunchRegs * kNumThreads - kNumTMARegisters * 128) / kNumMathThreads / 8 * 8;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Data on shared memory
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto &smem = *reinterpret_cast<Storage *>(smem_buffer);

    // Block scheduler
    auto scheduler = sched::make_scheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder, kUseStreamK,
                                           kAlignedSlices, kLayout, kNumExperts>(params, smem);
    using Tile = typename decltype(scheduler)::Tile;
    Tile tile;

    // Load the routing, overlapped with the barrier initialization
    if constexpr (kGrouped) {
        if (warp_idx == kNumMathThreads / 32 + 1) {
            cudaGridDependencySynchronize();
            scheduler.load_routing(params.expert_offsets, params.expert_counts, params.num_tokens_padded);
        }
    }

    // Issue the TMA loads of one K block
    auto issue_tma = [&](uint32_t stage, uint32_t k_block_idx, const Tile &t) {
        if constexpr (!kIndexed) g2s::load_activations(params, smem, stage, k_block_idx, t.row_begin);
        g2s::load_weights<SHAPE_N, SHAPE_K>(params, smem, stage, k_block_idx, t.n_block_idx, t.expert_id);
        smem.full_barriers[stage]->arrive_and_expect_tx(kTmaBytesPerStage);
    };

    if (warp_idx == kNumMathThreads / 32 && cute::elect_one_sync()) {
        // Prefetch TMA descriptors at the very beginning
        if constexpr (!kIndexed) {
            cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_a));
        }
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_sfb));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor *>(&params.tensor_map_d));

        // Initialize barriers
        for (uint32_t s = 0; s < kNumStages; s++) {
            smem.full_barriers[s]->init(kIndexed ? 1 + kNumGatherThreads : 1);
            smem.empty_barriers[s]->init(kNumMathThreads / 32);
        }
        for (uint32_t i = 0; i < 2; i++) {
            smem.sfa_full_barriers[i]->init(kIndexed ? kNumGatherThreads : 32);
            smem.sfa_empty_barriers[i]->init(kNumMathThreads);
        }
        if constexpr (kGrouped) {
            smem.routing_ready->init(kIndexed ? 1 : 32);
        }
        if constexpr (kIndexed) {
            smem.rows_consumed[0]->init(kNumMathThreads);
            smem.rows_consumed[1]->init(kNumMathThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
        cutlass::arch::fence_view_async_shared();

        // Issue the first K block of a dense CTA
        if constexpr (!kGrouped) {
            auto first_scheduler = scheduler;
            Tile first_tile;
            if (first_scheduler.get_next_block(first_tile)) {
                cudaGridDependencySynchronize();
                issue_tma(0, first_tile.k_block_idx, first_tile);
            }
        }
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

        // Derive the routing, then issue the first K block of a contiguous or masked CTA
        if constexpr (kGrouped) {
            if (tma_warp_idx == 1) {
                scheduler.prepare();
                if constexpr (!kIndexed) {
                    scheduler.wait_ready();
                    auto first_scheduler = scheduler;
                    Tile first_tile;
                    const bool has_first_tile = first_scheduler.get_next_block(first_tile);
                    if (has_first_tile && cute::elect_one_sync()) issue_tma(0, 0, first_tile);
                }
            }
        }

        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (tma_warp_idx == 0) {
            // TMA warp
            scheduler.wait_ready();
            uint32_t tile_count = 0;
            while (scheduler.get_next_block(tile)) {
                const uint32_t num_tokens = min(tile.row_end - tile.row_begin, BLOCK_M);
                const uint32_t buffer = tile_count % 2, buffer_phase = (tile_count / 2) & 1;

                // NOTES: unrolled twice, fully unrolled the producer's registers spill
                #pragma unroll 2
                for (uint32_t k_block = 0; k_block < tile.num_k_blocks; k_block++) {
                    // Wait consumer release
                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    smem.empty_barriers[next_stage]->wait(phase ^ (next_stage == 0) ^ 1);

                    // Issue TMA
                    const bool issued_early = !kIndexed && tile_count == 0 && k_block == 0;
                    if (!issued_early && cute::elect_one_sync()) issue_tma(stage_idx, tile.k_block_idx + k_block, tile);
                    __syncwarp();

                    // Load per-token scales
                    if constexpr (!kIndexed) {
                        if (k_block == 0) {
                            smem.sfa_empty_barriers[buffer]->wait(buffer_phase ^ 1);
                            g2s::load_token_scales(params, smem, buffer, tile.row_begin, num_tokens);
                            ptx::cp_async_arrive(smem.sfa_full_barriers[buffer]);
                        }
                    }
                    advance_pipeline();
                }
                tile_count++;
            }

            // Let the next kernel launch
            cudaTriggerProgrammaticLaunchCompletion();
        } else if constexpr (kIndexed) {
            // Gather warps
            scheduler.wait_ready();
            g2s::Gather<SHAPE_K, kNumGatherThreads, kGatherBarrier> gather(threadIdx.x - kNumMathThreads - 32);
            uint32_t tile_count = 0;
            while (scheduler.get_next_block(tile)) {
                const uint32_t buffer = tile_count % 2, buffer_phase = (tile_count / 2) & 1;

                // Load the tile's routes
                smem.rows_consumed[buffer]->wait(buffer_phase ^ 1);
                gather.load_routes(params, smem, tile.row_begin, buffer);

                #pragma unroll 2
                for (uint32_t k_block = 0; k_block < tile.num_k_blocks; k_block++) {
                    // Wait consumer release
                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    smem.empty_barriers[next_stage]->wait(phase ^ (next_stage == 0) ^ 1);

                    // Gather activation rows
                    gather.load_activations(params, smem, stage_idx, tile.k_block_idx + k_block, buffer);
                    ptx::cp_async_arrive(smem.full_barriers[stage_idx]);

                    // Load per-token scales
                    if (k_block == 0) {
                        smem.sfa_empty_barriers[buffer]->wait(buffer_phase ^ 1);
                        gather.load_token_scales(params, smem, buffer);
                        ptx::cp_async_arrive(smem.sfa_full_barriers[buffer]);
                    }
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
        uint32_t packed[kNumSubTiles][kNumBuffers][2];
        uint32_t scales[kNumSubTiles][kNumBuffers][4];
        uint32_t fragment[kNumSubTiles][kNumBuffers][4];
        float accum[kNumSubTiles][WGMMA::kNumAccum];

        // Read one k32 iteration's weights and scales from shared memory
        auto load_weights = [&](uint32_t stage, uint32_t k) {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                const uint32_t half = i % kHalvesPerTile, tile_n = i / kHalvesPerTile;
                fpma::load_weights(smem.b[stage] + tile_n * Storage::kTileBytesB, k, threadIdx.x + half * 128,
                                   packed[i][k % kNumBuffers]);
                fpma::load_scales(smem.sfb[stage] + tile_n * Storage::kTileBytesSFB, k, r_0 + half * 64,
                                  scales[i][k % kNumBuffers]);
            }
        };

        // Run one k32 iteration's weights through the FPMA bridge
        auto dequant = [&](uint32_t k) {
            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                fpma::dequant<kExactBridge>(packed[i][k % kNumBuffers], scales[i][k % kNumBuffers],
                                            fragment[i][k % kNumBuffers]);
            }
        };

        // WGMMA descriptor of one k32 iteration's activations
            auto make_a_desc = [&](uint32_t stage, uint32_t k) -> uint64_t {
            const uint32_t offset = (k / 4) * Storage::kKBlockBytesA + (k % 4) * WGMMA::K;
            if constexpr (kUseRingDescs) {
                const cute::GmmaDescriptor ring_desc = mma::sm90::make_smem_desc(smem.a[0]);
            const uint32_t start = ring_desc.reg32_[0] + (stage * Storage::SMEM_A_SIZE_PER_STAGE + offset) / 16;
                return static_cast<uint64_t>(ring_desc.reg32_[1]) << 32 | start;
            } else {
                return mma::sm90::make_smem_desc(smem.a[stage] + offset).desc_;
            }
        };

        // Persistently schedule over blocks
        scheduler.wait_ready();
        Tile next_tile;
        bool has_next = scheduler.get_next_block(next_tile);
        uint32_t tile_count = 0;
        while (has_next) {
            tile = next_tile;
            if constexpr (kLookAhead) has_next = scheduler.get_next_block(next_tile);

            for (uint32_t i = 0; i < kNumSubTiles; i++) {
                for (uint32_t j = 0; j < WGMMA::kNumAccum; j++) accum[i][j] = 0;
            }

            // NOTES: read before the mainloop, which hides the load's latency from the epilogue
            const float global_scale = params.global_scale[tile.expert_id];

            // Wait TMA arrivals
            smem.full_barriers[stage_idx]->wait(phase);

            // Prepare the first k32 iteration
            load_weights(stage_idx, 0);
            dequant(0);

            stage_idx = for_each_ring_slot<kNumStages>(stage_idx, tile.num_k_blocks, [&](uint32_t stage,
                                                                                        uint32_t k_blocks_left) {
                const uint32_t next_stage = (stage + 1) % kNumStages;
                #pragma unroll
                for (uint32_t k = 0; k < kNumKIters; k++) {
                    // Read the next k32 iteration's weights
                    // NOTES: past the tile's last K block they are read but never used, which saves a branch per K block
                    const bool has_next_k = !kOneKBlockTiles || k + 1 < kNumKIters;
                    if (has_next_k) load_weights(k + 1 == kNumKIters ? next_stage : stage, (k + 1) % kNumKIters);

                    // Commit WGMMA instructions
                    const uint64_t a_desc = make_a_desc(stage, k);
                    cute::warpgroup_arrive();
                    for (uint32_t i = 0; i < kNumSubTiles; i++) {
                        WGMMA::wgmma(fragment[i][k % kNumBuffers], a_desc, accum[i], true);
                    }
                    cute::warpgroup_commit_batch();

                    // Release this stage and wait for the next one
                    if (k == kNumKIters - 2) {
                        if (cute::elect_one_sync()) smem.empty_barriers[stage]->arrive();
                        __syncwarp();
                        if (k_blocks_left > 1) smem.full_barriers[next_stage]->wait(phase ^ (next_stage == 0));
                    }

                    // Dequantize the next k32 iteration while the WGMMA runs
                    if (has_next_k) dequant(k + 1);
                    cute::warpgroup_wait<0>();
                }
                phase ^= next_stage == 0;
            });

            // Scale, round to BF16 and store the tile
            epilogue::sm90_store_d<SHAPE_N, kLayout, kNumMathThreads, kMathBarrier, kHalvesPerTile>(
                params, smem, scheduler, tile, tile_count, global_scale, accum, r_0);
            if constexpr (!kLookAhead) has_next = scheduler.get_next_block(next_tile);
            tile_count++;
        }

        // Wait for the last TMA stores before the shared memory goes
        cute::tma_store_wait<0>();
    }

    __syncthreads();
}

}  // namespace tensorbridge
