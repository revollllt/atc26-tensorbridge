#pragma once

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <tensorbridge/common/params.h>
#include <tensorbridge/common/ptx.cuh>
#include <tensorbridge/common/utils.cuh>

// Global -> shared memory, into a stage of the ring (`layout/sm90.cuh`)

namespace tensorbridge::g2s {

constexpr auto kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);

// Weights and weight scales of one K block, one TMA per 128-deep K block
template <uint32_t SHAPE_N, uint32_t SHAPE_K, class Ring>
CUTLASS_DEVICE void load_weights(const GemmParams &params, Ring &smem, uint32_t stage, uint32_t k_block_idx,
                            uint32_t n_block_idx, uint32_t expert_id) {
    auto *barrier = reinterpret_cast<uint64_t *>(&smem.full_barriers[stage]);
    for (uint32_t i = 0; i < Ring::kKBlocks; i++) {
        const uint32_t k_idx = k_block_idx * Ring::kKBlocks + i;
        for (uint32_t t = 0; t < Ring::kTilesN; t++) {
            const uint32_t n_idx = n_block_idx * Ring::kBlockN + t * 128;
            // Weights: K in int32 words, N in rows
            cute::SM90_TMA_LOAD_2D::copy(&params.tensor_map_b, barrier, kCacheHint,
                                         smem.b[stage] + t * Ring::kTileBytesB + i * 128 * 64, k_idx * 16,
                                         expert_id * SHAPE_N + n_idx);
            if constexpr (Ring::kBlockScales) {
                // Scales: N in 8-byte rows of a K block's scale groups, K in blocks
                cute::SM90_TMA_LOAD_2D::copy(&params.tensor_map_sfb, barrier, kCacheHint,
                                             smem.sfb[stage] + t * Ring::kTileBytesSFB + i * 8 * 128, n_idx,
                                             expert_id * (SHAPE_K / 128) + k_idx);
            } else {
                // Scales: N in bytes, K in scale groups
                cute::SM90_TMA_LOAD_2D::copy(&params.tensor_map_sfb, barrier, kCacheHint,
                                             smem.sfb[stage] + t * Ring::kTileBytesSFB + i * 8 * 128, n_idx,
                                             expert_id * (SHAPE_K / 16) + k_idx * 8);
            }
        }
    }
}

// Activations of one K block
template <class Ring>
CUTLASS_DEVICE void load_activations(const GemmParams &params, Ring &smem, uint32_t stage, uint32_t k_block_idx,
                                uint32_t m_idx) {
    auto *barrier = reinterpret_cast<uint64_t *>(&smem.full_barriers[stage]);
    for (uint32_t i = 0; i < Ring::kKBlocks; i++) {
        cute::SM90_TMA_LOAD_2D::copy(&params.tensor_map_a, barrier, kCacheHint, smem.a[stage] + i * Ring::kKBlockBytesA,
                                     (k_block_idx * Ring::kKBlocks + i) * 128, m_idx);
    }
}

// Per-token scales of one tile, by the 32 lanes of the calling warp
template <class Ring>
CUTLASS_DEVICE void load_token_scales(const GemmParams &params, Ring &smem, uint32_t buffer, uint32_t m_idx,
                                 uint32_t num_tokens) {
    const uint32_t lane_idx = threadIdx.x % 32;
    for (uint32_t i = 0; i < math::ceil_div(Ring::kBlockM, 32); i++) {
        const uint32_t token = i * 32 + lane_idx;
        ptx::cp_async_load_pred(params.sfa + m_idx + token, smem.sfa[buffer] + token, token < num_tokens);
    }
}

// Indexed layout: the gather warps copy the activation rows and per-token scales of a tile's routes
template <uint32_t SHAPE_K, uint32_t kNumGatherThreads, uint32_t kGatherBarrier>
class Gather {
    static constexpr uint32_t kRowsPerPass = kNumGatherThreads / 8;
    static_assert(kRowsPerPass % 8 == 4, "The swizzle of a thread's rows alternates");

    const uint32_t thread, first_row, column, swizzled_column;

public:
    CUTLASS_DEVICE explicit Gather(uint32_t thread)
        : thread(thread), first_row(thread / 8), column(thread % 8), swizzled_column(column ^ first_row % 8) {}

    // Route IDs and the input rows they read
    template <class Storage>
    CUTLASS_DEVICE void load_routes(const GemmParams &params, Storage &smem, uint32_t row_begin, uint32_t buffer) {
        for (uint32_t row = thread; row < Storage::kBlockM; row += kNumGatherThreads) {
            const uint32_t route = params.sorted_ids[row_begin + row];
            smem.row_ids[buffer][row] = route;
            smem.input_rows[buffer][row] = route / params.top_k;
        }
        cutlass::arch::NamedBarrier::sync(kNumGatherThreads, kGatherBarrier);
    }

    // Thread `g` copies 16-byte column `g % 8` of rows `g / 8 + 12 j`
    // NOTES: the 128B swizzle XORs the column with the row modulo 8, which advances by 4 from row `r` to `r + 12`
    template <class Storage>
    CUTLASS_DEVICE void load_activations(const GemmParams &params, Storage &smem, uint32_t stage, uint32_t k_block_idx,
                                    uint32_t buffer) {
        constexpr uint32_t kRowsPerThread = math::ceil_div(Storage::kBlockM, kRowsPerPass);
        const uint8_t *k_block_a = params.a + k_block_idx * Storage::kBlockK + column * sizeof(int4);
        uint8_t *stage_a = smem.a[stage] + first_row * 128;
        // NOTES: unrolled twice to keep `j % 2` a constant, fully unrolled the gather warps' registers spill
        #pragma unroll 2
        for (uint32_t j = 0; j < kRowsPerThread; j++) {
            const uint32_t row = first_row + j * kRowsPerPass;
            if (row < Storage::kBlockM) {
                const uint32_t input_row = smem.input_rows[buffer][row];
                const uint32_t row_column = swizzled_column ^ (j % 2) * 4;
                for (uint32_t k_part = 0; k_part < Storage::kKBlocks; k_part++) {
                    ptx::cp_async_load_pred(
                        reinterpret_cast<const int4 *>(k_block_a + static_cast<uint64_t>(input_row) * SHAPE_K + k_part * 128),
                        reinterpret_cast<int4 *>(stage_a + k_part * Storage::kKBlockBytesA + j * kRowsPerPass * 128) +
                            row_column,
                        input_row < params.shape_m);
                }
            }
        }
    }

    template <class Storage>
    CUTLASS_DEVICE void load_token_scales(const GemmParams &params, Storage &smem, uint32_t buffer) {
        for (uint32_t row = thread; row < Storage::kBlockM; row += kNumGatherThreads) {
            const uint32_t input_row = smem.input_rows[buffer][row];
            ptx::cp_async_load_pred(params.sfa + input_row, smem.sfa[buffer] + row, input_row < params.shape_m);
        }
    }
};

}  // namespace tensorbridge::g2s
