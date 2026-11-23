#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>

#include <tensorbridge/common/params.h>
#include <tensorbridge/common/ptx.cuh>
#include <tensorbridge/common/utils.cuh>

// FP32 accumulators -> BF16 output tile in shared memory -> global memory

namespace tensorbridge::epilogue {

// Scale, round to BF16 and write one chunk of a 128-channel tile with `stmatrix.x2.trans`
template <uint32_t kRows>
CUTLASS_DEVICE void write_stsm(nv_bfloat16 *tile, const float *accum, const float2 *scales, uint32_t chunk) {
    static_assert(kRows % 8 == 0, "Whole 8-token groups");
    constexpr uint32_t kNumTokenGroups = kRows / 8;
    const uint32_t warp_idx = threadIdx.x / 32, lane_idx = threadIdx.x % 32;
    const auto *accum_pairs = reinterpret_cast<const float2 *>(accum);
    const uint32_t token = lane_idx % 8;
    const uint32_t channel_group = 2 * (warp_idx % 4) + (lane_idx / 8) % 2;
    const uint32_t swizzled_group = channel_group ^ ((cute::cast_smem_ptr_to_uint(tile) / 128 + token) % 8);
    nv_bfloat16 *row = tile + (warp_idx / 4) * (kRows * 64) + token * 64 + swizzled_group * 8;

    for (uint32_t group = 0; group < kNumTokenGroups; group++) {
        const uint32_t tile_group = chunk * kNumTokenGroups + group;
        uint32_t bf16_pairs[2];
        for (uint32_t r = 0; r < 2; r++) {
            float2 pair = accum_pairs[tile_group * 2 + r];
            pair.x *= scales[group].x;
            pair.y *= scales[group].y;
            const nv_bfloat162 rounded = __float22bfloat162_rn(pair);
            bf16_pairs[r] = *reinterpret_cast<const uint32_t *>(&rounded);
        }
        cute::SM90_U16x4_STSM_T::copy(bf16_pairs[0], bf16_pairs[1],
                                      *reinterpret_cast<cute::uint128_t *>(row + group * 8 * 64));
    }
}

// Store rows of the output tile with 16-byte copies, to `output_rows[row]` or from `m_idx`
template <uint32_t SHAPE_N, uint32_t BLOCK_N, uint32_t kRows, uint32_t kNumMathThreads>
CUTLASS_DEVICE void store_rows(const int4 *tile, nv_bfloat16 *output, const int32_t *output_rows, uint32_t m_idx,
                          uint32_t n_idx, uint32_t num_rows, uint32_t chunk) {
    constexpr uint32_t kNumBoxes = BLOCK_N / 64;
    constexpr uint32_t kVectorsPerRow = 64 * sizeof(nv_bfloat16) / sizeof(int4);
    constexpr uint32_t kBoxVectors = kRows * kVectorsPerRow;
    static_assert(kRows % 8 == 0 && kNumMathThreads % kVectorsPerRow == 0, "A thread's column and swizzle phase are fixed");
    const uint32_t column = threadIdx.x % kVectorsPerRow;
    const uint32_t swizzle_base = cute::cast_smem_ptr_to_uint(tile) / 128;
    for (uint32_t row = threadIdx.x / kVectorsPerRow; row < kRows; row += kNumMathThreads / kVectorsPerRow) {
        const uint32_t tile_row = chunk * kRows + row;
        uint32_t output_row;
        if (output_rows != nullptr) {
            output_row = output_rows[tile_row];
            if (output_row >= num_rows) continue;
        } else {
            if (tile_row >= num_rows) continue;
            output_row = m_idx + tile_row;
        }
        const int4 *source = tile + row * kVectorsPerRow + (column ^ (swizzle_base + row) % 8);
        auto *destination = reinterpret_cast<int4 *>(output + static_cast<uint64_t>(output_row) * SHAPE_N + n_idx) + column;
        for (uint32_t box = 0; box < kNumBoxes; box++) destination[box * kVectorsPerRow] = source[box * kBoxVectors];
    }
}

// One TMA store per 64-channel box, a reduce-add for a Stream-K slice
template <uint32_t BLOCK_N, uint32_t kRows>
CUTLASS_DEVICE void store_tma(const int4 *tile, const void *tensor_map_d, uint32_t m_idx, uint32_t n_idx, bool reduce,
                         uint32_t chunk) {
    constexpr uint32_t kNumBoxes = BLOCK_N / 64;
    const uint32_t box = threadIdx.x;
    if (box >= kNumBoxes) return;
    const int4 *source = tile + box * (kRows * 64 * sizeof(nv_bfloat16) / sizeof(int4));
    const int32_t box_n_idx = n_idx + box * 64;
    const int32_t box_m_idx = m_idx + chunk * kRows;
    if (reduce) {
        cute::SM90_TMA_REDUCE_ADD_2D::copy(tensor_map_d, source, box_n_idx, box_m_idx);
    } else {
        cute::SM90_TMA_STORE_2D::copy(tensor_map_d, source, box_n_idx, box_m_idx);
    }
    cute::tma_store_arrive();
}

// Stream-K: zero the tiles whose last K slice is this CTA's, then publish their negated slice counts
template <uint32_t OUTPUT_N, uint32_t BLOCK_M, uint32_t BLOCK_N, class Scheduler>
CUTLASS_DEVICE void zero_stream_k_tiles(Scheduler &scheduler, nv_bfloat16 *output, int32_t *locks) {
    constexpr uint32_t kVectorsPerRow = BLOCK_N * sizeof(nv_bfloat16) / sizeof(int4);
    static_assert(32 % kVectorsPerRow == 0, "A lane zeroes one column of the tile");
    const uint32_t lane_idx = threadIdx.x % 32;
    typename Scheduler::Tile tile;
    while (scheduler.get_next_block(tile)) {
        if (tile.is_full_depth() || tile.slice_id != 0) continue;
        const uint32_t num_rows = min(tile.row_end - tile.row_begin, BLOCK_M);
        const bool lane_in_output =
            OUTPUT_N % BLOCK_N == 0 || tile.n_block_idx * BLOCK_N + lane_idx % kVectorsPerRow * 8 < OUTPUT_N;
        for (uint32_t vector = lane_idx; lane_in_output && vector < num_rows * kVectorsPerRow; vector += 32) {
            nv_bfloat16 *row = output + static_cast<uint64_t>(tile.row_begin + vector / kVectorsPerRow) * OUTPUT_N +
                               tile.n_block_idx * BLOCK_N;
            reinterpret_cast<int4 *>(row)[vector % kVectorsPerRow] = make_int4(0, 0, 0, 0);
        }
        __syncwarp();
        if (cute::elect_one_sync()) {
            ptx::gmem_lock_publish(&locks[tile.lock_idx], -static_cast<int32_t>(tile.slice_count));
        }
    }
}

// Scale, round to BF16 and store one tile of the per-token kernel
template <uint32_t SHAPE_N, GroupedLayout kLayout, uint32_t kNumMathThreads, uint32_t kMathBarrier,
          uint32_t kHalvesPerTile, uint32_t kNumSubTiles, uint32_t kNumAccum, class Storage,
          class Scheduler>
CUTLASS_DEVICE void sm90_store_d(const GemmParams &params, Storage &smem, const Scheduler &scheduler,
                          const typename Scheduler::Tile &tile, uint32_t tile_count, float global_scale,
                          float (&accum)[kNumSubTiles][kNumAccum], uint32_t r_0) {
    constexpr bool kGrouped = kLayout != GroupedLayout::kDense;
    constexpr bool kIndexed = kLayout == GroupedLayout::kIndexed;
    constexpr uint32_t BLOCK_M = Storage::kBlockM, BLOCK_N = Storage::kBlockN, kOutputRows = Storage::kOutputRows;
    constexpr uint32_t kNumChunks = BLOCK_M / kOutputRows;
    const uint32_t lane_idx = threadIdx.x % 32;

    // Read the per-token scales of this thread's token pair in each 8-token group
    const uint32_t sfa_buffer = tile_count % 2;
    smem.sfa_full_barriers[sfa_buffer]->wait((tile_count / 2) & 1);
    const auto *token_scale_pairs = reinterpret_cast<const float2 *>(smem.sfa[sfa_buffer]) + lane_idx % 4;
    float2 token_scales[kOutputRows / 8];
    for (uint32_t group = 0; group < kOutputRows / 8; group++) token_scales[group] = token_scale_pairs[group * 4];
    if constexpr (kNumChunks == 1) smem.sfa_empty_barriers[sfa_buffer]->arrive();

    // Wait the WGMMAs
    cute::warpgroup_wait<0>();
    for (uint32_t i = 0; i < kNumSubTiles; i++) {
        for (uint32_t j = 0; j < kNumAccum; j++) cute::warpgroup_fence_operand(accum[i][j]);
    }

    const uint32_t m_idx = tile.row_begin, n_idx = tile.n_block_idx * BLOCK_N;
    const uint32_t num_tokens = min(tile.row_end - tile.row_begin, BLOCK_M);
    auto *output = static_cast<nv_bfloat16 *>(params.d);

    // A partial tile of one 8-token group is stored from registers
    constexpr bool kStorePartialFromRegs = kGrouped && !kIndexed && BLOCK_M == 8;
    if (kStorePartialFromRegs && num_tokens < BLOCK_M) {
        const float2 scale = make_float2(token_scales[0].x * global_scale, token_scales[0].y * global_scale);
        const uint32_t token = 2 * (lane_idx % 4);
        for (uint32_t i = 0; i < kNumSubTiles; i++) {
            for (uint32_t r = 0; r < 2; r++) {
                const nv_bfloat162 rounded =
                    __float22bfloat162_rn(make_float2(accum[i][2 * r] * scale.x, accum[i][2 * r + 1] * scale.y));
                nv_bfloat16 *column = output + static_cast<uint64_t>(m_idx + token) * SHAPE_N + n_idx +
                                      (i / kHalvesPerTile) * 128 + (i % kHalvesPerTile) * 64 + r_0 + 8 * r;
                if (token < num_tokens) column[0] = rounded.x;
                if (token + 1 < num_tokens) column[SHAPE_N] = rounded.y;
            }
        }
        return;
    }

    // A dense CTA's last tile writes its second chunk into the consumed ring
    bool last_tile = false;
    if constexpr (!kGrouped && kNumChunks == 2) {
        static_assert(sizeof(smem.a) >= sizeof(smem.d), "The ring's activations hold an output chunk");
        auto lookahead = scheduler;
        typename Scheduler::Tile next_tile;
        last_tile = !lookahead.get_next_block(next_tile);
    }

    #pragma unroll
    for (uint32_t chunk = 0; chunk < kNumChunks; chunk++) {
        const bool chunk_in_ring = last_tile && chunk == 1;
        uint8_t *chunk_tile = chunk_in_ring ? &smem.a[0][0] : &smem.d[0][0];
        const auto *smem_d = reinterpret_cast<const int4 *>(chunk_tile);

        // Fold the global scale into the chunk's scales
        for (uint32_t group = 0; group < kOutputRows / 8; group++) {
            const float2 pair = chunk == 0 ? token_scales[group] : token_scale_pairs[(chunk * (kOutputRows / 8) + group) * 4];
            token_scales[group] = make_float2(pair.x * global_scale, pair.y * global_scale);
        }
        if (chunk > 0 && chunk == kNumChunks - 1) smem.sfa_empty_barriers[sfa_buffer]->arrive();

        // Wait last TMA store to be finished
        if (!chunk_in_ring) cute::tma_store_wait<0>();
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        // Stream-K: wait for the tile to be zeroed
        if (chunk == 0 && !tile.is_full_depth()) {
            ptx::gmem_lock_acquire<kNumMathThreads, kMathBarrier>(&params.locks[tile.lock_idx], -1);
        }

        // Write to shared memory
        for (uint32_t i = 0; i < kNumSubTiles; i++) {
            auto *box = reinterpret_cast<nv_bfloat16 *>(chunk_tile + (i / kHalvesPerTile) * sizeof(smem.d[0])) +
                        (i % kHalvesPerTile) * kOutputRows * 64;
            write_stsm<kOutputRows>(box, accum[i], token_scales, chunk);
        }
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        // Store to global memory
        if constexpr (kIndexed) {
            store_rows<SHAPE_N, BLOCK_N, kOutputRows, kNumMathThreads>(smem_d, output, smem.row_ids[tile.route_buffer], 0,
                                                                       n_idx, params.num_routes, chunk);
        } else if constexpr (kGrouped) {
            if (num_tokens == BLOCK_M) {
                cute::tma_store_fence();
                store_tma<BLOCK_N, kOutputRows>(smem_d, &params.tensor_map_d, m_idx, n_idx, false, chunk);
            } else {
                store_rows<SHAPE_N, BLOCK_N, kOutputRows, kNumMathThreads>(smem_d, output, nullptr, m_idx, n_idx, num_tokens,
                                                                           chunk);
            }
        } else {
            cute::tma_store_fence();
            store_tma<BLOCK_N, kOutputRows>(smem_d, &params.tensor_map_d, m_idx, n_idx, !tile.is_full_depth(), chunk);
        }
    }
    cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

    // Stream-K: count this slice in, and release the row buffer
    if (!tile.is_full_depth() && threadIdx.x / 32 == 0 && cute::elect_one_sync()) {
        ptx::gmem_lock_increment(&params.locks[tile.lock_idx]);
    }
    if constexpr (kIndexed) smem.rows_consumed[tile.route_buffer]->arrive();
}

// Shared implementation of fused SwiGLU + group-128 dynamic FP8 E4M3 epilogue (`BLOCK_N == 256`).
// Packs 4 adjacent FP8 bytes across lanes `q, q+1, q+2, q+3` via `__shfl_xor_sync` + `__byte_perm` into
// bank-conflict-free 32-bit `STS.B32` stores into `fp8_smem` (144B row stride), then issues coalesced `STG.128`.
template <uint32_t OUTPUT_N, uint32_t BLOCK_M, uint32_t kNumMathThreads, uint32_t kMathBarrier,
          uint32_t kHalvesPerTile, uint32_t kNumSubTiles, uint32_t kNumAccum, class Storage>
CUTLASS_DEVICE void store_swiglu_fp8_g128_impl(void *dst_fp8_ptr, float *dst_scale_base, Storage &smem,
                                               uint32_t m_idx, uint32_t num_tokens, uint32_t group_idx,
                                               float global_scale, float (&final_accum)[kNumSubTiles][kNumAccum],
                                               uint32_t r_0) {
    static_assert(Storage::kBlockN == 256 && kNumSubTiles == 2 * kHalvesPerTile,
                  "Fused SwiGLU+FP8 requires BLOCK_N == 256 with interleaved (gate_128, up_128) tiles");
    static_assert(OUTPUT_N % 256 == 0, "Fused SwiGLU+FP8 requires OUTPUT_N % 256 == 0");

    constexpr uint32_t kOutputRows = Storage::kOutputRows;
    constexpr uint32_t kNumChunks = BLOCK_M / kOutputRows;
    constexpr uint32_t kChunkGroups = kOutputRows / 8;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    constexpr uint32_t kInterN = OUTPUT_N / 2;
    constexpr uint32_t kNumGroups = OUTPUT_N / 256;
    // 144B row stride = 9 * 16B: 16B aligned for `int4` STG.128 and coprime to 128B (32 banks) -> zero bank conflicts.
    constexpr uint32_t kSmemRowStride = 144;
    static_assert(kOutputRows * kSmemRowStride + kNumMathWarps * kOutputRows * sizeof(float) <= sizeof(smem.d),
                  "smem.d must hold FP8 staging tile + warp absmax scratch");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    auto *fp8_smem = reinterpret_cast<uint8_t *>(smem.d);
    auto *warp_amax = reinterpret_cast<float *>(fp8_smem + kOutputRows * kSmemRowStride);
    auto *dst_fp8_base = static_cast<uint8_t *>(dst_fp8_ptr);

    for (uint32_t chunk = 0; chunk < kNumChunks; chunk++) {
        if (chunk > 0) cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        // Step 1: In-register FP32 SwiGLU (`silu(gate) * up`, overwritten into `final_accum[0..kHalvesPerTile-1]`)
        // + intra-warp absmax across the 8 lanes sharing each token pair (`lane_idx % 4`).
        for (uint32_t tg = 0; tg < kChunkGroups; tg++) {
            const uint32_t group = chunk * kChunkGroups + tg;
            float amax0 = 0.0f, amax1 = 0.0f;
            for (uint32_t h = 0; h < kHalvesPerTile; h++) {
                float *g_ptr = final_accum[h] + group * 4;
                const float *u_ptr = final_accum[h + kHalvesPerTile] + group * 4;
                #pragma unroll
                for (uint32_t k = 0; k < 4; k++) {
                    const float g = g_ptr[k] * global_scale;
                    const float u = u_ptr[k] * global_scale;
                    const float v = __fdividef(g, 1.0f + __expf(-g)) * u;
                    g_ptr[k] = v;
                    if (k & 1) amax1 = fmaxf(amax1, fabsf(v));
                    else amax0 = fmaxf(amax0, fabsf(v));
                }
            }
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffff, amax0, 4));
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffff, amax0, 8));
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffff, amax0, 16));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffff, amax1, 4));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffff, amax1, 8));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffff, amax1, 16));
            if (lane_idx < 4) {
                const uint32_t local_r0 = tg * 8 + lane_idx * 2;
                warp_amax[warp_idx * kOutputRows + local_r0] = amax0;
                warp_amax[warp_idx * kOutputRows + local_r0 + 1] = amax1;
            }
        }

        // Wait for all math warps to publish their partial absmax values.
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        // Step 2: Cross-warp max over 128 channels -> scale + FP8 E4M3 quant + bank-conflict-free `STS.B32`.
        const uint32_t pair_offset = (lane_idx % 4) * 2;
        const bool is_quad_leader = (lane_idx & 12) == 0;
        for (uint32_t tg = 0; tg < kChunkGroups; tg++) {
            const uint32_t group = chunk * kChunkGroups + tg;
            const uint32_t local_r0 = tg * 8 + pair_offset;
            const uint32_t local_r1 = local_r0 + 1;
            float amax0 = warp_amax[local_r0];
            float amax1 = warp_amax[local_r1];
            #pragma unroll
            for (uint32_t w = 1; w < kNumMathWarps; w++) {
                amax0 = fmaxf(amax0, warp_amax[w * kOutputRows + local_r0]);
                amax1 = fmaxf(amax1, warp_amax[w * kOutputRows + local_r1]);
            }
            const float scale0 = fmaxf(amax0, 1e-30f) * (1.0f / 448.0f);
            const float scale1 = fmaxf(amax1, 1e-30f) * (1.0f / 448.0f);
            const float inv0 = 1.0f / scale0;
            const float inv1 = 1.0f / scale1;

            if (warp_idx == 0 && lane_idx < 4) {
                const uint32_t tile_r0 = chunk * kOutputRows + local_r0;
                if (tile_r0 < num_tokens) {
                    dst_scale_base[static_cast<uint64_t>(m_idx + tile_r0) * kNumGroups + group_idx] = scale0;
                }
                if (tile_r0 + 1 < num_tokens) {
                    dst_scale_base[static_cast<uint64_t>(m_idx + tile_r0 + 1) * kNumGroups + group_idx] = scale1;
                }
            }

            uint8_t *row0_smem = fp8_smem + local_r0 * kSmemRowStride;
            uint8_t *row1_smem = fp8_smem + local_r1 * kSmemRowStride;
            for (uint32_t h = 0; h < kHalvesPerTile; h++) {
                const float *v_ptr = final_accum[h] + group * 4;
                #pragma unroll
                for (uint32_t r = 0; r < 2; r++) {
                    const uint32_t pk0 =
                        ptx::cvt_f32x2_to_fp8_e4m3x2(v_ptr[2 * r] * inv0, v_ptr[2 * r + 1] * inv1);
                    const uint32_t pk1 = __shfl_xor_sync(0xffffffff, pk0, 4);
                    const uint32_t pk2 = __shfl_xor_sync(0xffffffff, pk0, 8);
                    const uint32_t pk3 = __shfl_xor_sync(0xffffffff, pk0, 12);
                    if (is_quad_leader) {
                        const uint32_t w0 = __byte_perm(__byte_perm(pk0, pk1, 0x0040),
                                                        __byte_perm(pk2, pk3, 0x0040), 0x5410);
                        const uint32_t w1 = __byte_perm(__byte_perm(pk0, pk1, 0x0051),
                                                        __byte_perm(pk2, pk3, 0x0051), 0x5410);
                        const uint32_t col = h * 64 + r_0 + 8 * r;
                        *reinterpret_cast<uint32_t *>(row0_smem + col) = w0;
                        *reinterpret_cast<uint32_t *>(row1_smem + col) = w1;
                    }
                }
            }
        }

        cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);

        // Step 3: Coalesced 16-byte (`STG.128`) global stores of the `[kOutputRows, 128]` FP8 tile.
        constexpr uint32_t kVectorsPerRow = 128 / sizeof(int4);  // 8 vectors per 128B FP8 row
        const uint32_t vec_col = threadIdx.x % kVectorsPerRow;
        for (uint32_t r = threadIdx.x / kVectorsPerRow; r < kOutputRows; r += kNumMathThreads / kVectorsPerRow) {
            const uint32_t tile_row = chunk * kOutputRows + r;
            if (tile_row >= num_tokens) continue;
            auto *dst_row = reinterpret_cast<int4 *>(
                dst_fp8_base + static_cast<uint64_t>(m_idx + tile_row) * kInterN + group_idx * 128);
            const auto *src_row = reinterpret_cast<const int4 *>(fp8_smem + r * kSmemRowStride);
            dst_row[vec_col] = src_row[vec_col];
        }
    }
    cutlass::arch::NamedBarrier::sync(kNumMathThreads, kMathBarrier);
}

// Fused SwiGLU + group-128 dynamic FP8 E4M3 epilogue wrapper for `sm90_w4a8_pgroup` (`BLOCK_N == 256`).
template <uint32_t OUTPUT_N, uint32_t BLOCK_M, uint32_t kNumMathThreads, uint32_t kMathBarrier,
          uint32_t kHalvesPerTile, uint32_t kNumSubTiles, uint32_t kNumAccum, class Storage, class Tile>
CUTLASS_DEVICE void store_swiglu_fp8_g128(const GemmParams &params, Storage &smem, const Tile &tile,
                                          float global_scale, float (&final_accum)[kNumSubTiles][kNumAccum],
                                          uint32_t r_0) {
    store_swiglu_fp8_g128_impl<OUTPUT_N, BLOCK_M, kNumMathThreads, kMathBarrier, kHalvesPerTile>(
        params.d, params.d_scale, smem, tile.row_begin, min(tile.row_end - tile.row_begin, BLOCK_M),
        tile.n_block_idx, global_scale, final_accum, r_0);
}

}  // namespace tensorbridge::epilogue
