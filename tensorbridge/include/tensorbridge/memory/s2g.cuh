#pragma once

#include <tensorbridge/memory/storage.cuh>
#include <tensorbridge/ptx/barrier.cuh>
#include <tensorbridge/ptx/tma.cuh>

namespace tensorbridge {

// Shared -> global: the second half of the epilogue.
//
// A Stream-K tile is computed as several K slices, possibly on different CTAs,
// and each slice contributes a partial sum. Slice 0 (the final K run, which
// its CTA reaches first) stores; the others add into global memory. One lock
// word per tile orders them: the storing slice takes the lock at 0 and leaves
// it at `1 - slice_count`; every adding slice needs a negative lock and
// increments it, so the word is back at 0 when the tile is complete.
template <class Config>
class OutputTileStore {
  static constexpr uint32_t kBlockM = Config::kBlockM;
  static constexpr uint32_t kChannelsPerBox = 128 / sizeof(nv_bfloat16);
  static constexpr uint32_t kNumBoxes = Tile::kBlockN / kChannelsPerBox;
  static constexpr uint32_t kBoxInt4s = kBlockM * kChannelsPerBox * sizeof(nv_bfloat16) / sizeof(int4);

  int4 *tile;
  const CUtensorMap *tensor_map_d;
  int4 *gmem_d;
  int32_t *locks;

  uint32_t m_idx, n_idx, num_tokens;

public:
  // `output` is the TMA descriptor of D, or D itself without TMA stores.
  TB_INLINE OutputTileStore(int4 *smem_d, const void *output, int32_t *locks) : tile(smem_d), locks(locks) {
    if constexpr (Config::kUseTmaStore) {
      tensor_map_d = reinterpret_cast<const CUtensorMap *>(output);
      if (threadIdx.x == 0) ptx::tma_prefetch_descriptor(tensor_map_d);
    } else {
      gmem_d = reinterpret_cast<int4 *>(const_cast<void *>(output));
    }
    sync_math_threads();
  }

  TB_INLINE void seek(uint32_t m_block_id, uint32_t n_block_id, uint32_t shape_m) {
    m_idx = m_block_id * kBlockM;
    n_idx = n_block_id * Tile::kBlockN;
    num_tokens = min_of(shape_m - m_idx, kBlockM);
  }

  TB_INLINE static void sync_math_threads() {
    ptx::sync_threads<Tile::kNumMathThreads, Tile::kNumThreads>();
  }

  TB_INLINE void lock_slice(uint32_t lock_idx, uint32_t slice_id) {
    ptx::gmem_lock_acquire<Tile::kNumMathThreads, Tile::kNumThreads>(&locks[lock_idx], slice_id == 0 ? 0 : -1);
  }

  TB_INLINE void unlock_slice(uint32_t lock_idx, uint32_t slice_id, uint32_t slice_count) {
    const int32_t val = slice_id == 0 ? 1 - static_cast<int32_t>(slice_count) : 0;
    ptx::gmem_lock_release<Tile::kNumMathThreads, Tile::kNumThreads>(&locks[lock_idx], val);
  }

  // One TMA store per 64-channel box, issued by threads 0 and 1.
  TB_INLINE void store_tma(uint32_t slice_id, uint32_t slice_count) {
    const uint32_t box = threadIdx.x;
    if (box >= kNumBoxes) return;
    int4 *src = tile + box * kBoxInt4s;
    const uint32_t box_n_idx = n_idx + box * kChannelsPerBox;
    if (slice_count == 1 || slice_id == 0) {
      ptx::tma_store_2d(src, tensor_map_d, box_n_idx, m_idx);
      if (slice_count > 1) ptx::tma_wait_store_group<0>();
    } else {
      ptx::tma_reduce_add_2d(src, tensor_map_d, box_n_idx, m_idx);
      if (slice_id != slice_count - 1) ptx::tma_wait_store_group<0>();
    }
  }

  // Per-thread stores of 16-byte vectors (eight BF16 values), undoing the 128B
  // swizzle by hand. Small tiles on very wide outputs prefer it to TMA.
  TB_INLINE void store_direct() {
    static_assert(!Config::kUseStreamK, "direct stores cannot accumulate Stream-K slices");
    constexpr uint32_t kTileInt4s = kNumBoxes * kBoxInt4s;
    constexpr uint32_t kIters = ceil_div(kTileInt4s, Tile::kNumMathThreads);
    const uint32_t swizzle_base = smem_addr(tile) / 128;
    int4 *dst = gmem_d + m_idx * (Config::kShapeN / 8) + n_idx / 8;

    TB_UNROLL
    for (uint32_t i = 0; i < kIters; i++) {
      const uint32_t idx = threadIdx.x + Tile::kNumMathThreads * i;
      if (kTileInt4s % Tile::kNumMathThreads == 0 || i != kIters - 1 || idx < kTileInt4s) {
        const uint32_t smem_row = idx / 8;
        const uint32_t smem_col = idx % 8;
        const uint32_t token = smem_row % kBlockM;
        const uint32_t channel_group = smem_row / kBlockM * 8 + smem_col;
        if (token >= num_tokens) continue;

        const uint32_t swizzled_col = smem_col ^ ((smem_row + swizzle_base) % 8);
        dst[token * (Config::kShapeN / 8) + channel_group] = tile[smem_row * 8 + swizzled_col];
      }
    }
  }
};

}  // namespace tensorbridge
