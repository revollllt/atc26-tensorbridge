#pragma once

#include <tensorbridge/common/tile.cuh>

namespace tensorbridge {

// Hands out work to one CTA group (a CTA, or a multicast cluster acting as one).
// Every thread of every CTA runs the same scheduler and derives its share from
// `blockIdx` alone, so no coordination is needed.
//
// Data parallel: MN tiles are dealt round-robin, CTA group `g` computes tiles
// `g, g + G, g + 2G, ...` over their full K depth.
//
// Stream-K tail: dealing whole tiles leaves the last wave underfilled whenever
// the tile count is not a multiple of `G`. With `kUseStreamK`, the tiles of that
// last wave are instead cut along K: their `tail_tiles * K_BLOCKS` K blocks are
// laid end to end and split evenly over all `G` groups. A group may then get
// the end of one tile and the start of the next; each contiguous run within a
// tile is one *slice*, and the slices of a tile are summed in global memory
// (`memory/s2g.cuh`).
template <class Config>
class Scheduler {
  static constexpr uint32_t kMulticast = Config::kMulticastA * Config::kMulticastB;
  static constexpr uint32_t kNumNBlocks = Config::kShapeN / Tile::kBlockN / Config::kMulticastA;
  static constexpr uint32_t kNumKBlocks = Config::kShapeK / Tile::kBlockK;

  const uint32_t num_groups = gridDim.x / kMulticast;
  const uint32_t group_id = blockIdx.x / kMulticast;
  const uint32_t cluster_rank = blockIdx.x % kMulticast;

  uint32_t num_m_blocks;
  uint32_t num_k_blocks_total;

  uint32_t dp_tiles_left, dp_next_tile, dp_tiles_per_group;
  uint32_t tail_share, tail_blocks_left, tail_next_block;

public:
  uint32_t m_block_id, n_block_id, k_block_id;
  uint32_t slice_num_k_blocks;        // K blocks to run now, starting at `k_block_id`
  uint32_t slice_count, slice_id;     // slices the tile is cut into; `slice_id == 0` reaches the end of K
  uint32_t lock_idx;                  // lock word of the tile, see `memory/s2g.cuh`

  TB_INLINE explicit Scheduler(uint32_t shape_m) {
    num_m_blocks = ceil_div(shape_m, Config::kBlockM * Config::kMulticastB);
    const uint32_t num_tiles = num_m_blocks * kNumNBlocks;
    num_k_blocks_total = num_tiles * kNumKBlocks;

    if constexpr (Config::kUseStreamK) {
      uint32_t tail_tiles = num_tiles;
      if (num_tiles > num_groups) {
        tail_tiles = num_tiles % num_groups;
        // A tail this thin would be all slicing overhead: widen it by one wave.
        if (tail_tiles && tail_tiles * 10 <= num_groups) tail_tiles += num_groups;
      }
      dp_tiles_left = (num_tiles - tail_tiles) / num_groups;

      tail_share = ceil_div(tail_tiles * kNumKBlocks, num_groups);
      tail_next_block = num_groups * dp_tiles_left * kNumKBlocks + tail_share * group_id;
      tail_blocks_left =
          tail_next_block >= num_k_blocks_total ? 0 : min_of(num_k_blocks_total - tail_next_block, tail_share);
    } else {
      dp_tiles_left = num_tiles / num_groups + (group_id < num_tiles % num_groups);
    }

    dp_tiles_per_group = dp_tiles_left;
    dp_next_tile = group_id;
  }

  TB_INLINE bool get_next_block() {
    if (dp_tiles_left) {
      set_tile(dp_next_tile);
      k_block_id = 0;
      slice_num_k_blocks = kNumKBlocks;
      slice_count = 1, slice_id = 0, lock_idx = 0;
      dp_next_tile += num_groups;
      dp_tiles_left--;
      return true;
    }
    if constexpr (Config::kUseStreamK) return get_next_tail_slice();
    return false;
  }

  TB_INLINE bool is_full_depth() const { return slice_count == 1; }

private:
  TB_INLINE void set_tile(uint32_t tile_idx) {
    if constexpr (Config::kMFastTileOrder && kMulticast == 1) {
      m_block_id = tile_idx % num_m_blocks;
      n_block_id = tile_idx / num_m_blocks;
    } else {
      m_block_id = tile_idx / kNumNBlocks;
      n_block_id = tile_idx % kNumNBlocks;
    }
    // Within a cluster the shared dimension is common, the other one is split.
    if constexpr (Config::kMulticastB > 1) {
      m_block_id = m_block_id * Config::kMulticastB + cluster_rank;
    } else if constexpr (Config::kMulticastA > 1) {
      n_block_id = n_block_id * Config::kMulticastA + cluster_rank;
    }
  }

  TB_INLINE bool get_next_tail_slice() {
    if (!tail_blocks_left) return false;
    const uint32_t tile_idx = tail_next_block / kNumKBlocks;
    set_tile(tile_idx);
    k_block_id = tail_next_block - tile_idx * kNumKBlocks;

    slice_num_k_blocks = min_of(kNumKBlocks - k_block_id, tail_blocks_left);
    tail_blocks_left -= slice_num_k_blocks;
    tail_next_block += slice_num_k_blocks;

    // Slice boundaries of a tile fall on multiples of `tail_share` in the
    // concatenated K-block sequence; recover this tile's cut from `k_block_id`.
    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = ceil_div(kNumKBlocks - slice_num_k_blocks, tail_share) + 1;
    } else {
      slice_id = k_block_id / tail_share;
      const uint32_t first_slice_blocks = k_block_id - slice_id * tail_share;
      slice_count = ceil_div(kNumKBlocks - first_slice_blocks, tail_share);
      if (first_slice_blocks) {
        slice_id++;
        slice_count++;
      }
    }
    // Number the slices from the end of K. A group's share begins with the
    // final run of a tile, so that run is computed first and is the one that
    // stores; the runs before it add.
    slice_id = slice_count - 1 - slice_id;

    lock_idx = (tile_idx - dp_tiles_per_group * num_groups) * kMulticast + cluster_rank;
    return true;
  }
};

}  // namespace tensorbridge
