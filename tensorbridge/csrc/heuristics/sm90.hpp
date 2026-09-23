#pragma once

#include <algorithm>

#include "config.hpp"
#include "sm90_tuned.hpp"

// Shape-only config selection for SM90 (H100). Nothing is profiled at run time:
// the config follows from `(M, N, K)` and the SM count.
//
// A GEMM is cut into `ceil(M / block_m) * N / 128` MN tiles, each K blocks deep.
// The device runs `num_sms` CTAs in lockstep *waves*; with plain data
// parallelism one CTA computes one whole tile, so a wave that is not full
// leaves CTAs idle for a tile's duration. Two decisions follow:
//
//   1. the token tile `block_m`, which sets the tile count, and
//   2. whether to run the underfilled last wave as a Stream-K tail, cutting its
//      tiles along K so that every CTA gets an equal share of K blocks.

namespace tensorbridge::sm90 {

// Thresholds below that are not derived from the wave model are empirical, from
// sweeps on H100, and marked as such.

// Widest tile `select_block_m` considers (empirical). The 256-token tile is
// only chosen by the explicit rules in `select_config`.
constexpr int64_t kMaxGenericBlockM = 176;

// Fewest token tiles first, then the smallest tile that achieves it: a tile
// wider than needed only pads the accumulators.
inline int64_t select_block_m(int64_t m) {
  int64_t best_block_m = 8;
  for (int64_t block_m = 16; block_m <= kMaxGenericBlockM; block_m += 8) {
    if (ceil_div(m, block_m) < ceil_div(m, best_block_m)) best_block_m = block_m;
  }
  return best_block_m;
}

struct WaveStats {
  int64_t num_groups;   // CTA groups: CTAs, or multicast clusters
  int64_t num_tiles;    // MN tiles
  int64_t tail_tiles;   // tiles a Stream-K tail would cut along K
  int64_t num_k_blocks;

  // Fraction of CTA-tile slots that data-parallel waves actually fill, as
  // `num / den`.
  int64_t fill_num() const { return num_tiles; }
  int64_t fill_den() const { return ceil_div(num_tiles, num_groups) * num_groups; }
};

inline WaveStats get_wave_stats(const GemmDesc &desc, const GemmConfig &config) {
  WaveStats stats;
  stats.num_groups = std::max<int64_t>(1, desc.num_sms / config.cluster_size());
  stats.num_tiles = ceil_div(desc.m, config.block_m * config.multicast_b) *
                    ceil_div(desc.n, kBlockN * config.multicast_a);
  stats.num_k_blocks = ceil_div(desc.k, kBlockK);

  // Mirrors `Scheduler` in `include/tensorbridge/scheduler/gemm.cuh`.
  stats.tail_tiles = stats.num_tiles;
  if (stats.num_tiles > stats.num_groups) {
    stats.tail_tiles = stats.num_tiles % stats.num_groups;
    if (stats.tail_tiles && stats.tail_tiles * 10 <= stats.num_groups) stats.tail_tiles += stats.num_groups;
  }
  return stats;
}

// Whether the Stream-K tail pays off. Its price is the reduction: a tile cut
// into `s` slices is written `s` times, `s - 1` of them as read-modify-write
// under a lock.
inline bool select_stream_k(const GemmDesc &desc, const GemmConfig &config) {
  const auto stats = get_wave_stats(desc, config);
  if (stats.tail_tiles == 0) return false;

  // Severely underfilled (fill <= 3/4): always worth it.
  if (stats.fill_num() * 4 <= 3 * stats.fill_den()) return true;
  // Nearly full (fill > 112/132): the idle time is smaller than any reduction.
  if (stats.fill_num() * 132 > 112 * stats.fill_den()) return false;

  // In between, bound the amount of output that would be reduced.
  constexpr int64_t kMaxReducedElements = 1500000;
  const int64_t k_blocks_per_group = ceil_div(stats.tail_tiles * stats.num_k_blocks, stats.num_groups);
  const int64_t slices_per_tile = std::max<int64_t>(1, ceil_div(stats.num_k_blocks, k_blocks_per_group));
  const int64_t reduced_tiles = stats.tail_tiles * (slices_per_tile - 1);
  return reduced_tiles * config.block_m * kBlockN <= kMaxReducedElements;
}

// `stream_k_override`: negative lets the model decide; otherwise Stream-K is
// forced off (0) or on (positive, as far as there is a tail to cut).
inline GemmConfig select_config(const GemmDesc &desc, int64_t stream_k_override = -1) {
  GemmConfig config;
  bool is_data_parallel_tile = false;

  const bool is_large_mlp = (desc.n >= 8192 && desc.k >= 3584) || (desc.n >= 3584 && desc.k >= 8192);
  if (desc.m >= 1024 && is_large_mlp) {
    // Prefill on the large projections: many full waves whatever the tile, so
    // take the largest one.
    config.block_m = 256;
    config.m_fast_tile_order = desc.m <= 2048 && desc.n >= 14336 && desc.k >= 3072;
  } else {
    config.block_m = select_block_m(desc.m);

    // Decode sizes on wide outputs: a deeper ring (empirical).
    if (desc.n > 8192 && desc.m <= 128) config.num_stages = 6;

    // Exactly 256 tokens on a wide, deep layer: one 256-token tile, data
    // parallel, rather than two 128-token tiles (empirical).
    if (desc.m == 256 && desc.n > 8192 && desc.k >= 4096) config.block_m = 256;
  }
  tuned::apply_tile_rules(desc, config);
  if (desc.m == 256 && config.block_m == 256) is_data_parallel_tile = true;

  // If 256-token tiles would fill data-parallel waves well enough
  // (fill >= 96/132), they beat 176-token tiles with or without a tail.
  if (config.block_m == kMaxGenericBlockM && desc.m >= 512) {
    GemmConfig wide = config;
    wide.block_m = 256;
    const auto stats = get_wave_stats(desc, wide);
    if (stats.fill_num() * 132 >= 96 * stats.fill_den()) {
      config.block_m = 256;
      is_data_parallel_tile = true;
    }
  }

  if (stream_k_override < 0) {
    config.use_stream_k = !is_data_parallel_tile && select_stream_k(desc, config);
  } else {
    config.use_stream_k = stream_k_override > 0 && get_wave_stats(desc, config).tail_tiles != 0;
  }

  if (!config.use_stream_k) {
    // Without a tail, 256-token tiles beat 176-token ones for long inputs (empirical).
    if (config.block_m == kMaxGenericBlockM && ((desc.m == 512 && desc.n >= 13824) || desc.m >= 2048))
      config.block_m = 256;
    // Dequant variant of the 256-token tile (empirical), see `dequant/fpma.cuh`.
    config.prebroadcast_scale = config.block_m == 256 && desc.k > 512;
  }
  tuned::apply_knob_rules(desc, config);
  return config;
}

}  // namespace tensorbridge::sm90
