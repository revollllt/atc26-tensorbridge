#pragma once

#include <limits>
#include <vector>

#include "config.hpp"

// Measured overrides for H100, kept apart from the shape-only model in
// `sm90.hpp`. Each rule names the knob it replaces and the shapes it was
// measured on; nothing here is needed for correctness, and deleting a rule only
// moves its shapes back to the model's choice.

namespace tensorbridge::tuned {

constexpr int64_t kAny = std::numeric_limits<int64_t>::max();

struct ShapeRange {
  int64_t m;
  int64_t n_min, n_max;
  int64_t k_min, k_max;

  bool contains(const GemmDesc &desc) const {
    return desc.m == m && desc.n >= n_min && desc.n <= n_max && desc.k >= k_min && desc.k <= k_max;
  }
};

constexpr ShapeRange exactly(int64_t m, int64_t n, int64_t k) { return {m, n, n, k, k}; }

struct Rule {
  const char *what;
  int64_t block_m;  // tile the rule was measured with; 0 if it does not depend on one
  void (*apply)(GemmConfig &);
  std::vector<ShapeRange> shapes;

  bool matches(const GemmDesc &desc, const GemmConfig &config) const {
    if (block_m != 0 && block_m != config.block_m) return false;
    for (const auto &shape : shapes)
      if (shape.contains(desc)) return true;
    return false;
  }
};

// Applied before the Stream-K decision, which depends on the token tile.
inline const Rule kTileRules[] = {
    {"one 256-token tile rather than two 128-token tiles, below the model's K threshold", 0,
     [](GemmConfig &c) { c.block_m = 256; },
     {exactly(256, 32768, 512), exactly(256, 24576, 1536)}},

    {"a fifth ring slot", 0,
     [](GemmConfig &c) { c.num_stages = 5; },
     {exactly(128, 57344, 8192), exactly(128, 28672, 8192), exactly(128, 13824, 5120), exactly(128, 15360, 5120)}},
};

// Applied after it, and to data-parallel configs only.
inline const Rule kKnobRules[] = {
    {"M-fast tile order", 256,
     [](GemmConfig &c) { c.m_fast_tile_order = true; },
     {exactly(4096, 32768, 512), exactly(8192, 32768, 512), exactly(1024, 4096, 7168), exactly(1024, 8192, 28672),
      exactly(2048, 8192, 28672), exactly(4096, 24576, 4096),
      {512, 8192, kAny, 4096, 18432}, {512, 0, kAny, 8192, 18432}}},

    {"M-fast tile order and prebroadcast scales on the 128-token tile", 128,
     [](GemmConfig &c) { c.m_fast_tile_order = true, c.prebroadcast_scale = true; },
     {{256, 7168, 7168, 5120, kAny}, {256, 8192, 8192, 5120, kAny}}},

    {"per-thread output stores", 0,
     [](GemmConfig &c) { c.use_tma_store = false; },
     {exactly(16, 32768, 512), exactly(32, 32768, 512), exactly(64, 32768, 512), exactly(128, 32768, 512),
      exactly(128, 57344, 8192), exactly(128, 28672, 8192), exactly(128, 13824, 5120), exactly(128, 15360, 5120)}},

    {"per-thread output stores", 256,
     [](GemmConfig &c) { c.use_tma_store = false; },
     {exactly(512, 8192, 28672), exactly(2048, 8192, 28672), exactly(2048, 4096, 12288)}},

    {"56 producer registers", 256,
     [](GemmConfig &c) { c.num_producer_regs = 56; },
     {exactly(1024, 4096, 7168),  exactly(1024, 8192, 28672), exactly(1024, 15360, 5120), exactly(1024, 17408, 5120),
      exactly(1024, 18432, 7168), exactly(1024, 22016, 4096), exactly(1024, 24576, 4096), exactly(1024, 25600, 5120),
      exactly(1024, 27648, 5120), exactly(1024, 28672, 8192), exactly(1024, 34816, 5120), exactly(1024, 36864, 5120),
      exactly(1024, 36864, 7168), exactly(1024, 51200, 5120), exactly(2048, 7168, 18432), exactly(2048, 8192, 5120),
      exactly(2048, 12288, 4096), exactly(2048, 22016, 4096), exactly(2048, 25600, 5120), exactly(2048, 28672, 8192),
      exactly(2048, 34816, 5120), exactly(2048, 51200, 5120), exactly(4096, 8192, 5120),  exactly(4096, 8192, 8192)}},

    {"weight-tile multicast", 256,
     [](GemmConfig &c) { c.multicast_b = 2; },
     {exactly(4096, 36864, 7168), exactly(8192, 36864, 7168), exactly(4096, 51200, 5120), exactly(8192, 51200, 5120),
      exactly(4096, 57344, 8192), exactly(8192, 57344, 8192)}},
};

inline void apply_tile_rules(const GemmDesc &desc, GemmConfig &config) {
  for (const auto &rule : kTileRules)
    if (rule.matches(desc, config)) rule.apply(config);
}

inline void apply_knob_rules(const GemmDesc &desc, GemmConfig &config) {
  if (config.use_stream_k) return;
  for (const auto &rule : kKnobRules)
    if (rule.matches(desc, config)) rule.apply(config);
}

}  // namespace tensorbridge::tuned
