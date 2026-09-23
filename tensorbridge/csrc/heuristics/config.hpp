#pragma once

#include <cstdint>
#include <vector>

namespace tensorbridge {

// The problem: `D[M, N] = A[M, K] @ B[N, K].T` on a device with `num_sms` SMs,
// one CTA per SM.
struct GemmDesc {
  int64_t m, n, k;
  int64_t num_sms;
};

// Everything the kernel template is specialized on besides the shape. The
// channel and depth tiles are fixed at 128 by the WGMMA instruction (see
// `include/tensorbridge/common/tile.cuh`), so tiling comes down to `block_m`.
struct GemmConfig {
  int64_t block_m = 128;            // token tile
  int64_t num_stages = 4;           // shared-memory ring slots, one K block each
  // Multicast: a cluster of CTAs receives one tile by a single load and splits
  // the other dimension. Only `multicast_b` is ever selected (by measured rules
  // in `sm90_tuned.hpp`); sharing activation tiles was slower than independent
  // CTAs on every shape tried. Data parallel only, see the kernel.
  int64_t multicast_a = 1;          // CTAs sharing one activation tile (split over N)
  int64_t multicast_b = 1;          // CTAs sharing one weight tile (split over M)
  int64_t num_producer_regs = 40;   // registers left to the producer warp group
  bool use_stream_k = false;        // cut the underfilled last wave along K
  bool use_tma_store = true;        // TMA output stores (else per-thread stores)
  bool m_fast_tile_order = false;   // deal tiles M-fast instead of N-fast
  bool prebroadcast_scale = false;  // dequant variant, see `dequant/fpma.cuh`

  int64_t cluster_size() const { return multicast_a * multicast_b; }

  // Field order is mirrored by `GemmConfig` in `tensorbridge/jit/codegen.py`.
  std::vector<int64_t> to_vector() const {
    return {block_m,       num_stages,    multicast_a,       multicast_b,       num_producer_regs,
            use_stream_k,  use_tma_store, m_fast_tile_order, prebroadcast_scale};
  }

  static GemmConfig from_vector(const std::vector<int64_t> &v) {
    GemmConfig c;
    c.block_m = v.at(0), c.num_stages = v.at(1), c.multicast_a = v.at(2), c.multicast_b = v.at(3);
    c.num_producer_regs = v.at(4);
    c.use_stream_k = v.at(5), c.use_tma_store = v.at(6), c.m_fast_tile_order = v.at(7);
    c.prebroadcast_scale = v.at(8);
    return c;
  }
};

constexpr int64_t kBlockN = 128;
constexpr int64_t kBlockK = 128;

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace tensorbridge
