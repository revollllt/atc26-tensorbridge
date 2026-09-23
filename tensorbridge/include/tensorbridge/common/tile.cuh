#pragma once

#include <tensorbridge/common/base.cuh>

namespace tensorbridge {

// The CTA evaluates the transposed product `Y^T = W X^T` (operand swap): the
// weight fragment is built in registers, so it has to sit on the WGMMA operand
// that accepts registers (`m64`), and the token dimension maps to the flexible
// WGMMA `n`. One WGMMA call is `m64 x nBLOCK_M x k32` over FP8 E4M3.
//
// Everything except the token tile `BLOCK_M` is fixed by that instruction:
//
//   channel tile  128 = 8 consumer warps x 16 channels
//   depth tile    128 = 4 WGMMA calls    x k32
//   one thread        = 2 channel rows (r, r + 8) x 4 consecutive K, twice
//                       (K-low and K-high half of the k32 call)
struct Tile {
  static constexpr uint32_t kBlockN = 128;
  static constexpr uint32_t kBlockK = 128;

  static constexpr uint32_t kWgmmaK = 32;
  static constexpr uint32_t kKIters = kBlockK / kWgmmaK;

  // NVFP4: one E4M3 scale per 16 weights, so two scale groups per k32 call.
  static constexpr uint32_t kScaleGroupSize = 16;
  static constexpr uint32_t kScaleGroupsPerBlock = kBlockK / kScaleGroupSize;

  static constexpr uint32_t kWarpSize = 32;
  static constexpr uint32_t kChannelsPerWarp = 16;
  static constexpr uint32_t kNumMathThreads = kBlockN / kChannelsPerWarp * kWarpSize;
  static constexpr uint32_t kNumLoadThreads = 128;
  static constexpr uint32_t kNumThreads = kNumMathThreads + kNumLoadThreads;
};

// First channel row owned by a math thread; it also owns row `+ 8`.
TB_INLINE uint32_t thread_channel_base() {
  const uint32_t warp_id = threadIdx.x / Tile::kWarpSize;
  const uint32_t lane_id = threadIdx.x % Tile::kWarpSize;
  return lane_id / 4 + (warp_id % 4) * 16 + (warp_id / 4) * 64;
}

// Token row (within an 8-token group) owned by a math thread. The two halves of
// each accumulator pair are adjacent tokens, hence the parity term.
TB_INLINE uint32_t thread_token_in_group() {
  const uint32_t lane_id = threadIdx.x % Tile::kWarpSize;
  return (lane_id % 4) * 2 + (lane_id % 8) / 4;
}

// Token tiles whose accumulators leave room for four weight register buffers,
// which is what lets the mainloop keep one WGMMA group in flight while it
// dequantizes the next (see `impls/sm90_fp8_nvfp4_gemm.cuh`). Other tiles run
// the two-buffer loop.
constexpr bool has_four_register_buffers(uint32_t block_m) {
  return block_m == 16 || block_m == 32 || block_m == 64 || block_m == 128 || block_m == 256;
}

// Token tiles whose output can leave the registers through `stmatrix.x2.trans`,
// which stores pairs of 8-token groups.
constexpr bool has_stsm_epilogue(uint32_t block_m) {
  return block_m == 32 || block_m == 64 || block_m == 128 || block_m == 256;
}

}  // namespace tensorbridge
