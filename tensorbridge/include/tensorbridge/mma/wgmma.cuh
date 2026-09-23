#pragma once

#include <tensorbridge/common/tile.cuh>
#include <tensorbridge/ptx/wgmma.cuh>

namespace tensorbridge {

// Accumulators of one thread: `m64 x nBLOCK_M` FP32 results spread over the 32
// lanes of its warp, i.e. 2 channel rows x BLOCK_M / 4 token pairs per thread.
template <class Config>
constexpr uint32_t kNumAccumulators = Config::kBlockM / 2;

// Issue k32 call `k_iter` of a K block as its own commit group:
//
//     accum += fragment (registers, FP8 weights) x stage_a[:, 32 * k_iter ...] (shared, FP8 activations)
//
// The call is asynchronous; `fragment` must stay untouched until the group has
// been waited for. `Config::wgmma` is the instruction itself, which the JIT
// emits for the tile (`jit/codegen.py`).
template <class Config>
TB_INLINE void issue_wgmma(int4 *stage_a, uint32_t k_iter, uint32_t *fragment, float *accum) {
  constexpr uint32_t kInt4sPerCall = Tile::kWgmmaK / sizeof(int4);
  uint64_t activations = ptx::wgmma_smem_operand(stage_a + k_iter * kInt4sPerCall);
  ptx::wgmma_fence();
  Config::wgmma(fragment, activations, accum);
  ptx::wgmma_commit_group();
}

}  // namespace tensorbridge
