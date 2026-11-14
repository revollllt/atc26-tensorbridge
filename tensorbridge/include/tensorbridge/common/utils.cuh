#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cute/arch/util.hpp>
#include <cutlass/detail/helper_macros.hpp>

namespace tensorbridge {

// Mirrored by `GroupedLayout` in `csrc/heuristics/config.hpp` and `LAYOUT_IDS` in `jit/codegen.py`.
enum class GroupedLayout { kDense, kIndexed, kContiguous, kMasked };

namespace math {

constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace math

// Walk a tile's K blocks through a ring of `kNumStages` stages from `first_stage`, with `body(stage, k_blocks_left)`
// per K block, and return the stage after the last one.
// NOTES: the stage is a compile-time constant once the loops unroll, so its addresses fold into constants
template <uint32_t kNumStages, class Body>
CUTLASS_DEVICE uint32_t for_each_ring_slot(uint32_t first_stage, uint32_t num_k_blocks, Body &&body) {
    uint32_t k_blocks_left = num_k_blocks;
    #pragma unroll
    for (uint32_t stage = 0; stage < kNumStages; stage++) {
        if (stage >= first_stage && k_blocks_left) body(stage, k_blocks_left--);
    }
    while (k_blocks_left) {
        #pragma unroll
        for (uint32_t stage = 0; stage < kNumStages; stage++) {
            body(stage, k_blocks_left--);
            if (!k_blocks_left) break;
        }
    }
    return (first_stage + num_k_blocks) % kNumStages;
}

}  // namespace tensorbridge
