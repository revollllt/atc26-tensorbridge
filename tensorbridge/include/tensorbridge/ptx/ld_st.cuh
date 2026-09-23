#pragma once

#include <tensorbridge/common/base.cuh>

namespace tensorbridge::ptx {

// Transposed register -> shared store of two 8x8 BF16 matrices. Each thread of
// the warp contributes one row of 8 values per matrix, and the transpose lands
// them as columns: this is what turns the `[channel, token]` accumulators into
// a `[token, channel]` output tile without a separate permutation pass.
TB_INLINE void st_shared_trans_x2(uint32_t addr, const uint32_t *regs) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1,%2};\n"
               :: "r"(addr), "r"(regs[0]), "r"(regs[1]));
}

}  // namespace tensorbridge::ptx
