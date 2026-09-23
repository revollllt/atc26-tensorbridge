#pragma once

#include <tensorbridge/common/base.cuh>

// Warp-Group MMA control. WGMMA calls are asynchronous: they are issued into a
// commit group, and `wait<N>` returns once at most `N` groups remain in flight.
// The instruction itself is emitted by the JIT (`jit/codegen.py`), because its
// mnemonic and operand count encode the token tile.

namespace tensorbridge::ptx {

TB_INLINE void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::
                   : "memory");
}

TB_INLINE void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::
                   : "memory");
}

template <uint32_t kGroupsInFlight>
TB_INLINE void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(kGroupsInFlight)
               : "memory");
}

// Descriptor of the shared-memory operand: 128B-swizzled FP8 rows.
TB_INLINE uint64_t wgmma_smem_operand(void *smem_ptr) {
  constexpr uint64_t kSwizzle128 = 1;
  constexpr uint64_t kStride = (128 * 8) >> 4;
  uint64_t desc = (kSwizzle128 << 62) | (kStride << 32);
  reinterpret_cast<uint32_t *>(&desc)[0] = smem_addr(smem_ptr) >> 4;
  return desc;
}

}  // namespace tensorbridge::ptx
