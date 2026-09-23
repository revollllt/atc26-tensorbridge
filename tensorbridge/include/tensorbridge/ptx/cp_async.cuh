#pragma once

#include <tensorbridge/common/base.cuh>

// Per-thread asynchronous copies from global to shared memory. They carry the
// small side inputs (weight scales, per-token scales) that are too narrow to be
// worth a TMA descriptor.

namespace tensorbridge::ptx {

// `T` is 16 bytes (`cg`, cache-line granular) or 4 bytes (`ca`).
template <typename T>
TB_INLINE void cp_async_load_pred(const T *gmem_ptr, T *smem_ptr, bool pred) {
  const uint32_t dst = smem_addr(smem_ptr);
  if constexpr (sizeof(T) == 16) {
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.s32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n"
                 :
                 : "r"((uint32_t)pred), "r"(dst), "l"(gmem_ptr), "n"(16)
                 : "memory");
  } else {
    static_assert(sizeof(T) == 4);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.s32 p, %0, 0;\n"
                 "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                 "}\n"
                 :
                 : "r"((uint32_t)pred), "r"(dst), "l"(gmem_ptr), "n"(4)
                 : "memory");
  }
}

// Every copying thread arrives once per round.
TB_INLINE void cp_async_arrive(uint64_t *mbar_ptr) {
  const uint32_t mbar = smem_addr(mbar_ptr);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
               :
               : "r"(mbar)
               : "memory");
}

}  // namespace tensorbridge::ptx
