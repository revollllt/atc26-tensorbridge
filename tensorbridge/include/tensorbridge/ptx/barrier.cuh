#pragma once

#include <cooperative_groups.h>
#include <cuda_awbarrier_primitives.h>  // __mbarrier_init

#include <tensorbridge/common/base.cuh>

// Two kinds of synchronization are used by the kernel:
//
//   mbarrier   a 64-bit shared-memory barrier between the producer warp group
//              and the consumer warp groups of one CTA. Each barrier flips a
//              phase parity per round, so a waiter names the parity it expects.
//   gmem lock  a global-memory word that orders the Stream-K slices of one
//              output tile across CTAs.

namespace tensorbridge::ptx {

// Barrier over the first `kNumSyncThreads` threads only; the producer and the
// consumers live in the same launch but never wait on each other this way.
template <uint32_t kNumSyncThreads, uint32_t kNumThreads, uint32_t kBarrierId = 1>
TB_INLINE void sync_threads() {
  if constexpr (kNumSyncThreads == kNumThreads) {
    __syncthreads();
  } else {
    asm volatile("bar.sync %0, %1;" ::"r"(kBarrierId), "r"(kNumSyncThreads));
  }
}

TB_INLINE void mbarrier_wait(void *barrier, bool phase_parity) {
  const uint32_t addr = smem_addr(barrier);
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(addr), "r"((uint32_t)phase_parity)
               : "memory");
}

// Non-blocking probe; returns 1 if the barrier is already passed.
TB_INLINE uint32_t mbarrier_try_wait(void *barrier, bool phase_parity) {
  const uint32_t addr = smem_addr(barrier);
  uint32_t ready = 0;
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "  selp.u32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(ready)
               : "r"(addr), "r"((uint32_t)phase_parity)
               : "memory");
  return ready;
}

TB_INLINE void mbarrier_arrive(void *barrier) {
  const uint32_t addr = smem_addr(barrier);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
               :
               : "r"(addr)
               : "memory");
}

// Arrive at a barrier that lives in another CTA of the same cluster.
TB_INLINE void mbarrier_arrive_cluster(void *barrier, uint32_t dst_cta_id, bool pred) {
  if (pred) {
    const uint32_t addr = smem_addr(barrier);
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(addr), "r"(dst_cta_id)
        : "memory");
  }
}

// Spin until the lock word satisfies `state <= count`, holding it on exit.
template <uint32_t kNumSyncThreads, uint32_t kNumThreads>
TB_INLINE void gmem_lock_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = 1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state > count);
  }
  sync_threads<kNumSyncThreads, kNumThreads>();
}

// Negative `val` stores it; otherwise the word is incremented by one.
template <uint32_t kNumSyncThreads, uint32_t kNumThreads>
TB_INLINE void gmem_lock_release(int *lock, int32_t val) {
  sync_threads<kNumSyncThreads, kNumThreads>();
  if (threadIdx.x == 0) {
    if (val < 0) {
      __stcg(&lock[0], val);
    } else {
      int32_t one = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(one));
    }
  }
}

}  // namespace tensorbridge::ptx
