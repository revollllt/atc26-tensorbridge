#pragma once

#include <cutlass/arch/barrier.h>

#include <tensorbridge/common/utils.cuh>

namespace tensorbridge::ptx {

// Async copy of `T` (16 or 4 bytes) when `pred` holds
template <typename T>
CUTLASS_DEVICE void cp_async_load_pred(const T *gmem_ptr, T *smem_ptr, bool pred) {
    const uint32_t dst = cute::cast_smem_ptr_to_uint(smem_ptr);
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

// Arrive on `barrier` once the calling thread's async copies have landed
CUTLASS_DEVICE void cp_async_arrive(cutlass::arch::ClusterBarrier *barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(cute::cast_smem_ptr_to_uint(barrier)) : "memory");
}

// Stream-K: spin until the lock word is at most `count`, then order this CTA's TMA accesses after it
template <uint32_t kNumThreads, uint32_t kBarrierId>
CUTLASS_DEVICE void gmem_lock_acquire(int32_t *lock, int32_t count) {
    if (threadIdx.x == 0) {
        int32_t state = 1;
        do {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock) : "memory");
        } while (state > count);
    }
    cutlass::arch::NamedBarrier::sync(kNumThreads, kBarrierId);
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// Stream-K: publish the lock word after the calling warp's writes
CUTLASS_DEVICE void gmem_lock_publish(int32_t *lock, int32_t value) {
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    asm volatile("st.relaxed.gpu.global.b32 [%0], %1;\n" ::"l"(lock), "r"(value) : "memory");
}

// Stream-K: add one to the lock word
CUTLASS_DEVICE void gmem_lock_increment(int32_t *lock) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" ::"l"(lock), "r"(1) : "memory");
}

// Route B streaming overlap: GPU-scope acquire load, release store, atomic increment, TMA global completion wait,
// and polite spin backoff.
CUTLASS_DEVICE uint32_t ld_acquire_gpu_u32(const uint32_t *ptr) {
    uint32_t val;
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
    return val;
}

CUTLASS_DEVICE void gmem_signal_release(uint32_t *ptr, uint32_t value) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(ptr), "r"(value) : "memory");
}

CUTLASS_DEVICE uint32_t gmem_atomic_add_u32(uint32_t *ptr, uint32_t inc) {
    uint32_t old;
    asm volatile("atom.global.acq_rel.gpu.add.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(ptr), "r"(inc) : "memory");
    return old;
}

// Wait until all previously committed TMA bulk stores from this thread have completed writing to global memory
// (unlike `cute::tma_store_wait<0>()`, which emits `.read` and only waits for SMEM reads to finish).
CUTLASS_DEVICE void tma_store_wait_global() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

CUTLASS_DEVICE void async_proxy_fence_global() {
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

CUTLASS_DEVICE void nanosleep_ns(uint32_t ns) {
    asm volatile("nanosleep.u32 %0;\n" ::"r"(ns));
}

}  // namespace tensorbridge::ptx
