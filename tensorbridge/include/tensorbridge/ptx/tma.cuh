#pragma once

#include <tensorbridge/common/base.cuh>

// Tensor Map Access: bulk copies between global and shared memory, described by
// a host-built `CUtensorMap` (see `csrc/launcher/tma.h`). A load completes a
// byte count on an mbarrier; the consumer passes the barrier once the producer's
// `expect_tx` total has arrived.

namespace tensorbridge::ptx {

TB_INLINE void tma_prefetch_descriptor(const void *desc_ptr) {
  const uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile("prefetch.tensormap [%0];"
               :
               : "l"(desc)
               : "memory");
}

// `kMulticast > 1` delivers the same tile to every CTA of the cluster.
template <uint32_t kMulticast = 1>
TB_INLINE void tma_load_2d(const void *desc_ptr, void *smem_ptr, void *mbar_ptr, uint32_t crd0, uint32_t crd1) {
  const uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
  const uint32_t mbar = smem_addr(mbar_ptr);
  const uint32_t dst = smem_addr(smem_ptr);

  if constexpr (kMulticast == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :
                 : "r"(dst), "l"(desc), "r"(mbar), "r"(crd0), "r"(crd1)
                 : "memory");
  } else {
    constexpr uint16_t cast_mask = (1 << kMulticast) - 1;
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%4, %5}], [%2], %3;"
                 :
                 : "r"(dst), "l"(desc), "r"(mbar), "h"(cast_mask), "r"(crd0), "r"(crd1)
                 : "memory");
  }
}

TB_INLINE void tma_expect_tx(void *mbar_ptr, uint32_t bytes) {
  const uint32_t mbar = smem_addr(mbar_ptr);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(mbar), "r"(bytes));
}

TB_INLINE void tma_store_2d(void *smem_ptr, const void *desc_ptr, uint32_t crd0, uint32_t crd1) {
  const uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
  const uint32_t src = smem_addr(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
               " [%0, {%2, %3}], [%1];"
               :
               : "l"(desc), "r"(src), "r"(crd0), "r"(crd1)
               : "memory");
}

// Store that adds into global memory; Stream-K slices accumulate with it.
TB_INLINE void tma_reduce_add_2d(void *smem_ptr, const void *desc_ptr, uint32_t crd0, uint32_t crd1) {
  const uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
  const uint32_t src = smem_addr(smem_ptr);
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
               " [%0, {%2, %3}], [%1];"
               :
               : "l"(desc), "r"(src), "r"(crd0), "r"(crd1)
               : "memory");
}

// TMA stores read shared memory through the async proxy; publish the writes
// made through the generic proxy before issuing one.
TB_INLINE void tma_fence_shared_writes() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <uint32_t kGroupsInFlight, bool kOnlyWaitRead = false>
TB_INLINE void tma_wait_store_group() {
  if constexpr (kOnlyWaitRead) {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(kGroupsInFlight));
  } else {
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(kGroupsInFlight));
  }
}

}  // namespace tensorbridge::ptx
