#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>
#include <tensorbridge/common/megamoe_params.h>

namespace tensorbridge::nvlink_ep {

CUTLASS_DEVICE void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;\n" ::: "memory");
}

CUTLASS_DEVICE uint32_t ld_acquire_sys_u32(const uint32_t *ptr) {
    uint32_t v;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

CUTLASS_DEVICE void st_release_sys_u32(uint32_t *ptr, uint32_t v) {
    asm volatile("st.release.sys.global.b32 [%0], %1;\n" ::"l"(ptr), "r"(v) : "memory");
}

CUTLASS_DEVICE uint32_t atomic_add_sys_u32(uint32_t *ptr, uint32_t inc) {
    uint32_t old;
    asm volatile("atom.global.acq_rel.sys.add.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(ptr), "r"(inc) : "memory");
    return old;
}

template <typename T>
CUTLASS_DEVICE T *symm_ptr(const MegaMoeParams &p, uint32_t rank, uint64_t byte_off) {
    return reinterpret_cast<T *>(p.peer_base[rank] + byte_off);
}

}  // namespace tensorbridge::nvlink_ep
