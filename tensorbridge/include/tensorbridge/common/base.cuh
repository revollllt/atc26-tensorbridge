#pragma once

#include <cstdint>
#include <cuda.h>

#define TB_STR(x) #x
#define TB_UNROLL _Pragma(TB_STR(unroll))
#define TB_INLINE __device__ __forceinline__

namespace tensorbridge {

constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
constexpr uint32_t min_of(uint32_t a, uint32_t b) { return a < b ? a : b; }

// PTX addresses shared memory with 32-bit offsets, not generic pointers.
template <typename T>
TB_INLINE uint32_t smem_addr(T *smem_ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

}  // namespace tensorbridge
