#pragma once

#include <ATen/core/Tensor.h>
#include <cuda.h>
#include <vector>

// Host side of Tensor Map Access: a `CUtensorMap` tells the device how a 2D
// tensor in global memory is cut into shared-memory boxes, so that a kernel can
// move one box with a single instruction and two coordinates.

inline CUtensorMapDataType get_tma_dtype(at::ScalarType type) {
  switch (type) {
    case at::ScalarType::BFloat16: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case at::ScalarType::Int: return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case at::ScalarType::Float8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    default: TORCH_CHECK(false, "Unsupported torch dtype for TMA");
  }
}

inline CUtensorMapSwizzle get_tma_swizzle(uint32_t swizzle_bytes) {
  if (swizzle_bytes == 64) return CU_TENSOR_MAP_SWIZZLE_64B;
  if (swizzle_bytes == 128) return CU_TENSOR_MAP_SWIZZLE_128B;
  TORCH_CHECK(false, "Swizzle bytes must be 64 or 128");
}

// `box_inner` x `box_outer` elements per box (inner = the tensor's last
// dimension); one box row must be exactly `swizzle_bytes` wide.
inline CUtensorMap make_tma_desc(const at::Tensor &tensor, uint32_t box_inner, uint32_t box_outer, uint32_t swizzle_bytes) {
  TORCH_CHECK(tensor.dim() == 2, "TMA descriptors are built over 2D tensors");
  TORCH_CHECK(box_inner * tensor.element_size() == swizzle_bytes, "A box row must fill one swizzle row");

  // TMA orders dimensions innermost first.
  std::vector<uint64_t> gmem_dims = {static_cast<uint64_t>(tensor.size(1)), static_cast<uint64_t>(tensor.size(0))};
  std::vector<uint64_t> gmem_strides = {static_cast<uint64_t>(tensor.stride(0) * tensor.element_size())};
  std::vector<uint32_t> smem_dims = {box_inner, box_outer};
  std::vector<uint32_t> element_strides = {1, 1};

  CUtensorMap tmap;
  CUresult res = cuTensorMapEncodeTiled(
      &tmap,
      get_tma_dtype(tensor.scalar_type()),
      2,
      tensor.data_ptr(),
      gmem_dims.data(),
      gmem_strides.data(),
      smem_dims.data(),
      element_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      get_tma_swizzle(swizzle_bytes),
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    const char *message;
    cuGetErrorString(res, &message);
    TORCH_CHECK(false, "cuTensorMapEncodeTiled failed: ", message);
  }
  return tmap;
}
