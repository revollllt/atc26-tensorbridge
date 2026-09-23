#define USE_CUDA 1

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/library.h>
#include <unordered_map>

#include "../heuristics/sm90.hpp"
#include "./tma.h"

// Torch library `tensorbridge`: config selection, and loading and launching the
// kernels that the Python JIT compiled (`tensorbridge/jit`). Launches go to the
// current CUDA stream through the driver API, so they can be graph-captured.

using tensorbridge::GemmConfig;

struct Kernel {
  CUfunction func;
  GemmConfig config;
  int64_t shape_n, shape_k;
  uint32_t smem_bytes;
};

static std::unordered_map<int64_t, Kernel> g_kernels;

inline void check_curesult(const CUresult res, const char *func_name) {
  if (res == CUDA_SUCCESS) return;
  const char *name, *message;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &message);
  TORCH_CHECK(false, func_name, " failed with error: ", name, " (", message, ")");
}

int64_t get_num_sms(int64_t dev) {
  int32_t num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
  return num_sms;
}

std::vector<int64_t> select_config(int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t stream_k_override) {
  return tensorbridge::sm90::select_config({m, n, k, num_sms}, stream_k_override).to_vector();
}

// `smem_bytes` is the kernel's shared-memory footprint, which the JIT reads
// back from the cubin.
int64_t register_kernel(
    const std::string &cubin_path, const std::string &func_name,
    int64_t shape_n, int64_t shape_k, std::vector<int64_t> config, int64_t smem_bytes) {
  CUmodule module;
  CUfunction func;
  check_curesult(cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
  check_curesult(cuModuleGetFunction(&func, module, func_name.c_str()), "cuModuleGetFunction");

  const int64_t kernel_id = static_cast<int64_t>(g_kernels.size());
  g_kernels[kernel_id] = {func, GemmConfig::from_vector(config), shape_n, shape_k, static_cast<uint32_t>(smem_bytes)};
  return kernel_id;
}

inline void check_tensor(const at::Tensor &tensor, const char *name, int64_t dev, at::ScalarType dtype, std::vector<int64_t> shape) {
  TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == dev, name, " must be on the device of `a`");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has dtype ", at::toString(tensor.scalar_type()),
               ", expected ", at::toString(dtype));
  TORCH_CHECK(tensor.dim() == static_cast<int64_t>(shape.size()), name, " has the wrong rank");
  for (size_t i = 0; i < shape.size(); i++) {
    TORCH_CHECK(tensor.size(i) == shape[i], name, ".size(", i, ") is ", tensor.size(i), ", expected ", shape[i]);
  }
}

// D[M, N] = (A[M, K] * sfa[M, 1]) @ dequant(B, sfb).T * global_scale, see
// `include/tensorbridge/impls/sm90_fp8_nvfp4_gemm.cuh` for the layouts.
at::Tensor launch_kernel(
    int64_t kernel_id, at::Tensor a, at::Tensor sfa, at::Tensor b, at::Tensor sfb, at::Tensor global_scale, at::Tensor locks,
    std::optional<at::Tensor> d_) {
  TORCH_CHECK(g_kernels.find(kernel_id) != g_kernels.end(), "Unknown kernel id ", kernel_id);
  const Kernel &kernel = g_kernels[kernel_id];
  const GemmConfig &config = kernel.config;

  const int64_t dev = a.get_device();
  const int64_t m = a.size(0), n = kernel.shape_n, k = kernel.shape_k;
  at::Tensor d = d_.has_value() ? d_.value() : at::empty({m, n}, a.options().dtype(at::ScalarType::BFloat16));

  check_tensor(a, "a", dev, at::ScalarType::Float8_e4m3fn, {m, k});
  check_tensor(sfa, "sfa", dev, at::ScalarType::Float, {m, 1});
  check_tensor(b, "b", dev, at::ScalarType::Int, {n, k / 8});
  check_tensor(sfb, "sfb", dev, at::ScalarType::Float8_e4m3fn, {k / 16, n});
  check_tensor(global_scale, "global_scale", dev, at::ScalarType::Float, {1});
  check_tensor(d, "d", dev, at::ScalarType::BFloat16, {m, n});
  TORCH_CHECK(locks.is_cuda() && locks.scalar_type() == at::ScalarType::Int, "locks must be an int32 CUDA tensor");

  // Boxes: A in 128B rows of `block_m` tokens, B in 64B rows of 128 channels
  // (counted in int32 words), D in 64-channel BF16 rows of `block_m` tokens.
  const auto block_m = static_cast<uint32_t>(config.block_m);
  CUtensorMap tensor_map_a = make_tma_desc(a, 128, block_m, 128);
  CUtensorMap tensor_map_b = make_tma_desc(b, 16, 128, 64);
  CUtensorMap tensor_map_d = config.use_tma_store ? make_tma_desc(d, 64, block_m, 128) : CUtensorMap();

  void *d_ptr = d.data_ptr();
  void *sfa_ptr = sfa.data_ptr();
  void *sfb_ptr = sfb.data_ptr();
  void *global_scale_ptr = global_scale.data_ptr();
  void *locks_ptr = locks.data_ptr();
  uint32_t shape_m = static_cast<uint32_t>(m);
  void *kernel_args[] = {
      &tensor_map_a,
      &tensor_map_b,
      config.use_tma_store ? static_cast<void *>(&tensor_map_d) : static_cast<void *>(&d_ptr),
      &sfa_ptr,
      &sfb_ptr,
      &global_scale_ptr,
      &locks_ptr,
      &shape_m};

  // One CTA per SM; a multicast cluster is a group of adjacent CTAs.
  CUlaunchConfig launch = {};
  launch.gridDimX = static_cast<uint32_t>(get_num_sms(dev));
  launch.gridDimY = launch.gridDimZ = 1;
  launch.blockDimX = 384;  // 256 consumer + 128 producer threads, `common/tile.cuh`
  launch.blockDimY = launch.blockDimZ = 1;
  launch.sharedMemBytes = kernel.smem_bytes;
  launch.hStream = at::cuda::getCurrentCUDAStream(dev);

  CUlaunchAttribute attrs[1];
  if (config.cluster_size() > 1) {
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[0].value.clusterDim.x = static_cast<uint32_t>(config.cluster_size());
    attrs[0].value.clusterDim.y = attrs[0].value.clusterDim.z = 1;
    launch.attrs = attrs;
    launch.numAttrs = 1;
  }

  CUfunction func = kernel.func;
  check_curesult(
      cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel.smem_bytes),
      "cuFuncSetAttribute");
  check_curesult(cuLaunchKernelEx(&launch, func, kernel_args, nullptr), "cuLaunchKernelEx");
  return d;
}

TORCH_LIBRARY(tensorbridge, m) {
  m.def("select_config(int m, int n, int k, int num_sms, int stream_k_override) -> int[]");
  m.def("get_num_sms(int device) -> int");
  m.def("register_kernel(str cubin_path, str func_name, int shape_n, int shape_k, int[] config, int smem_bytes) -> int");
  m.def("launch_kernel(int kernel_id, Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor global_scale, "
        "Tensor locks, Tensor? d) -> Tensor");
};

TORCH_LIBRARY_IMPL(tensorbridge, CUDA, m) {
  m.impl("launch_kernel", &launch_kernel);
};

TORCH_LIBRARY_IMPL(tensorbridge, Undefined, m) {
  m.impl("select_config", &select_config);
  m.impl("get_num_sms", &get_num_sms);
  m.impl("register_kernel", &register_kernel);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){};
