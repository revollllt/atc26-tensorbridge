#pragma once

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../heuristics/sm90.hpp"
#include "./driver.h"

using tensorbridge::GemmConfig;
using tensorbridge::GroupedLayout;

inline void check_curesult(const CUresult res, const char *func_name) {
    if (res == CUDA_SUCCESS) return;
    const char *name, *message;
    driver().cuGetErrorName(res, &name);
    driver().cuGetErrorString(res, &message);
    TORCH_CHECK(false, func_name, " failed with error: ", name, " (", message, ")");
}

// SM resources of the current stream. `TENSORBRIDGE_NUM_SMS` may cap the worker budget.
inline int64_t get_num_sms(int64_t dev) {
    static const int64_t sm_limit = [] {
        const char *setting = std::getenv("TENSORBRIDGE_NUM_SMS");
        return setting ? std::max<int64_t>(1, std::atoi(setting)) : std::numeric_limits<int64_t>::max();
    }();
    const c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(dev));
    CUcontext context;
    CUdevResource resource{};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
    CUresult stream_context = driver().cuStreamGetCtx(stream, &context);
    if (stream_context == CUDA_ERROR_INVALID_CONTEXT) {
        const cudaError_t initialized = cudaSetDevice(static_cast<int>(dev));
        TORCH_CHECK(initialized == cudaSuccess, "cudaSetDevice: ", cudaGetErrorString(initialized));
        stream_context = driver().cuStreamGetCtx(stream, &context);
    }
    check_curesult(stream_context, "cuStreamGetCtx");
    check_curesult(driver().cuCtxGetDevResource(context, &resource, CU_DEV_RESOURCE_TYPE_SM), "cuCtxGetDevResource");
    return std::min<int64_t>(resource.sm.smCount, sm_limit);
}

// The device class of `device_name`: an unregistered GPU borrows the registered one
// with the closest SM count, with one warning per process.
inline const tensorbridge::sm90::DeviceClass &get_device_class(const std::string &device_name, int64_t num_sms) {
    const auto choice = tensorbridge::sm90::choose_device(device_name, num_sms);
    if (!choice.exact) {
        TORCH_WARN_ONCE("TensorBridge has no device class registered for ", device_name, "; using the ",
                        choice.device_class->name, " class of ", choice.device_class->device_name,
                        " (csrc/heuristics/sm90.hpp registers one)");
    }
    return *choice.device_class;
}

inline std::string device_class_for(const std::string &device_name, int64_t num_sms) {
    return get_device_class(device_name, num_sms).name;
}

struct DeviceInfo {
    int64_t l2_bytes, smem_bytes;
    const tensorbridge::sm90::DeviceClass *device_class;
};

inline const DeviceInfo &get_device_info(int32_t dev) {
    static std::mutex mutex;
    static std::map<int32_t, DeviceInfo> devices;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = devices.find(dev); it != devices.end()) return it->second;
    int32_t l2_bytes, smem_bytes;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev);
    cudaDeviceGetAttribute(&smem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, dev);
    return devices.emplace(dev, DeviceInfo{l2_bytes, smem_bytes,
                                            &get_device_class(properties.name, properties.multiProcessorCount)})
        .first->second;
}

inline std::pair<tensorbridge::GemmDesc, const tensorbridge::sm90::DeviceClass *> get_problem(
    int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t num_experts, int64_t layout, bool group_scales) {
    int32_t dev;
    cudaGetDevice(&dev);
    const DeviceInfo &info = get_device_info(dev);
    return {{m, n, k, num_sms, info.l2_bytes, info.smem_bytes, num_experts, static_cast<GroupedLayout>(layout),
             group_scales},
            info.device_class};
}

inline void check_tensor(const at::Tensor &tensor, const char *name, int64_t dev, at::ScalarType dtype,
                         std::vector<int64_t> shape) {
    TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == dev, name, " must be on the expected CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has dtype ", at::toString(tensor.scalar_type()),
                ", expected ", at::toString(dtype));
    TORCH_CHECK(tensor.dim() == static_cast<int64_t>(shape.size()), name, " has the wrong rank");
    for (size_t i = 0; i < shape.size(); i++) {
        TORCH_CHECK(tensor.size(i) == shape[i], name, ".size(", i, ") is ", tensor.size(i), ", expected ", shape[i]);
    }
}
