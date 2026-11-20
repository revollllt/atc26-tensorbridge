#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "./driver.h"

// Hardware SM partitioning via CUDA Green Contexts (Route B: GC-MoE).
// Splits a GPU into a dedicated communication partition and a remaining compute partition, each with its
// own CUgreenCtx and non-blocking CUstream. Kernels launched on `compute_stream` automatically report
// `compute_sms` via `get_num_sms` in `launcher.cpp` and select matching Route-A profiles.

namespace tensorbridge::green_ctx {

struct Partition {
    int64_t device;
    int64_t comm_sms;
    int64_t compute_sms;
    CUgreenCtx comm_ctx;
    CUgreenCtx compute_ctx;
    CUstream comm_stream;
    CUstream compute_stream;
};

inline std::mutex &partition_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<int64_t, Partition> &partitions() {
    static std::unordered_map<int64_t, Partition> map;
    return map;
}

inline void check_cu(CUresult res, const char *fn) {
    if (res == CUDA_SUCCESS) return;
    const char *name = nullptr, *msg = nullptr;
    driver().cuGetErrorName(res, &name);
    driver().cuGetErrorString(res, &msg);
    TORCH_CHECK(false, fn, " failed: ", name ? name : "?", " (", msg ? msg : "?", ")");
}

// Create a (comm, compute) Green Context partition on `device` with at least `min_comm_sms` SMs for comm.
// Returns `{partition_id, comm_stream_ptr, compute_stream_ptr, comm_sms, compute_sms}`.
inline std::vector<int64_t> create_partition(int64_t device, int64_t min_comm_sms) {
    TORCH_CHECK(min_comm_sms > 0, "min_comm_sms must be positive");
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    // Ensure primary context is initialized on this device before querying resources.
    cudaError_t err = cudaFree(nullptr);
    TORCH_CHECK(err == cudaSuccess, "cudaFree(0) failed: ", cudaGetErrorString(err));

    const CUdevice cu_dev = static_cast<CUdevice>(device);
    const auto &gc_api = green_ctx_driver();
    CUdevResource sm_resource{};
    check_cu(gc_api.cuDeviceGetDevResource(cu_dev, &sm_resource, CU_DEV_RESOURCE_TYPE_SM),
             "cuDeviceGetDevResource");
    TORCH_CHECK(min_comm_sms < sm_resource.sm.smCount, "min_comm_sms (", min_comm_sms,
                ") must be smaller than total device SMs (", sm_resource.sm.smCount, ")");

    CUdevResource comm_res{};
    CUdevResource compute_res{};
    unsigned int nb_groups = 1;
    check_cu(gc_api.cuDevSmResourceSplitByCount(&comm_res, &nb_groups, &sm_resource, &compute_res, 0,
                                                static_cast<unsigned int>(min_comm_sms)),
             "cuDevSmResourceSplitByCount");
    TORCH_CHECK(nb_groups == 1 && comm_res.sm.smCount > 0 && compute_res.sm.smCount > 0,
                "cuDevSmResourceSplitByCount returned empty partition (comm=", comm_res.sm.smCount,
                ", compute=", compute_res.sm.smCount, ")");

    CUdevResourceDesc comm_desc{}, compute_desc{};
    check_cu(gc_api.cuDevResourceGenerateDesc(&comm_desc, &comm_res, 1), "cuDevResourceGenerateDesc(comm)");
    check_cu(gc_api.cuDevResourceGenerateDesc(&compute_desc, &compute_res, 1),
             "cuDevResourceGenerateDesc(compute)");

    // RAII cleanup guard so partial failures never leak CUDA Green Contexts or streams.
    struct Cleanup {
        const GreenCtxDriverApi &api;
        CUgreenCtx comm_ctx = nullptr, compute_ctx = nullptr;
        CUstream comm_stream = nullptr, compute_stream = nullptr;
        bool committed = false;
        ~Cleanup() {
            if (committed) return;
            if (comm_stream) api.cuStreamDestroy(comm_stream);
            if (compute_stream) api.cuStreamDestroy(compute_stream);
            if (comm_ctx) api.cuGreenCtxDestroy(comm_ctx);
            if (compute_ctx) api.cuGreenCtxDestroy(compute_ctx);
        }
    } res{gc_api};

    check_cu(gc_api.cuGreenCtxCreate(&res.comm_ctx, comm_desc, cu_dev, CU_GREEN_CTX_DEFAULT_STREAM),
             "cuGreenCtxCreate(comm)");
    check_cu(gc_api.cuGreenCtxCreate(&res.compute_ctx, compute_desc, cu_dev, CU_GREEN_CTX_DEFAULT_STREAM),
             "cuGreenCtxCreate(compute)");

    check_cu(gc_api.cuGreenCtxStreamCreate(&res.comm_stream, res.comm_ctx, CU_STREAM_NON_BLOCKING, 0),
             "cuGreenCtxStreamCreate(comm)");
    check_cu(gc_api.cuGreenCtxStreamCreate(&res.compute_stream, res.compute_ctx, CU_STREAM_NON_BLOCKING, 0),
             "cuGreenCtxStreamCreate(compute)");

    std::lock_guard<std::mutex> lock(partition_mutex());
    static int64_t next_id = 1;
    const int64_t pid = next_id++;
    partitions()[pid] = Partition{
        device,
        static_cast<int64_t>(comm_res.sm.smCount),
        static_cast<int64_t>(compute_res.sm.smCount),
        res.comm_ctx,
        res.compute_ctx,
        res.comm_stream,
        res.compute_stream,
    };
    res.committed = true;
    return {
        pid,
        reinterpret_cast<int64_t>(res.comm_stream),
        reinterpret_cast<int64_t>(res.compute_stream),
        static_cast<int64_t>(comm_res.sm.smCount),
        static_cast<int64_t>(compute_res.sm.smCount),
    };
}

inline void destroy_partition(int64_t pid) {
    Partition p{};
    {
        std::lock_guard<std::mutex> lock(partition_mutex());
        auto it = partitions().find(pid);
        if (it == partitions().end()) return;
        p = it->second;
        partitions().erase(it);
    }
    try {
        const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(p.device));
        const auto &gc_api = green_ctx_driver();
        if (p.comm_stream) gc_api.cuStreamDestroy(p.comm_stream);
        if (p.compute_stream) gc_api.cuStreamDestroy(p.compute_stream);
        if (p.comm_ctx) gc_api.cuGreenCtxDestroy(p.comm_ctx);
        if (p.compute_ctx) gc_api.cuGreenCtxDestroy(p.compute_ctx);
    } catch (...) {
        // Ignore teardown errors during process/CUDA runtime shutdown.
    }
}

}  // namespace tensorbridge::green_ctx
