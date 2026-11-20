#pragma once

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

// The CUDA driver functions the launcher calls, taken from the driver that the CUDA runtime loaded
// instead of linking libcuda.so: where (and whether) a host keeps libcuda.so differs between
// clusters and linkers, while the runtime finds the driver wherever it runs.
// Each pointer has the ABI of the CUDA version in its typedef, which is the version it is requested at.
struct DriverApi {
    PFN_cuGetErrorName_v6000 cuGetErrorName;
    PFN_cuGetErrorString_v6000 cuGetErrorString;
    PFN_cuStreamGetCtx_v9020 cuStreamGetCtx;
    PFN_cuCtxGetDevResource_v12040 cuCtxGetDevResource;
    PFN_cuModuleLoad_v2000 cuModuleLoad;
    PFN_cuModuleGetFunction_v2000 cuModuleGetFunction;
    PFN_cuFuncSetAttribute_v9000 cuFuncSetAttribute;
    PFN_cuLaunchKernelEx_v11060 cuLaunchKernelEx;
    PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled;
};

// CUDA 12.4 / 12.5 Green Context entry points are loaded lazily so non-Green-Context GEMMs continue
// to work on older drivers that lack `cuGreenCtxStreamCreate` (CUDA 12.5).
struct GreenCtxDriverApi {
    PFN_cuDeviceGetDevResource_v12040 cuDeviceGetDevResource;
    PFN_cuDevSmResourceSplitByCount_v12040 cuDevSmResourceSplitByCount;
    PFN_cuDevResourceGenerateDesc_v12040 cuDevResourceGenerateDesc;
    PFN_cuGreenCtxCreate_v12040 cuGreenCtxCreate;
    PFN_cuGreenCtxStreamCreate_v12050 cuGreenCtxStreamCreate;
    PFN_cuGreenCtxDestroy_v12040 cuGreenCtxDestroy;
    PFN_cuStreamDestroy_v4000 cuStreamDestroy;
};

template <typename FnPtr>
inline void load_driver_symbol(FnPtr &func, const char *symbol, unsigned int cuda_version) {
    cudaDriverEntryPointQueryResult status;
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        symbol, reinterpret_cast<void **>(&func), cuda_version, cudaEnableDefault, &status);
    TORCH_CHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess,
                "CUDA driver entry point ", symbol, " (CUDA ", cuda_version, ") not found: ",
                cudaGetErrorString(err), ", query result ", static_cast<int>(status));
}

inline const DriverApi &driver() {
    static const DriverApi api = [] {
        DriverApi api;
        load_driver_symbol(api.cuGetErrorName, "cuGetErrorName", 6000);
        load_driver_symbol(api.cuGetErrorString, "cuGetErrorString", 6000);
        load_driver_symbol(api.cuStreamGetCtx, "cuStreamGetCtx", 9020);
        load_driver_symbol(api.cuCtxGetDevResource, "cuCtxGetDevResource", 12040);
        load_driver_symbol(api.cuModuleLoad, "cuModuleLoad", 2000);
        load_driver_symbol(api.cuModuleGetFunction, "cuModuleGetFunction", 2000);
        load_driver_symbol(api.cuFuncSetAttribute, "cuFuncSetAttribute", 9000);
        load_driver_symbol(api.cuLaunchKernelEx, "cuLaunchKernelEx", 11060);
        load_driver_symbol(api.cuTensorMapEncodeTiled, "cuTensorMapEncodeTiled", 12000);
        return api;
    }();
    return api;
}

inline const GreenCtxDriverApi &green_ctx_driver() {
    static const GreenCtxDriverApi api = [] {
        GreenCtxDriverApi api;
        load_driver_symbol(api.cuDeviceGetDevResource, "cuDeviceGetDevResource", 12040);
        load_driver_symbol(api.cuDevSmResourceSplitByCount, "cuDevSmResourceSplitByCount", 12040);
        load_driver_symbol(api.cuDevResourceGenerateDesc, "cuDevResourceGenerateDesc", 12040);
        load_driver_symbol(api.cuGreenCtxCreate, "cuGreenCtxCreate", 12040);
        load_driver_symbol(api.cuGreenCtxStreamCreate, "cuGreenCtxStreamCreate", 12050);
        load_driver_symbol(api.cuGreenCtxDestroy, "cuGreenCtxDestroy", 12040);
        load_driver_symbol(api.cuStreamDestroy, "cuStreamDestroy", 4000);
        return api;
    }();
    return api;
}
