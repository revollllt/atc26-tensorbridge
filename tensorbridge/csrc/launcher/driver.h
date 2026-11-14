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

inline const DriverApi &driver() {
    static const DriverApi api = [] {
        auto load = [](auto &func, const char *symbol, unsigned int cuda_version) {
            cudaDriverEntryPointQueryResult status;
            const cudaError_t err = cudaGetDriverEntryPointByVersion(
                symbol, reinterpret_cast<void **>(&func), cuda_version, cudaEnableDefault, &status);
            TORCH_CHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess,
                        "CUDA driver entry point ", symbol, " (CUDA ", cuda_version, ") not found: ",
                        cudaGetErrorString(err), ", query result ", static_cast<int>(status));
        };
        DriverApi api;
        load(api.cuGetErrorName, "cuGetErrorName", 6000);
        load(api.cuGetErrorString, "cuGetErrorString", 6000);
        load(api.cuStreamGetCtx, "cuStreamGetCtx", 9020);
        load(api.cuCtxGetDevResource, "cuCtxGetDevResource", 12040);
        load(api.cuModuleLoad, "cuModuleLoad", 2000);
        load(api.cuModuleGetFunction, "cuModuleGetFunction", 2000);
        load(api.cuFuncSetAttribute, "cuFuncSetAttribute", 9000);
        load(api.cuLaunchKernelEx, "cuLaunchKernelEx", 11060);
        load(api.cuTensorMapEncodeTiled, "cuTensorMapEncodeTiled", 12000);
        return api;
    }();
    return api;
}
