#define USE_CUDA 1

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/library.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <tensorbridge/common/params.h>

#include "../heuristics/sm90.hpp"
#include "./tma.h"

// Torch library `tensorbridge`: config selection, and loading and launching the
// kernels that the Python JIT compiled (`tensorbridge/jit`). Launches go to the
// current CUDA stream through the driver API, so they can be graph-captured.
// With `enable_pdl` a launch may start before the previous kernel on the stream
// has finished (programmatic dependent launch): the kernel waits for it before
// its first global memory access (`impls/sm90_w4a8_ptoken.cuh`).

using tensorbridge::GemmConfig;

using tensorbridge::GroupedLayout;

struct Kernel {
    CUfunction func;
    GemmConfig config;
    // The weight's shape: A has K columns and D has N; the kernel runs on them rounded up to multiples of 128, the
    // shape of the zero-padded weight (`quant/nvfp4.py`).
    int64_t n, k;
    uint32_t smem_bytes;
    int64_t num_experts;
    GroupedLayout layout;
    // Activation scales per token: 1 (`impls/sm90_w4a8_ptoken.cuh`), or one per 128-deep K block
    // (`impls/sm90_w4a8_pgroup.cuh`), M-major with M padded to 4 or K-major as compiled.
    int64_t activation_scale_groups;
    bool scale_m_major;
    // The weight scales by K blocks, each channel's 8 scale groups of a block in a row (`impls/sm90_w4a8_pgroup.cuh`,
    // `tensorbridge.experimental.prepare`).
    bool block_scales;
};

static std::unordered_map<int64_t, Kernel> g_kernels;

inline void check_curesult(const CUresult res, const char *func_name) {
    if (res == CUDA_SUCCESS) return;
    const char *name, *message;
    driver().cuGetErrorName(res, &name);
    driver().cuGetErrorString(res, &message);
    TORCH_CHECK(false, func_name, " failed with error: ", name, " (", message, ")");
}

// SM resources of the current stream, including a Green Context's partition.
// TENSORBRIDGE_NUM_SMS may lower the worker budget further; it does not partition SMs.
int64_t get_num_sms(int64_t dev) {
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
        // A new host thread can select device 0 before it has a current context.
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
const tensorbridge::sm90::DeviceClass &get_device_class(const std::string &device_name, int64_t num_sms) {
    const auto choice = tensorbridge::sm90::choose_device(device_name, num_sms);
    if (!choice.exact) {
        TORCH_WARN_ONCE("TensorBridge has no device class registered for ", device_name, "; using the ",
                        choice.device_class->name, " class of ", choice.device_class->device_name,
                        " (csrc/heuristics/sm90.hpp registers one)");
    }
    return *choice.device_class;
}

std::string device_class_for(const std::string &device_name, int64_t num_sms) {
    return get_device_class(device_name, num_sms).name;
}

// What selection needs to know of a device, read once per device.
struct DeviceInfo {
    int64_t l2_bytes, smem_bytes;
    const tensorbridge::sm90::DeviceClass *device_class;
};

const DeviceInfo &get_device_info(int32_t dev) {
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

// The problem on the current device: its L2 and shared memory, and its class.
std::pair<tensorbridge::GemmDesc, const tensorbridge::sm90::DeviceClass *> get_problem(
    int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t num_experts, int64_t layout, bool group_scales) {
    int32_t dev;
    cudaGetDevice(&dev);
    const DeviceInfo &info = get_device_info(dev);
    return {{m, n, k, num_sms, info.l2_bytes, info.smem_bytes, num_experts, static_cast<GroupedLayout>(layout),
             group_scales},
            info.device_class};
}

std::vector<int64_t> select_config(int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t stream_k_override,
                                   int64_t num_experts, int64_t layout, bool group_scales) {
    const auto [desc, device_class] = get_problem(m, n, k, num_sms, num_experts, layout, group_scales);
    return tensorbridge::sm90::select_config(desc, *device_class, stream_k_override).to_vector();
}

// Autotune's candidates (`sm90_rule.hpp`), flattened, `GemmConfig::to_vector()` each.
std::vector<int64_t> autotune_candidates(int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t count,
                                         int64_t num_experts, int64_t layout, bool group_scales) {
    const auto [desc, device_class] = get_problem(m, n, k, num_sms, num_experts, layout, group_scales);
    std::vector<int64_t> flat;
    for (const GemmConfig &config :
         tensorbridge::sm90::autotune_candidates(desc, device_class->rule_class(), count)) {
        const std::vector<int64_t> values = config.to_vector();
        flat.insert(flat.end(), values.begin(), values.end());
    }
    return flat;
}

// Full legal config enumeration (`sm90_rule.hpp`), flattened, `GemmConfig::to_vector()` each.
std::vector<int64_t> all_legal_configs(int64_t m, int64_t n, int64_t k, int64_t num_sms, bool check_clean,
                                       int64_t num_experts, int64_t layout, bool group_scales) {
    const auto [desc, device_class] = get_problem(m, n, k, num_sms, num_experts, layout, group_scales);
    (void)device_class;
    std::vector<int64_t> flat;
    for (const GemmConfig &config : tensorbridge::sm90::all_legal_configs(desc, check_clean)) {
        const std::vector<int64_t> values = config.to_vector();
        flat.insert(flat.end(), values.begin(), values.end());
    }
    return flat;
}

// Why `config` cannot run a weight of `(n, k)` on a device with `smem_bytes` of
// opt-in shared memory per block, or "" if it can;
// without `check_clean`, only the resources count, as the census asks.
std::string illegal_reason(int64_t n, int64_t k, int64_t smem_bytes, std::vector<int64_t> config, bool check_clean,
                             int64_t num_experts, int64_t layout, bool group_scales) {
    const char *reason = tensorbridge::sm90::illegal_reason(
        GemmConfig::from_vector(config),
        {1, n, k, 1, 0, smem_bytes, num_experts, static_cast<GroupedLayout>(layout), group_scales}, check_clean);
    return reason ? reason : "";
}

int64_t clean_aligned_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t block_k, int64_t num_stages) {
    return tensorbridge::sm90::clean_aligned_block_ms(ctas_per_sm, block_n, block_k, num_stages);
}

int64_t clean_one_warpgroup_block_ms(int64_t block_k, int64_t num_stages) {
    return tensorbridge::sm90::clean_one_warpgroup_block_ms(block_k, num_stages);
}

int64_t clean_pgroup_one_warpgroup_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t num_stages) {
    return tensorbridge::sm90::clean_pgroup_one_warpgroup_block_ms(ctas_per_sm, block_n, num_stages);
}

int64_t clean_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t block_k, bool use_stream_k, int64_t layout,
                       int64_t num_stages, bool group_scales) {
    return tensorbridge::sm90::clean_block_ms(ctas_per_sm, block_n, block_k, use_stream_k,
                                              static_cast<GroupedLayout>(layout), num_stages, group_scales);
}

// The selector's account of a kernel's shared memory, which the census checks against the cubin's.
int64_t config_smem_bytes(std::vector<int64_t> config, int64_t num_experts, int64_t layout, bool group_scales) {
    return tensorbridge::sm90::get_smem_bytes(GemmConfig::from_vector(config), num_experts,
                                              static_cast<GroupedLayout>(layout), group_scales);
}

// `smem_bytes` is the kernel's shared-memory footprint, which the JIT reads
// back from the cubin; `shape_n` and `shape_k` are the weight's shape (`Kernel::n`, `Kernel::k`).
int64_t register_kernel(
    const std::string &cubin_path, const std::string &func_name,
    int64_t shape_n, int64_t shape_k, std::vector<int64_t> config, int64_t smem_bytes,
    int64_t num_experts, int64_t layout, int64_t activation_scale_groups, bool scale_m_major, bool block_scales) {
    CUmodule module;
    CUfunction func;
    check_curesult(driver().cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
    check_curesult(driver().cuModuleGetFunction(&func, module, func_name.c_str()), "cuModuleGetFunction");

    const int64_t kernel_id = static_cast<int64_t>(g_kernels.size());
    g_kernels[kernel_id] = {func, GemmConfig::from_vector(config), shape_n, shape_k, static_cast<uint32_t>(smem_bytes),
                            num_experts, static_cast<GroupedLayout>(layout), activation_scale_groups, scale_m_major,
                            block_scales};
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

// D[M, N] = (A[M, K] * a_scale[M, 1 or K / 128]) @ dequant(B, sfb).T * global_scale, see
// `include/tensorbridge/common/params.h` for the layouts.
// `enable_pdl` pays off only when the previous kernel on the stream lets its
// dependents launch early (`tensorbridge.quant.fp8.quantize` does); after any
// other kernel it costs about half a microsecond.
at::Tensor launch_kernel_impl(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale, at::Tensor locks,
    std::optional<at::Tensor> d_, bool enable_pdl,
    std::optional<at::Tensor> sorted_ids, std::optional<at::Tensor> expert_ids,
    std::optional<at::Tensor> num_tokens_padded, std::optional<at::Tensor> expert_layout, int64_t top_k) {
    TORCH_CHECK(g_kernels.find(kernel_id) != g_kernels.end(), "Unknown kernel id ", kernel_id);
    const Kernel &kernel = g_kernels[kernel_id];
    const GemmConfig &config = kernel.config;

    const int64_t dev = a.get_device();
    // Persistent workers: as many CTAs per SM as the kernel was compiled for.
    const int64_t worker_ctas = get_num_sms(dev) * config.ctas_per_sm;
    const int64_t m = a.size(0), n = kernel.n, k = kernel.k;
    const int64_t padded_n = tensorbridge::ceil_div(n, tensorbridge::kBlockN) * tensorbridge::kBlockN;
    const int64_t padded_k = tensorbridge::ceil_div(k, tensorbridge::kBlockK) * tensorbridge::kBlockK;
    const bool indexed = kernel.layout == GroupedLayout::Indexed;
    TORCH_CHECK(top_k >= 1, "top_k must be positive");
    const int64_t output_rows = indexed ? m * top_k : m;
    at::Tensor d = d_.has_value() ? d_.value() : at::empty({output_rows, n}, a.options().dtype(at::ScalarType::BFloat16));

    check_tensor(a, "a", dev, at::ScalarType::Float8_e4m3fn, {m, k});
    if (kernel.block_scales && kernel.layout != GroupedLayout::Dense) {
        TORCH_CHECK(a_scale.is_cuda() && a_scale.get_device() == dev && a_scale.scalar_type() == at::ScalarType::Float,
                    "a_scale must be FP32 on the device of a");
        if (a_scale.dim() == 3 && kernel.layout == GroupedLayout::Masked) {
            TORCH_CHECK(a_scale.size(0) == kernel.num_experts && a_scale.size(1) * kernel.num_experts == m &&
                        a_scale.size(2) == kernel.activation_scale_groups, "masked scales must be [E, capacity, K/128]");
        } else {
            TORCH_CHECK(a_scale.dim() == 2 && a_scale.size(0) == m &&
                        a_scale.size(1) == kernel.activation_scale_groups, "scales must be [M, K/128]");
        }
    } else if (kernel.activation_scale_groups == 1) {
        check_tensor(a_scale, "a_scale", dev, at::ScalarType::Float, {m, 1});
    } else if (kernel.scale_m_major) {
        TORCH_CHECK(a_scale.is_cuda() && a_scale.get_device() == dev && a_scale.scalar_type() == at::ScalarType::Float,
                    "a_scale must be FP32 on the device of `a`");
        TORCH_CHECK(a_scale.dim() == 2 && a_scale.size(0) == m && a_scale.size(1) == kernel.activation_scale_groups,
                    "a_scale must have shape [M, K / 128]");
        TORCH_CHECK(a_scale.stride(0) == 1 && a_scale.stride(1) == (m + 3) / 4 * 4,
                    "a_scale must be M-major with M padded to a multiple of 4, as the kernel was compiled");
    } else {
        check_tensor(a_scale, "a_scale", dev, at::ScalarType::Float, {m, kernel.activation_scale_groups});
    }
    if (kernel.layout == GroupedLayout::Dense) {
        check_tensor(b, "b", dev, at::ScalarType::Int, {padded_n, padded_k / 8});
        if (kernel.block_scales) {
            check_tensor(sfb, "sfb", dev, at::ScalarType::Float8_e4m3fn, {padded_k / 128, padded_n, 8});
        } else {
            check_tensor(sfb, "sfb", dev, at::ScalarType::Float8_e4m3fn, {padded_k / 16, padded_n});
        }
        check_tensor(global_scale, "global_scale", dev, at::ScalarType::Float, {1});
    } else {
        const int64_t experts = kernel.num_experts;
        check_tensor(b, "b", dev, at::ScalarType::Int, {experts, padded_n, padded_k / 8});
        if (kernel.block_scales) {
            check_tensor(sfb, "sfb", dev, at::ScalarType::Float8_e4m3fn, {experts, padded_k / 128, padded_n, 8});
        } else {
            check_tensor(sfb, "sfb", dev, at::ScalarType::Float8_e4m3fn, {experts, padded_k / 16, padded_n});
        }
        check_tensor(global_scale, "global_scale", dev, at::ScalarType::Float, {experts});
        if (indexed) {
            TORCH_CHECK(sorted_ids && expert_ids && num_tokens_padded, "indexed routing needs IDs and padded count");
            check_tensor(*sorted_ids, "sorted_ids", dev, at::ScalarType::Int, {sorted_ids->numel()});
            check_tensor(*expert_ids, "expert_ids", dev, at::ScalarType::Int, {expert_ids->numel()});
            check_tensor(*num_tokens_padded, "num_tokens_padded", dev, at::ScalarType::Int, {1});
            TORCH_CHECK(expert_ids->numel() >= (sorted_ids->numel() + config.block_m - 1) / config.block_m,
                        "indexed routing must provide one expert per aligned block");
        } else {
            TORCH_CHECK(expert_layout.has_value(), "grouped routing needs expert offsets or counts");
            if (kernel.layout == GroupedLayout::Contiguous) {
                check_tensor(*expert_layout, "expert_offsets", dev, at::ScalarType::Long, {experts + 1});
            } else {
                check_tensor(*expert_layout, "expert_counts", dev, at::ScalarType::Int, {experts});
                TORCH_CHECK(m % experts == 0, "masked activation rows must divide into equal expert capacities");
            }
        }
    }
    check_tensor(d, "d", dev, at::ScalarType::BFloat16, {output_rows, n});
    TORCH_CHECK(locks.is_cuda() && locks.scalar_type() == at::ScalarType::Int, "locks must be an int32 CUDA tensor");
    if (m == 0) return d;

    // Boxes: A in 128B rows of `block_m` tokens, B in 64B rows of 128 channels
    // (counted in int32 words), D in 64-channel BF16 rows of `block_m` tokens. A's K and D's N columns
    // bound their maps: a box of A past K is filled with zeros, which meet the zero weights of the
    // padding, and a box of D past N is not stored. TMA strides are multiples of 16 bytes, so K is a
    // multiple of 16 and N of 8 (`quant/nvfp4.prepare` checks).
    const auto block_m = static_cast<uint32_t>(config.block_m);
    CUtensorMap tensor_map_a = make_tma_desc(a, 128, block_m, 128);
    CUtensorMap tensor_map_b = make_tma_desc(b.view({-1, padded_k / 8}), 16, 128, 64);
    // A tile of a multiple of 128 rows is stored in chunks of 128 (`layout::PtokenRing::kOutputRows`).
    CUtensorMap tensor_map_d = make_tma_desc(d, 64, block_m % 128 == 0 ? 128 : block_m, 128);
    // The weight scales of one K block: 8 scale groups x 128 channels, unswizzled; by K blocks, 128 channels of one
    // 64-bit row each.
    CUtensorMap tensor_map_sfb = kernel.block_scales
                                     ? make_tma_desc(sfb.view(at::ScalarType::Long).view({-1, padded_n}), 128, 1, 0)
                                     : make_tma_desc(sfb.view({-1, padded_n}), 128, 8, 0);

    const uint32_t shape_m = static_cast<uint32_t>(m);
    void *offsets_ptr = kernel.layout == GroupedLayout::Contiguous ? expert_layout->data_ptr() : nullptr;
    void *counts_ptr = kernel.layout == GroupedLayout::Masked ? expert_layout->data_ptr() : nullptr;
    void *sorted_ptr = indexed ? sorted_ids->data_ptr() : nullptr;
    void *experts_ptr = indexed ? expert_ids->data_ptr() : nullptr;
    void *padded_ptr = indexed ? num_tokens_padded->data_ptr() : nullptr;
    void *activations_ptr = a.data_ptr();
    void *output_ptr = d.data_ptr();
    const uint32_t num_routes = static_cast<uint32_t>(output_rows);
    const uint32_t route_top_k = static_cast<uint32_t>(top_k);
    // The Stream-K split, as the selector models it: the share of the tail's K blocks per CTA, or
    // with aligned slices the tail tiles and their slices (one slice leaves the tiles whole).
    tensorbridge::StreamKSplit stream_k = {};
    if (config.use_stream_k && kernel.layout == GroupedLayout::Dense) {
        const auto schedule = tensorbridge::sm90::get_schedule({m, padded_n, padded_k, get_num_sms(dev), 0, 0}, config);
        const int64_t num_ctas = schedule.num_groups;
        stream_k.dp_tiles_per_cta = static_cast<uint32_t>(schedule.dp_tiles);
        if (config.stream_k_aligned) {
            stream_k.tail_divisor = static_cast<uint32_t>(schedule.tail_tiles);
            stream_k.aligned_slices = static_cast<uint32_t>(
                schedule.tail_tiles ? std::min({num_ctas / schedule.tail_tiles, schedule.k_blocks,
                                                tensorbridge::sm90::kMaxBf16StreamKSlices})
                                    : 0);
        } else {
            stream_k.tail_divisor = static_cast<uint32_t>(schedule.stream_k ? schedule.share : 0);
        }
        // Round-up reciprocal of the divisor scaled by 2^(31 + ceil(log2 divisor)), as CUTLASS's `FastDivmod`.
        if (stream_k.tail_divisor > 1) {
            const uint32_t log2_divisor = 32 - __builtin_clz(stream_k.tail_divisor - 1);
            const uint64_t scale = uint64_t(1) << (31 + log2_divisor);
            stream_k.divisor_multiplier =
                static_cast<uint32_t>((scale + stream_k.tail_divisor - 1) / stream_k.tail_divisor);
            stream_k.divisor_shift = log2_divisor - 1;
        }
    }
    tensorbridge::GemmParams params = {
            tensor_map_a, tensor_map_b, tensor_map_sfb, tensor_map_d,
            a_scale.data_ptr<float>(), global_scale.data_ptr<float>(), locks.data_ptr<int32_t>(),
            static_cast<uint32_t>(a_scale.stride(a_scale.dim() - 2)),
            static_cast<uint32_t>(a_scale.stride(a_scale.dim() - 1)),
            static_cast<const int64_t *>(offsets_ptr), static_cast<const int32_t *>(counts_ptr),
            static_cast<const int32_t *>(sorted_ptr), static_cast<const int32_t *>(experts_ptr),
            static_cast<const int32_t *>(padded_ptr), static_cast<const uint8_t *>(activations_ptr), output_ptr,
            shape_m, num_routes, route_top_k, stream_k,
            static_cast<uint64_t>(a_scale.dim() == 3 ? a_scale.stride(0) :
                                 kernel.num_experts ? m / kernel.num_experts * a_scale.stride(0) : 0)};
    void *params_args[] = {&params};

    CUlaunchConfig launch = {};
    launch.gridDimX = static_cast<uint32_t>(worker_ctas);
    launch.gridDimY = launch.gridDimZ = 1;
    // The math warp groups and the producer's (`impls/sm90_w4a8_ptoken.cuh`; the group-scale kernel has two math ones).
    launch.blockDimX = static_cast<uint32_t>(128 * (config.math_warpgroups + 1));
    launch.blockDimY = launch.blockDimZ = 1;
    launch.sharedMemBytes = kernel.smem_bytes;
    launch.hStream = at::cuda::getCurrentCUDAStream(dev);

    CUlaunchAttribute attrs[1];
    uint32_t num_attrs = 0;
    if (enable_pdl) {
        attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attrs[num_attrs].value.programmaticStreamSerializationAllowed = 1;
        num_attrs++;
    }
    launch.attrs = attrs;
    launch.numAttrs = num_attrs;

    CUfunction func = kernel.func;
    check_curesult(
        driver().cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel.smem_bytes),
        "cuFuncSetAttribute");
    check_curesult(driver().cuLaunchKernelEx(&launch, func, params_args, nullptr),
                   "cuLaunchKernelEx");
    return d;
}

at::Tensor launch_kernel(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale, at::Tensor locks,
    std::optional<at::Tensor> d, bool enable_pdl) {
    return launch_kernel_impl(kernel_id, a, a_scale, b, sfb, global_scale, locks, d, enable_pdl,
                              std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1);
}

at::Tensor launch_grouped_kernel(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale, at::Tensor locks,
    std::optional<at::Tensor> sorted_ids, std::optional<at::Tensor> expert_ids,
    std::optional<at::Tensor> num_tokens_padded, std::optional<at::Tensor> expert_layout, int64_t top_k,
    std::optional<at::Tensor> d, bool enable_pdl) {
    return launch_kernel_impl(kernel_id, a, a_scale, b, sfb, global_scale, locks, d, enable_pdl,
                              sorted_ids, expert_ids, num_tokens_padded, expert_layout, top_k);
}

TORCH_LIBRARY(tensorbridge, m) {
    m.def("select_config(int m, int n, int k, int num_sms, int stream_k_override, int num_experts=0, int layout=0, "
          "bool group_scales=False) -> int[]");
    m.def("get_num_sms(int device) -> int");
    m.def("autotune_candidates(int m, int n, int k, int num_sms, int count, int num_experts=0, int layout=0, "
          "bool group_scales=False) -> int[]");
    m.def("all_legal_configs(int m, int n, int k, int num_sms, bool check_clean=True, int num_experts=0, int layout=0, "
          "bool group_scales=False) -> int[]");
    m.def("illegal_reason(int n, int k, int smem_bytes, int[] config, bool check_clean=True, int num_experts=0, "
          "int layout=0, bool group_scales=False) -> str");
    m.def("clean_block_ms(int ctas_per_sm, int block_n, int block_k, bool use_stream_k, int layout=0, "
          "int num_stages=0, bool group_scales=False) -> int");
    m.def("clean_aligned_block_ms(int ctas_per_sm, int block_n, int block_k, int num_stages) -> int");
    m.def("clean_one_warpgroup_block_ms(int block_k, int num_stages) -> int");
    m.def("clean_pgroup_one_warpgroup_block_ms(int ctas_per_sm, int block_n, int num_stages) -> int");
    m.def("config_smem_bytes(int[] config, int num_experts=0, int layout=0, bool group_scales=False) -> int");
    m.def("device_class_for(str device_name, int num_sms) -> str");
    m.def("register_kernel(str cubin_path, str func_name, int shape_n, int shape_k, int[] config, int smem_bytes, "
          "int num_experts=0, int layout=0, int activation_scale_groups=1, bool scale_m_major=False, "
          "bool block_scales=False) -> int");
    m.def("launch_kernel(int kernel_id, Tensor a, Tensor a_scale, Tensor b, Tensor sfb, Tensor global_scale, "
          "Tensor locks, Tensor? d, bool enable_pdl=False) -> Tensor");
    m.def("launch_grouped_kernel(int kernel_id, Tensor a, Tensor a_scale, Tensor b, Tensor sfb, Tensor global_scale, "
          "Tensor locks, Tensor? sorted_ids, Tensor? expert_ids, Tensor? num_tokens_padded, Tensor? expert_layout, "
          "int top_k, Tensor? d=None, bool enable_pdl=False) -> Tensor");
};

TORCH_LIBRARY_IMPL(tensorbridge, CUDA, m) {
    m.impl("launch_kernel", &launch_kernel);
    m.impl("launch_grouped_kernel", &launch_grouped_kernel);
};

TORCH_LIBRARY_IMPL(tensorbridge, Undefined, m) {
    m.impl("select_config", &select_config);
    m.impl("get_num_sms", &get_num_sms);
    m.impl("autotune_candidates", &autotune_candidates);
    m.impl("all_legal_configs", &all_legal_configs);
    m.impl("illegal_reason", &illegal_reason);
    m.impl("clean_block_ms", &clean_block_ms);
    m.impl("clean_aligned_block_ms", &clean_aligned_block_ms);
    m.impl("clean_one_warpgroup_block_ms", &clean_one_warpgroup_block_ms);
    m.impl("clean_pgroup_one_warpgroup_block_ms", &clean_pgroup_one_warpgroup_block_ms);
    m.impl("config_smem_bytes", &config_smem_bytes);
    m.impl("device_class_for", &device_class_for);
    m.impl("register_kernel", &register_kernel);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_num_sms", &get_num_sms);
}
