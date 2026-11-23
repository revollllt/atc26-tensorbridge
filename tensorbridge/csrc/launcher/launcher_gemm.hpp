#pragma once

#include <ATen/ops/empty.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <tensorbridge/common/params.h>

#include "./launcher_common.hpp"
#include "./tma.h"

struct Kernel {
    CUfunction func;
    GemmConfig config;
    int64_t n, k;
    uint32_t smem_bytes;
    int64_t num_experts;
    GroupedLayout layout;
    int64_t activation_scale_groups;
    bool scale_m_major;
    bool block_scales;
    bool fuse_swiglu_fp8;
};

static std::unordered_map<int64_t, Kernel> g_kernels;

inline std::vector<int64_t> select_config(int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t stream_k_override,
                                          int64_t num_experts, int64_t layout, bool group_scales) {
    const auto [desc, device_class] = get_problem(m, n, k, num_sms, num_experts, layout, group_scales);
    return tensorbridge::sm90::select_config(desc, *device_class, stream_k_override).to_vector();
}

inline std::vector<int64_t> autotune_candidates(int64_t m, int64_t n, int64_t k, int64_t num_sms, int64_t count,
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

inline std::vector<int64_t> all_legal_configs(int64_t m, int64_t n, int64_t k, int64_t num_sms, bool check_clean,
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

inline std::string illegal_reason(int64_t n, int64_t k, int64_t smem_bytes, std::vector<int64_t> config,
                                  bool check_clean, int64_t num_experts, int64_t layout, bool group_scales) {
    const char *reason = tensorbridge::sm90::illegal_reason(
        GemmConfig::from_vector(config),
        {1, n, k, 1, 0, smem_bytes, num_experts, static_cast<GroupedLayout>(layout), group_scales}, check_clean);
    return reason ? reason : "";
}

inline int64_t clean_aligned_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t block_k, int64_t num_stages) {
    return tensorbridge::sm90::clean_aligned_block_ms(ctas_per_sm, block_n, block_k, num_stages);
}

inline int64_t clean_one_warpgroup_block_ms(int64_t block_k, int64_t num_stages) {
    return tensorbridge::sm90::clean_one_warpgroup_block_ms(block_k, num_stages);
}

inline int64_t clean_pgroup_one_warpgroup_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t num_stages) {
    return tensorbridge::sm90::clean_pgroup_one_warpgroup_block_ms(ctas_per_sm, block_n, num_stages);
}

inline int64_t clean_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t block_k, bool use_stream_k, int64_t layout,
                              int64_t num_stages, bool group_scales) {
    return tensorbridge::sm90::clean_block_ms(ctas_per_sm, block_n, block_k, use_stream_k,
                                              static_cast<GroupedLayout>(layout), num_stages, group_scales);
}

inline int64_t config_smem_bytes(std::vector<int64_t> config, int64_t num_experts, int64_t layout, bool group_scales) {
    return tensorbridge::sm90::get_smem_bytes(GemmConfig::from_vector(config), num_experts,
                                              static_cast<GroupedLayout>(layout), group_scales);
}

inline int64_t register_kernel(
    const std::string &cubin_path, const std::string &func_name,
    int64_t shape_n, int64_t shape_k, std::vector<int64_t> config, int64_t smem_bytes,
    int64_t num_experts, int64_t layout, int64_t activation_scale_groups, bool scale_m_major, bool block_scales,
    bool fuse_swiglu_fp8) {
    const auto grouped_layout = static_cast<GroupedLayout>(layout);
    TORCH_CHECK(!fuse_swiglu_fp8 || grouped_layout != GroupedLayout::Indexed,
                "register_kernel: fuse_swiglu_fp8 cannot be combined with Indexed layout");
    CUmodule module;
    CUfunction func;
    check_curesult(driver().cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
    check_curesult(driver().cuModuleGetFunction(&func, module, func_name.c_str()), "cuModuleGetFunction");

    const int64_t kernel_id = static_cast<int64_t>(g_kernels.size());
    g_kernels[kernel_id] = {func, GemmConfig::from_vector(config), shape_n, shape_k, static_cast<uint32_t>(smem_bytes),
                            num_experts, grouped_layout, activation_scale_groups, scale_m_major,
                            block_scales, fuse_swiglu_fp8};
    return kernel_id;
}

inline at::Tensor launch_kernel_impl(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale,
    at::Tensor locks, std::optional<at::Tensor> d_, bool enable_pdl,
    std::optional<at::Tensor> sorted_ids, std::optional<at::Tensor> expert_ids,
    std::optional<at::Tensor> num_tokens_padded, std::optional<at::Tensor> expert_layout, int64_t top_k,
    std::optional<at::Tensor> d_scale_ = std::nullopt) {
    TORCH_CHECK(g_kernels.find(kernel_id) != g_kernels.end(), "Unknown kernel id ", kernel_id);
    const Kernel &kernel = g_kernels[kernel_id];
    const GemmConfig &config = kernel.config;

    const int64_t dev = a.get_device();
    const int64_t worker_ctas = get_num_sms(dev) * config.ctas_per_sm;
    const int64_t m = a.size(0), n = kernel.n, k = kernel.k;
    const int64_t padded_n = tensorbridge::ceil_div(n, tensorbridge::kBlockN) * tensorbridge::kBlockN;
    const int64_t padded_k = tensorbridge::ceil_div(k, tensorbridge::kBlockK) * tensorbridge::kBlockK;
    const bool indexed = kernel.layout == GroupedLayout::Indexed;
    TORCH_CHECK(top_k >= 1, "top_k must be positive");
    const int64_t output_rows = indexed ? m * top_k : m;
    at::Tensor d = d_.has_value()
                       ? d_.value()
                       : (kernel.fuse_swiglu_fp8
                              ? at::empty({output_rows, n / 2}, a.options().dtype(at::ScalarType::Float8_e4m3fn))
                              : at::empty({output_rows, n}, a.options().dtype(at::ScalarType::BFloat16)));
    at::Tensor d_scale;
    if (kernel.fuse_swiglu_fp8) {
        TORCH_CHECK(n % 256 == 0, "fused SwiGLU+FP8 kernel requires N % 256 == 0");
        TORCH_CHECK(d_scale_.has_value(), "fused SwiGLU+FP8 kernel requires d_scale output tensor");
        d_scale = d_scale_.value();
        check_tensor(d_scale, "d_scale", dev, at::ScalarType::Float, {output_rows, n / 256});
    } else {
        TORCH_CHECK(!d_scale_.has_value(), "d_scale was supplied to a non-fused kernel");
    }

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
    if (kernel.fuse_swiglu_fp8) {
        check_tensor(d, "d", dev, at::ScalarType::Float8_e4m3fn, {output_rows, n / 2});
    } else {
        check_tensor(d, "d", dev, at::ScalarType::BFloat16, {output_rows, n});
    }
    TORCH_CHECK(locks.is_cuda() && locks.scalar_type() == at::ScalarType::Int, "locks must be an int32 CUDA tensor");
    if (m == 0) return d;

    const auto block_m = static_cast<uint32_t>(config.block_m);
    CUtensorMap tensor_map_a = make_tma_desc(a, 128, block_m, 128);
    CUtensorMap tensor_map_b = make_tma_desc(b.view({-1, padded_k / 8}), 16, 128, 64);
    CUtensorMap tensor_map_d = {};
    if (!kernel.fuse_swiglu_fp8) {
        tensor_map_d = make_tma_desc(d, 64, block_m % 128 == 0 ? 128 : block_m, 128);
    }
    CUtensorMap tensor_map_sfb = kernel.block_scales
                                     ? make_tma_desc(sfb.view(at::ScalarType::Long).view({-1, padded_n}), 128, 1, 0)
                                     : make_tma_desc(sfb.view({-1, padded_n}), 128, 8, 0);

    const uint32_t shape_m = static_cast<uint32_t>(m);
    void *offsets_ptr = kernel.layout == GroupedLayout::Contiguous && expert_layout ? expert_layout->data_ptr() : nullptr;
    void *counts_ptr = kernel.layout == GroupedLayout::Masked && expert_layout ? expert_layout->data_ptr() : nullptr;
    void *sorted_ptr = indexed ? sorted_ids->data_ptr() : nullptr;
    void *experts_ptr = indexed ? expert_ids->data_ptr() : nullptr;
    void *padded_ptr = indexed ? num_tokens_padded->data_ptr() : nullptr;
    void *activations_ptr = a.data_ptr();
    void *output_ptr = d.data_ptr();
    const uint32_t num_routes = static_cast<uint32_t>(output_rows);
    const uint32_t route_top_k = static_cast<uint32_t>(top_k);
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
                             kernel.num_experts ? m / kernel.num_experts * a_scale.stride(0) : 0),
        kernel.fuse_swiglu_fp8 ? d_scale.data_ptr<float>() : nullptr};
    void *params_args[] = {&params};

    CUlaunchConfig launch = {};
    launch.gridDimX = static_cast<uint32_t>(worker_ctas);
    launch.gridDimY = launch.gridDimZ = 1;
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
    check_curesult(driver().cuLaunchKernelEx(&launch, func, params_args, nullptr), "cuLaunchKernelEx");
    return d;
}

inline at::Tensor launch_kernel(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale,
    at::Tensor locks, std::optional<at::Tensor> d, bool enable_pdl) {
    return launch_kernel_impl(kernel_id, a, a_scale, b, sfb, global_scale, locks, d, enable_pdl,
                              std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1);
}

inline at::Tensor launch_grouped_kernel(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale,
    at::Tensor locks, std::optional<at::Tensor> sorted_ids, std::optional<at::Tensor> expert_ids,
    std::optional<at::Tensor> num_tokens_padded, std::optional<at::Tensor> expert_layout, int64_t top_k,
    std::optional<at::Tensor> d, bool enable_pdl) {
    return launch_kernel_impl(kernel_id, a, a_scale, b, sfb, global_scale, locks, d, enable_pdl,
                              sorted_ids, expert_ids, num_tokens_padded, expert_layout, top_k);
}

inline at::Tensor launch_fused_swiglu_grouped_kernel(
    int64_t kernel_id, at::Tensor a, at::Tensor a_scale, at::Tensor b, at::Tensor sfb, at::Tensor global_scale,
    at::Tensor locks, at::Tensor d, at::Tensor d_scale, std::optional<at::Tensor> expert_layout, bool enable_pdl) {
    return launch_kernel_impl(kernel_id, a, a_scale, b, sfb, global_scale, locks, d, enable_pdl, std::nullopt,
                              std::nullopt, std::nullopt, expert_layout, 1, d_scale);
}
