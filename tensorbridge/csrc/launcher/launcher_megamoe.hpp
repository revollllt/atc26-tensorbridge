#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <tensorbridge/common/megamoe_params.h>

#include "./launcher_common.hpp"
#include "./tma.h"

struct MegaMoeKernel {
    CUfunction func;
    int64_t n_gu, k_gu, n_dn, k_dn, num_experts;
    int64_t block_m, num_stages, ctas_per_sm, math_warpgroups;
    uint32_t smem_bytes;
};

static std::unordered_map<int64_t, MegaMoeKernel> g_megamoe_kernels;

inline int64_t register_megamoe_kernel(
    const std::string &cubin_path, const std::string &func_name,
    int64_t n_gu, int64_t k_gu, int64_t n_dn, int64_t k_dn, int64_t num_experts,
    int64_t block_m, int64_t num_stages, int64_t ctas_per_sm, int64_t math_warpgroups, int64_t smem_bytes) {
    CUmodule module;
    CUfunction func;
    check_curesult(driver().cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
    check_curesult(driver().cuModuleGetFunction(&func, module, func_name.c_str()), "cuModuleGetFunction");
    const int64_t kid = static_cast<int64_t>(g_megamoe_kernels.size());
    g_megamoe_kernels[kid] = {func, n_gu, k_gu, n_dn, k_dn, num_experts,
                              block_m, num_stages, ctas_per_sm, math_warpgroups, static_cast<uint32_t>(smem_bytes)};
    return kid;
}

inline at::Tensor launch_megamoe_kernel(
    int64_t kernel_id,
    std::vector<int64_t> peer_ptrs, std::vector<int64_t> symm_offsets,
    at::Tensor local_x_fp8, at::Tensor local_x_scale, at::Tensor topk_ids, at::Tensor topk_weights,
    at::Tensor local_final_out, at::Tensor local_gen, at::Tensor local_phase_done,
    at::Tensor a_gu, at::Tensor sfa_gu,
    at::Tensor w_gu, at::Tensor sfb_gu, at::Tensor gs_gu,
    at::Tensor inter_fp8, at::Tensor inter_scale,
    at::Tensor w_dn, at::Tensor sfb_dn, at::Tensor gs_dn,
    at::Tensor expert_counts,
    at::Tensor ready_ring, at::Tensor write_head, at::Tensor dispatch_done,
    at::Tensor gu_work_cursor, at::Tensor gu_tile_done,
    at::Tensor down_ready_ring, at::Tensor down_write_cursor,
    int64_t ep_rank, int64_t local_e, int64_t capacity,
    bool enable_pdl, int64_t reserve_sms) {
    TORCH_CHECK(g_megamoe_kernels.find(kernel_id) != g_megamoe_kernels.end(), "Unknown megamoe kernel id ", kernel_id);
    const MegaMoeKernel &k = g_megamoe_kernels[kernel_id];
    const int64_t dev = a_gu.get_device();
    const int64_t m = a_gu.size(0);
    const int64_t e = k.num_experts;
    TORCH_CHECK(e == local_e, "MegaMoE num_experts (", e, ") must match local_e (", local_e, ")");
    TORCH_CHECK(m == e * capacity, "MegaMoE masked rows (", m, ") must equal local_e * capacity (", e * capacity, ")");
    const int64_t ep_size = static_cast<int64_t>(peer_ptrs.size());
    TORCH_CHECK(ep_size >= 1 && ep_size <= static_cast<int64_t>(tensorbridge::kMaxNvlinkEpRanks),
                "peer_ptrs.size() must be in [1, ", tensorbridge::kMaxNvlinkEpRanks, "], got ", ep_size);
    TORCH_CHECK(ep_rank >= 0 && ep_rank < ep_size, "ep_rank out of range: ", ep_rank, " vs ", ep_size);
    TORCH_CHECK(local_e >= 1 && local_e <= 128, "local_e must be in [1, 128], got ", local_e);
    TORCH_CHECK(symm_offsets.size() == 8, "symm_offsets must have 8 entries, got ", symm_offsets.size());

    const int64_t num_tokens = local_x_fp8.size(0);
    const int64_t hidden = local_x_fp8.size(1);
    const int64_t top_k = topk_ids.size(1);
    TORCH_CHECK(hidden == k.k_gu && hidden == k.n_dn,
                "MegaMoE hidden (", hidden, ") must match k_gu (", k.k_gu, ") and n_dn (", k.n_dn, ")");
    TORCH_CHECK(top_k >= 1 && top_k <= 255, "top_k must be in [1, 255], got ", top_k);
    TORCH_CHECK(num_tokens * top_k <= 2048,
                "MegaMoE routes (", num_tokens * top_k, ") exceeds SMEM route limit (2048); chunk tokens per call");

    check_tensor(local_x_fp8, "local_x_fp8", dev, at::ScalarType::Float8_e4m3fn, {num_tokens, hidden});
    check_tensor(local_x_scale, "local_x_scale", dev, at::ScalarType::Float, {num_tokens, hidden / 128});
    check_tensor(topk_ids, "topk_ids", dev, at::ScalarType::Int, {num_tokens, top_k});
    check_tensor(topk_weights, "topk_weights", dev, at::ScalarType::Float, {num_tokens, top_k});
    check_tensor(local_final_out, "local_final_out", dev, at::ScalarType::BFloat16, {num_tokens, hidden});
    check_tensor(local_gen, "local_gen", dev, at::ScalarType::Int, {1});
    check_tensor(local_phase_done, "local_phase_done", dev, at::ScalarType::Int, {2});
    check_tensor(a_gu, "a_gu", dev, at::ScalarType::Float8_e4m3fn, {m, k.k_gu});
    check_tensor(sfa_gu, "sfa_gu", dev, at::ScalarType::Float, {m, k.k_gu / 128});
    check_tensor(w_gu, "w_gu", dev, at::ScalarType::Int, {e, k.n_gu, k.k_gu / 8});
    check_tensor(sfb_gu, "sfb_gu", dev, at::ScalarType::Float8_e4m3fn, {e, k.k_gu / 128, k.n_gu, 8});
    check_tensor(gs_gu, "gs_gu", dev, at::ScalarType::Float, {e});
    check_tensor(inter_fp8, "inter_fp8", dev, at::ScalarType::Float8_e4m3fn, {m, k.k_dn});
    check_tensor(inter_scale, "inter_scale", dev, at::ScalarType::Float, {m, k.k_dn / 128});
    check_tensor(w_dn, "w_dn", dev, at::ScalarType::Int, {e, k.n_dn, k.k_dn / 8});
    check_tensor(sfb_dn, "sfb_dn", dev, at::ScalarType::Float8_e4m3fn, {e, k.k_dn / 128, k.n_dn, 8});
    check_tensor(gs_dn, "gs_dn", dev, at::ScalarType::Float, {e});
    check_tensor(expert_counts, "expert_counts", dev, at::ScalarType::Int, {e});

    const int64_t total_sms = get_num_sms(dev);
    TORCH_CHECK(reserve_sms >= 0 && reserve_sms < total_sms,
                "reserve_sms (", reserve_sms, ") must be in [0, ", total_sms, ")");
    const int64_t worker_ctas = (total_sms - reserve_sms) * k.ctas_per_sm;
    TORCH_CHECK(worker_ctas >= ep_size + 1,
                "MegaMoE requires worker_ctas (", worker_ctas, ") >= ep_size + 1 (", ep_size + 1, ")");

    auto check_ctrl = [&](const at::Tensor &t, const char *name, int64_t min_numel) {
        TORCH_CHECK(t.is_cuda() && t.get_device() == dev && t.is_contiguous() &&
                        t.scalar_type() == at::ScalarType::Int && t.numel() >= min_numel,
                    name, " must be contiguous int32 CUDA tensor on device with >= ", min_numel, " elements");
    };
    TORCH_CHECK(ready_ring.is_cuda() && ready_ring.get_device() == dev && ready_ring.is_contiguous() &&
                    ready_ring.dim() == 2 && ready_ring.size(1) == 4 &&
                    ready_ring.scalar_type() == at::ScalarType::Int,
                "ready_ring must be contiguous int32 [ring_capacity, 4] CUDA tensor");
    const int64_t ring_cap = ready_ring.size(0);
    const int64_t tasks_per_m_tile = k.n_gu / 256 + k.n_dn / 256;
    check_ctrl(write_head, "write_head", 1);
    check_ctrl(dispatch_done, "dispatch_done", 1);
    check_ctrl(gu_work_cursor, "gu_work_cursor", 1);
    check_ctrl(gu_tile_done, "gu_tile_done", ring_cap);
    check_ctrl(down_ready_ring, "down_ready_ring", ring_cap * tasks_per_m_tile + worker_ctas);
    check_ctrl(down_write_cursor, "down_write_cursor", 1);

    const uint32_t bm = static_cast<uint32_t>(k.block_m);
    tensorbridge::MegaMoeParams p = {};
    p.tensor_map_a_gu = make_tma_desc(a_gu, 128, bm, 128);
    p.tensor_map_b_gu = make_tma_desc(w_gu.view({-1, k.k_gu / 8}), 16, 128, 64);
    p.tensor_map_sfb_gu = make_tma_desc(sfb_gu.view(at::ScalarType::Long).view({-1, k.n_gu}), 128, 1, 0);
    p.tensor_map_a_dn = make_tma_desc(inter_fp8, 128, bm, 128);
    p.tensor_map_b_dn = make_tma_desc(w_dn.view({-1, k.k_dn / 8}), 16, 128, 64);
    p.tensor_map_sfb_dn = make_tma_desc(sfb_dn.view(at::ScalarType::Long).view({-1, k.n_dn}), 128, 1, 0);
    for (int64_t r = 0; r < ep_size; r++) {
        p.peer_base[r] = static_cast<uint64_t>(peer_ptrs[r]);
    }
    p.sfa_gu = sfa_gu.data_ptr<float>();
    p.global_scale_gu = gs_gu.data_ptr<float>();
    p.inter_fp8 = inter_fp8.data_ptr();
    p.inter_scale = inter_scale.data_ptr<float>();
    p.global_scale_dn = gs_dn.data_ptr<float>();
    p.local_x_fp8 = reinterpret_cast<const uint8_t *>(local_x_fp8.data_ptr());
    p.local_x_scale = local_x_scale.data_ptr<float>();
    p.topk_ids = topk_ids.data_ptr<int32_t>();
    p.topk_weights = topk_weights.data_ptr<float>();
    p.local_final_out = local_final_out.data_ptr();
    p.expert_counts = expert_counts.data_ptr<int32_t>();
    p.ready_ring = reinterpret_cast<tensorbridge::MegaMoeTile *>(ready_ring.data_ptr());
    p.write_head = reinterpret_cast<uint32_t *>(write_head.data_ptr());
    p.dispatch_done = reinterpret_cast<uint32_t *>(dispatch_done.data_ptr());
    p.gu_work_cursor = reinterpret_cast<uint32_t *>(gu_work_cursor.data_ptr());
    p.gu_tile_done = reinterpret_cast<uint32_t *>(gu_tile_done.data_ptr());
    p.down_ready_ring = reinterpret_cast<uint32_t *>(down_ready_ring.data_ptr());
    p.down_write_cursor = reinterpret_cast<uint32_t *>(down_write_cursor.data_ptr());
    p.local_gen = reinterpret_cast<uint32_t *>(local_gen.data_ptr());
    p.local_phase_done = reinterpret_cast<uint32_t *>(local_phase_done.data_ptr());
    p.sfa_gu_stride_expert = static_cast<uint64_t>(capacity * sfa_gu.stride(0));
    p.sfa_dn_stride_expert = static_cast<uint64_t>(capacity * inter_scale.stride(0));
    p.off_recv_fp8 = static_cast<uint64_t>(symm_offsets[0]);
    p.off_recv_scale = static_cast<uint64_t>(symm_offsets[1]);
    p.off_recv_meta = static_cast<uint64_t>(symm_offsets[2]);
    p.off_combine_recv = static_cast<uint64_t>(symm_offsets[3]);
    p.off_recv_slot_count = static_cast<uint64_t>(symm_offsets[4]);
    p.off_expert_arrived = static_cast<uint64_t>(symm_offsets[5]);
    p.off_combine_arrived = static_cast<uint64_t>(symm_offsets[6]);
    p.off_barrier_flags = static_cast<uint64_t>(symm_offsets[7]);
    p.sfa_gu_stride_m = static_cast<uint32_t>(sfa_gu.stride(0));
    p.sfa_gu_stride_k = static_cast<uint32_t>(sfa_gu.stride(1));
    p.sfa_dn_stride_m = static_cast<uint32_t>(inter_scale.stride(0));
    p.sfa_dn_stride_k = static_cast<uint32_t>(inter_scale.stride(1));
    p.ep_rank = static_cast<uint32_t>(ep_rank);
    p.ep_size = static_cast<uint32_t>(ep_size);
    p.local_e = static_cast<uint32_t>(local_e);
    p.num_tokens = static_cast<uint32_t>(num_tokens);
    p.capacity = static_cast<uint32_t>(capacity);
    p.hidden = static_cast<uint32_t>(hidden);
    p.top_k = static_cast<uint32_t>(top_k);

    void *args[] = {&p};
    CUlaunchConfig launch = {};
    launch.gridDimX = static_cast<uint32_t>(worker_ctas);
    launch.gridDimY = launch.gridDimZ = 1;
    launch.blockDimX = static_cast<uint32_t>(128 * (k.math_warpgroups + 1));
    launch.blockDimY = launch.blockDimZ = 1;
    launch.sharedMemBytes = k.smem_bytes;
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

    check_curesult(
        driver().cuFuncSetAttribute(k.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes),
        "cuFuncSetAttribute(megamoe)");
    check_curesult(driver().cuLaunchKernelEx(&launch, k.func, args, nullptr), "cuLaunchKernelEx(megamoe)");
    return local_final_out;
}
