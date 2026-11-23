#define USE_CUDA 1

#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "./launcher_common.hpp"
#include "./launcher_gemm.hpp"
#include "./launcher_megamoe.hpp"

// Torch library `tensorbridge`: SM90 heuristics, dense/grouped NVFP4xFP8 GEMM launcher,
// and MegaMoE launcher (`launcher_gemm.hpp`, `launcher_megamoe.hpp`).

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
          "bool block_scales=False, bool fuse_swiglu_fp8=False) -> int");
    m.def("launch_kernel(int kernel_id, Tensor a, Tensor a_scale, Tensor b, Tensor sfb, Tensor global_scale, "
          "Tensor locks, Tensor? d, bool enable_pdl=False) -> Tensor");
    m.def("launch_grouped_kernel(int kernel_id, Tensor a, Tensor a_scale, Tensor b, Tensor sfb, Tensor global_scale, "
          "Tensor locks, Tensor? sorted_ids, Tensor? expert_ids, Tensor? num_tokens_padded, Tensor? expert_layout, "
          "int top_k, Tensor? d=None, bool enable_pdl=False) -> Tensor");
    m.def("launch_fused_swiglu_grouped_kernel(int kernel_id, Tensor a, Tensor a_scale, Tensor b, Tensor sfb, "
          "Tensor global_scale, Tensor locks, Tensor d, Tensor d_scale, Tensor? expert_layout=None, "
          "bool enable_pdl=False) -> Tensor");
    m.def("register_megamoe_kernel(str cubin_path, str func_name, int n_gu, int k_gu, int n_dn, int k_dn, "
          "int num_experts, int block_m, int num_stages, int ctas_per_sm, int math_warpgroups, int smem_bytes) -> int");
    m.def("launch_megamoe_kernel(int kernel_id, int[] peer_ptrs, int[] symm_offsets, Tensor local_x_fp8, "
          "Tensor local_x_scale, Tensor topk_ids, Tensor topk_weights, Tensor local_final_out, Tensor local_gen, "
          "Tensor local_phase_done, Tensor a_gu, Tensor sfa_gu, Tensor w_gu, Tensor sfb_gu, Tensor gs_gu, "
          "Tensor inter_fp8, Tensor inter_scale, Tensor w_dn, Tensor sfb_dn, Tensor gs_dn, Tensor expert_counts, "
          "Tensor ready_ring, Tensor write_head, Tensor dispatch_done, Tensor gu_work_cursor, Tensor gu_tile_done, "
          "Tensor down_ready_ring, Tensor down_write_cursor, int ep_rank=0, int local_e=1, int capacity=1, "
          "bool enable_pdl=False, int reserve_sms=0) -> Tensor");
};

TORCH_LIBRARY_IMPL(tensorbridge, CUDA, m) {
    m.impl("launch_kernel", &launch_kernel);
    m.impl("launch_grouped_kernel", &launch_grouped_kernel);
    m.impl("launch_fused_swiglu_grouped_kernel", &launch_fused_swiglu_grouped_kernel);
    m.impl("launch_megamoe_kernel", &launch_megamoe_kernel);
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
    m.impl("register_megamoe_kernel", &register_megamoe_kernel);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_num_sms", &get_num_sms);
}
