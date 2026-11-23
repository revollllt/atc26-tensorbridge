#pragma once

#include <cuda.h>
#include <cstdint>

namespace tensorbridge {

static constexpr uint32_t kMaxNvlinkEpRanks = 16;

// One ready masked M-tile published into `MegaMoeParams::ready_ring`.
struct alignas(16) MegaMoeTile {
    uint32_t expert_id;
    uint32_t row_begin;
    uint32_t row_end;
    uint32_t m_tile_idx;
};

// Parameter block for the MegaMoE kernel (`impls/sm90_w4a8_megamoe.cuh`):
// fuses 4x route-striped NVLink P2P Dispatch, GateUp + SwiGLU + group-128 FP8 quant,
// Down (group-128) with direct 512B-warp NVLink P2P scatter epilogue, weighted BF16 reduction,
// inline ring self-reset, and a single NVLink barrier.
struct alignas(128) MegaMoeParams {
    // 6 TMA descriptors = 768 bytes
    CUtensorMap tensor_map_a_gu;    // FP8 [E * cap, K_gu]
    CUtensorMap tensor_map_b_gu;    // NVFP4 interleaved [E * N_gu, K_gu / 8]
    CUtensorMap tensor_map_sfb_gu;  // FP8 [E * K_gu/128, N_gu] (64-bit rows)
    CUtensorMap tensor_map_a_dn;    // FP8 [E * cap, K_dn] (intermediate output of GateUp+SwiGLU)
    CUtensorMap tensor_map_b_dn;    // NVFP4 [E * N_dn, K_dn / 8]
    CUtensorMap tensor_map_sfb_dn;  // FP8 [E * K_dn/128, N_dn] (64-bit rows)

    // Symmetric NVLink P2P base pointers (128 bytes)
    uint64_t peer_base[kMaxNvlinkEpRanks];

    // Activation, weight scale, intermediate, and local I/O pointers
    const float *sfa_gu;            // [E * cap, K_gu / 128]
    const float *global_scale_gu;   // [E]
    void *inter_fp8;                // FP8 [E * cap, K_dn] (backing tensor_map_a_dn)
    float *inter_scale;             // FP32 [E * cap, K_dn / 128] (sfa_dn)
    const float *global_scale_dn;   // [E]
    const uint8_t *local_x_fp8;     // [num_tokens, hidden] FP8 E4M3
    const float *local_x_scale;     // [num_tokens, hidden / 128] FP32
    const int32_t *topk_ids;        // [num_tokens, top_k] global expert ids
    const float *topk_weights;      // [num_tokens, top_k] routing weights
    void *local_final_out;          // [num_tokens, hidden] BF16 final combined output

    // Zero-CAS lock-free task ring & intra-kernel synchronization scratch
    int32_t *expert_counts;         // [local_e] valid rows per expert
    MegaMoeTile *ready_ring;        // [ring_capacity] M-tiles published by dispatch
    uint32_t *write_head;           // [1] published M-tile count
    uint32_t *dispatch_done;        // [1] dispatch completion flag
    uint32_t *gu_work_cursor;       // [1] unified monotonic ticket cursor (`atomicAdd`)
    uint32_t *gu_tile_done;         // [ring_capacity] GateUp N-tiles finished per M-tile
    uint32_t *down_ready_ring;      // [task_ring_len] packed tasks (init 0xFFFFFFFF)
    uint32_t *down_write_cursor;    // [1] unified task ring write tail (`atomicAdd`)
    uint32_t *local_gen;            // [1] monotonic call epoch counter (CUDA-Graph safe)
    uint32_t *local_phase_done;     // [2] CTA arrival counters

    // Strides & symmetric NVLink P2P byte offsets
    uint64_t sfa_gu_stride_expert;
    uint64_t sfa_dn_stride_expert;
    uint64_t off_recv_fp8;          // uint8  [local_e * capacity, hidden]
    uint64_t off_recv_scale;        // float  [local_e * capacity, hidden / 128]
    uint64_t off_recv_meta;         // uint32 [local_e * capacity] packed (src_rank<<24 | k<<16 | src_t)
    uint64_t off_combine_recv;      // bf16   [max_tokens * top_k, hidden]
    uint64_t off_recv_slot_count;   // uint32 [local_e]
    uint64_t off_expert_arrived;    // uint32 [local_e]
    uint64_t off_combine_arrived;   // uint32 [1]
    uint64_t off_barrier_flags;     // uint32 [kMaxNvlinkEpRanks]

    // Runtime strides & dimensions
    uint32_t sfa_gu_stride_m;
    uint32_t sfa_gu_stride_k;
    uint32_t sfa_dn_stride_m;
    uint32_t sfa_dn_stride_k;
    uint32_t ep_rank;
    uint32_t ep_size;
    uint32_t local_e;
    uint32_t num_tokens;
    uint32_t capacity;
    uint32_t hidden;
    uint32_t top_k;
};

static_assert(sizeof(MegaMoeParams) == 1280 && alignof(MegaMoeParams) == 128,
              "MegaMoeParams layout must match between launcher and kernel");

}  // namespace tensorbridge
