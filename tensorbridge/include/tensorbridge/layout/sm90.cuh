#pragma once

#include <cutlass/arch/barrier.h>

#include <tensorbridge/common/utils.cuh>

// Shared memory of the kernels under `impls/`, mirrored by `get_smem_bytes` in `csrc/heuristics/sm90_common.hpp`

namespace tensorbridge::layout {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// NOTES: a raw word: a CUTLASS barrier member would make the structs non-POD, and a derived struct could then place
// members in a base's tail padding, away from the selector's account
struct BarrierWord {
    uint64_t word;

    CUTLASS_DEVICE Barrier *operator->() { return reinterpret_cast<Barrier *>(&word); }
    CUTLASS_DEVICE operator Barrier *() { return reinterpret_cast<Barrier *>(&word); }
};

// Contiguous, masked: each expert's rows and first M tile, and the M tiles ending with each lane's experts
template <GroupedLayout kLayout, uint32_t BLOCK_M, uint32_t kNumExperts>
struct Routing {
    uint32_t lane_ends[32];
    uint32_t row_starts[kNumExperts];
    uint32_t row_ends[kNumExperts];
    uint32_t m_block_offsets[kNumExperts + 1];
    uint32_t num_m_blocks;
    alignas(8) BarrierWord routing_ready;

};

// Indexed: two buffers of a tile's route IDs and the input rows they read
template <uint32_t BLOCK_M, uint32_t kNumExperts>
struct Routing<GroupedLayout::kIndexed, BLOCK_M, kNumExperts> {
    int32_t row_ids[2][BLOCK_M];
    uint32_t input_rows[2][BLOCK_M];
    uint32_t num_m_blocks;
    alignas(8) BarrierWord routing_ready;
    BarrierWord rows_consumed[2];

};

template <uint32_t BLOCK_M, uint32_t kNumExperts>
struct Routing<GroupedLayout::kDense, BLOCK_M, kNumExperts> {};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumStages>
struct PtokenRing {
    static constexpr uint32_t kBlockM = BLOCK_M, kBlockN = BLOCK_N, kBlockK = BLOCK_K;
    static constexpr bool kBlockScales = false;
    static constexpr uint32_t kKBlocks = BLOCK_K / 128;
    static constexpr uint32_t kTilesN = BLOCK_N / 128;
    static constexpr uint32_t kOutputRows = BLOCK_M % 128 == 0 ? 128 : BLOCK_M;
    static constexpr uint32_t kKBlockBytesA = BLOCK_M * 128;
    static constexpr uint32_t kTileBytesB = kKBlocks * 128 * 64;
    static constexpr uint32_t kTileBytesSFB = kKBlocks * 8 * 128;
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = kKBlocks * kKBlockBytesA;
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = kTilesN * kTileBytesB;
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = kTilesN * kTileBytesSFB;

    // NOTES: the arrays of the ring are multiples of 1024 bytes, which aligns `d` for its 128B swizzle
    alignas(128) uint8_t a[kNumStages][SMEM_A_SIZE_PER_STAGE];
    uint8_t b[kNumStages][SMEM_B_SIZE_PER_STAGE];
    uint8_t sfb[kNumStages][SMEM_SFB_SIZE_PER_STAGE];
    uint8_t d[kTilesN][kOutputRows * 128 * sizeof(nv_bfloat16)];
    float sfa[2][BLOCK_M];

    alignas(128) BarrierWord full_barriers[kNumStages];
    BarrierWord sfa_full_barriers[2];
    BarrierWord empty_barriers[kNumStages];
    BarrierWord sfa_empty_barriers[2];
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumStages, GroupedLayout kLayout,
          uint32_t kNumExperts>
struct PtokenStorage : PtokenRing<BLOCK_M, BLOCK_N, BLOCK_K, kNumStages>, Routing<kLayout, BLOCK_M, kNumExperts> {};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages>
struct PgroupRing {
    static constexpr uint32_t kBlockM = BLOCK_M, kBlockK = 128;
    static constexpr uint32_t kBlockN = BLOCK_N;
    static constexpr bool kBlockScales = true;
    static constexpr uint32_t kKBlocks = 1;
    static constexpr uint32_t kTilesN = BLOCK_N / 128;
    static constexpr uint32_t kOutputRows = BLOCK_M % 128 == 0 ? 128 : BLOCK_M;
    static constexpr uint32_t kKBlockBytesA = BLOCK_M * 128;
    static constexpr uint32_t kTileBytesB = 128 * 64;
    static constexpr uint32_t kTileBytesSFB = 8 * 128;

    alignas(128) uint8_t a[kNumStages][BLOCK_M * 128];
    uint8_t b[kNumStages][kTilesN * kTileBytesB];
    uint8_t sfb[kNumStages][kTilesN * kTileBytesSFB];
    uint8_t d[kTilesN][kOutputRows * 128 * sizeof(nv_bfloat16)];
    float sfa[kNumStages][BLOCK_M];

    alignas(128) BarrierWord full_barriers[kNumStages];
    BarrierWord empty_barriers[kNumStages];
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages,
          GroupedLayout kLayout = GroupedLayout::kDense, uint32_t kNumExperts = 0>
struct PgroupStorage : PgroupRing<BLOCK_M, BLOCK_N, kNumStages>, Routing<kLayout, BLOCK_M, kNumExperts> {};

}  // namespace tensorbridge::layout
