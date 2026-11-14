#pragma once

#include <cute/arch/copy_sm80.hpp>
#include <cute/util/type_traits.hpp>
#include <cutlass/arch/barrier.h>

#include <tensorbridge/common/params.h>
#include <tensorbridge/common/ptx.cuh>
#include <tensorbridge/common/utils.cuh>
#include <tensorbridge/layout/sm90.cuh>

// Tile schedulers: each deals output tiles to the CTAs from `blockIdx`, and each role of a CTA walks the same tiles

namespace tensorbridge::sched {

template <uint32_t kNumNBlocks, bool kMFastTileOrder>
CUTLASS_DEVICE void tile_coordinates(uint32_t tile_idx, uint32_t num_m_blocks, uint32_t &m_block_idx,
                                uint32_t &n_block_idx) {
    if constexpr (kMFastTileOrder) {
        m_block_idx = tile_idx % num_m_blocks;
        n_block_idx = tile_idx / num_m_blocks;
    } else {
        m_block_idx = tile_idx / kNumNBlocks;
        n_block_idx = tile_idx % kNumNBlocks;
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          bool kMFastTileOrder>
class DataParallelScheduler {
    static constexpr uint32_t kNumNBlocks = SHAPE_N / BLOCK_N;
    const uint32_t shape_m, num_m_blocks, num_tiles;
    uint32_t next_tile = blockIdx.x;

public:
    struct Tile {
        uint32_t m_block_idx, n_block_idx, row_begin, row_end;
        static constexpr uint32_t expert_id = 0, route_buffer = 0;
        static constexpr uint32_t k_block_idx = 0, num_k_blocks = SHAPE_K / BLOCK_K;
        static constexpr uint32_t slice_id = 0, slice_count = 1, lock_idx = 0;
        static constexpr bool is_full_depth() { return true; }
    };

    template <class Storage>
    CUTLASS_DEVICE DataParallelScheduler(const GemmParams &params, Storage &)
        : shape_m(params.shape_m), num_m_blocks(math::ceil_div(shape_m, BLOCK_M)), num_tiles(num_m_blocks * kNumNBlocks) {}

    CUTLASS_DEVICE void wait_ready() {}

    CUTLASS_DEVICE bool get_next_block(Tile &tile) {
        if (next_tile >= num_tiles) return false;
        tile_coordinates<kNumNBlocks, kMFastTileOrder>(next_tile, num_m_blocks, tile.m_block_idx, tile.n_block_idx);
        tile.row_begin = tile.m_block_idx * BLOCK_M;
        tile.row_end = shape_m;
        next_tile += gridDim.x;
        return true;
    }
};

// Data-parallel waves, then the last wave's tiles cut along K over all CTAs.
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          bool kMFastTileOrder, bool kAlignedSlices>
class StreamKScheduler {
    static constexpr uint32_t kNumNBlocks = SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumKBlocks = SHAPE_K / BLOCK_K;

    const uint32_t num_ctas = gridDim.x;
    const uint32_t cta_id = blockIdx.x;
    const uint32_t shape_m;
    const uint32_t num_m_blocks;
    const uint32_t dp_tiles_per_cta, tail_divisor, divisor_multiplier, divisor_shift;
    uint32_t dp_tiles_left, dp_next_tile;
    uint32_t tail_blocks_left, tail_next_block;
    uint32_t aligned_tile, aligned_slice, aligned_slices;
    bool aligned_pending;

    CUTLASS_DEVICE uint32_t div_tail(uint32_t x) const {
        return divisor_multiplier ? __umulhi(x, divisor_multiplier) >> divisor_shift : x;
    }

public:
    struct Tile {
        uint32_t m_block_idx, n_block_idx, row_begin, row_end;
        static constexpr uint32_t expert_id = 0, route_buffer = 0;
        uint32_t k_block_idx, num_k_blocks;
        uint32_t slice_id, slice_count;
        uint32_t lock_idx;
        CUTLASS_DEVICE bool is_full_depth() const { return slice_count == 1; }
    };

    template <class Storage>
    CUTLASS_DEVICE StreamKScheduler(const GemmParams &params, Storage &)
        : shape_m(params.shape_m), num_m_blocks(math::ceil_div(shape_m, BLOCK_M)),
          dp_tiles_per_cta(params.stream_k.dp_tiles_per_cta), tail_divisor(params.stream_k.tail_divisor),
          divisor_multiplier(params.stream_k.divisor_multiplier), divisor_shift(params.stream_k.divisor_shift) {
        const uint32_t num_k_blocks_total = num_m_blocks * kNumNBlocks * kNumKBlocks;
        dp_tiles_left = dp_tiles_per_cta;
        dp_next_tile = cta_id;

        if constexpr (kAlignedSlices) {
            aligned_slices = params.stream_k.aligned_slices;
            aligned_pending = cta_id < aligned_slices * tail_divisor;
            aligned_slice = aligned_pending ? div_tail(cta_id) : 0;
            aligned_tile = aligned_pending ? cta_id - aligned_slice * tail_divisor : 0;
            return;
        }
        tail_next_block = num_ctas * dp_tiles_per_cta * kNumKBlocks + tail_divisor * cta_id;
        tail_blocks_left =
            tail_next_block >= num_k_blocks_total ? 0 : min(num_k_blocks_total - tail_next_block, tail_divisor);
    }

    CUTLASS_DEVICE void wait_ready() {}

    CUTLASS_DEVICE bool get_next_block(Tile &tile) {
        tile.row_end = shape_m;
        if (dp_tiles_left) {
            tile_coordinates<kNumNBlocks, kMFastTileOrder>(dp_next_tile, num_m_blocks, tile.m_block_idx, tile.n_block_idx);
            tile.row_begin = tile.m_block_idx * BLOCK_M;
            tile.k_block_idx = 0;
            tile.num_k_blocks = kNumKBlocks;
            tile.slice_id = 0, tile.slice_count = 1, tile.lock_idx = 0;
            dp_next_tile += num_ctas;
            dp_tiles_left--;
            return true;
        }
        if constexpr (kAlignedSlices) {
            if (!aligned_pending) return false;
            aligned_pending = false;
            const uint32_t tile_idx = dp_tiles_per_cta * num_ctas + aligned_tile;
            tile_coordinates<kNumNBlocks, kMFastTileOrder>(tile_idx, num_m_blocks, tile.m_block_idx, tile.n_block_idx);
            tile.row_begin = tile.m_block_idx * BLOCK_M;
            tile.k_block_idx = aligned_slice * kNumKBlocks / aligned_slices;
            tile.num_k_blocks = (aligned_slice + 1) * kNumKBlocks / aligned_slices - tile.k_block_idx;
            tile.slice_id = aligned_slices - 1 - aligned_slice;
            tile.slice_count = aligned_slices;
            tile.lock_idx = aligned_tile;
            return true;
        }
        if (!tail_blocks_left) return false;

        const uint32_t tile_idx = tail_next_block / kNumKBlocks;
        tile_coordinates<kNumNBlocks, kMFastTileOrder>(tile_idx, num_m_blocks, tile.m_block_idx, tile.n_block_idx);
        tile.row_begin = tile.m_block_idx * BLOCK_M;
        tile.k_block_idx = tail_next_block - tile_idx * kNumKBlocks;
        tile.num_k_blocks = min(kNumKBlocks - tile.k_block_idx, tail_blocks_left);
        tail_blocks_left -= tile.num_k_blocks;
        tail_next_block += tile.num_k_blocks;

        if (tile.k_block_idx == 0) {
            tile.slice_id = 0;
            tile.slice_count = div_tail(kNumKBlocks - tile.num_k_blocks + tail_divisor - 1) + 1;
        } else {
            tile.slice_id = div_tail(tile.k_block_idx);
            const uint32_t first_slice_blocks = tile.k_block_idx - tile.slice_id * tail_divisor;
            tile.slice_count = div_tail(kNumKBlocks - first_slice_blocks + tail_divisor - 1);
            if (first_slice_blocks) {
                tile.slice_id++;
                tile.slice_count++;
            }
        }
        tile.slice_id = tile.slice_count - 1 - tile.slice_id;
        tile.lock_idx = tile_idx - dp_tiles_per_cta * num_ctas;
        return true;
    }
};

// The M tiles of every expert, each with its full K
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          bool kMFastTileOrder, GroupedLayout kLayout, uint32_t kNumExperts>
class GroupedScheduler {
    static_assert(kLayout != GroupedLayout::kDense && kNumExperts > 0, "A grouped GEMM needs experts");
    static constexpr uint32_t kNumNBlocks = SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumChunks = math::ceil_div(kNumExperts, 32);
    static constexpr uint32_t kExpertsPerLane = kNumChunks;
    static_assert(kExpertsPerLane <= 32, "Two ballots cover at most 1024 experts");
    using Storage = layout::Routing<kLayout, BLOCK_M, kNumExperts>;

    Storage &routing;
    const int32_t *expert_ids;
    const uint32_t shape_m;
    uint32_t next_tile = blockIdx.x;
    uint32_t tile_sequence = 0;

public:
    struct Tile {
        uint32_t m_block_idx, n_block_idx, expert_id, row_begin, row_end, route_buffer;
        static constexpr uint32_t k_block_idx = 0, num_k_blocks = SHAPE_K / BLOCK_K;
        static constexpr uint32_t slice_id = 0, slice_count = 1, lock_idx = 0;
        static constexpr bool is_full_depth() { return true; }
    };

    CUTLASS_DEVICE GroupedScheduler(const GemmParams &params, Storage &routing)
        : routing(routing), expert_ids(params.expert_ids), shape_m(params.shape_m) {}

    // Load the routing into shared memory
    CUTLASS_DEVICE void load_routing(const int64_t *offsets, const int32_t *counts, const int32_t *num_tokens_padded) {
        const uint32_t lane_idx = threadIdx.x % 32;
        if constexpr (kLayout == GroupedLayout::kIndexed) {
            if (lane_idx == 0) {
                ptx::cp_async_load_pred(reinterpret_cast<const uint32_t *>(num_tokens_padded), &routing.num_m_blocks, true);
            }
        } else {
            for (uint32_t chunk = 0; chunk < kNumChunks; chunk++) {
                const uint32_t expert = chunk * 32 + lane_idx;
                if (expert >= kNumExperts) continue;
                if constexpr (kLayout == GroupedLayout::kContiguous) {
                    const auto *offset_words = reinterpret_cast<const uint32_t *>(offsets);
                    ptx::cp_async_load_pred(offset_words + 2 * expert, &routing.row_starts[expert], true);
                    ptx::cp_async_load_pred(offset_words + 2 * (expert + 1), &routing.row_ends[expert], true);
                } else {
                    ptx::cp_async_load_pred(reinterpret_cast<const uint32_t *>(counts) + expert, &routing.row_ends[expert], true);
                }
            }
        }
    }

    // Prefix sum of each expert's M tiles: per lane over its experts, then a warp-wide scan over the lanes
    CUTLASS_DEVICE void prepare() {
        const uint32_t lane_idx = threadIdx.x % 32;
        cute::cp_async_wait<0>();
        if constexpr (kLayout == GroupedLayout::kIndexed) {
            if (lane_idx != 0) return;
            routing.num_m_blocks = math::ceil_div(routing.num_m_blocks, BLOCK_M);
            routing.routing_ready->arrive();
        } else {
            const uint32_t first_expert = lane_idx * kExpertsPerLane;
            uint32_t lane_ends[kExpertsPerLane];
            uint32_t lane_blocks = 0;
            for (uint32_t i = 0; i < kExpertsPerLane; i++) {
                const uint32_t expert = first_expert + i;
                uint32_t rows = 0;
                if (expert < kNumExperts) {
                    rows = kLayout == GroupedLayout::kContiguous ? routing.row_ends[expert] - routing.row_starts[expert]
                                                                 : routing.row_ends[expert];
                }
                lane_blocks += math::ceil_div(rows, BLOCK_M);
                lane_ends[i] = lane_blocks;
            }
            uint32_t lane_end = lane_blocks;
            for (uint32_t delta = 1; delta < 32; delta *= 2) {
                const uint32_t preceding = __shfl_up_sync(0xffffffff, lane_end, delta);
                lane_end += lane_idx >= delta ? preceding : 0;
            }
            const uint32_t lane_start = lane_end - lane_blocks;
            for (uint32_t i = 0; i < kExpertsPerLane; i++) {
                const uint32_t expert = first_expert + i;
                if (expert < kNumExperts) {
                    if constexpr (kLayout == GroupedLayout::kMasked) {
                        routing.row_starts[expert] = expert * (shape_m / kNumExperts);
                        routing.row_ends[expert] += expert * (shape_m / kNumExperts);
                    }
                    routing.m_block_offsets[expert + 1] = lane_start + lane_ends[i];
                }
            }
            routing.lane_ends[lane_idx] = lane_end;
            if (lane_idx == 0) routing.m_block_offsets[0] = 0;
            if (lane_idx == 31) routing.num_m_blocks = lane_end;
            routing.routing_ready->arrive();
        }
    }

    CUTLASS_DEVICE void wait_ready() {
        routing.routing_ready->wait(0);
    }

    CUTLASS_DEVICE bool get_next_block(Tile &tile) {
        const uint32_t num_m_blocks = routing.num_m_blocks;
        if (next_tile >= num_m_blocks * kNumNBlocks) return false;
        tile_coordinates<kNumNBlocks, kMFastTileOrder>(next_tile, num_m_blocks, tile.m_block_idx, tile.n_block_idx);
        next_tile += gridDim.x;
        tile.route_buffer = tile_sequence++ % 2;
        if constexpr (kLayout == GroupedLayout::kIndexed) {
            tile.expert_id = expert_ids[tile.m_block_idx];
            tile.row_begin = tile.m_block_idx * BLOCK_M;
            tile.row_end = num_m_blocks * BLOCK_M;
        } else {
            // Find the expert whose M tiles contain `m_block_id`, with two ballots over the warp
            const uint32_t lane_idx = threadIdx.x % 32;
            const uint32_t chunk = __ffs(__ballot_sync(0xffffffff, routing.lane_ends[lane_idx] > tile.m_block_idx)) - 1;
            const uint32_t expert_in_chunk = chunk * kExpertsPerLane + lane_idx;
            const uint32_t experts = __ballot_sync(0xffffffff, lane_idx < kExpertsPerLane && expert_in_chunk < kNumExperts &&
                                                                   routing.m_block_offsets[expert_in_chunk + 1] > tile.m_block_idx);
            const uint32_t expert = chunk * kExpertsPerLane + __ffs(experts) - 1;
            tile.expert_id = expert;
            tile.row_begin = routing.row_starts[expert] + (tile.m_block_idx - routing.m_block_offsets[expert]) * BLOCK_M;
            tile.row_end = routing.row_ends[expert];
        }
        return true;
    }
};

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, bool kMFastTileOrder,
          bool kUseStreamK, bool kAlignedSlices, GroupedLayout kLayout, uint32_t kNumExperts>
using Scheduler = cute::conditional_t<
    kLayout != GroupedLayout::kDense,
    GroupedScheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder, kLayout, kNumExperts>,
    cute::conditional_t<kUseStreamK,
                        StreamKScheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder, kAlignedSlices>,
                        DataParallelScheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder>>>;

// NOTES: constructed directly in a kernel, NVRTC addresses the grouped routing with generic loads and stores
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, bool kMFastTileOrder,
          bool kUseStreamK, bool kAlignedSlices, GroupedLayout kLayout, uint32_t kNumExperts, class Storage>
CUTLASS_DEVICE auto make_scheduler(const GemmParams &params, Storage &smem) {
    return Scheduler<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kMFastTileOrder, kUseStreamK, kAlignedSlices, kLayout,
                     kNumExperts>(params, smem);
}

}  // namespace tensorbridge::sched
