#pragma once

#include <tensorbridge/memory/storage.cuh>
#include <tensorbridge/ptx/barrier.cuh>
#include <tensorbridge/ptx/cp_async.cuh>
#include <tensorbridge/ptx/tma.cuh>

namespace tensorbridge {

// Global -> shared: the producer warp group. It streams the K blocks of a tile
// into the stage ring, staying `kNumStages - 1` slots ahead of the consumers:
// activations and packed weights by TMA (one issuing thread), weight scales and
// per-token scales by per-thread async copies.
//
// With multicast, one tile is shared by the CTAs of a cluster: along `A` the
// cluster shares the activation tile and splits the channel blocks, along `B`
// it shares the weight tile and splits the token blocks. Only the cluster
// leader issues the shared load; it lands in every peer's shared memory.
template <class Config>
class Producer {
  using Storage = SharedStorage<Config>;
  static constexpr uint32_t kNumStages = Config::kNumStages;
  static constexpr uint32_t kMulticast = Config::kMulticastA * Config::kMulticastB;
  static constexpr uint32_t kTmaBytesPerStage = Storage::kStageBytesA + Storage::kStageBytesB;

  // Weight scales are `[K / 16, N]` in global memory, 16 bytes per copy.
  static constexpr uint32_t kScaleRowInt4s = Config::kShapeN / 16;
  static constexpr uint32_t kScaleTileInt4s = Storage::kStageBytesBS / sizeof(int4);
  static constexpr uint32_t kScaleTileRowInt4s = Tile::kBlockN / 16;

  Storage &smem;
  const CUtensorMap *tensor_map_a;
  const CUtensorMap *tensor_map_b;
  const int4 *gmem_bs;
  const float *gmem_as;

  const uint32_t thread_id = threadIdx.x - Tile::kNumMathThreads;
  const uint32_t cluster_rank = blockIdx.x % kMulticast;
  const bool is_leader = kMulticast == 1 || cluster_rank == 0;

  // Cursor of the next K block to load, and the token rows of the current tile.
  uint32_t m_idx, n_block_id, k_block_id, num_tokens;

  uint32_t stage_phase[kNumStages + 1] = {0};
  uint32_t empty_phase[kNumStages] = {0};

public:
  TB_INLINE
  Producer(Storage &smem, const CUtensorMap *tensor_map_a, const CUtensorMap *tensor_map_b,
           const float *gmem_as, const uint8_t *gmem_bs)
      : smem(smem), tensor_map_a(tensor_map_a), tensor_map_b(tensor_map_b),
        gmem_bs(reinterpret_cast<const int4 *>(gmem_bs)), gmem_as(gmem_as) {
    if (thread_id == 0) {
      ptx::tma_prefetch_descriptor(tensor_map_a);
      ptx::tma_prefetch_descriptor(tensor_map_b);
    }
    __syncwarp();
  }

  // Thread `i` initializes barrier slot `i`. A load barrier opens after every
  // copying thread plus the TMA issuer arrived; a math barrier after one lane
  // per consumer warp did.
  TB_INLINE void init_barriers() {
    if (thread_id < kNumStages + 1) __mbarrier_init(&smem.load_mbar[thread_id], Tile::kNumLoadThreads + 1);
    if (thread_id == Storage::kTokenScaleSlot) __mbarrier_init(&smem.load_mbar[thread_id], Tile::kNumLoadThreads);
    if (thread_id < kNumStages + 1)
      __mbarrier_init(&smem.math_mbar[thread_id], Tile::kNumMathThreads / Tile::kWarpSize);
    if constexpr (kMulticast > 1) {
      if (thread_id < kNumStages) __mbarrier_init(&smem.empty_mbar[thread_id], kMulticast - 1);
    }
  }

  TB_INLINE void seek(uint32_t m_block_id, uint32_t n_block_id_, uint32_t k_block_id_, uint32_t shape_m) {
    m_idx = m_block_id * Config::kBlockM;
    n_block_id = n_block_id_;
    k_block_id = k_block_id_;
    num_tokens = min_of(shape_m - m_idx, Config::kBlockM);
  }

  // Load the next K block into ring slot `stage_id` and advance the cursor.
  template <bool kIsFirst = false>
  TB_INLINE void load_stage(uint32_t stage_id, bool pred = true) {
    if (!pred) return;
    stage_id = stage_id % kNumStages;
    uint64_t *mbar = &smem.load_mbar[kIsFirst ? Storage::kFirstStageSlot : stage_id];

    // A peer's barrier may receive the leader's multicast bytes before its own,
    // and PTX requires `expect_tx` to precede them.
    if constexpr (kMulticast > 1) {
      if (!is_leader && thread_id == 0) ptx::tma_expect_tx(mbar, kTmaBytesPerStage);
      if (!is_leader) __syncwarp();
    }

    load_activations(smem.a[stage_id], mbar);
    load_weights(smem.b[stage_id], mbar);
    load_weight_scales(smem.bs[stage_id]);
    k_block_id++;

    ptx::cp_async_arrive(mbar);
    if (is_leader && thread_id == 0) ptx::tma_expect_tx(mbar, kTmaBytesPerStage);
    if (is_leader) __syncwarp();
  }

  TB_INLINE void load_token_scales() {
    float *smem_as = reinterpret_cast<float *>(smem.as);
    TB_UNROLL
    for (uint32_t i = 0; i < ceil_div(Config::kBlockM, Tile::kNumLoadThreads); i++) {
      const uint32_t token = i * Tile::kNumLoadThreads + thread_id;
      ptx::cp_async_load_pred(gmem_as + m_idx + token, smem_as + token, token < num_tokens);
    }
    ptx::cp_async_arrive(&smem.load_mbar[Storage::kTokenScaleSlot]);
  }

  TB_INLINE void wait_stage_consumed(uint32_t stage_id) {
    ptx::mbarrier_wait(&smem.math_mbar[stage_id], stage_phase[stage_id]);
    stage_phase[stage_id] ^= 1;
    if constexpr (kMulticast > 1) {
      if (cluster_rank == 0) {
        ptx::mbarrier_wait(&smem.empty_mbar[stage_id], empty_phase[stage_id]);
        empty_phase[stage_id] ^= 1;
      }
    }
  }

  // The output tile overlays the ring, so it must be stored before a refill.
  TB_INLINE void wait_tile_stored() {
    ptx::mbarrier_wait(&smem.math_mbar[Storage::kEpilogueSlot], stage_phase[Storage::kEpilogueSlot]);
    stage_phase[Storage::kEpilogueSlot] ^= 1;
  }

private:
  TB_INLINE void load_activations(int4 *smem_ptr, void *mbar) {
    if (thread_id != 0) return;
    const uint32_t k_idx = k_block_id * Tile::kBlockK;
    if constexpr (Config::kMulticastA == 1) {
      ptx::tma_load_2d(tensor_map_a, smem_ptr, mbar, k_idx, m_idx);
    } else if (blockIdx.x % Config::kMulticastA == 0) {
      ptx::tma_load_2d<Config::kMulticastA>(tensor_map_a, smem_ptr, mbar, k_idx, m_idx);
    }
  }

  // The weight descriptor counts K in int32 words (16 per block) and N in rows.
  TB_INLINE void load_weights(int4 *smem_ptr, void *mbar) {
    if (thread_id != 0) return;
    const uint32_t k_word_idx = k_block_id * (Tile::kBlockK / 8);
    const uint32_t n_idx = n_block_id * Tile::kBlockN;
    if constexpr (Config::kMulticastB == 1) {
      ptx::tma_load_2d(tensor_map_b, smem_ptr, mbar, k_word_idx, n_idx);
    } else if (blockIdx.x % Config::kMulticastB == 0) {
      ptx::tma_load_2d<Config::kMulticastB>(tensor_map_b, smem_ptr, mbar, k_word_idx, n_idx);
    }
  }

  TB_INLINE void load_weight_scales(int4 *smem_ptr) {
    const uint32_t group_row = thread_id / kScaleTileRowInt4s;
    const uint32_t col = thread_id % kScaleTileRowInt4s;
    const int4 *src = gmem_bs + (k_block_id * Tile::kScaleGroupsPerBlock + group_row) * kScaleRowInt4s +
                      n_block_id * kScaleTileRowInt4s + col;
    ptx::cp_async_load_pred(src, smem_ptr + thread_id, thread_id < kScaleTileInt4s);
  }
};

}  // namespace tensorbridge
