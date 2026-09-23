#pragma once

#include <tensorbridge/memory/storage.cuh>
#include <tensorbridge/ptx/barrier.cuh>

namespace tensorbridge {

// Shared -> register: the consumer warp groups. A consumer may read ring slot
// `s` only after passing `load_mbar[s]`. Releasing `s` through `math_mbar[s]`
// lets the producer refill the slot *behind* it, so `s` itself stays readable
// until the next release.

// Registers holding the weights of a k32 call, before and after the bridge. A
// tile with four buffers can hold a whole K block, which lets the mainloop load
// and dequantize ahead of the WGMMAs in flight; see `common/tile.cuh`.
template <class Config>
struct WeightRegisters {
  static constexpr uint32_t kNumBuffers = has_four_register_buffers(Config::kBlockM) ? 4 : 2;
  static constexpr uint32_t kNumScaleWords = Config::kPrebroadcastScale ? 4 : 1;

  uint32_t packed[kNumBuffers][2];                // NVFP4 nibbles: channel rows r, r + 8
  uint32_t scales[kNumBuffers][kNumScaleWords];   // prefolded E4M3 scales, see `dequant/fpma.cuh`
  uint32_t fragment[kNumBuffers][4];              // FP8 WGMMA operand
};

template <class Config>
class Consumer {
  using Storage = SharedStorage<Config>;
  static constexpr uint32_t kNumStages = Config::kNumStages;
  static constexpr uint32_t kMulticast = Config::kMulticastA * Config::kMulticastB;

  // Offline repacking (`quant/nvfp4.py`) lays a weight tile out in thread order:
  // two 4096-byte halves, each holding one 16-byte vector per math thread that
  // covers two consecutive k32 calls.
  static constexpr uint32_t kBytesPerCallPair = Tile::kNumMathThreads * sizeof(uint4);

  Storage &smem;
  const uint32_t lane_id = threadIdx.x % Tile::kWarpSize;
  const uint32_t cluster_rank = blockIdx.x % kMulticast;
  uint32_t load_phase[kNumStages + 2] = {0};

public:
  TB_INLINE explicit Consumer(Storage &smem) : smem(smem) {}

  template <bool kIsFirst = false>
  TB_INLINE void wait_stage(uint32_t stage_id) {
    const uint32_t slot = kIsFirst ? Storage::kFirstStageSlot : stage_id % kNumStages;
    ptx::mbarrier_wait(&smem.load_mbar[slot], load_phase[slot]);
    load_phase[slot] ^= 1;
  }

  // Probe early, while WGMMAs are still in flight, so that the blocking wait at
  // the stage seam is skipped when the next slot has already landed.
  TB_INLINE uint32_t probe_stage(uint32_t stage_id) {
    return ptx::mbarrier_try_wait(&smem.load_mbar[stage_id], load_phase[stage_id]);
  }

  TB_INLINE void wait_stage(uint32_t stage_id, uint32_t already_passed) {
    if (!already_passed) {
      ptx::mbarrier_wait(&smem.load_mbar[stage_id], load_phase[stage_id]);
    } else {
      asm volatile("" ::: "memory");
    }
    load_phase[stage_id] ^= 1;
  }

  TB_INLINE void wait_token_scales() {
    ptx::mbarrier_wait(&smem.load_mbar[Storage::kTokenScaleSlot], load_phase[Storage::kTokenScaleSlot]);
    load_phase[Storage::kTokenScaleSlot] ^= 1;
  }

  // `stage_id` may also be `kEpilogueSlot`: the output tile has left the ring.
  TB_INLINE void release(uint32_t stage_id) {
    if (lane_id == 0) ptx::mbarrier_arrive(&smem.math_mbar[stage_id]);
    if constexpr (kMulticast > 1) {
      if (stage_id < kNumStages)
        ptx::mbarrier_arrive_cluster(&smem.empty_mbar[stage_id], 0, cluster_rank >= 1 && threadIdx.x == 0);
    }
    __syncwarp();
  }

  // Weights of k32 calls `k_iter` and `k_iter + 1` in one 16-byte read.
  TB_INLINE void load_weights_pair(uint32_t stage_id, uint32_t k_iter, uint32_t *packed_0, uint32_t *packed_1) {
    const auto *tile = reinterpret_cast<const uint8_t *>(smem.b[stage_id]);
    const uint4 v = *reinterpret_cast<const uint4 *>(
        tile + (k_iter / 2) * kBytesPerCallPair + threadIdx.x * sizeof(uint4));
    packed_0[0] = v.x, packed_0[1] = v.y;
    packed_1[0] = v.z, packed_1[1] = v.w;
  }

  TB_INLINE void load_weights(uint32_t stage_id, uint32_t k_iter, uint32_t *packed) {
    const auto *tile = reinterpret_cast<const uint8_t *>(smem.b[stage_id]);
    const uint2 v = *reinterpret_cast<const uint2 *>(
        tile + (k_iter / 2) * kBytesPerCallPair + threadIdx.x * sizeof(uint4) + (k_iter & 1) * sizeof(uint2));
    packed[0] = v.x, packed[1] = v.y;
  }

  // The four scales of k32 call `k_iter` for this thread, in fragment order:
  // {row r K-low, row r + 8 K-low, row r K-high, row r + 8 K-high}.
  TB_INLINE void load_scales(uint32_t stage_id, uint32_t k_iter, uint32_t *scales) {
    const auto *k_low = reinterpret_cast<const uint8_t *>(smem.bs[stage_id]) +
                        (k_iter * 2) * Tile::kBlockN + thread_channel_base();
    const auto *k_high = k_low + Tile::kBlockN;
    if constexpr (Config::kPrebroadcastScale) {
      scales[0] = uint32_t(k_low[0]) * 0x01010101u;
      scales[1] = uint32_t(k_low[8]) * 0x01010101u;
      scales[2] = uint32_t(k_high[0]) * 0x01010101u;
      scales[3] = uint32_t(k_high[8]) * 0x01010101u;
    } else {
      scales[0] = uint32_t(k_low[0]) | (uint32_t(k_low[8]) << 8) |
                  (uint32_t(k_high[0]) << 16) | (uint32_t(k_high[8]) << 24);
    }
  }

  // One scale per 8-token group: the token row this thread owns in each group.
  TB_INLINE void load_token_scales(float *token_scales) {
    const float *smem_as = reinterpret_cast<const float *>(smem.as);
    TB_UNROLL
    for (uint32_t i = 0; i < Config::kBlockM / 8; i++) {
      token_scales[i] = smem_as[i * 8 + thread_token_in_group()];
    }
  }
};

}  // namespace tensorbridge
