#pragma once

#include <tensorbridge/memory/storage.cuh>
#include <tensorbridge/ptx/ld_st.cuh>

namespace tensorbridge {

// Register -> shared: the first half of the epilogue. Because of the operand
// swap the FP32 accumulators are `[channel, token]`; the output tile is BF16
// `[token, channel]`, 128B-swizzled the way the TMA-D box expects. Both writers
// below scale by the (global-scale-folded) per-token scale, round to BF16 and
// transpose on the way out.
//
// Accumulator layout of one thread, as pairs of adjacent tokens:
//
//     accum_pairs[token_group * 2 + row]     row 0 = channel r, row 1 = r + 8
//
// where each 8-token group holds this thread's token `thread_token_in_group()`
// in `.x` / `.y` of the lane that owns its even / odd half.
template <class Config>
class OutputTileWriter {
  static constexpr uint32_t kBlockM = Config::kBlockM;
  static constexpr uint32_t kNumTokenGroups = kBlockM / 8;
  // One TMA-D box is one 128B swizzle row: 64 BF16 channels in eight 8-channel
  // groups, the group index XORed with the low three bits of the token row.
  static constexpr uint32_t kChannelsPerBox = 128 / sizeof(nv_bfloat16);
  static constexpr uint32_t kWarpsPerBox = kChannelsPerBox / Tile::kChannelsPerWarp;

  nv_bfloat16 *tile;
  const uint32_t lane_id = threadIdx.x % Tile::kWarpSize;
  const uint32_t warp_id = threadIdx.x / Tile::kWarpSize;

public:
  TB_INLINE explicit OutputTileWriter(int4 *smem_d) : tile(reinterpret_cast<nv_bfloat16 *>(smem_d)) {}

  // `stmatrix.x2.trans`: each call stores this thread's row of two 8x8 BF16
  // matrices, i.e. two 8-token groups of one 8-channel group. No value leaves
  // the registers before it is final, and the swizzle needs no address math
  // beyond the group XOR.
  TB_INLINE void write_stsm(const float *accum, const float *token_scales) {
    static_assert(has_stsm_epilogue(kBlockM));
    // Token groups are stored in blocks of four, two per `x2` store.
    constexpr uint32_t kGroupsPerBlock = 4;
    constexpr uint32_t kNumBlocks = kNumTokenGroups / kGroupsPerBlock;
    const auto *accum_pairs = reinterpret_cast<const float2 *>(accum);
    const uint32_t box = warp_id / kWarpsPerBox;
    const uint32_t parity = (lane_id % 8) / 4;

    TB_UNROLL
    for (uint32_t store = 0; store < 2 * kNumBlocks; store++) {
      const uint32_t row = store / kNumBlocks;
      const uint32_t block = store % kNumBlocks;
      const uint32_t channel_group = 2 * (warp_id % kWarpsPerBox) + row;

      uint32_t bf16_pairs[kGroupsPerBlock];
      TB_UNROLL
      for (uint32_t i = 0; i < kGroupsPerBlock; i++) {
        const uint32_t token_group = block * kGroupsPerBlock + i;
        float2 v = accum_pairs[token_group * 2 + row];
        // The odd token's scale lives in the registers of lane ^ 4.
        const float scale_self = token_scales[token_group];
        const float scale_other = __shfl_xor_sync(0xffffffff, scale_self, 4);
        v.x *= parity ? scale_other : scale_self;
        v.y *= parity ? scale_self : scale_other;
        const nv_bfloat162 rounded = __float22bfloat162_rn(v);
        bf16_pairs[i] = *reinterpret_cast<const uint32_t *>(&rounded);
      }

      TB_UNROLL
      for (uint32_t pair = 0; pair < 2; pair++) {
        // The store takes its addresses from lanes 0-15: 0-7 name the first
        // token group of the pair, 8-15 the second.
        const uint32_t token_group = block * kGroupsPerBlock + pair * 2 + (lane_id / 8) % 2;
        const uint32_t token = token_group * 8 + lane_id % 8;
        const uint32_t swizzled_group = channel_group ^ (token % 8);
        const uint32_t offset = box * (kBlockM * kChannelsPerBox) + token * kChannelsPerBox + swizzled_group * 8;
        ptx::st_shared_trans_x2(smem_addr(tile + offset), &bf16_pairs[pair * 2]);
      }
    }
  }

  // Scalar fallback for every other tile: one BF16 pair per store. The halves
  // of a pair are first regrouped across lanes so both belong to this thread's
  // token row, then placed with explicit swizzle arithmetic.
  TB_INLINE void write_scalar(const float *accum, const float *token_scales) {
    const auto *accum_pairs = reinterpret_cast<const float2 *>(accum);
    auto *tile_pairs = reinterpret_cast<nv_bfloat162 *>(tile);
    const uint32_t swizzle_base = smem_addr(tile) / 128;
    const uint32_t token_in_group = thread_token_in_group();

    TB_UNROLL
    for (uint32_t row = 0; row < 2; row++) {
      TB_UNROLL
      for (uint32_t token_group = 0; token_group < kNumTokenGroups; token_group++) {
        float2 v = accum_pairs[token_group * 2 + row];
        regroup_across_lanes(v);
        v.x *= token_scales[token_group];
        v.y *= token_scales[token_group];

        const uint32_t token = token_group * 8 + token_in_group;
        const uint32_t channel_group = (warp_id % kWarpsPerBox) * 2 + row;
        const uint32_t swizzled_group = channel_group ^ ((token_in_group + swizzle_base) % 8);
        const uint32_t box_offset = (warp_id / kWarpsPerBox) * (kBlockM * kChannelsPerBox / 2);
        tile_pairs[token * 32 + swizzled_group * 4 + lane_id / 8 + box_offset] = __float22bfloat162_rn(v);
      }
    }
  }

private:
  // Exchange one half of the pair with lane ^ 4, which holds the same channels
  // of the adjacent token.
  TB_INLINE static void regroup_across_lanes(float2 &v) {
    auto *halves = reinterpret_cast<uint32_t *>(&v);
    // Selects rather than a computed index: the pair must stay in registers.
    const bool send_x = (threadIdx.x / 4) % 2;
    const uint32_t received = __shfl_xor_sync(0xffffffff, send_x ? halves[0] : halves[1], 4);
    if (send_x) {
      halves[0] = received;
    } else {
      halves[1] = received;
    }
  }
};

}  // namespace tensorbridge
