#pragma once

#include <tensorbridge/common/tile.cuh>

namespace tensorbridge {

// Shared-memory layout of one CTA. Inputs are staged in a ring of `kNumStages`
// slots, one K block (128 deep) per slot:
//
//     a   [BLOCK_M, 128]   FP8 activations, 128B-swizzled rows
//     b   [128, 64]        packed NVFP4 weights in thread order (`quant/nvfp4.py`)
//     bs  [8, 128]         prefolded E4M3 weight scales, one row per scale group
//
// The per-token activation scales are only needed by the epilogue and are
// staged once per tile. The BF16 output tile `d` overlays the ring: the producer
// does not refill it until the consumers have stored the tile, so the ring can
// take the whole shared memory while K blocks stream through it.
template <class Config>
struct SharedStorage {
  static constexpr uint32_t kNumStages = Config::kNumStages;

  static constexpr uint32_t kStageBytesA = Config::kBlockM * Tile::kBlockK;
  static constexpr uint32_t kStageBytesB = Tile::kBlockN * Tile::kBlockK / 2;
  static constexpr uint32_t kStageBytesBS = Tile::kScaleGroupsPerBlock * Tile::kBlockN;
  static constexpr uint32_t kBytesAS = Config::kBlockM * sizeof(float);
  static constexpr uint32_t kBytesD = Config::kBlockM * Tile::kBlockN * sizeof(nv_bfloat16);

  // Barrier slots beyond the per-stage ones.
  static constexpr uint32_t kFirstStageSlot = kNumStages;   // load: first K block of a tile
  static constexpr uint32_t kTokenScaleSlot = kNumStages + 1;  // load: per-token scales
  static constexpr uint32_t kEpilogueSlot = kNumStages;     // math: output tile stored

  union alignas(128) {
    struct {
      int4 a[kNumStages][kStageBytesA / sizeof(int4)];
      int4 b[kNumStages][kStageBytesB / sizeof(int4)];
      int4 bs[kNumStages][kStageBytesBS / sizeof(int4)];
      int4 as[kBytesAS / sizeof(int4)];
    };
    int4 d[kBytesD / sizeof(int4)];
  };

  // load_mbar[s]   producer -> consumers: slot `s` holds a loaded K block
  // math_mbar[s]   consumers -> producer: slot `s` was consumed
  // empty_mbar[s]  cluster peers -> leader: the multicast copy was consumed too
  alignas(128) uint64_t load_mbar[kNumStages + 2];
  uint64_t math_mbar[kNumStages + 1];
  uint64_t empty_mbar[kNumStages];
};

}  // namespace tensorbridge
