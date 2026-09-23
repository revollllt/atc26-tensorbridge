#pragma once

#include <tensorbridge/common/base.cuh>

// FPMA bridge: NVFP4 (E2M1 value + E4M3 scale per 16 values) -> FP8 E4M3,
// entirely in registers, with integer instructions only.
//
// An E2M1 magnitude `eem` and an E4M3 byte `seeeemmm` line up if the magnitude
// is shifted left by two: its exponent lands on the low E4M3 exponent bits and
// its mantissa bit on the top E4M3 mantissa bit. Multiplying by the scale is
// then an *addition in the encoding domain*:
//
//     byte = prefolded_scale + (sign << 7 | magnitude << 2)
//
// where `prefolded_scale = raw_e4m3_scale - 0x1C` is computed offline. The
// result decodes to approximately `scale * value / 6`: mantissas are added
// rather than multiplied, which is the approximation the paper quantifies. The
// host folds the `6` into the global scale (`quant/nvfp4.py`). Scales are kept
// in `[0x1C, 0x7E]`, so the largest sum is `0x62 + 0x9C = 0xFE` and the four
// byte lanes of a word never carry into each other.
//
// Two E2M1 codes break the pattern and are recoded offline (SNC, "stored
// nonzero convention"):
//
//     0.5  is E2M1's subnormal: exponent field 0, mantissa bit set. Shifted as
//          is, that bit would land on the E4M3 mantissa and decode to 2/3 of
//          1.0. It is stored as magnitude 0 instead, one exponent step below
//          1.0 (magnitude 2), which is exactly half.
//     0    is stored as the magnitude 1 this frees up, and cleared by a mask.
//
// One packed word holds eight nibbles: the low nibble of each byte belongs to
// the K-low half of the k32 call, the high nibble to the K-high half. The two
// halves fall into different scale groups.

namespace tensorbridge::fpma {

// `(a & b) | c` in one LOP3.
TB_INLINE uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"((0xF0 & 0xCC) | 0xAA));
  return res;
}

// 0xFF per byte lane whose (shifted) magnitude is not the zero marker.
TB_INLINE uint32_t nonzero_mask(uint32_t magnitude_shifted) {
  uint32_t mask;
  asm volatile(
      "{ .reg .u32 t; "
      "xor.b32 t, %1, 0x04040404; "
      "add.u32 t, t, 0x7f7f7f7f; "
      "prmt.b32 %0, t, 0, 0xBA98; }"
      : "=r"(mask)
      : "r"(magnitude_shifted));
  return mask;
}

// Same masks for both halves, looked up from the nibbles directly: four PRMTs
// and no arithmetic, at the price of two more live registers.
TB_INLINE uint2 nonzero_masks_lut(uint32_t packed) {
  uint32_t masks_01, masks_23, masks_lo, masks_hi;
  asm volatile("prmt.b32 %0, 0xffff00ff, 0xffffffff, %1;"
               : "=r"(masks_01)
               : "r"(packed));
  asm volatile("prmt.b32 %0, 0xffff00ff, 0xffffffff, %1;"
               : "=r"(masks_23)
               : "r"(packed >> 16));
  asm volatile("prmt.b32 %0, %1, %2, 0x6420;"
               : "=r"(masks_lo)
               : "r"(masks_01), "r"(masks_23));
  asm volatile("prmt.b32 %0, %1, %2, 0x7531;"
               : "=r"(masks_hi)
               : "r"(masks_01), "r"(masks_23));
  return {masks_lo, masks_hi};
}

// One packed word -> the K-low and K-high FP8 words. `scale_lo4` / `scale_hi4`
// carry the prefolded scale byte replicated into all four lanes.
template <bool kLutMasks>
TB_INLINE void dequant_word(uint32_t packed, uint32_t scale_lo4, uint32_t scale_hi4,
                            uint32_t &fp8_lo, uint32_t &fp8_hi) {
  const uint32_t lo_magnitude = (packed & 0x07070707u) << 2;
  const uint32_t hi_magnitude = (packed & 0x70707070u) >> 2;

  const uint32_t lo_addend = and_or(packed << 4, 0x80808080u, lo_magnitude);
  const uint32_t hi_addend = and_or(packed, 0x80808080u, hi_magnitude);
  const uint32_t lo_sum = scale_lo4 + lo_addend;
  const uint32_t hi_sum = scale_hi4 + hi_addend;

  if constexpr (kLutMasks) {
    const uint2 masks = nonzero_masks_lut(packed);
    fp8_lo = lo_sum & masks.x;
    fp8_hi = hi_sum & masks.y;
  } else {
    fp8_lo = lo_sum & nonzero_mask(lo_magnitude);
    fp8_hi = hi_sum & nonzero_mask(hi_magnitude);
  }
}

// One k32 call's worth of weights for this thread: two packed words (channel
// rows `r` and `r + 8`) -> the four-register WGMMA operand
//
//     fragment[0] = row r,     K-low      fragment[2] = row r,     K-high
//     fragment[1] = row r + 8, K-low      fragment[3] = row r + 8, K-high
//
// `scales` holds the four prefolded scales in the same order, either as four
// bytes of one word or, when `kPrebroadcast`, as four lane-replicated words.
// Prebroadcasting moves the replication off the dequant path into the register
// load (`memory/s2r.cuh`) and costs three registers per buffer.
template <bool kPrebroadcast>
TB_INLINE void dequant(const uint32_t *packed, const uint32_t *scales, uint32_t *fragment) {
  TB_UNROLL
  for (uint32_t row = 0; row < 2; row++) {
    uint32_t scale_lo4, scale_hi4;
    if constexpr (kPrebroadcast) {
      scale_lo4 = scales[row];
      scale_hi4 = scales[2 + row];
    } else {
      const auto *bytes = reinterpret_cast<const uint8_t *>(scales);
      scale_lo4 = uint32_t(bytes[row]) * 0x01010101u;
      scale_hi4 = uint32_t(bytes[2 + row]) * 0x01010101u;
    }
    dequant_word<kPrebroadcast>(packed[row], scale_lo4, scale_hi4, fragment[row], fragment[2 + row]);
  }
}

}  // namespace tensorbridge::fpma
