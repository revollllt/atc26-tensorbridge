#pragma once

#include <tensorbridge/common/utils.cuh>

// FPMA bridge: NVFP4 -> FP8 E4M3 in registers with integer instructions.

namespace tensorbridge::fpma {

// `(a & b) | c` in one LOP3
// NOTES: the asm here is not `volatile`, so the compiler is free to schedule it
CUTLASS_DEVICE uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res)
        : "r"(a), "r"(b), "r"(c), "n"((0xF0 & 0xCC) | 0xAA));
    return res;
}

// `(sign & 0x80808080) | (positive & 0x7F7F7F7F)` in one LOP3
CUTLASS_DEVICE uint32_t merge_sign(uint32_t sign_bits, uint32_t positive) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res)
        : "r"(sign_bits), "r"(positive), "n"(0x80808080), "n"((0xF0 & 0xAA) | (0xCC & ~0xAA & 0xFF)));
    return res;
}

// Four FP8 bytes of one half-word, by a table lookup on the prefolded scale
template <bool kExact = false>
CUTLASS_DEVICE uint32_t lut_half(uint32_t selectors, uint32_t sign_bits, uint32_t scale) {
    uint32_t positive;
    if constexpr (kExact) {
        const uint32_t lut_03 = scale * 0x01010001u + 0x19100000u;  // [Q, 0, Q + 16, Q + 25]
        const uint32_t lut_47 = scale * 0x01010101u + 0x39302920u;  // [Q + 32, Q + 41, Q + 48, Q + 57]
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(positive) : "r"(lut_03), "r"(lut_47), "r"(selectors));
        return merge_sign(sign_bits, positive >> 1);
    } else {
        const uint32_t lut_03 = scale * 0x01010001u + 0x0C080000u;  // [s, 0, s + 8, s + 12]
        const uint32_t lut_47 = scale * 0x01010101u + 0x1C181410u;  // [s + 16, s + 20, s + 24, s + 28]
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(positive) : "r"(lut_03), "r"(lut_47), "r"(selectors));
        return and_or(sign_bits, 0x80808080u, positive);
    }
}

// One k32 call's FP8 WGMMA operand from two packed words and four prefolded scales
template <bool kExact = false>
CUTLASS_DEVICE void dequant(const uint32_t *packed, const uint32_t *scales, uint32_t *fragment) {
    for (uint32_t row = 0; row < 2; row++) {
        const uint32_t selectors = packed[row] & 0x77777777u;
        fragment[row] = lut_half<kExact>(selectors, packed[row] << 4, scales[row]);
        fragment[2 + row] = lut_half<kExact>(selectors >> 16, packed[row], scales[2 + row]);
    }
}

// Shared memory -> registers: the weights of two k32 calls, or one
CUTLASS_DEVICE void load_weights_pair(const uint8_t *smem_b, uint32_t k_iter, uint32_t thread_idx, uint32_t *packed_0,
                                 uint32_t *packed_1) {
    const uint4 words = *reinterpret_cast<const uint4 *>(smem_b + (k_iter / 2) * 4096 + thread_idx * sizeof(uint4));
    packed_0[0] = words.x, packed_0[1] = words.y;
    packed_1[0] = words.z, packed_1[1] = words.w;
}

CUTLASS_DEVICE void load_weights(const uint8_t *smem_b, uint32_t k_iter, uint32_t thread_idx, uint32_t *packed) {
    const uint2 words = *reinterpret_cast<const uint2 *>(smem_b + (k_iter / 2) * 4096 + thread_idx * sizeof(uint4) +
                                                         (k_iter & 1) * sizeof(uint2));
    packed[0] = words.x, packed[1] = words.y;
}

// Shared memory -> registers: one k32 call's four scale bytes, in `dequant`'s order
CUTLASS_DEVICE void load_scales(const uint8_t *smem_sfb, uint32_t k_iter, uint32_t r_0, uint32_t *scales) {
    const uint8_t *k_low = smem_sfb + (k_iter * 2) * 128 + r_0;
    const uint8_t *k_high = k_low + 128;
    scales[0] = k_low[0];
    scales[1] = k_low[8];
    scales[2] = k_high[0];
    scales[3] = k_high[8];
}

// Byte `kIdx` of `word`, zero-extended, in one PRMT
template <uint32_t kIdx>
CUTLASS_DEVICE uint32_t byte_of(uint32_t word) {
    uint32_t res;
    const uint32_t zero = 0;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(word), "r"(zero), "n"(0x4440 + kIdx));
    return res;
}

// Shared memory -> registers: a K block's scale bytes of the thread's two channels, 8 per channel
// (`tensorbridge.experimental.prepare`), as words `[row 0 low, row 1 low, row 0 high, row 1 high]`
CUTLASS_DEVICE void load_scale_rows(const uint8_t *smem_sfb, uint32_t r_0, uint32_t *rows) {
    const uint2 row_0 = *reinterpret_cast<const uint2 *>(smem_sfb + r_0 * 8);
    const uint2 row_1 = *reinterpret_cast<const uint2 *>(smem_sfb + (r_0 + 8) * 8);
    rows[0] = row_0.x, rows[1] = row_1.x, rows[2] = row_0.y, rows[3] = row_1.y;
}

// One k32 call's four scale bytes of `load_scale_rows`, in `dequant`'s order
CUTLASS_DEVICE void scale_bytes(const uint32_t *rows, uint32_t k_iter, uint32_t *scales) {
    const uint32_t word_0 = rows[k_iter / 2 * 2], word_1 = rows[k_iter / 2 * 2 + 1];
    if (k_iter % 2 == 0) {
        scales[0] = byte_of<0>(word_0), scales[1] = byte_of<0>(word_1);
        scales[2] = byte_of<1>(word_0), scales[3] = byte_of<1>(word_1);
    } else {
        scales[0] = byte_of<2>(word_0), scales[1] = byte_of<2>(word_1);
        scales[2] = byte_of<3>(word_0), scales[3] = byte_of<3>(word_1);
    }
}

}  // namespace tensorbridge::fpma
