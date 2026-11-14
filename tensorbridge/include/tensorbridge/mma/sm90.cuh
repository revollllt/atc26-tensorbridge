#pragma once

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

#include <tensorbridge/common/utils.cuh>

// SM90 WGMMA over FP8 E4M3 with the A operand in registers, after DeepGEMM's `mma/sm90.cuh` (MIT license)

namespace tensorbridge::mma::sm90 {

template <uint32_t N_, typename MMA>
struct FP8RSMMA {
    template <size_t... Idx>
    CUTLASS_DEVICE static void call_fma(const uint32_t *a, uint64_t desc_b, float *accum, bool scale_d,
                                   cute::index_sequence<Idx...>) {
        using cute::SM90::GMMA::ScaleOut;
        MMA::fma(a[0], a[1], a[2], a[3], desc_b, accum[Idx]..., scale_d ? ScaleOut::One : ScaleOut::Zero);
    }

    CUTLASS_DEVICE static void wgmma(const uint32_t *a, uint64_t desc_b, float *accum, bool scale_d) {
        call_fma(a, desc_b, accum, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr uint32_t M = 64;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = 32;
    static constexpr uint32_t kNumAccum = M * N / 128;
};

template <uint32_t N>
struct FP8RSMMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 136) return MMA_64x136x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 168) return MMA_64x168x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 176) return MMA_64x176x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 184) return MMA_64x184x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 200) return MMA_64x200x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 208) return MMA_64x208x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 216) return MMA_64x216x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 224) return MMA_64x224x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 232) return MMA_64x232x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 240) return MMA_64x240x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 248) return MMA_64x248x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_RS_TN();
    }

    using type = FP8RSMMA<N, decltype(select_mma())>;
};

// Descriptor of a K-major shared memory operand of 128B-swizzled rows
CUTLASS_DEVICE cute::GmmaDescriptor make_smem_desc(const void *smem_ptr) {
    cute::GmmaDescriptor desc;
    desc.bitfield.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(smem_ptr) >> 4);
    desc.bitfield.leading_byte_offset_ = 0;
    desc.bitfield.stride_byte_offset_ = 1024 >> 4;
    desc.bitfield.base_offset_ = 0;
    desc.bitfield.layout_type_ = static_cast<uint8_t>(cute::SM90::GMMA::LayoutType::B128);
    return desc;
}

}  // namespace tensorbridge::mma::sm90
