#pragma once

#include <cstdint>
#include <cuda.h>

namespace tensorbridge {

// Arguments of the kernels under `impls/`, passed as one `__grid_constant__`
// struct; the launcher fills it (`csrc/launcher/launcher.cpp`), so both sides
// compile this header. Device code uses the tensor maps by address and never
// copies the struct: a TMA descriptor must stay in the parameter space. CUDA
// 13's `cuda.h` aligns `CUtensorMap` to 128 bytes, the JIT's declaration to 64
// (`jit/compiler.py`); the struct's own alignment makes both sides agree.
//
// The Stream-K split of the grid's CTAs (`sched::StreamKScheduler`), computed by the launcher with
// the selector's `get_schedule` (`csrc/heuristics/sm90_common.hpp`), so that the CTAs do not divide
// by `gridDim.x`. `x / tail_divisor` is `__umulhi(x, divisor_multiplier) >> divisor_shift` for
// `x < 2^31`, or `x` when `divisor_multiplier` is 0 (`tail_divisor` 1).
struct StreamKSplit {
    uint32_t dp_tiles_per_cta;  // whole tiles per CTA before the tail
    uint32_t tail_divisor;      // K blocks of the tail per CTA, or the tail tiles with aligned slices; 0 without a tail
    uint32_t divisor_multiplier;
    uint32_t divisor_shift;
    uint32_t aligned_slices;    // aligned slices: the slices of each tail tile
};

// A grouped kernel (`GroupedLayout`, `common/utils.cuh`) also reads the routing
// of its layout; the other routing pointers are null.
struct alignas(128) GemmParams {
    CUtensorMap tensor_map_a;    // A [M, K] FP8: boxes of 128 B x BLOCK_M tokens, 128B swizzle
    CUtensorMap tensor_map_b;    // B [E x N, K / 8] int32: boxes of 16 words x 128 channels, 64B swizzle
    CUtensorMap tensor_map_sfb;  // SFB [E x K / 16, N] E4M3: boxes of 128 channels x 8 scale groups; or, by K
                                 // blocks, [K / 128, N] rows of 8 scale groups: boxes of 128 channels
    CUtensorMap tensor_map_d;    // D [M, N] BF16: boxes of 64 channels x BLOCK_M tokens, 128B swizzle
    const float *sfa;            // [M] per-token activation scales
    const float *global_scale;   // [E], one per expert ([1] dense)
    int32_t *locks;              // Stream-K lock words, one per tile of the tail (the kernel's epilogue)
    uint32_t sfa_stride_m, sfa_stride_k;  // group-scale element strides
    const int64_t *expert_offsets;     // contiguous: [E + 1] first row of each expert
    const int32_t *expert_counts;      // masked: [E] rows of each expert
    const int32_t *sorted_ids;         // indexed: route IDs, sorted by expert and padded per tile
    const int32_t *expert_ids;         // indexed: the expert of each tile
    const int32_t *num_tokens_padded;  // indexed: [1] routes after padding
    const uint8_t *a;                  // indexed: A, gathered by rows
    void *d;                           // grouped: D, written by rows (BF16)
    uint32_t shape_m;
    uint32_t num_routes;  // output rows: M x top_k indexed, else M
    uint32_t top_k;       // indexed: routes per input row
    StreamKSplit stream_k;  // Stream-K kernels only
    uint64_t sfa_stride_expert;  // masked scales may have a strided expert dimension
    float *d_scale;              // fused SwiGLU+FP8 epilogue: [M, OUTPUT_N / 256] FP32 group-128 output scales
};

// Every member is naturally aligned, so size and alignment pin the layout (NVRTC has no `offsetof`).
static_assert(sizeof(CUtensorMap) == 128 && sizeof(GemmParams) == 768 && alignof(GemmParams) == 128,
              "The launcher and the kernels must agree on the layout");

}  // namespace tensorbridge
