#pragma once

#include <algorithm>
#include <cstdint>

#include "config.hpp"

// What selection on SM90 needs of the kernels: the token tiles, the shared-memory footprint, the census of clean
// token tiles, legality and the device scheduler's mirror. The rule itself is in `sm90_rule.hpp`.

namespace tensorbridge::sm90 {

// Token tiles of 8 to 256 in steps of 8.
constexpr int64_t kBlockMStep = 8, kMaxBlockM = 256;
// The token tiles of the group-scale kernel (`static_assert` of `impls/sm90_w4a8_pgroup.cuh`).
constexpr int64_t kMinGroupScaleBlockM = 16, kMaxGroupScaleBlockM = 192;

// `sizeof(layout::PtokenStorage<...>)` of `layout/sm90.cuh`: the ring, then the output tile (in 128-row chunks
// for 128- and 256-token tiles) and two per-token scale buffers beside it, the barriers, and a grouped
// layout's routing (`layout::Routing`).
// With group scales, `sizeof(layout::PgroupStorage<...>)` of `layout/sm90.cuh`: each stage holds
// its K block's activation scales, and there are no per-token scale buffers.
inline int64_t get_smem_bytes(const GemmConfig &config, int64_t num_experts = 0,
                              GroupedLayout layout = GroupedLayout::Dense, bool group_scales = false) {
    const int64_t stage = config.block_m * config.block_k + config.block_n * config.block_k / 2 +
                          config.block_k / 16 * config.block_n;
    const int64_t output_rows = config.block_m % 128 == 0 ? 128 : config.block_m;
    if (group_scales) {
        const int64_t data = config.num_stages * (stage + config.block_m * 4) + output_rows * config.block_n * 2;
        const int64_t stages = ceil_div(data, 128) * 128 + ceil_div(2 * config.num_stages * 8, 128) * 128;
        if (layout == GroupedLayout::Dense) return stages;
        const int64_t routing = layout == GroupedLayout::Indexed ? 16 * config.block_m + 32 : 12 * num_experts + 144;
        return stages + ceil_div(routing, 128) * 128;
    }
    const int64_t data = config.num_stages * stage + output_rows * config.block_n * 2 + 2 * config.block_m * 4;
    const int64_t stages = ceil_div(data, 128) * 128 + ceil_div(2 * (config.num_stages + 2) * 8, 128) * 128;
    if (layout == GroupedLayout::Dense) return stages;
    const int64_t routing = layout == GroupedLayout::Indexed ? 16 * config.block_m + 32 : 12 * num_experts + 144;
    return stages + ceil_div(routing, 128) * 128;
}

// Token tiles that compile without spilling registers and without WGMMAs that ptxas serializes (C7511 to C7513), by
// CTAs per SM, block shape, Stream-K, layout and ring depth: bit `i` stands for `block_m = 8 * (i + 1)`.
// Contiguous and masked kernels share a kind, which keeps a token tile only where both are clean.

// The deepest ring the census covers.
constexpr int64_t kMaxCensusStages = 16;

inline uint32_t clean_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t block_k, bool use_stream_k,
                               GroupedLayout layout, int64_t num_stages, bool group_scales = false) {
    // Indexed by (dense, indexed, contiguous and masked, or dense with group scales; num_stages - 3;
    // ctas_per_sm - 1; block_n / 256; block_k / 256; use_stream_k).
    constexpr uint32_t kMasks[4][kMaxCensusStages - 2][2][2][2][2] = {
            {
            {{{{0xffffffff, 0xffffffff}, {0x001fffff, 0x001fffff}}, {{0x007fffff, 0x001fffff}, {0x000007ff, 0x000007ff}}}, {{{0x00000fff, 0x00000fff}, {0x0000007f, 0x0000007f}}, {{0x0000003f, 0x0000003f}, {0x00000000, 0x00000000}}}},
            {{{{0xffffffff, 0xffffffff}, {0x00007fff, 0x00007fff}}, {{0x0007ffff, 0x0007ffff}, {0x0000003f, 0x0000003f}}}, {{{0x00000fff, 0x00000fff}, {0x0000000f, 0x0000000f}}, {{0x0000001f, 0x0000001f}, {0x00000000, 0x00000000}}}},
            {{{{0x01ffffff, 0x01ffffff}, {0x000007ff, 0x000007ff}}, {{0x00007fff, 0x00007fff}, {0x00000007, 0x00000007}}}, {{{0x000001ff, 0x000001ff}, {0x00000001, 0x00000001}}, {{0x00000003, 0x00000003}, {0x00000000, 0x00000000}}}},
            {{{{0x001fffff, 0x001fffff}, {0x000000ff, 0x000000ff}}, {{0x000007ff, 0x000007ff}, {0x00000000, 0x00000000}}}, {{{0x0000007f, 0x0000007f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0003ffff, 0x0003ffff}, {0x0000003f, 0x0000003f}}, {{0x000001ff, 0x000001ff}, {0x00000000, 0x00000000}}}, {{{0x0000000f, 0x0000001f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x00007fff, 0x00007fff}, {0x0000000f, 0x0000000f}}, {{0x0000003f, 0x0000003f}, {0x00000000, 0x00000000}}}, {{{0x0000000f, 0x0000000f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x00001fff, 0x00001fff}, {0x00000007, 0x00000007}}, {{0x0000000f, 0x0000000f}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000003}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x000007ff, 0x000007ff}, {0x00000003, 0x00000003}}, {{0x00000007, 0x00000007}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000001}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x000001ff, 0x000001ff}, {0x00000001, 0x00000001}}, {{0x00000001, 0x00000001}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000001}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x000000ff, 0x000000ff}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000007f, 0x0000007f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000003f, 0x0000003f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000001f, 0x0000001f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000000f, 0x0000000f}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            },
            {
            {{{{0xffffffff, 0x00000000}, {0x001fffff, 0x00000000}}, {{0x001fffff, 0x00000000}, {0x000007ff, 0x00000000}}}, {{{0x00000fff, 0x00000000}, {0x0000007f, 0x00000000}}, {{0x0000001f, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0xbfffffff, 0x00000000}, {0x00007fff, 0x00000000}}, {{0x0003ffff, 0x00000000}, {0x0000003f, 0x00000000}}}, {{{0x00000fff, 0x00000000}, {0x0000000f, 0x00000000}}, {{0x0000000f, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x01ffffff, 0x00000000}, {0x000007ff, 0x00000000}}, {{0x00003fff, 0x00000000}, {0x00000007, 0x00000000}}}, {{{0x000001ff, 0x00000000}, {0x00000001, 0x00000000}}, {{0x00000003, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x001fffff, 0x00000000}, {0x000000ff, 0x00000000}}, {{0x000007ff, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x0000001f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0001ffff, 0x00000000}, {0x0000003f, 0x00000000}}, {{0x000000ff, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000007, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x00007fff, 0x00000000}, {0x0000000f, 0x00000000}}, {{0x0000003f, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000007, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x00001fff, 0x00000000}, {0x00000007, 0x00000000}}, {{0x0000000f, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x000007ff, 0x00000000}, {0x00000003, 0x00000000}}, {{0x00000007, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x000001ff, 0x00000000}, {0x00000001, 0x00000000}}, {{0x00000001, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x000000ff, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000007f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000003f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000001f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            {{{{0x0000000f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            },
            {
                    {{{{0xffffffff, 0x00000000}, {0x001fffff, 0x00000000}}, {{0x007fffff, 0x00000000}, {0x000007ff, 0x00000000}}}, {{{0x00000fff, 0x00000000}, {0x0000003f, 0x00000000}}, {{0x0000003f, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0xbfffffff, 0x00000000}, {0x00007fff, 0x00000000}}, {{0x0003ffff, 0x00000000}, {0x0000003f, 0x00000000}}}, {{{0x00000fff, 0x00000000}, {0x00000007, 0x00000000}}, {{0x0000000f, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x01ffffff, 0x00000000}, {0x000007ff, 0x00000000}}, {{0x00003fff, 0x00000000}, {0x00000007, 0x00000000}}}, {{{0x000001ff, 0x00000000}, {0x00000001, 0x00000000}}, {{0x00000003, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x001fffff, 0x00000000}, {0x000000ff, 0x00000000}}, {{0x000007ff, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x0000003f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0001ffff, 0x00000000}, {0x0000003f, 0x00000000}}, {{0x000000ff, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x0000001f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x00007fff, 0x00000000}, {0x0000000f, 0x00000000}}, {{0x0000003f, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000007, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x00000fff, 0x00000000}, {0x00000007, 0x00000000}}, {{0x0000000f, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000003, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x000007ff, 0x00000000}, {0x00000001, 0x00000000}}, {{0x00000007, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000001, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x000001ff, 0x00000000}, {0x00000001, 0x00000000}}, {{0x00000001, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x000000ff, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000007f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000003f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000001f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000000f, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            },
            {
                    {{{{0x007ffffe, 0x001ffffe}, {0x00000000, 0x00000000}}, {{0x000001fe, 0x000001fe}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0003fffe, 0x001ffffe}, {0x00000000, 0x00000000}}, {{0x0000013e, 0x000001fe}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x007ffffe, 0x001ffffe}, {0x00000000, 0x00000000}}, {{0x000001fe, 0x000001fe}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x001ffffe, 0x001ffffe}, {0x00000000, 0x00000000}}, {{0x000000fe, 0x000001fe}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0001fffe, 0x0001fffe}, {0x00000000, 0x00000000}}, {{0x000000fe, 0x000000fe}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x00007ffe, 0x00007ffe}, {0x00000000, 0x00000000}}, {{0x0000003e, 0x0000003e}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x00000ffe, 0x00000ffe}, {0x00000000, 0x00000000}}, {{0x0000000e, 0x0000000e}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x000007fe, 0x000007fe}, {0x00000000, 0x00000000}}, {{0x00000006, 0x00000006}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x000001fe, 0x000001fe}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x000000fe, 0x000000fe}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000007e, 0x0000007e}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000003e, 0x0000003e}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000001e, 0x0000001e}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
                    {{{{0x0000000e, 0x0000000e}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}, {{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}},
            },
    };
    const bool known = (ctas_per_sm == 1 || ctas_per_sm == 2) && (block_n == 128 || block_n == 256) &&
                       (block_k == 128 || block_k == 256) && num_stages >= 3 && num_stages <= kMaxCensusStages;
    if (!known) return 0;
    const int64_t kind = group_scales ? 3 : layout == GroupedLayout::Dense ? 0 : layout == GroupedLayout::Indexed ? 1 : 2;
    return kMasks[kind][num_stages - 3][ctas_per_sm - 1][block_n / 256][block_k / 256][use_stream_k];
}

// The clean token tiles of the dense Stream-K kernels with aligned slices (`stream_k_aligned`), by
// (num_stages - 3; ctas_per_sm - 1; block_n / 256; block_k / 256), as in `clean_block_ms`, 3 to 6 slots.
inline uint32_t clean_aligned_block_ms(int64_t ctas_per_sm, int64_t block_n, int64_t block_k, int64_t num_stages) {
    constexpr uint32_t kMasks[4][2][2][2] = {
            {{{0xffffffff, 0x001fffff}, {0x003fffff, 0x000007ff}}, {{0x00000fff, 0x0000007f}, {0x0000003f, 0x00000000}}},
            {{{0xffffffff, 0x00007fff}, {0x0007ffff, 0x0000003f}}, {{0x00000fff, 0x0000000f}, {0x0000001f, 0x00000000}}},
            {{{0x01ffffff, 0x000007ff}, {0x00007fff, 0x00000007}}, {{0x000001ff, 0x00000001}, {0x00000003, 0x00000000}}},
            {{{0x001fffff, 0x000000ff}, {0x000007ff, 0x00000000}}, {{0x0000007f, 0x00000000}, {0x00000000, 0x00000000}}},
    };
    const bool known = (ctas_per_sm == 1 || ctas_per_sm == 2) && (block_n == 128 || block_n == 256) &&
                       (block_k == 128 || block_k == 256) && num_stages >= 3 && num_stages <= 6;
    if (!known) return 0;
    return kMasks[num_stages - 3][ctas_per_sm - 1][block_n / 256][block_k / 256];
}

// The clean token tiles of the one-math-warp-group kernels (`math_warpgroups` 1) that rule item 6 of `sm90_rule.hpp`
// can pick: contiguous and masked (their kernels share the masks as in `clean_block_ms`, ANDed), 3 CTAs per SM,
// BN 128, no Stream-K; by (block_k / 256; num_stages - 3) for 3 and 4 ring slots, bit i
// for BM 8 (i + 1), BM 8 to 64. Tiles left out do not fit 3 CTAs' shared memory.
constexpr int64_t kMaxOneWarpgroupBlockM = 64;
inline uint32_t clean_one_warpgroup_block_ms(int64_t block_k, int64_t num_stages) {
    constexpr uint32_t kMasks[2][2] = {
            {0x000000ff, 0x0000001f},
            {0x00000003, 0x00000000},
    };
    if ((block_k != 128 && block_k != 256) || num_stages < 3 || num_stages > 4) return 0;
    return kMasks[block_k / 256][num_stages - 3];
}

// Why `config` cannot run `desc`, or null if it can: the kernel template must take the config, the block must
// divide the weight, one CTA's shared memory must fit a block's opt-in limit and `ctas_per_sm` of them an SM (each
// reserves 1 KB besides, as `jit/runtime.py` checks), and the token tile must be clean; the census itself asks
// without the last condition (`check_clean`).
inline const char *illegal_reason(const GemmConfig &config, const GemmDesc &desc, bool check_clean = true) {
    if (desc.num_experts && config.use_stream_k) return "grouped GEMM does not support Stream-K";
    // A tile of one K block has nothing to split along K: every slice is a whole tile, so a Stream-K tail
    // only adds the zeroing and the locks to the data-parallel schedule.
    if (config.use_stream_k && desc.k < 2 * config.block_k) return "Stream-K needs at least two K blocks per tile";
    // The template of `impls/sm90_w4a8_pgroup.cuh`.
    if (config.math_warpgroups != 1 && config.math_warpgroups != 2) return "one or two math warp groups";
    if (desc.group_scales && config.math_warpgroups != 2) return "the group-scale kernel has two math warp groups";
    if (config.ctas_per_sm > (config.math_warpgroups == 2 ? 2 : 4))
      return "one or two CTAs per SM with two math warp groups, one to four with one";
    if (config.math_warpgroups == 1 && config.ctas_per_sm < 2) return "one math warp group takes two CTAs per SM or more";
    if (desc.group_scales && config.use_stream_k && config.stream_k_aligned) return "the group-scale kernel cuts the tail as Stream-K";
    if (desc.group_scales && (config.block_k != 128 || config.ctas_per_sm != 1))
      return "the group-scale kernel takes block_k 128 at one CTA per SM";
    if (desc.group_scales && (config.block_m < kMinGroupScaleBlockM || config.block_m > kMaxGroupScaleBlockM))
      return "the group-scale kernel takes token tiles of 16 to 192";
    // The producer warp group's registers, as the kernel derives them from the layout (`kNumProducerRegs`).
    const int64_t producer_regs = desc.layout == GroupedLayout::Indexed ? 64 : 40;
    if (desc.num_experts) {
        const int64_t math_threads = 128 * config.math_warpgroups, threads = math_threads + 128;
        const int64_t launch_regs = 65536 / (threads * config.ctas_per_sm) / 8 * 8;
        const int64_t consumer_regs = (launch_regs * threads - producer_regs * 128) / math_threads / 8 * 8;
        // All accumulators and four FP8 source registers must coexist at WGMMA issue.
        if (config.block_m * config.block_n / math_threads + 4 > consumer_regs)
          return "accumulators and WGMMA sources exceed the consumer register budget";
    }
    if (desc.k % config.block_k != 0) return "K is not a multiple of block_k";
    if (desc.n % config.block_n != 0) return "N is not a multiple of block_n";
    const int64_t smem_bytes = get_smem_bytes(config, desc.num_experts, desc.layout, desc.group_scales);
    if (smem_bytes > desc.smem_bytes) return "one CTA needs more shared memory than a block may have";
    if (config.ctas_per_sm * (smem_bytes + 1024) > desc.smem_bytes + 1024)
      return "ctas_per_sm CTAs need more shared memory than an SM has";
    if (!check_clean) return nullptr;
    const bool one_warpgroup_census = (desc.layout == GroupedLayout::Contiguous || desc.layout == GroupedLayout::Masked) &&
                                      config.ctas_per_sm == 3 && config.block_n == 128 &&
                                      config.block_m <= kMaxOneWarpgroupBlockM;
    if (config.math_warpgroups == 1 && !one_warpgroup_census)
      return "the census covers one math warp group for contiguous and masked kernels at 3 CTAs per SM, BN 128 and BM "
             "up to 64";
    const bool aligned = config.use_stream_k && config.stream_k_aligned && desc.layout == GroupedLayout::Dense;
    const uint32_t clean = config.math_warpgroups == 1 ? clean_one_warpgroup_block_ms(config.block_k, config.num_stages)
                           : aligned ? clean_aligned_block_ms(config.ctas_per_sm, config.block_n, config.block_k,
                                                              config.num_stages)
                                     : clean_block_ms(config.ctas_per_sm, config.block_n, config.block_k,
                                                      config.use_stream_k, desc.layout, config.num_stages,
                                                      desc.group_scales);
    if (config.block_m % kBlockMStep != 0 || config.block_m < kBlockMStep || config.block_m > kMaxBlockM ||
        !(clean >> (config.block_m / kBlockMStep - 1) & 1))
      return "the token tile spills, serializes its WGMMAs or does not compile with this block shape and CTAs per SM";
    // A 3-slot ring is the fallback where 4 slots do not fit beside the output tile or spill.
    if (config.num_stages == 3) {
        GemmConfig deeper = config;
        deeper.num_stages = 4;
        if (illegal_reason(deeper, desc, check_clean) == nullptr) return "a 3-slot ring only where 4 slots cannot run";
    }
    return nullptr;
}

inline bool is_legal(const GemmConfig &config, const GemmDesc &desc) { return illegal_reason(config, desc) == nullptr; }

// Stream-K slices of one tile with BF16 partial sums: at most 32 kept the relative RMS error below 0.95%, and 40
// reached 1.25%. `get_schedule` cuts no tile into more.
constexpr int64_t kMaxBf16StreamKSlices = 32;

// The work of the busiest CTA, as the schedulers deal it
// (`include/tensorbridge/scheduler/sm90.cuh`): the
// launcher's grid is `num_sms * ctas_per_sm` CTAs, called groups below, a tile
// is `block_m x block_n` and a K block `block_k` deep.
struct Schedule {
    int64_t num_groups, m_blocks, n_blocks, k_blocks;
    int64_t waves, last_wave;  // data parallel: waves, and groups busy in the last one
    bool stream_k;             // a Stream-K tail exists and is used
    int64_t full_tiles;        // Stream-K: whole tiles per group before the tail
    int64_t share;             // K blocks of the tail per group
    int64_t tail_groups;       // groups with tail work
    int64_t runs;              // slices (runs within one tile) of the busiest group
    int64_t slices;            // slices of the most-cut tile
    int64_t dp_tiles;          // Stream-K kernels: whole tiles per group before the tail, even without a tail
    int64_t tail_tiles;        // Stream-K kernels: tiles of the tail (0 without one)
};

// The schedule of a dense call.
inline Schedule get_schedule(const GemmDesc &desc, const GemmConfig &config) {
    Schedule s{};
    s.num_groups = desc.num_sms * config.ctas_per_sm;
    s.m_blocks = ceil_div(desc.m, config.block_m);
    s.n_blocks = desc.n / config.block_n;
    s.k_blocks = desc.k / config.block_k;
    const int64_t tiles = s.m_blocks * s.n_blocks;
    s.waves = ceil_div(tiles, s.num_groups);
    s.last_wave = tiles - (s.waves - 1) * s.num_groups;

    // Aligned: the same number of slices per tail tile, one per group; one slice leaves the tiles whole.
    const bool aligned = config.use_stream_k && config.stream_k_aligned;
    int64_t tail = tiles;
    if (tiles > s.num_groups) {
        tail = tiles % s.num_groups;
        if (!aligned && tail && tail * 10 <= s.num_groups) tail += s.num_groups;
    }
    const int64_t aligned_slices =
        aligned && tail ? std::min({s.num_groups / tail, s.k_blocks, kMaxBf16StreamKSlices}) : 0;
    s.dp_tiles = (tiles - tail) / s.num_groups;
    s.tail_tiles = tail;
    s.stream_k = config.use_stream_k && tail > 0 && (!aligned || aligned_slices > 1);
    if (!s.stream_k) return s;
    s.full_tiles = (tiles - tail) / s.num_groups;
    if (aligned) {
        s.share = ceil_div(s.k_blocks, aligned_slices);
        s.tail_groups = tail * aligned_slices;
        s.runs = 1;
        s.slices = aligned_slices;
        return s;
    }

    // The tail tiles' K blocks are laid end to end and dealt in shares: the even share over the groups, or the
    // smallest larger one that cuts no tile into more than `kMaxBf16StreamKSlices` slices (the groups past the
    // tail's last share then idle).
    const int64_t tail_blocks = tail * s.k_blocks;
    auto most_slices = [&](int64_t share) {
        int64_t most = 0;
        for (int64_t t = 0; t < tail; t++) most = std::max(most, ((t + 1) * s.k_blocks - 1) / share - t * s.k_blocks / share + 1);
        return most;
    };
    s.share = ceil_div(tail_blocks, s.num_groups);
    while (most_slices(s.share) > kMaxBf16StreamKSlices) s.share++;
    s.slices = most_slices(s.share);
    s.tail_groups = ceil_div(tail_blocks, s.share);
    for (int64_t start = 0; start < tail_blocks; start += s.share) {
        const int64_t end = std::min(start + s.share, tail_blocks);
        s.runs = std::max(s.runs, (end - 1) / s.k_blocks - start / s.k_blocks + 1);
    }
    return s;
}

}  // namespace tensorbridge::sm90
