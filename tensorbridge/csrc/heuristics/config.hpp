#pragma once

#include <cstdint>
#include <vector>

namespace tensorbridge {

enum class GroupedLayout { Dense, Indexed, Contiguous, Masked };

// The problem: `D[M, N] = A[M, K] @ B[N, K].T` with `num_sms` available SMs,
// `l2_bytes` of L2 and `smem_bytes` of opt-in shared memory per block.
struct GemmDesc {
    int64_t m, n, k;
    int64_t num_sms;
    int64_t l2_bytes;
    int64_t smem_bytes;
    int64_t num_experts = 0;
    GroupedLayout layout = GroupedLayout::Dense;
    // `[1, 128]` activation scales, one per token and 128-deep K block: the kernel of
    // `impls/sm90_w4a8_pgroup.cuh`.
    bool group_scales = false;
};

// Everything the kernel template is specialized on besides the shape. The weight layout's unit is a 128 x 128
// tile (see `include/tensorbridge/impls/sm90_w4a8_ptoken.cuh`): a ring stage holds `block_k / 128` consecutive
// K blocks, a CTA computes `block_n / 128` adjacent channel tiles, each in the layout of one, and `ctas_per_sm`
// CTAs share an SM.
struct GemmConfig {
    int64_t block_m = 128;            // token tile
    int64_t num_stages = 4;           // shared-memory ring slots
    bool use_stream_k = false;        // cut the underfilled last wave along K
    bool m_fast_tile_order = false;   // deal tiles M-fast instead of N-fast
    int64_t block_k = 128;            // K depth of a ring stage: 128 or 256
    int64_t block_n = 128;            // channels of a CTA's output tile: 128 or 256
    int64_t ctas_per_sm = 1;          // CTAs per SM, compiled into the kernel; the grid is num_sms * ctas_per_sm
    // the Stream-K tail cut into the same slices per tile, at most one per CTA; no-op without Stream-K
    bool stream_k_aligned = false;
    // math warp groups per CTA: two split each 128-channel tile, one computes it alone (rule item 6 of `sm90_rule.hpp`;
    // one takes 2 to 4 CTAs per SM)
    int64_t math_warpgroups = 2;

    // Field order is mirrored by `GemmConfig` in `tensorbridge/jit/codegen.py`.
    std::vector<int64_t> to_vector() const {
        return {block_m, num_stages, use_stream_k, m_fast_tile_order, block_k, block_n, ctas_per_sm, stream_k_aligned,
                math_warpgroups};
    }

    static GemmConfig from_vector(const std::vector<int64_t> &v) {
        GemmConfig c;
        c.block_m = v.at(0), c.num_stages = v.at(1), c.use_stream_k = v.at(2), c.m_fast_tile_order = v.at(3);
        c.block_k = v.at(4), c.block_n = v.at(5), c.ctas_per_sm = v.at(6), c.stream_k_aligned = v.at(7);
        c.math_warpgroups = v.at(8);
        return c;
    }
};

constexpr int64_t kBlockN = 128;
constexpr int64_t kBlockK = 128;  // one K block, the unit of the weight layout

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace tensorbridge
