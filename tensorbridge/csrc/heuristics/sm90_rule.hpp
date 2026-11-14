#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "config.hpp"
#include "sm90_common.hpp"

// The selection rule for SM90, one for dense and grouped calls; a dense call is one expert of M rows.
//
// With `rows` routed rows over `experts` local experts (even routes: min(rows, experts) occupied, the busiest
// with ceil(rows / experts) rows, r), S SMs and n output tiles along N over the occupied experts:
//   1. BM: r split evenly into the fewest tiles of at most 256 rows, rounded up to 8, m0 of them. Unless item 4
//      splits the call at that BM, m = min(ceil(r / 8), floor(W * S / n)) tiles with W = ceil(m0 * n / S) waves:
//      the fewest waves, filled with M tiles; BM = ceil8(r / m).
//   2. BN 128; on a class with `dense_bn256`, dense calls below the device's ridge point take BN 256 (BM by item
//      1 at BN 256) where the BN 256 tiles still fill one wave of SMs.
//   3. CTAs per SM: one, with the deepest ring the census accepts, when the data-parallel tiles fit one wave of SMs;
//      else two where two fit an SM's shared memory at ring depth 4, else one.
//   4. Stream-K (dense only) when the idle share of the last wave of the item-3 CTA slots times K / 128 reaches
//      `kStreamKIdleKBlocks`; a split call takes the CTA count of the shared-memory fit; aligned slices where their
//      busiest CTA gets no more K blocks than with plain ones.
//   5. BK 256 for grouped calls below the device's ridge point with K >= `kBlockK256MinK`, else 128.
//      Ring depth 4, N-fast tile order.
// `rule_preferences` lists the rule's config first, then the fallbacks for configs the census refuses:
// shallower rings, BK 128, plain Stream-K, the other CTA count, BN 128.
//   6. On a class with `grouped_one_warpgroup`, a contiguous or masked call whose pick (items 1 to 5 and the
//      census) runs 2 CTAs per SM, has at least `kOneWarpgroupMinTilesPerSm` output tiles per SM and a token
//      tile of at most `kOneWarpgroupMaxBlockM` rows takes one math warp group at 3 CTAs per SM instead
//      (`math_warpgroups` 1; BM, BN and BK kept, the ring lowered from the pick's depth until
//      selection may take it), where the tile's K blocks (K / BK) fit that ring.
// The group-scale kernel (`impls/sm90_w4a8_pgroup.cuh`: one CTA per SM, BK 128, plain Stream-K) has its own rule
// (`group_scale_preferences`). A dense call has M rows and n output tiles along N on S SMs:
//   G1. BM: M split evenly into the fewest tiles of at most the class's `group_scale_block_m` rows (and at most the
//       largest token tile the census accepts at any ring depth), rounded up to 8, at least 16; unless G3 splits
//       the call, item 1's wave split; a call G3 splits takes at least as many M tiles as give each SM a Stream-K
//       share of the class's `group_scale_stream_k_share` K blocks (tiles x K / 128 / S; 0: the fewest tiles).
//       With the class's accumulator bound A (`group_scale_tile_elements`; 0: none), a BN 128 call whose tiles fit
//       one wave of SMs takes BM x 128 <= A.
//   G2. BN 128; BN 256 where N is a multiple of 256 and the BN 256 token tile has BM x 256 <= A.
//   G3. Stream-K when K >= 256 and the idle share of the last wave of S slots times K / 128 reaches
//       `kStreamKIdleKBlocks`.
//   G4. Ring depth 5, falling back to 4 and 3. N-fast tile order.
// The list continues at lower BM, then the data-parallel copy of a Stream-K list, then BN 128 after BN 256.
// A grouped call (`num_experts` > 0) sizes the tile from the busiest expert. With r = ceil(rows / experts) plus
// `grouped_group_scale_headroom` when rows >= experts, and n output tiles times the occupied experts:
//   Gg1. BM: r split as in item 1 into tiles of at most `grouped_group_scale_block_m` rows (BN 256:
//        `grouped_group_scale_bn256_block_m`), and at most the largest data-parallel tile the census accepts,
//        rounded up to 8, at least 16.
//   Gg2. BN 128; BN 256 where N is a multiple of 256, the BN 256 cap is positive, and either
//        `grouped_group_scale_bn256_always` is set or r is at most that cap.
//   Gg3. No Stream-K.
//   Gg4. Ring depth 5, then 4, then 3. Depth 4 is not clean for BN 256 at BM 56 and 64, which are the tiles the
//        H20 cap selects, so depth 5 stays first. N-fast tile order.
// The list continues at lower BM, then BN 128 after BN 256.

namespace tensorbridge::sm90 {

// Stream-K when (1 - last-wave fill) x K / 128 reaches this (items 4 and G3).
constexpr double kStreamKIdleKBlocks = 6;
// BK 256 for grouped calls below the ridge from this K (item 5).
constexpr int64_t kBlockK256MinK = 2048;
// Item 6's least output tiles per SM and largest token tile.
constexpr int64_t kOneWarpgroupMinTilesPerSm = 4;
constexpr int64_t kOneWarpgroupMaxBlockM = 64;

// The constants of a device class the rule reads (`DeviceClass` of `sm90.hpp`): the ridge point (peak FP8
// FLOP/s over HBM bandwidth), item 2's and G2's BN 256, and G1's largest group-scale token tile.
struct RuleClass {
    double ridge_flop_per_byte;
    bool dense_bn256;
    bool grouped_one_warpgroup;  // item 6
    int64_t group_scale_block_m;
    int64_t group_scale_tile_elements;
    int64_t group_scale_stream_k_share;
    int64_t grouped_group_scale_block_m;
    int64_t grouped_group_scale_bn256_block_m;
    bool grouped_group_scale_bn256_always;
    int64_t grouped_group_scale_headroom;
};

// The largest group-scale token tile the census accepts at BN `block_n` with or without Stream-K at some ring depth
// whose shared memory fits `desc` (BN 128: 184 data parallel, 168 Stream-K; BN 256: 72 and 72, on 227 KB).
inline int64_t group_scale_census_block_m(const GemmDesc &desc, int64_t block_n, bool stream_k) {
    int64_t largest = 0;
    for (int64_t stages = 3; stages <= kMaxCensusStages; stages++) {
        const uint32_t clean = clean_block_ms(1, block_n, 128, stream_k, GroupedLayout::Dense, stages, true);
        for (int64_t block_m = kMinGroupScaleBlockM; block_m <= kMaxGroupScaleBlockM; block_m += kBlockMStep) {
            GemmConfig config;
            config.block_m = block_m, config.block_n = block_n, config.num_stages = stages;
            const bool fits = get_smem_bytes(config, 0, GroupedLayout::Dense, true) <= desc.smem_bytes;
            if (fits && (clean >> (block_m / kBlockMStep - 1) & 1)) largest = std::max(largest, block_m);
        }
    }
    return largest;
}

// The group-scale rule (G1 to G4 above) for `desc`; `stream_k_override` as in `rule_preferences`.
inline std::vector<GemmConfig> group_scale_preferences(const GemmDesc &desc, const RuleClass &rule_class,
                                                       int64_t stream_k_override) {
    if (desc.num_experts > 0) {
        const int64_t active = std::min(desc.m, desc.num_experts);
        const int64_t m = ceil_div(desc.m, desc.num_experts) +
                          (desc.m >= desc.num_experts ? rule_class.grouped_group_scale_headroom : 0);
        auto grouped_tile = [&](int64_t block_n) {
            const int64_t cap_cfg = block_n == 256 ? rule_class.grouped_group_scale_bn256_block_m
                                                   : rule_class.grouped_group_scale_block_m;
            const int64_t cap = std::min(cap_cfg, group_scale_census_block_m(desc, block_n, false));
            const int64_t n_tiles = desc.n / block_n * active;
            const int64_t pieces = ceil_div(m, cap);
            const int64_t waves = ceil_div(pieces * n_tiles, desc.num_sms);
            const int64_t split = std::max<int64_t>(1, std::min(ceil_div(m, 8), waves * desc.num_sms / n_tiles));
            return std::min(cap, std::max(kMinGroupScaleBlockM, ceil_div(ceil_div(m, split), 8) * 8));
        };
        const bool wide = rule_class.grouped_group_scale_bn256_block_m > 0 && desc.n % 256 == 0 &&
                          (rule_class.grouped_group_scale_bn256_always ||
                           m <= rule_class.grouped_group_scale_bn256_block_m);
        std::vector<GemmConfig> preferences;
        for (const int64_t block_n : wide ? std::vector<int64_t>{256, 128} : std::vector<int64_t>{128}) {
            const int64_t block_m = grouped_tile(block_n);
            for (int64_t lower = block_m; lower >= kMinGroupScaleBlockM; lower -= kBlockMStep) {
                // Depth 5 is clean through BN 256 BM 72; depth 4 drops BN 256 at BM 56 and 64.
                for (const int64_t stages : {5, 4, 3}) {
                    GemmConfig config;
                    config.block_m = lower, config.num_stages = stages, config.block_n = block_n;
                    preferences.push_back(config);
                }
            }
        }
        return preferences;
    }
    const int64_t m = desc.m;
    auto tile = [&](int64_t block_n) {  // (BM, Stream-K) at BN `block_n`
        const int64_t n_tiles = desc.n / block_n;
        auto fewest = [&](bool stream_k) {  // the fewest M tiles of at most the cap
            const int64_t cap = std::min(rule_class.group_scale_block_m, group_scale_census_block_m(desc, block_n, stream_k));
            return ceil_div(m, cap);
        };
        auto even = [&](int64_t pieces) { return std::max(kMinGroupScaleBlockM, ceil_div(ceil_div(m, pieces), 8) * 8); };
        const int64_t pieces = fewest(false);
        const int64_t tiles = ceil_div(m, even(pieces)) * n_tiles;
        const int64_t tail = tiles > desc.num_sms ? tiles % desc.num_sms : tiles;
        const double idle = tail ? double(desc.num_sms - tail) / desc.num_sms : 0.0;
        const bool stream_k = stream_k_override < 0
                                  ? desc.k >= 256 && idle * (desc.k / 128) >= kStreamKIdleKBlocks
                                  : stream_k_override > 0;
        const int64_t waves = ceil_div(pieces * n_tiles, desc.num_sms);
        const int64_t split = std::max<int64_t>(1, std::min(ceil_div(m, 8), waves * desc.num_sms / n_tiles));
        // The Stream-K share: M tiles enough for `group_scale_stream_k_share` K blocks per SM.
        const int64_t share_pieces = ceil_div(rule_class.group_scale_stream_k_share * desc.num_sms, n_tiles * (desc.k / 128));
        const int64_t split_pieces = std::min(std::max(fewest(true), share_pieces), ceil_div(m, kMinGroupScaleBlockM));
        int64_t block_m = stream_k ? even(split_pieces) : even(split);
        // The accumulator bound of a sub-wave BN 128 call.
        const int64_t bound = rule_class.group_scale_tile_elements;
        if (bound && block_n == 128 && ceil_div(m, block_m) * n_tiles <= desc.num_sms) {
            block_m = std::min(block_m, even(ceil_div(m, bound / block_n)));
        }
        return std::make_pair(block_m, stream_k);
    };
    const int64_t bound = rule_class.group_scale_tile_elements;
    const bool wide = bound && desc.n % 256 == 0 && tile(256).first * 256 <= bound;
    std::vector<GemmConfig> preferences;
    for (const int64_t block_n : wide ? std::vector<int64_t>{256, 128} : std::vector<int64_t>{128}) {
        const auto [block_m, stream_k] = tile(block_n);
        for (const bool split : stream_k ? std::vector<bool>{true, false} : std::vector<bool>{false}) {
            for (int64_t lower = block_m; lower >= kMinGroupScaleBlockM; lower -= kBlockMStep) {
                for (const int64_t stages : {5, 4, 3}) {
                    GemmConfig config;
                    config.block_m = lower, config.num_stages = stages, config.block_n = block_n;
                    config.use_stream_k = split;
                    preferences.push_back(config);
                }
            }
        }
    }
    return preferences;
}

// One pass of the rule for `desc`: `wave_split` splits M by wave utilization (item 1), `wide` allows BN 256
// (item 2). `rule_preferences` combines the passes.
inline std::vector<GemmConfig> rule_pass(const GemmDesc &desc, double ridge_flop_per_byte, int64_t stream_k_override,
                                         bool wave_split, bool wide) {
    const bool dense = desc.num_experts == 0;
    const int64_t experts = std::max<int64_t>(desc.num_experts, 1);
    const int64_t active = std::min(desc.m, experts);
    const int64_t most_rows = ceil_div(desc.m, experts);
    // FLOP per DRAM byte: NVFP4 weights of the occupied experts (0.5625 B each), FP8 activations, BF16 output.
    const double intensity = double(2 * desc.m * desc.n * desc.k) /
                             (double(active * desc.n * desc.k * 9) / 16 + double(desc.m * desc.k) +
                              double(2 * desc.m * desc.n));
    const bool below_ridge = intensity < ridge_flop_per_byte;

    // BM for `n_tiles` output tiles along N (over the occupied experts): the fewest M tiles of at most
    // `kMaxBlockM` rows, or with `wave_split` as many as the waves of those tiles hold.
    auto token_tile = [&](int64_t n_tiles) {
        const int64_t fewest = ceil_div(most_rows, kMaxBlockM);
        const int64_t waves = ceil_div(fewest * n_tiles, desc.num_sms);
        const int64_t pieces = wave_split ? std::min(ceil_div(most_rows, 8), waves * desc.num_sms / n_tiles) : fewest;
        return std::max(kBlockMStep, ceil_div(ceil_div(most_rows, pieces), 8) * 8);
    };
    const int64_t narrow_block_m = token_tile(active * (desc.n / 128));
    const bool wide_allowed = wide && below_ridge && desc.n % 256 == 0;
    const int64_t wide_block_m = wide_allowed ? token_tile(active * (desc.n / 256)) : 0;
    const bool wide_fills = wide_allowed && active * ceil_div(most_rows, wide_block_m) * (desc.n / 256) >= desc.num_sms;
    GemmConfig base;
    base.block_n = wide_fills ? 256 : 128;
    base.block_m = wide_fills ? wide_block_m : narrow_block_m;
    base.block_k = !dense && below_ridge && desc.k >= kBlockK256MinK ? 256 : 128;
    const int64_t tiles = active * ceil_div(most_rows, base.block_m) * (desc.n / base.block_n);

    GemmConfig two_ctas = base;
    two_ctas.ctas_per_sm = 2, two_ctas.num_stages = 4;
    const int64_t two_ctas_smem = get_smem_bytes(two_ctas, desc.num_experts, desc.layout);
    const int64_t fit_ctas = 2 * (two_ctas_smem + 1024) <= desc.smem_bytes + 1024 ? 2 : 1;
    // Data-parallel tiles that fit one wave of SMs take one CTA (item 3).
    int64_t ctas = tiles <= desc.num_sms ? 1 : fit_ctas;

    // Stream-K (dense only, item 4): the test over the item-3 CTA slots; a split call takes the CTA count of the
    // shared-memory fit, and its slices follow from that count.
    const int64_t test_slots = desc.num_sms * ctas;
    const int64_t test_tail = tiles > test_slots ? tiles % test_slots : tiles;
    const double idle = test_tail ? double(test_slots - test_tail) / test_slots : 0.0;
    const bool stream_k = dense && (stream_k_override < 0 ? idle * (desc.k / 128) >= kStreamKIdleKBlocks
                                                           : stream_k_override > 0);
    ctas = stream_k ? fit_ctas : ctas;
    const int64_t slots = desc.num_sms * ctas;
    const int64_t tail = tiles > slots ? tiles % slots : tiles;
    const int64_t k_blocks = desc.k / base.block_k;
    // Aligned slices only where one per slot cuts no tile into more than `kMaxBf16StreamKSlices`; a tail they would
    // cut further takes plain Stream-K, which `get_schedule` clamps to that many.
    const int64_t slices = tail ? std::min(slots / tail, k_blocks) : 0;
    const bool aligned = stream_k && slices > 1 && slices <= kMaxBf16StreamKSlices &&
                         ceil_div(k_blocks, slices) <= ceil_div(tail * k_blocks, slots);
    // Data-parallel tiles in one wave of SMs: one CTA with the deepest ring the census accepts.
    const bool one_wave = !stream_k && tiles <= desc.num_sms;
    ctas = one_wave ? 1 : ctas;
    const std::vector<int64_t> depths = one_wave ? std::vector<int64_t>{8, 7, 6, 5, 4, 3} : std::vector<int64_t>{4, 3};

    // A Stream-K list ends with its data-parallel copy, for Stream-K tiles the census refuses.
    std::vector<GemmConfig> preferences;
    auto add = [&](const GemmConfig &config) {
        // Each config once, at its first place.
        const bool seen = std::find_if(preferences.begin(), preferences.end(), [&](const GemmConfig &c) {
                                                return c.to_vector() == config.to_vector();
                                            }) != preferences.end();
        if (seen) return;
        preferences.push_back(config);
    };
    for (const bool split : {stream_k, false}) {
        for (const int64_t cta_count : {ctas, 3 - ctas}) {
            for (const int64_t stages : depths) {
                for (const int64_t block_k : {base.block_k, int64_t{128}}) {
                    for (const bool split_aligned : {split && aligned, false}) {
                        GemmConfig config = base;
                        config.ctas_per_sm = cta_count, config.num_stages = stages, config.block_k = block_k;
                        config.use_stream_k = split, config.stream_k_aligned = split_aligned;
                        add(config);
                    }
                }
            }
        }
    }
    if (base.block_n != 256) return preferences;
    // BN 256 refused (BM above 128 at one CTA): the BN 128 pass.
    for (const GemmConfig &config : rule_pass(desc, ridge_flop_per_byte, stream_k_override, wave_split, false)) {
        add(config);
    }
    return preferences;
}

// The rule's config, then its fallbacks, for `desc` on a device of `rule_class`. `stream_k_override`: negative
// lets the rule decide; otherwise Stream-K is off (0) or on (positive). A call the pass without the wave split
// cuts with Stream-K keeps that pass (item 1). The group-scale kernel takes its own rule.
inline std::vector<GemmConfig> rule_preferences(const GemmDesc &desc, const RuleClass &rule_class,
                                                int64_t stream_k_override = -1) {
    if (desc.group_scales) return group_scale_preferences(desc, rule_class, stream_k_override);
    const double ridge = rule_class.ridge_flop_per_byte;
    const bool wide = rule_class.dense_bn256 && desc.num_experts == 0;
    std::vector<GemmConfig> unsplit = rule_pass(desc, ridge, stream_k_override, false, wide);
    if (unsplit.front().use_stream_k) return unsplit;
    return rule_pass(desc, ridge, stream_k_override, true, wide);
}

// Whether selection may take a config: `illegal_reason` accepts it (`get_schedule` cuts a Stream-K tile into at
// most `kMaxBf16StreamKSlices` slices).
inline bool selectable(const GemmConfig &config, const GemmDesc &desc) { return is_legal(config, desc); }

// Item 6 on `pick`, the config items 1 to 5 select: the one-math-warp-group config, or `pick`.
inline GemmConfig one_warpgroup_config(const GemmDesc &desc, const GemmConfig &pick) {
    const bool grouped = desc.layout == GroupedLayout::Contiguous || desc.layout == GroupedLayout::Masked;
    const int64_t experts = std::max<int64_t>(desc.num_experts, 1);
    const int64_t tiles = std::min(desc.m, experts) * ceil_div(ceil_div(desc.m, experts), pick.block_m) *
                          (desc.n / pick.block_n);
    if (!grouped || pick.ctas_per_sm != 2 || pick.block_m > kOneWarpgroupMaxBlockM ||
        tiles < kOneWarpgroupMinTilesPerSm * desc.num_sms) {
        return pick;
    }
    for (int64_t stages = pick.num_stages; stages >= 3; stages--) {
        GemmConfig config = pick;
        config.math_warpgroups = 1, config.ctas_per_sm = 3, config.num_stages = stages;
        if (!selectable(config, desc)) continue;
        return desc.k / config.block_k <= stages ? config : pick;
    }
    return pick;
}

// The shapes selection takes: M, N and K positive, N and K multiples of 128 (the kernels' weight unit).
inline void check_shape(const GemmDesc &desc) {
    if (desc.m <= 0 || desc.n <= 0 || desc.k <= 0 || desc.n % kBlockN != 0 || desc.k % kBlockK != 0) {
        throw std::invalid_argument("TensorBridge needs M > 0 and N, K positive multiples of 128");
    }
}

// The first preference selection may take; where the census refuses all of them, the same list with the token
// tile one step of 8 lower, step by step down to the kernel's smallest tile. With `stream_k_override` > 0 only
// Stream-K entries count. Then item 6 on a class that takes it.
inline GemmConfig rule_config(const GemmDesc &desc, const RuleClass &rule_class, int64_t stream_k_override = -1) {
    check_shape(desc);
    const std::vector<GemmConfig> preferences = rule_preferences(desc, rule_class, stream_k_override);
    const int64_t min_block_m = desc.group_scales ? kMinGroupScaleBlockM : kBlockMStep;
    for (int64_t lower = 0; lower <= kMaxBlockM; lower += kBlockMStep) {
        for (GemmConfig config : preferences) {
            config.block_m -= lower;
            const bool counted = stream_k_override <= 0 || config.use_stream_k;
            if (config.block_m >= min_block_m && counted && selectable(config, desc)) {
                return rule_class.grouped_one_warpgroup ? one_warpgroup_config(desc, config) : config;
            }
        }
    }
    throw std::runtime_error(stream_k_override > 0 ? "No legal Stream-K configuration for this shape"
                                                   : "No legal GEMM configuration for this shape");
}

// Autotune's candidates (opt-in, `jit/autotune.py`): the rule's config, then its neighbours that selection may take, up to
// `count`: the other CTA count, the other Stream-K modes (dense calls), the other BK, BN 256, ring depths 3 to 8, token
// tiles one step of 8 smaller and larger and half as tall, M-fast order.
inline std::vector<GemmConfig> autotune_candidates(const GemmDesc &desc, const RuleClass &rule_class, int64_t count) {
    check_shape(desc);
    const GemmConfig pick = rule_config(desc, rule_class);
    std::vector<GemmConfig> neighbours;
    auto with = [&](auto change) {
        GemmConfig config = pick;
        change(config);
        neighbours.push_back(config);
    };
    with([](GemmConfig &c) { c.ctas_per_sm = 3 - c.ctas_per_sm, c.num_stages = 4; });
    // The other Stream-K modes (`selectable` refuses them on grouped calls).
    with([](GemmConfig &c) { c.use_stream_k = false, c.stream_k_aligned = false; });
    with([](GemmConfig &c) { c.use_stream_k = true, c.stream_k_aligned = false; });
    with([](GemmConfig &c) { c.use_stream_k = true, c.stream_k_aligned = true; });
    with([](GemmConfig &c) { c.block_k = 384 - c.block_k; });
    with([](GemmConfig &c) { c.block_n = 256, c.block_m = std::min<int64_t>(c.block_m, 128); });
    for (int64_t stages = 3; stages <= 8; stages++) with([&](GemmConfig &c) { c.num_stages = stages; });
    with([](GemmConfig &c) { c.block_m -= kBlockMStep; });
    with([](GemmConfig &c) { c.block_m += kBlockMStep; });
    with([](GemmConfig &c) { c.block_m = std::max(kBlockMStep, c.block_m / 2 / kBlockMStep * kBlockMStep); });
    with([](GemmConfig &c) { c.m_fast_tile_order = !c.use_stream_k; });

    std::vector<GemmConfig> candidates = {pick};
    for (const GemmConfig &config : neighbours) {
        if (static_cast<int64_t>(candidates.size()) >= count) break;
        const bool seen = std::find_if(candidates.begin(), candidates.end(), [&](const GemmConfig &c) {
                                                return c.to_vector() == config.to_vector();
                                            }) != candidates.end();
        if (seen || !selectable(config, desc)) continue;
        candidates.push_back(config);
    }
    return candidates;
}

}  // namespace tensorbridge::sm90
