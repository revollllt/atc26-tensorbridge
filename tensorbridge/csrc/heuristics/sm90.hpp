#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "config.hpp"
#include "sm90_common.hpp"
#include "sm90_rule.hpp"

// Config selection for SM90: the rule of `sm90_rule.hpp`, a function of the shape (and a grouped call's local
// experts), the device's SM count and shared memory, and its class's `RuleClass` constants. Nothing is
// measured at run time unless autotune is enabled (`jit/autotune.py`, opt-in).

namespace tensorbridge::sm90 {

// A device class: devices of one compute-to-bandwidth balance share a class and its rule constants. H20, whose
// tensor throughput is a fraction of H100's at a similar bandwidth, is a class of its own. The rule reads a class's
// ridge point (peak FP8 TFLOPS over HBM read GB/s) and the constants below.
struct DeviceClass {
    const char *name;
    const char *device_name;  // the device whose peaks these are
    int64_t num_sms;          // for the fallback of unregistered devices
    double peak_fp8_tflops;   // large per-tensor FP8 GEMM
    double hbm_read_gbps;
    bool dense_bn256;         // rule item 2 (`sm90_rule.hpp`)
    bool grouped_one_warpgroup;  // rule item 6
    int64_t group_scale_block_m;  // G1's largest group-scale token tile
    // G1 and G2's accumulator bound (BM x BN of a group-scale tile; 0: none) and G1's Stream-K share in K blocks per
    // SM (0: none).
    int64_t group_scale_tile_elements;
    int64_t group_scale_stream_k_share;
    // Grouped `[1, 128]` calls: largest BM at BN 128 and BN 256, whether BN 256 applies at any M, and routing
    // imbalance headroom added to ceil(rows / experts) when rows >= experts.
    int64_t grouped_group_scale_block_m;
    int64_t grouped_group_scale_bn256_block_m;
    bool grouped_group_scale_bn256_always;
    int64_t grouped_group_scale_headroom;
    double ridge_flop_per_byte() const { return peak_fp8_tflops * 1e12 / (hbm_read_gbps * 1e9); }
    RuleClass rule_class() const {
        return {ridge_flop_per_byte(), dense_bn256, grouped_one_warpgroup, group_scale_block_m,
                group_scale_tile_elements, group_scale_stream_k_share, grouped_group_scale_block_m,
                grouped_group_scale_bn256_block_m, grouped_group_scale_bn256_always,
                grouped_group_scale_headroom};
    }
};

// The registry: a device's class by its exact `cudaDeviceProp::name`, never a substring of it (`NVIDIA H200`
// contains `NVIDIA H20`). A device that is not listed takes the class of the listed device with the closest SM
// count, with a warning (`get_device_class` of `launcher.cpp`).
constexpr DeviceClass kDevices[] = {
        {"h100", "NVIDIA H100 80GB HBM3", 132, 1460.0, 3170.0, false, false, 160, 0, 3, 176, 64, false, 16},
        {"h20", "NVIDIA H20", 78, 284.0, 3510.0, true, true, 128, 8192, 0, 64, 56, true, 8},
};

// The class of a device (`exact`), else of the registered device with the closest SM count.
struct DeviceChoice {
    const DeviceClass *device_class;
    bool exact;
};

inline DeviceChoice choose_device(const std::string &device_name, int64_t num_sms) {
    const DeviceClass *closest = &kDevices[0];
    for (const DeviceClass &entry : kDevices) {
        if (device_name == entry.device_name) return {&entry, true};
        if (std::abs(entry.num_sms - num_sms) < std::abs(closest->num_sms - num_sms)) closest = &entry;
    }
    return {closest, false};
}

// The rule's config for `desc`, computed once per shape, SM count and override.
inline GemmConfig select_config(const GemmDesc &desc, const DeviceClass &device_class,
                                int64_t stream_k_override = -1) {
    static std::mutex mutex;
    static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const DeviceClass *, int64_t,
                               int64_t, int64_t, bool>,
                    GemmConfig>
        cache;
    const auto key = std::make_tuple(desc.m, desc.n, desc.k, desc.num_sms, desc.l2_bytes, desc.smem_bytes,
                                     &device_class, stream_k_override, desc.num_experts,
                                     static_cast<int64_t>(desc.layout), desc.group_scales);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(key); it != cache.end()) return it->second;
    return cache[key] = rule_config(desc, device_class.rule_class(), stream_k_override);
}

}  // namespace tensorbridge::sm90
