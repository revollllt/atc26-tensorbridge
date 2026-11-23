"""Source generation: one translation unit per (N, K, GemmConfig).

All kernel logic lives in `include/tensorbridge`. The generated file includes the kernel and reads
back its shared memory size; the NVRTC name expression pins the template arguments of the
instantiation (`impls/sm90_w4a8_ptoken.cuh`). The bridge mode is one of them, so both
modes compile the same file. The kernel computes on the weight padded to the 128 x 128 unit
(`pad_to_unit`, `quant/nvfp4.py`) and writes the weight's N output columns.
"""

import dataclasses
from typing import Literal

# `fpma`: the FPMA table, for FPMA-aware checkpoints (the paper's pairing);
# `exact`: the flagged table, equal to the RNE E4M3 rounding, for standard NVFP4 checkpoints; the package's default.
Bridge = Literal["fpma", "exact"]
GroupedLayout = Literal["dense", "indexed", "contiguous", "masked"]
LAYOUT_IDS: dict[GroupedLayout, int] = {"dense": 0, "indexed": 1, "contiguous": 2, "masked": 3}
LAYOUT_NAMES: dict[GroupedLayout, str] = {
    "dense": "kDense", "indexed": "kIndexed", "contiguous": "kContiguous", "masked": "kMasked",
}
# The kernels' weight unit: a tile of 128 channels by 128 K (`impls/sm90_w4a8_ptoken.cuh`).
WEIGHT_UNIT = 128


def cpp(value: bool) -> str:
    """A C++ boolean literal."""
    return "true" if value else "false"


def pad_to_unit(size: int) -> int:
    """`size` (N or K of a weight) rounded up to the weight unit: the shape the kernel and the selection rule
    see; `quant/nvfp4.prepare` zero-pads the weight to it."""
    return -(-size // WEIGHT_UNIT) * WEIGHT_UNIT


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    """Mirror of `GemmConfig` in `csrc/heuristics/config.hpp`: the fields in order, with the same defaults."""

    block_m: int = 128
    num_stages: int = 4
    use_stream_k: bool = False
    m_fast_tile_order: bool = False
    block_k: int = 128  # K depth of a ring stage: 128 or 256
    block_n: int = 128  # channels of a CTA's output tile: 128 or 256
    ctas_per_sm: int = 1  # CTAs per SM, compiled into the kernel; the grid is num_sms * ctas_per_sm
    # the Stream-K tail cut into the same slices per tile, at most one per CTA; no-op without Stream-K
    stream_k_aligned: bool = False
    math_warpgroups: int = 2  # math warp groups per CTA: two split each 128-channel tile, one (2 to 4 CTAs) computes it

    @classmethod
    def from_list(cls, values: list[int]) -> "GemmConfig":
        fields = dataclasses.fields(cls)
        assert len(values) == len(fields), "GemmConfig is out of sync with config.hpp"
        return cls(*(field.type(value) for field, value in zip(fields, values)))

    def to_list(self) -> list[int]:
        return [int(value) for value in dataclasses.astuple(self)]


@dataclasses.dataclass(frozen=True)
class KernelSource:
    source: str
    name: str  # NVRTC name expression of the kernel instantiation


def generate(
    n: int, k: int, config: GemmConfig, num_experts: int = 0, layout: GroupedLayout = "dense",
    bridge: Bridge = "exact",
) -> KernelSource:
    """The kernel for one weight shape `(n, k)` and config: `SHAPE_N` and `SHAPE_K` are the shape padded to the
    weight unit, `OUTPUT_N` is `n`."""
    layout_name = f"tensorbridge::GroupedLayout::{LAYOUT_NAMES[layout]}"
    tile = f"{config.block_m}, {config.block_n}, {config.block_k}, {config.num_stages}"
    name = (
        f"tensorbridge::sm90_w4a8_ptoken<{pad_to_unit(n)}, {pad_to_unit(k)}, {n}, {tile}, {config.ctas_per_sm}, "
        f"{config.math_warpgroups}, {cpp(config.use_stream_k)}, "
        f"{cpp(config.use_stream_k and config.stream_k_aligned)}, "
        f"{cpp(config.m_fast_tile_order)}, {layout_name}, "
        f"{num_experts}, {cpp(bridge == 'exact')}>"
    )
    source = f"""\
#include <tensorbridge/impls/sm90_w4a8_ptoken.cuh>

// Read back by the JIT: the launch has to reserve this much shared memory.
extern "C" __constant__ uint32_t SMEM_BYTES =
    sizeof(tensorbridge::layout::PtokenStorage<{tile}, {layout_name}, {num_experts}>);
"""
    return KernelSource(source, name)


def generate_pgroup(
    n: int, k: int, config: GemmConfig, scale_m_major: bool, bridge: Bridge = "exact",
    num_experts: int = 0, layout: GroupedLayout = "dense", *, fuse_swiglu_fp8: bool = False,
) -> KernelSource:
    """The `[1, 128]` activation-scale kernel (`impls/sm90_w4a8_pgroup.cuh`) for one weight shape `(n, k)` and
    config, padded as `generate` pads it: dense or grouped, one 128-deep K block per stage.
    `scale_m_major` describes dense activation scales; grouped kernels receive their actual strides."""
    max_ctas = 2 if config.math_warpgroups == 2 else 4
    if (
        config.block_k != 128
        or config.stream_k_aligned
        or config.math_warpgroups not in (1, 2)
        or config.ctas_per_sm < 1
        or config.ctas_per_sm > max_ctas
        or (config.math_warpgroups == 1 and config.ctas_per_sm < 2)
    ):
        raise ValueError(
            f"the group-scale kernel takes block_k 128, plain Stream-K, and valid (math_warpgroups, ctas_per_sm), "
            f"not {config}"
        )
    if fuse_swiglu_fp8:
        if config.block_n != 256 or n % 256 != 0 or layout == "indexed" or config.use_stream_k:
            raise ValueError(
                f"fuse_swiglu_fp8 requires block_n=256, n%256==0, non-indexed, non-Stream-K; "
                f"got n={n}, layout={layout}, config={config}"
            )
    fuse_arg = ", true" if fuse_swiglu_fp8 else ""
    name = (
        f"tensorbridge::sm90_w4a8_pgroup<{pad_to_unit(n)}, {pad_to_unit(k)}, {n}, {config.block_m}, {config.block_n}, "
        f"{config.num_stages}, {config.ctas_per_sm}, {config.math_warpgroups}, {cpp(config.use_stream_k)}, "
        f"{cpp(config.m_fast_tile_order)}, {cpp(scale_m_major)}, {cpp(bridge == 'exact')}, "
        f"tensorbridge::GroupedLayout::{LAYOUT_NAMES[layout]}, {num_experts}{fuse_arg}>"
    )
    source = f"""\
#include <tensorbridge/impls/sm90_w4a8_pgroup.cuh>

// Read back by the JIT: the launch has to reserve this much shared memory.
extern "C" __constant__ uint32_t SMEM_BYTES =
    sizeof(tensorbridge::layout::PgroupStorage<{config.block_m}, {config.block_n}, {config.num_stages},
        tensorbridge::GroupedLayout::{LAYOUT_NAMES[layout]}, {num_experts}>);
"""
    return KernelSource(source, name)


def generate_megamoe(
    n_gu: int, k_gu: int, n_dn: int, k_dn: int, num_experts: int,
    block_m: int, num_stages: int, ctas_per_sm: int, math_warpgroups: int = 2, bridge: Bridge = "exact",
) -> KernelSource:
    """Single SM90 persistent W4A8 MegaMoE kernel (`impls/sm90_w4a8_megamoe.cuh`) fusing
    4x route-striped NVLink P2P Dispatch, GateUp+SwiGLU+FP8g128, Down(g128) with direct 512B-warp NVLink
    scatter epilogue, weighted BF16 reduction, inline self-reset, and a single NVLink barrier."""
    if n_gu % 256 != 0 or n_dn % 256 != 0 or k_gu % 128 != 0 or k_dn % 128 != 0 or k_dn != n_gu // 2:
        raise ValueError(f"Invalid MegaMoE dimensions: gu=({n_gu}, {k_gu}), dn=({n_dn}, {k_dn})")
    if num_experts < 1:
        raise ValueError(f"MegaMoE requires num_experts >= 1, got {num_experts}")
    if block_m < 8 or block_m > 64 or block_m % 8 != 0:
        raise ValueError(f"MegaMoE requires block_m in {{8,16,...,64}}, got {block_m}")
    if num_stages < 3 or num_stages > 8:
        raise ValueError(f"MegaMoE requires num_stages in 3..8, got {num_stages}")
    if math_warpgroups not in (1, 2):
        raise ValueError(f"MegaMoE requires math_warpgroups in (1, 2), got {math_warpgroups}")
    max_ctas = 2 if math_warpgroups == 2 else 4
    min_ctas = 1 if math_warpgroups == 2 else 2
    if not (min_ctas <= ctas_per_sm <= max_ctas):
        raise ValueError(
            f"MegaMoE requires ctas_per_sm in {min_ctas}..{max_ctas} for WG={math_warpgroups}, got {ctas_per_sm}"
        )
    name = (
        f"tensorbridge::sm90_w4a8_megamoe<{n_gu}, {k_gu}, {n_dn}, {k_dn}, {block_m}, {num_stages}, "
        f"{ctas_per_sm}, {math_warpgroups}, {cpp(bridge == 'exact')}, {num_experts}>"
    )
    source = f"""\
#include <tensorbridge/impls/sm90_w4a8_megamoe.cuh>

extern "C" __constant__ uint32_t SMEM_BYTES =
    sizeof(tensorbridge::layout::MegaMoeStorage<{block_m}, {num_stages}>);
"""
    return KernelSource(source, name)

