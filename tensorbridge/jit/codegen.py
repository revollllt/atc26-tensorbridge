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
    num_experts: int = 0, layout: GroupedLayout = "dense",
) -> KernelSource:
    """The `[1, 128]` activation-scale kernel (`impls/sm90_w4a8_pgroup.cuh`) for one weight shape `(n, k)` and
    config, padded as `generate` pads it: dense or grouped, one 128-deep K block per stage, one CTA per SM.
    `scale_m_major` describes dense activation scales; grouped kernels receive their actual strides."""
    if config.block_k != 128 or config.ctas_per_sm != 1 or config.stream_k_aligned or config.math_warpgroups != 2:
        raise ValueError(f"the group-scale kernel takes block_k 128 at one CTA per SM, plain Stream-K and two math "
                         f"warp groups, not {config}")
    name = (
        f"tensorbridge::sm90_w4a8_pgroup<{pad_to_unit(n)}, {pad_to_unit(k)}, {n}, {config.block_m}, {config.block_n}, "
        f"{config.num_stages}, {cpp(config.use_stream_k)}, {cpp(config.m_fast_tile_order)}, {cpp(scale_m_major)}, "
        f"{cpp(bridge == 'exact')}, tensorbridge::GroupedLayout::{LAYOUT_NAMES[layout]}, {num_experts}>"
    )
    source = f"""\
#include <tensorbridge/impls/sm90_w4a8_pgroup.cuh>

// Read back by the JIT: the launch has to reserve this much shared memory.
extern "C" __constant__ uint32_t SMEM_BYTES =
    sizeof(tensorbridge::layout::PgroupStorage<{config.block_m}, {config.block_n}, {config.num_stages},
        tensorbridge::GroupedLayout::{LAYOUT_NAMES[layout]}, {num_experts}>);
"""
    return KernelSource(source, name)

