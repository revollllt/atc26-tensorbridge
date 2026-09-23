"""Source generation: one translation unit per (N, K, GemmConfig).

All kernel logic lives in `include/tensorbridge`. The generated file only pins
the template parameters and supplies the one thing C++ cannot template: the
WGMMA instruction, whose mnemonic and operand list encode the token tile.
"""

import dataclasses

KERNEL_NAME = "tensorbridge::sm90_fp8_nvfp4_gemm<Config>"


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    """Mirror of `GemmConfig` in `csrc/heuristics/config.hpp`, in field order."""

    block_m: int
    num_stages: int
    multicast_a: int
    multicast_b: int
    num_producer_regs: int
    use_stream_k: bool
    use_tma_store: bool
    m_fast_tile_order: bool
    prebroadcast_scale: bool

    @classmethod
    def from_list(cls, values: list[int]) -> "GemmConfig":
        fields = dataclasses.fields(cls)
        assert len(values) == len(fields), "GemmConfig is out of sync with config.hpp"
        return cls(*(field.type(value) for field, value in zip(fields, values)))

    def to_list(self) -> list[int]:
        return [int(value) for value in dataclasses.astuple(self)]


def generate_wgmma(block_m: int) -> str:
    """`accum += a (registers) x b (shared memory)`, m64 x n`block_m` x k32.

    Operands: `block_m / 2` FP32 accumulator registers, the four FP8 registers
    of the weight fragment, and the shared-memory descriptor. The trailing
    constants are the output scale (1: accumulate) and the operand scales.
    """
    num_accum = block_m // 2
    accum = ", ".join(f"%{i}" for i in range(num_accum))
    fragment = ", ".join(f"%{num_accum + i}" for i in range(4))
    accum_binds = ", ".join(f'"+f"(d[{i}])' for i in range(num_accum))
    fragment_binds = ", ".join(f'"r"(a[{i}])' for i in range(4))
    return (
        f'    asm volatile(\n'
        f'        "wgmma.mma_async.sync.aligned.m64n{block_m}k32.f32.e4m3.e4m3 "\n'
        f'        "{{{accum}}}, {{{fragment}}}, %{num_accum + 4}, 1, 1, 1;\\n"\n'
        f'        : {accum_binds}\n'
        f'        : {fragment_binds}, "l"(desc));'
    )


def generate(shape_n: int, shape_k: int, config: GemmConfig) -> str:
    def cpp(value: int | bool) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    return f"""\
#include <tensorbridge/common/base.cuh>

struct Config {{
  static constexpr uint32_t kShapeN = {shape_n};
  static constexpr uint32_t kShapeK = {shape_k};
  static constexpr uint32_t kBlockM = {config.block_m};
  static constexpr uint32_t kNumStages = {config.num_stages};
  static constexpr uint32_t kMulticastA = {config.multicast_a};
  static constexpr uint32_t kMulticastB = {config.multicast_b};
  static constexpr uint32_t kNumProducerRegs = {config.num_producer_regs};
  static constexpr bool kUseStreamK = {cpp(config.use_stream_k)};
  static constexpr bool kUseTmaStore = {cpp(config.use_tma_store)};
  static constexpr bool kMFastTileOrder = {cpp(config.m_fast_tile_order)};
  static constexpr bool kPrebroadcastScale = {cpp(config.prebroadcast_scale)};

  TB_INLINE static void wgmma(uint32_t *a, uint64_t &desc, float *d) {{
{generate_wgmma(config.block_m)}
  }}
}};

#include <tensorbridge/impls/sm90_fp8_nvfp4_gemm.cuh>

// Read back by the JIT: the launch has to reserve this much shared memory.
extern "C" __constant__ uint32_t SMEM_BYTES = sizeof(tensorbridge::SharedStorage<Config>);
"""
