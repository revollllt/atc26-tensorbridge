"""TensorBridge: NVFP4-weight x FP8-activation GEMMs on Hopper FP8 tensor cores.

The dense and grouped (MoE) GEMMs, the weight layouts they take, the linear layer, and config selection. Weights
come from `tensorbridge.quant.nvfp4`, activations from `tensorbridge.quant.fp8`; `[1, 128]` activation scales are in
`tensorbridge.experimental`.
"""

from tensorbridge.gemm import (
    ContiguousRouting,
    IndexedRouting,
    MaskedRouting,
    Routing,
    fp8_nvfp4_gemm,
    fp8_nvfp4_grouped_gemm,
)
from tensorbridge.jit.autotune import autotune, autotune_grouped
from tensorbridge.jit.codegen import Bridge, GemmConfig
from tensorbridge.jit.runtime import select_config
from tensorbridge.linear import NVFP4Linear
from tensorbridge.quant.nvfp4 import NVFP4Experts, NVFP4Weight

__all__ = [
    "Bridge", "ContiguousRouting", "GemmConfig", "IndexedRouting", "MaskedRouting", "NVFP4Experts", "NVFP4Linear",
    "NVFP4Weight", "Routing", "autotune", "autotune_grouped", "fp8_nvfp4_gemm", "fp8_nvfp4_grouped_gemm",
    "select_config",
]
