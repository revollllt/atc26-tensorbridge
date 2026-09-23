"""`D = A @ B.T` with FP8 activations and NVFP4 weights."""

import torch

from tensorbridge.jit import runtime
from tensorbridge.jit.codegen import GemmConfig
from tensorbridge.quant.nvfp4 import NVFP4Weight


def fp8_nvfp4_gemm(
    a: torch.Tensor,
    sfa: torch.Tensor,
    weight: NVFP4Weight,
    out: torch.Tensor | None = None,
    config: GemmConfig | None = None,
) -> torch.Tensor:
    """BF16 `[M, N]` output of `(a * sfa) @ dequant(weight).T`.

    `a` is FP8 E4M3 `[M, K]` with per-token FP32 scales `sfa` `[M, 1]`
    (`tensorbridge.quant.fp8.quantize`), `weight` comes from
    `tensorbridge.quant.nvfp4.prepare`. The kernel is selected from the shape
    alone and JIT-compiled on first use; pass `config` to pin it instead.
    """
    n, k = weight.shape
    if config is None:
        config = runtime.select_config(a.size(0), n, k)
    kernel_id = runtime.get_kernel(n, k, config)
    return torch.ops.tensorbridge.launch_kernel(
        kernel_id, a, sfa, weight.weight, weight.scale, weight.global_scale, weight.locks, out
    )
