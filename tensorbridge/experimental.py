"""Explicit opt-in to the dense SM90 activation `[1,128]` quantization prototype."""

import dataclasses

import torch
import triton

from tensorbridge.jit import codegen, runtime
from tensorbridge.jit.codegen import Bridge, GemmConfig
from tensorbridge.quant import nvfp4
from tensorbridge.quant.nvfp4 import NVFP4Weight


def prepare(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    clamp_scale: bool = False,
    bridge: Bridge = "exact",
) -> NVFP4Weight:
    """`tensorbridge.quant.nvfp4.prepare` for `fp8_nvfp4_group_quant_gemm`: the same weight with its scales by K
    blocks, `[K / 128, N, 8]`, the 8 scale groups of a channel in a 128-deep K block next to each other, which a
    thread of the group-scale kernel reads in one load. `tensorbridge.fp8_nvfp4_gemm` does not take this layout."""
    weight = nvfp4.prepare(packed, scale, global_scale, clamp_scale, bridge)
    groups, padded_n = weight.scale.shape
    return dataclasses.replace(weight, scale=weight.scale.view(groups // 8, 8, padded_n).permute(0, 2, 1).contiguous())


def quantize(
    x: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `x` (`[M, K]`, K a multiple of 128) to FP8 E4M3 with one FP32 scale per token and 128 of K:
    `[M, K]` FP8 and `[M, K / 128]` FP32 scales, M-major (the layout `fp8_nvfp4_group_quant_gemm` reads fastest).

    Runs vLLM's CUDA group-128 quantizer, which the `experimental` extra installs. ``out`` supplies the FP8 and
    FP32 buffers; no scale transpose or extra copy is launched. The scale is ``max(absmax, 1e-30) / 448``; UE8M0
    rounding is off. vLLM's quantizer takes K multiples of 128 only; `fp8_nvfp4_group_quant_gemm` also takes other
    K, with scales from elsewhere.
    """
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA matrix")
    tokens, shape_k = x.shape
    if shape_k == 0 or shape_k % 128 != 0:
        raise ValueError("K must be a positive multiple of 128 for vLLM's group quantizer")
    if out is None:
        quantized = torch.empty((tokens, shape_k), device=x.device, dtype=torch.float8_e4m3fn)
        scale_storage = torch.empty((shape_k // 128, triton.cdiv(tokens, 4) * 4), device=x.device, dtype=torch.float32)
        scales = scale_storage[:, :tokens].t()
    else:
        quantized, scales = out
    if tokens == 0:
        return quantized, scales
    import vllm._custom_ops  # noqa: F401  registers torch.ops._C

    column_major = scales.stride(0) == 1
    tma_aligned = column_major and scales.stride(1) == triton.cdiv(tokens, 4) * 4
    torch.ops._C.per_token_group_fp8_quant(x, quantized, scales, 128, 1e-30, -448.0, 448.0, False, column_major,
                                          tma_aligned)
    return quantized, scales


def fp8_nvfp4_group_quant_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: NVFP4Weight,
    out: torch.Tensor | None = None,
    config: GemmConfig | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """BF16 `[M, N]` output of `(a * a_scale) @ dequant(weight).T` with `[1, 128]` activation scales: `a` is FP8
    E4M3 `[M, K]` and `a_scale` FP32 `[M, ceil(K / 128)]`, one scale per token and 128 of K, the last one for the K
    columns from the last multiple of 128 on; M-major as `quantize` writes it, or K-major. `weight` comes from
    `prepare` of this module. Each WGMMA output subtile keeps one partial sum per 128 of K. `config` and
    `enable_pdl` as for `tensorbridge.fp8_nvfp4_gemm`."""
    from tensorbridge.jit import compiler

    shape_n, shape_k = weight.shape
    if config is None:
        config = runtime.select_config(a.size(0), shape_n, shape_k, device=a.get_device(), group_scales=True)
    # The group-scale kernel (`impls/sm90_w4a8_pgroup.cuh`) for the weight shape and the scales' layout, compiled
    # and registered on the first request.
    scale_m_major = a_scale.stride(0) == 1 and a_scale.stride(1) == triton.cdiv(a.size(0), 4) * 4
    kernel_id = runtime.register_device_kernel(
        a.get_device(), codegen.generate_pgroup(shape_n, shape_k, config, scale_m_major, weight.bridge),
        compiler.INCLUDE_DIR, shape_n, shape_k, config,
        activation_scale_groups=codegen.pad_to_unit(shape_k) // 128, scale_m_major=scale_m_major, block_scales=True,
    )
    return torch.ops.tensorbridge.launch_kernel(
        kernel_id, a, a_scale, weight.weight, weight.scale, weight.global_scale, weight.locks, out, enable_pdl,
    )
