"""Pure-torch NVFP4 W4A8 reference (no TensorBridge kernel invoked).

Layout: weight = fp4 e2m1 packed two-per-byte, scale = fp8 e4m3 per group of 16,
optional fp32 global_scale, activation = fp8 e4m3 (dynamic per-row), output bf16/fp16.
This is the exact (non-FPMA) numeric reference used to check the fused kernel.
"""

import torch

# fp4 e2m1 magnitude LUT indexed by the 3-bit magnitude (low 3 bits of nibble):
# 0, 0.5, 1, 1.5, 2, 3, 4, 6 (OCP / NVFP4 E2M1 spec).
_FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4e2m1_codes_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    """Decode 4-bit fp4 e2m1 codes (low nibble of each byte) to fp32."""
    table = torch.tensor(
        list(_FP4_MAG_LUT) + [-x for x in _FP4_MAG_LUT],
        dtype=torch.float32,
        device=codes.device,
    )
    table[8] = 0.0  # canonicalize -0 -> 0
    idx = codes.long() & 0xF
    return table[idx]


def unpack_fp4_packed_to_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (..., k//2) packed fp4 bytes to (..., k) nibble codes.

    Byte b holds [low nibble = element 2b, high nibble = element 2b+1].
    """
    assert packed.dtype in (torch.uint8, torch.int8)
    bytes_view = packed.view(torch.uint8)
    low = bytes_view & 0xF
    high = (bytes_view >> 4) & 0xF
    out = torch.stack([low, high], dim=-1)
    return out.flatten(-2)


def dequant_nvfp4_weight_reference(
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    global_scale: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """(n, k//2) packed + (n, k//G) fp8e4m3 scales (+opt fp32 global) -> (n, k) fp32."""
    assert weight_packed.dim() == 2
    assert weight_scale_e4m3.dim() == 2
    n, k_half = weight_packed.shape
    k = k_half * 2
    assert weight_scale_e4m3.shape == (n, k // group_size)
    assert weight_scale_e4m3.dtype == torch.float8_e4m3fn

    codes = unpack_fp4_packed_to_codes(weight_packed)
    weight_fp = fp4e2m1_codes_to_fp32(codes)
    scale_fp = weight_scale_e4m3.float().repeat_interleave(group_size, dim=-1)
    weight_dq = weight_fp * scale_fp

    if global_scale is not None:
        gs = global_scale.float()
        if gs.dim() == 0 or gs.numel() == 1:
            weight_dq = weight_dq * gs.reshape(())
        else:
            weight_dq = weight_dq * gs.view(-1, 1)
    return weight_dq


def nvfp4_w4a8_reference(
    inputs_hp: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    global_scale: torch.Tensor | None,
    group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """End-to-end exact reference: fp8e4m3 dynamic per-row input quant, exact
    fp4*scale weight dequant, fp32 GEMM, cast to out_dtype."""
    f = torch.finfo(torch.float8_e4m3fn)
    a_amax = inputs_hp.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    a_scale = a_amax / f.max
    a_q = (inputs_hp.float() / a_scale).clamp(f.min, f.max).to(torch.float8_e4m3fn)
    a_dq = a_q.float() * a_scale

    w_dq = dequant_nvfp4_weight_reference(
        weight_packed, weight_scale_e4m3, global_scale, group_size
    )
    out = a_dq @ w_dq.T
    return out.to(out_dtype)
