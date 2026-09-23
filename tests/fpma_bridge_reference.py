"""Bit-level Python reproduction of the device NVFP4 W4A8 FPMA dequant bridge.

The device bridge (tensorbridge/include/tensorbridge/dequant/fpma.cuh) does NOT do
an exact FP4*scale dequant. For each FP4 nibble it packs sign/magnitude into an
fp8e4m3 bit layout and INTEGER-ADDS a "prefolded" scale byte (`raw_e4m3 - 0x1C`),
then masks the FP4-zero lanes to fp8 +0. The resulting byte is interpreted as
fp8e4m3 and fed to WGMMA; the implicit /6 from the layout is compensated by
`global_scale *= 6` in `tensorbridge.quant.nvfp4.prepare`.

This module reproduces that integer pipeline exactly so we can build a
BRIDGE-ACCURATE W4A8 reference and SEPARATE the bridge's approximation error from
WGMMA-accumulation / bf16-rounding noise. the FPMA bridge test shows kernel↔bridge-ref
is far tighter than kernel↔exact-ref, which is the evidence the reproduction is right.

Per-nibble scalar map (the device low- and high-nibble paths are bit-symmetric):
    sign = nibble bit3 ; mag = nibble bits0..2 (0..7)
    addend = (sign << 7) | (mag << 2)
    byte   = (prefolded_scale + addend) & 0xFF      # per-byte add, safe domain => no carry
    byte   = 0 if mag == 0 else byte                # FP4 zero -> fp8 +0 (device nonzero mask)
value = fp8e4m3_decode(byte) ; effective weight = value * (global_scale * 6).

With `snc=True` the map also models the offline SNC recoding the kernel runs with:
the subnormal 0.5 (mag == 1) is stored as mag 0, so its addend carries no magnitude
bits and the byte decodes to half of what 1.0 decodes to (mag << 2 would give 2/3).

Safe domain (device comment): raw in [0x1C, 0x7E] -> prefolded in [0, 0x62];
addend max = 0x80 | (7<<2) = 0x9C; max sum = 0x62 + 0x9C = 0xFE <= 0xFF (no byte carry).
"""

import torch

PREFOLD_DELTA = 0x1C  # kE4M3HalfMinusSixCodeDelta


def prefold_scale_and_global(weight_scale_e4m3, global_scale):
    """Reproduce the scale prefold of `nvfp4.prepare`: prefolded = raw - 0x1C (clamp 0..255),
    global *= 6. Returns (prefolded_uint8, global_x6_or_None)."""
    raw = weight_scale_e4m3.view(torch.uint8).to(torch.int16)
    prefolded = (raw - PREFOLD_DELTA).clamp_(min=0, max=255).to(torch.uint8)
    g6 = (global_scale.float() * 6.0) if global_scale is not None else None
    return prefolded, g6


def fpma_bridge_fp8_bytes(codes, prefolded_scale, group_size, snc=False):
    """codes: (..., k) uint8 FP4 nibbles in 0..15. prefolded_scale: (..., k//group_size)
    uint8. Returns the fp8e4m3 bytes (uint8) the device bridge feeds to WGMMA."""
    codes = codes.to(torch.int32)
    sign = (codes >> 3) & 0x1
    mag = codes & 0x7
    stored_mag = torch.where(mag == 1, torch.zeros_like(mag), mag) if snc else mag
    addend = (sign << 7) | (stored_mag << 2)
    pref = prefolded_scale.to(torch.int32).repeat_interleave(group_size, dim=-1)
    s = pref + addend
    max_sum = int(s.max().item())
    assert max_sum <= 0xFF, f"bridge integer add overflowed a byte (max={max_sum:#x}); scale domain violated"
    byte = s & 0xFF
    byte = torch.where(mag == 0, torch.zeros_like(byte), byte)
    return byte.to(torch.uint8)


def fp8e4m3_decode(byte_uint8):
    return byte_uint8.contiguous().view(torch.float8_e4m3fn).float()


def dequant_nvfp4_weight_bridge(codes, weight_scale_e4m3, global_scale, group_size, snc=False):
    """Bridge-accurate weight dequant: the fp8 values WGMMA multiplies, times global*6.
    codes (n,k) uint8; weight_scale (n, k//g) fp8e4m3; global per-channel/scalar/None."""
    prefolded, g6 = prefold_scale_and_global(weight_scale_e4m3, global_scale)
    fp8_bytes = fpma_bridge_fp8_bytes(codes, prefolded, group_size, snc)
    w = fp8e4m3_decode(fp8_bytes)
    if g6 is not None:
        g6 = g6.float()
        if g6.dim() == 0 or g6.numel() == 1:
            w = w * g6.reshape(())
        else:
            w = w * g6.view(-1, 1)
    return w


def nvfp4_w4a8_bridge_reference(inputs_hp, codes, weight_scale_e4m3, global_scale,
                                group_size, out_dtype, snc=False):
    """End-to-end bridge-accurate W4A8 reference matching the kernel datapath:
    fp8 activation operand (scale deferred) x bridge fp8 weight, fp32 accumulate,
    then * per-row a_scale (global*6 already folded into the weight). Cast to out_dtype.
    """
    f = torch.finfo(torch.float8_e4m3fn)
    a_amax = inputs_hp.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    a_scale = a_amax / f.max
    a_q = (inputs_hp.float() / a_scale).clamp(f.min, f.max).to(torch.float8_e4m3fn)
    a_q_val = a_q.float()  # fp8 operand value WGMMA reads; per-row scale applied in epilogue

    w_bridge = dequant_nvfp4_weight_bridge(codes, weight_scale_e4m3, global_scale, group_size, snc)
    psum = a_q_val @ w_bridge.T
    out = psum * a_scale
    return out.to(out_dtype)
