"""On-load quantization to standard ModelOpt NVFP4 tensors, on CPU or CUDA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, get_args

import torch

if TYPE_CHECKING:
    from tensorbridge.jit.codegen import Bridge

QuantMode = Literal["mse_7ulp", "four_over_six", "bridge_aware", "rtn"]
GROUP_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
RAW_SCALE_MIN, RAW_SCALE_MAX = 0x1C, 0x7E
E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def parse_quant_mode(value: object) -> QuantMode:
    if value not in get_args(QuantMode):
        raise ValueError(f"Unknown nvfp4_quant_mode {value!r}")
    return cast(QuantMode, value)


def decoded_levels(device: torch.device, bridge: Bridge | None) -> torch.Tensor:
    """Normalized positive levels for every finite nonnegative E4M3 scale byte."""
    raw = torch.arange(RAW_SCALE_MAX + 1, device=device, dtype=torch.uint8)[:, None]
    scale = raw.view(torch.float8_e4m3fn).float()
    levels = torch.tensor(E2M1_POS, device=device, dtype=torch.float32)
    if bridge is None:
        return scale * levels
    if bridge == "exact":
        return (scale * (levels / 6)).to(torch.float8_e4m3fn).float()
    if bridge != "fpma":
        raise ValueError(f"Unknown bridge {bridge!r}")
    # Standard magnitudes 1..7 address SNC selectors 0,2,3,4,5,6,7 in fpma::lut_half.
    offsets = torch.tensor([0, 0, 8, 12, 16, 20, 24, 28], device=device, dtype=torch.int16)
    encoded = (raw.short() - RAW_SCALE_MIN + offsets).clamp_min(0).byte()
    encoded[:, 0] = 0
    return encoded.masked_fill(raw < RAW_SCALE_MIN, 0).view(torch.float8_e4m3fn).float()


def quantize_groups(
    groups: torch.Tensor,
    global_scale: torch.Tensor,
    mode: QuantMode,
    bridge: Bridge | None,
    table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bounded [experts, groups, 16] tile using one candidate at a time."""
    magnitude = groups.abs()
    amax = magnitude.amax(-1)
    s6 = (amax / (6 * global_scale)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    if mode == "rtn":
        effective = (s6.float() * global_scale).clamp_min(1e-30)
        boundaries = torch.tensor(E2M1_MIDPOINTS, dtype=torch.float32, device=groups.device)
        best_codes = torch.bucketize(magnitude / effective.unsqueeze(-1), boundaries).byte()
        best_raw = s6.view(torch.uint8)
    else:
        base = s6.view(torch.uint8).short()
        offsets = range(-2, 5) if mode == "mse_7ulp" else range(2 if mode == "four_over_six" else 1)
        for index, offset in enumerate(offsets):
            if mode == "mse_7ulp":
                raw = (base + offset).clamp(RAW_SCALE_MIN if bridge is not None else 0, RAW_SCALE_MAX).byte()
                raw = raw.masked_fill(amax == 0, 0)
            elif mode == "four_over_six" and index == 1:
                raw = (amax / (4 * global_scale)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
            else:
                raw = s6.view(torch.uint8)
            levels = table[raw.long()] * (global_scale if bridge is None else 6 * global_scale).unsqueeze(-1)
            boundaries = levels[..., :-1] * 0.5 + levels[..., 1:] * 0.5
            codes = torch.searchsorted(boundaries, magnitude)
            error = (magnitude - levels.gather(-1, codes)).square().sum(-1)
            if index == 0:
                best_raw, best_codes, best_error = raw, codes.byte(), error
            else:
                better = error < best_error
                best_raw = torch.where(better, raw, best_raw)
                best_codes = torch.where(better.unsqueeze(-1), codes.byte(), best_codes)
                best_error = torch.minimum(error, best_error)
    signed = best_codes | (torch.signbit(groups).byte() << 3)
    signed.masked_fill_((amax == 0).unsqueeze(-1), 0)
    return signed[..., ::2] | (signed[..., 1::2] << 4), best_raw.view(torch.float8_e4m3fn)


@torch.no_grad()
def convert_weights(
    weight: torch.Tensor,
    scales: torch.Tensor | None,
    block_size: tuple[int, int],
    gated: bool,
    chunk_experts: int,
    mode: QuantMode,
    bridge: Bridge | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parse_quant_mode(mode)
    matrix = weight.ndim == 2
    if weight.ndim not in (2, 3) or weight.size(-1) % GROUP_SIZE:
        raise ValueError("Expected [N,K] or [E,N,K] weights with K divisible by 16")
    if matrix:
        weight = weight.unsqueeze(0)
        scales = None if scales is None else scales.unsqueeze(0)
    experts, rows, width = weight.shape
    shards = 2 if gated and not matrix else 1
    if chunk_experts < 1 or (shards == 2 and rows % 2):
        raise ValueError("Expected positive chunk_experts and equal gate/up shards")
    packed = torch.empty((experts, rows, width // 2), device=weight.device, dtype=torch.uint8)
    local = torch.empty((experts, rows, width // GROUP_SIZE), device=weight.device, dtype=torch.float8_e4m3fn)
    globals_out = torch.empty((experts, shards), device=weight.device, dtype=torch.float32)
    table = decoded_levels(weight.device, bridge)
    block_n, block_k = block_size
    for first in range(0, experts, chunk_experts):
        source = weight[first : first + chunk_experts]
        batch = len(source)
        # Cap each FP32 tile at ~4 MiB for GLM. No full FP32 expert or seven-candidate stack.
        row_step = max(1, (1 << 20) // (batch * width))
        for shard in range(shards):
            begin, end = shard * rows // shards, (shard + 1) * rows // shards

            def read_rows(start: int, stop: int) -> torch.Tensor:
                values = source[:, start:stop].float()
                if scales is None:
                    return values
                factors = scales[first : first + batch].float()
                if factors.ndim == 3:
                    # Use absolute rows even when a gate/up boundary splits an FP8 block.
                    block_rows = torch.arange(start, stop, device=weight.device) // block_n
                    return values * factors[:, block_rows, :].repeat_interleave(block_k, -1)[..., :width]
                if shards == 2 and factors.ndim == 2 and factors.size(1) == 2:
                    return values * factors[:, shard, None, None]
                return values * factors.reshape(batch, 1, 1)

            peak = torch.zeros(batch, device=weight.device, dtype=torch.float32)
            for start in range(begin, end, row_step):
                peak = torch.maximum(peak, read_rows(start, min(start + row_step, end)).abs().amax(dim=(1, 2)))
            divisor = 6 if mode in ("rtn", "bridge_aware") else 4
            multiplier = (peak / (divisor * E4M3_MAX)).clamp_min(1e-30)
            globals_out[first : first + batch, shard] = multiplier
            for start in range(begin, end, row_step):
                stop = min(start + row_step, end)
                groups = read_rows(start, stop).reshape(batch, -1, GROUP_SIZE)
                codes, group_scales = quantize_groups(groups, multiplier[:, None], mode, bridge, table)
                packed[first : first + batch, start:stop] = codes.reshape(batch, stop - start, -1)
                local[first : first + batch, start:stop] = group_scales.reshape(batch, stop - start, -1)
    if matrix:
        return packed[0], local[0], globals_out.flatten()
    return packed, local, globals_out if shards == 2 else globals_out.flatten()


def quantize_modelopt(
    weight: torch.Tensor,
    *,
    mode: QuantMode = "mse_7ulp",
    bridge: Bridge | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return standard packed E2M1, group-16 E4M3 scales and a per-matrix FP32 multiplier."""
    return convert_weights(weight, None, (128, 128), False, 4, mode, bridge)


def quantize_fp8_block_to_nvfp4_modelopt(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    *,
    is_gated_w13: bool = False,
    chunk_experts: int = 4,
    mode: QuantMode = "mse_7ulp",
    bridge: Bridge | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize FP8 blocks in row tiles. Gate/up retain separate global multipliers per expert."""
    return convert_weights(weight_fp8, scale_inv, block_size, is_gated_w13, chunk_experts, mode, bridge)
