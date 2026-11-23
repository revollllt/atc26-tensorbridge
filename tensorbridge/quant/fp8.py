"""Dynamic per-token/group-128 FP8 E4M3 quantization, optionally fused with MoE SiLU-mul or scatter."""

from typing import Literal

import torch
import triton
import triton.language as tl

InputQuant = Literal["per_token", "group_128"]


# Batch lengths only bound accesses; new prefill lengths reuse the same kernel.
@triton.jit(do_not_specialize=["rows", "output_rows"])
def scatter_fp8_rows(
    input_ptr: tl.tensor, scale_ptr: tl.tensor, output_ptr: tl.tensor, output_scale_ptr: tl.tensor,
    scatter_ptr: tl.tensor, valid_rows_ptr: tl.tensor, input_stride: tl.constexpr,
    scale_stride_m: tl.constexpr, scale_stride_k: tl.constexpr,
    rows: tl.tensor, width: tl.constexpr, output_rows: tl.tensor, groups: tl.constexpr, top_k: tl.constexpr,
    block_k: tl.constexpr, block_g: tl.constexpr, bounded: tl.constexpr,
) -> None:
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    if row >= rows:
        return
    destination_rows = tl.load(valid_rows_ptr).to(tl.int64) if bounded else output_rows.to(tl.int64)
    has_local = 0
    for route in tl.static_range(top_k):
        target = tl.load(scatter_ptr + row * top_k + route).to(tl.int64)
        has_local += tl.where((target >= 0) & (target < destination_rows), 1, 0)
    if has_local > 0:
        g_offs = tl.arange(0, block_g)
        g_mask = g_offs < groups
        scales = tl.load(scale_ptr + row * scale_stride_m + g_offs * scale_stride_k, g_mask, other=1.0)
        for route in tl.static_range(top_k):
            target = tl.load(scatter_ptr + row * top_k + route).to(tl.int64)
            if (target >= 0) & (target < destination_rows):
                tl.store(output_scale_ptr + target * groups + g_offs, scales, g_mask)
        for k_start in tl.static_range(0, width, block_k):
            cols = k_start + tl.arange(0, block_k)
            c_mask = cols < width
            vals = tl.load(input_ptr + row * input_stride + cols, c_mask, other=0.0)
            for route in tl.static_range(top_k):
                target = tl.load(scatter_ptr + row * top_k + route).to(tl.int64)
                if (target >= 0) & (target < destination_rows):
                    tl.store(output_ptr + target * width + cols, vals, c_mask)


def scatter_fp8(
    inputs: torch.Tensor, input_scale: torch.Tensor, scatter_idx: torch.Tensor,
    *, out: tuple[torch.Tensor, torch.Tensor], num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter pre-quantized FP8 `[T, K]` rows and FP32 `[T, groups]` scales to expert-ordered
    `out` (`[R, K]` FP8, `[R, groups]` FP32) at `scatter_idx` (`[T, top_k]` int32), skipping
    targets outside the local expert prefix (all R rows when no bound is supplied)."""
    if not (inputs.is_cuda and inputs.dtype == torch.float8_e4m3fn and inputs.dim() == 2 and inputs.stride(1) == 1):
        raise ValueError("inputs must be an FP8 E4M3 CUDA matrix with unit stride along K")
    if not (input_scale.is_cuda and input_scale.dtype == torch.float32 and input_scale.dim() == 2):
        raise ValueError("input_scale must be a 2D FP32 CUDA matrix")
    if not (scatter_idx.dtype == torch.int32 and scatter_idx.is_contiguous()):
        raise ValueError("scatter_idx must be a contiguous int32 tensor")
    rows, width = inputs.shape
    groups = input_scale.size(1)
    output, output_scale = out
    block_k = min(2048, triton.next_power_of_2(max(128, width)))
    block_g = triton.next_power_of_2(max(1, groups))
    warps = 8 if block_k >= 2048 else 4
    if rows > 0:
        scatter_fp8_rows[(rows,)](
            inputs, input_scale, output, output_scale, scatter_idx,
            num_valid_rows if num_valid_rows is not None else scatter_idx,
            inputs.stride(0), input_scale.stride(0), input_scale.stride(1),
            rows, width, output.size(0), groups, scatter_idx.size(1),
            block_k, block_g, num_valid_rows is not None, num_warps=warps, num_stages=1,
        )
    return output, output_scale


# Batch lengths only bound accesses; new prefill lengths reuse the same kernel.
@triton.jit(do_not_specialize=["rows", "output_rows"])
def quantize_moe_rows(
    input_ptr: tl.tensor, output_ptr: tl.tensor, scale_ptr: tl.tensor, counts_ptr: tl.tensor,
    scatter_ptr: tl.tensor, input_stride: tl.constexpr, rows: tl.tensor, width: tl.constexpr,
    capacity: tl.constexpr, output_rows: tl.tensor, groups: tl.constexpr, top_k: tl.constexpr, block: tl.constexpr,
    activation: tl.constexpr, masked: tl.constexpr, bounded: tl.constexpr, scatter: tl.constexpr,
    m_major_scales: tl.constexpr = False, scale_stride_k: tl.constexpr = 0,
    rows_per_tile: tl.constexpr = 4, interleaved_gate_up: tl.constexpr = False,
) -> None:
    tl.extra.cuda.gdc_launch_dependents()
    row_start = (tl.program_id(0) * rows_per_tile).to(tl.int64)
    n_valid = tl.load(counts_ptr).to(tl.int64) if bounded else row_start + rows_per_tile
    if scatter or (not bounded) or (row_start < n_valid):
        group = tl.program_id(1)
        columns = group * block + tl.arange(0, block)
        col_mask = columns < width
        gate_cols = (group * (2 * block) + tl.arange(0, block)) if (activation and interleaved_gate_up) else columns
        up_cols = (gate_cols + block) if (activation and interleaved_gate_up) else (width + columns)
        if rows_per_tile == 1 and not scatter:
            if masked:
                valid = (row_start < rows) & (
                    row_start % capacity < tl.load(counts_ptr + row_start // capacity, row_start < rows, other=0)
                )
            elif bounded:
                valid = (row_start < rows) & (row_start < n_valid)
            else:
                valid = row_start < rows
            if valid:
                input_values = tl.load(input_ptr + row_start * input_stride + gate_cols, col_mask, other=0).to(tl.float32)
                if activation:
                    up = tl.load(
                        input_ptr + row_start * input_stride + up_cols, col_mask, other=0
                    ).to(tl.float32)
                    silu = tl.extra.cuda.libdevice.div_rn(
                        input_values, 1.0 + tl.extra.cuda.libdevice.exp(-input_values)).to(tl.bfloat16)
                    values = (silu.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
                else:
                    values = input_values
                scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-30) / 448
                quantized = (values / scale).to(tl.float8e4nv)
                tl.store(output_ptr + row_start * width + columns, quantized, col_mask)
                if m_major_scales:
                    tl.store(scale_ptr + group * scale_stride_k + row_start, scale)
                else:
                    tl.store(scale_ptr + row_start * groups + group, scale)
        else:
            row = row_start + tl.arange(0, rows_per_tile).to(tl.int64)
            if masked:
                valid = (row < rows) & (row % capacity < tl.load(counts_ptr + row // capacity, row < rows, other=0))
            elif bounded and not scatter:
                valid = (row < rows) & (row < n_valid)
            else:
                valid = row < rows
            input_values = tl.load(input_ptr + row[:, None] * input_stride + gate_cols[None, :],
                                   valid[:, None] & col_mask[None, :], other=0).to(tl.float32)
            if activation:
                up = tl.load(input_ptr + row[:, None] * input_stride + up_cols[None, :],
                             valid[:, None] & col_mask[None, :], other=0).to(tl.float32)
                silu = tl.extra.cuda.libdevice.div_rn(
                    input_values, 1.0 + tl.extra.cuda.libdevice.exp(-input_values)).to(tl.bfloat16)
                values = (silu.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
            else:
                values = input_values
            scales = tl.maximum(tl.max(tl.abs(values), axis=1), 1e-30) / 448
            quantized = (values / scales[:, None]).to(tl.float8e4nv)
            if scatter:
                for route in tl.static_range(top_k):
                    target = tl.load(scatter_ptr + row * top_k + route, row < rows, other=output_rows)
                    destination_rows = n_valid if bounded else output_rows
                    local = valid & (target >= 0) & (target < destination_rows)
                    tl.store(output_ptr + target[:, None].to(tl.int64) * width + columns[None, :], quantized,
                             local[:, None] & col_mask[None, :])
                    tl.store(scale_ptr + target.to(tl.int64) * groups + group, scales, local)
            else:
                tl.store(output_ptr + row[:, None] * width + columns[None, :], quantized,
                         valid[:, None] & col_mask[None, :])
                if m_major_scales:
                    tl.store(scale_ptr + group * scale_stride_k + row, scales, valid)
                else:
                    tl.store(scale_ptr + row * groups + group, scales, valid)


@triton.jit
def quantize_moe_masked_rows(
    input_ptr: tl.tensor, output_ptr: tl.tensor, scale_ptr: tl.tensor, counts_ptr: tl.tensor,
    input_stride: tl.constexpr, width: tl.constexpr, capacity: tl.constexpr, groups: tl.constexpr,
    block: tl.constexpr, rows_per_tile: tl.constexpr, blocks_per_expert: tl.constexpr,
    activation: tl.constexpr,
) -> None:
    """Compact persistent masked-MoE quantizer: launches O(E * blocks_per_expert * groups) CTAs
    instead of O(E * capacity * groups), skipping all invalid capacity slots with zero wasted math."""
    tl.extra.cuda.gdc_launch_dependents()
    expert = tl.program_id(0).to(tl.int64)
    cnt = tl.maximum(tl.minimum(tl.load(counts_ptr + expert).to(tl.int64), capacity), 0)
    tile_id = tl.program_id(1).to(tl.int64)
    r_start = tile_id * rows_per_tile
    if r_start < cnt:
        group = tl.program_id(2)
        columns = group * block + tl.arange(0, block)
        col_mask = columns < width
        base_row = expert * capacity
        step = blocks_per_expert * rows_per_tile
        while r_start < cnt:
            if rows_per_tile == 1:
                row = base_row + r_start
                input_values = tl.load(input_ptr + row * input_stride + columns, col_mask, other=0.0).to(tl.float32)
                if activation:
                    up = tl.load(input_ptr + row * input_stride + width + columns, col_mask, other=0.0).to(tl.float32)
                    silu = tl.extra.cuda.libdevice.div_rn(
                        input_values, 1.0 + tl.extra.cuda.libdevice.exp(-input_values)).to(tl.bfloat16)
                    values = (silu.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
                else:
                    values = input_values
                scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-30) / 448
                quantized = (values / scale).to(tl.float8e4nv)
                tl.store(output_ptr + row * width + columns, quantized, col_mask)
                tl.store(scale_ptr + row * groups + group, scale)
            else:
                local_r = r_start + tl.arange(0, rows_per_tile).to(tl.int64)
                valid = local_r < cnt
                row = base_row + local_r
                mask2d = valid[:, None] & col_mask[None, :]
                input_values = tl.load(input_ptr + row[:, None] * input_stride + columns[None, :],
                                       mask2d, other=0.0).to(tl.float32)
                if activation:
                    up = tl.load(input_ptr + row[:, None] * input_stride + width + columns[None, :],
                                 mask2d, other=0.0).to(tl.float32)
                    silu = tl.extra.cuda.libdevice.div_rn(
                        input_values, 1.0 + tl.extra.cuda.libdevice.exp(-input_values)).to(tl.bfloat16)
                    values = (silu.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
                else:
                    values = input_values
                scales = tl.maximum(tl.max(tl.abs(values), axis=1), 1e-30) / 448
                quantized = (values / scales[:, None]).to(tl.float8e4nv)
                tl.store(output_ptr + row[:, None] * width + columns[None, :], quantized, mask2d)
                tl.store(scale_ptr + row * groups + group, scales, valid)
            r_start += step


def quantize_group_128(
    x: torch.Tensor, *, out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token group-128 dynamic FP8 E4M3 quantization for `[M, K]` (any `K % 16 == 0`), releasing
    PDL dependents early. When `out` is omitted, allocates M-major TMA-aligned scales (`stride(0) == 1`,
    `stride(1) == ceil(M / 4) * 4`) matching `sm90_w4a8_pgroup.cuh`'s fast path."""
    if not (x.is_cuda and x.dim() == 2 and x.stride(1) == 1):
        raise ValueError("x must be a CUDA matrix with unit stride along K")
    rows, width = x.shape
    groups = triton.cdiv(width, 128)
    aligned_rows = triton.cdiv(rows, 4) * 4
    if out is None:
        quantized = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
        scale_storage = torch.empty((groups, aligned_rows), dtype=torch.float32, device=x.device)
        scales = scale_storage[:, :rows].t()
    else:
        quantized, scales = out
    if rows == 0:
        return quantized, scales
    m_major = scales.stride(0) == 1 and (groups == 1 or scales.stride(1) >= rows)
    quantize_moe_rows[(triton.cdiv(rows, 4), groups)](
        x, quantized, scales, scales, scales, x.stride(0), rows, width, 1, rows, groups, 1, 128,
        False, False, False, False, m_major, scales.stride(1) if m_major else 0,
        num_warps=4, num_stages=1,
    )
    return quantized, scales


def quantize_moe(
    inputs: torch.Tensor, mode: InputQuant, *, activation: bool = False,
    expert_counts: torch.Tensor | None = None, num_valid_rows: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None, out: tuple[torch.Tensor, torch.Tensor],
    interleaved_gate_up: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize valid MoE rows, optionally fusing SiLU-mul or expert scatter.

    Masked input is flattened [E * capacity, K]; counts bound each expert separately.
    Scales are FP32 [output rows, 1 or ceil(K/128)]. With scatter, num_valid_rows bounds
    destination indices. Every source token can still have a local route.
    When `interleaved_gate_up=True` (with `activation=True` and `mode=="group_128"`),
    expects 128-channel interleaved `[gate_0, up_0, gate_1, up_1, ...]` layout.
    """
    rows = inputs.size(0)
    width = inputs.size(1) // (2 if activation else 1)
    output, scales = out
    if rows == 0:
        return output, scales
    if interleaved_gate_up and not (activation and mode == "group_128" and expert_counts is None and scatter_idx is None):
        raise ValueError("interleaved_gate_up requires activation=True, mode='group_128', and contiguous layout")
    groups = triton.cdiv(width, 128) if mode == "group_128" else 1
    block = 128 if mode == "group_128" else triton.next_power_of_2(width)
    if expert_counts is not None and scatter_idx is None:
        num_experts = expert_counts.numel()
        capacity = rows // num_experts
        if num_experts == 0 or capacity == 0:
            return output, scales
        if mode == "per_token":
            rows_per_tile = 1
            blocks_per_expert = min(capacity, max(4, triton.cdiv(512, num_experts)))
            warps = num_warps(block)
        else:
            rows_per_tile = 4
            max_tiles = triton.cdiv(capacity, rows_per_tile)
            blocks_per_expert = min(max_tiles, max(1, 2048 // max(1, num_experts * groups)))
            warps = 4
        quantize_moe_masked_rows[(num_experts, blocks_per_expert, groups)](
            inputs, output, scales, expert_counts,
            inputs.stride(0), width, capacity, groups,
            block, rows_per_tile, blocks_per_expert, activation,
            num_warps=warps, num_stages=1,
        )
        return output, scales

    capacity = rows // expert_counts.numel() if expert_counts is not None else 1
    if mode == "per_token" and scatter_idx is None:
        rows_per_tile = 1
    elif mode == "group_128" and activation and scatter_idx is None and expert_counts is None and rows >= 1024:
        rows_per_tile = 8
    else:
        rows_per_tile = 4
    warps = num_warps(block) if rows_per_tile == 1 else 4
    quantize_moe_rows[(triton.cdiv(rows, rows_per_tile), groups)](
        inputs, output, scales, expert_counts if expert_counts is not None else num_valid_rows, scatter_idx,
        inputs.stride(0), rows, width, capacity, output.size(0), groups,
        1 if scatter_idx is None else scatter_idx.size(1), block,
        activation, expert_counts is not None, num_valid_rows is not None, scatter_idx is not None,
        False, 0, rows_per_tile, interleaved_gate_up,
        num_warps=warps, num_stages=1,
    )
    return output, scales


def num_warps(block: int) -> int:
    """Warps of a row kernel over `block` columns: one per 256 columns, 1 to 8."""
    return min(max(block // 256, 1), 8)


@triton.jit
def quantize_rows(x_ptr, xq_ptr, scale_ptr, stride_x, K: tl.constexpr, BLOCK: tl.constexpr):
    # A kernel launched next with PDL may start now; it waits for this one to
    # finish before reading the quantized rows.
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)  # row * stride exceeds 2^31 on large masked buffers
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1e-30) / 448  # E4M3 maximum
    inv_scale = 1 / scale
    tl.store(xq_ptr + row * K + cols, (x * inv_scale).to(tl.float8e4nv), mask=mask)
    tl.store(scale_ptr + row, scale)


def quantize(
    x: torch.Tensor, *, out: tuple[torch.Tensor, torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quantization: `x ~= x_fp8 * scale`.

    `[M, K]` BF16/FP16/FP32 in; `[M, K]` FP8 E4M3 and `[M, 1]` FP32 scales out.
    The kernel lets the next kernel on the stream launch as soon as it starts, so
    a following `fp8_nvfp4_gemm(..., enable_pdl=True)` overlaps its launch with it.
    `out` supplies contiguous FP8 `[M, K]` and FP32 `[M, 1]` buffers.
    """
    if not (x.is_cuda and x.dim() == 2 and x.stride(1) == 1):
        raise ValueError("x must be a CUDA matrix with unit stride along K")
    m, k = x.shape
    if out is None:
        x_fp8 = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    else:
        x_fp8, scale = out
    block = triton.next_power_of_2(k)
    quantize_rows[(m,)](
        x, x_fp8, scale, x.stride(0), k, block, num_warps=num_warps(block), num_stages=1,
    )
    return x_fp8, scale


@triton.jit
def silu_mul_quantize_rows(x_ptr, xq_ptr, scale_ptr, valid_rows_ptr, stride_x, I: tl.constexpr,  # noqa: E741
                           BLOCK: tl.constexpr, HAS_VALID_ROWS: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)  # row * stride exceeds 2^31 on large masked buffers
    cols = tl.arange(0, BLOCK)
    mask = cols < I
    if HAS_VALID_ROWS:
        valid = row < tl.load(valid_rows_ptr)
        mask = mask & valid
    gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + I + cols, mask=mask, other=0.0).to(tl.float32)
    # BF16 SiLU, then the BF16 product, rounded as torch rounds `silu(gate) * up` on BF16 tensors.
    silu = tl.extra.cuda.libdevice.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate)).to(tl.bfloat16)
    x = (silu.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1e-30) / 448  # as `quantize_rows`
    inv_scale = 1 / scale
    tl.store(xq_ptr + row * I + cols, (x * inv_scale).to(tl.float8e4nv), mask=mask)
    if HAS_VALID_ROWS:
        tl.store(scale_ptr + row, scale, mask=valid)
    else:
        tl.store(scale_ptr + row, scale)


def silu_mul_quantize(
    gate_up: torch.Tensor, num_valid_rows: torch.Tensor | None = None,
    *, out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`quantize(silu(gate) * up)` in one kernel, with the same BF16 rounding: `gate_up` is BF16
    `[R, 2I]` with the gate in the first `I` columns; out FP8 `[R, I]` and FP32 `[R, 1]`.
    `num_valid_rows`, a device integer `[1]`, leaves the rows from it on unread and unwritten.
    `out` supplies the FP8 and FP32 buffers instead of allocating them."""
    if not (gate_up.is_cuda and gate_up.dtype == torch.bfloat16 and gate_up.dim() == 2 and gate_up.stride(1) == 1):
        raise ValueError("gate_up must be a BF16 CUDA matrix with unit stride along its columns")
    rows, two_i = gate_up.shape
    i = two_i // 2
    if out is None:
        x_fp8 = torch.empty((rows, i), dtype=torch.float8_e4m3fn, device=gate_up.device)
        scale = torch.empty((rows, 1), dtype=torch.float32, device=gate_up.device)
    else:
        x_fp8, scale = out
    block = triton.next_power_of_2(i)
    silu_mul_quantize_rows[(rows,)](
        gate_up, x_fp8, scale, num_valid_rows if num_valid_rows is not None else scale, gate_up.stride(0), i, block,
        num_valid_rows is not None, num_warps=num_warps(block), num_stages=1,
    )
    return x_fp8, scale


@triton.jit
def scatter_quantize_rows(x_ptr, xq_ptr, scale_ptr, scatter_ptr, stride_x, rows, K: tl.constexpr,
                          TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + token * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1e-30) / 448  # as `quantize_rows`
    inv_scale = 1 / scale
    xq = (x * inv_scale).to(tl.float8e4nv)
    for j in tl.static_range(TOP_K):
        target = tl.load(scatter_ptr + token * TOP_K + j)
        local = target < rows
        tl.store(xq_ptr + target.to(tl.int64) * K + cols, xq, mask=mask & local)
        tl.store(scale_ptr + target, scale, mask=local)


def scatter_quantize(
    hidden_states: torch.Tensor, scatter_idx: torch.Tensor, rows: int,
    *, out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each token of `hidden_states` `[T, K]` once, as `quantize` does, and write it to each
    row `scatter_idx[t, j]` of the `[rows, K]` FP8 and `[rows, 1]` FP32 outputs; targets from `rows`
    on (experts on other ranks) are skipped and rows no token targets are left unwritten.
    `scatter_idx` is int32 `[T, top_k]`; `out` supplies the FP8 and FP32 buffers."""
    if not (hidden_states.is_cuda and hidden_states.dim() == 2 and hidden_states.stride(1) == 1):
        raise ValueError("hidden_states must be a CUDA matrix with unit stride along K")
    if not (scatter_idx.dtype == torch.int32 and scatter_idx.is_contiguous()):
        raise ValueError("scatter_idx must be a contiguous int32 tensor")
    tokens, k = hidden_states.shape
    if out is None:
        x_fp8 = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=hidden_states.device)
        scale = torch.empty((rows, 1), dtype=torch.float32, device=hidden_states.device)
    else:
        x_fp8, scale = out
    block = triton.next_power_of_2(k)
    scatter_quantize_rows[(tokens,)](
        hidden_states, x_fp8, scale, scatter_idx, hidden_states.stride(0), rows, k, scatter_idx.size(1), block,
        num_warps=num_warps(block), num_stages=1,
    )
    return x_fp8, scale
