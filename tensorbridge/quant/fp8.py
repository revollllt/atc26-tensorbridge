"""Dynamic per-token FP8 E4M3 activation quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(x_ptr, xq_ptr, scale_ptr, stride_x, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1e-30) / 448  # E4M3 maximum
    inv_scale = 1 / scale
    tl.store(xq_ptr + row * K + cols, (x * inv_scale).to(tl.float8e4nv), mask=mask)
    tl.store(scale_ptr + row, scale)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quantization: `x ~= x_fp8 * scale`.

    `[M, K]` BF16/FP16/FP32 in; `[M, K]` FP8 E4M3 and `[M, 1]` FP32 scales out.
    """
    assert x.is_cuda and x.dim() == 2 and x.stride(1) == 1
    m, k = x.shape
    x_fp8 = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(k)
    _quantize_rows[(m,)](
        x, x_fp8, scale, x.stride(0), k, block,
        num_warps=min(max(block // 256, 1), 8), num_stages=1,
    )
    return x_fp8, scale
