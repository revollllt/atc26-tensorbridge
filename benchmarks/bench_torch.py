#!/usr/bin/env python3
"""PyTorch: `torch._scaled_mm` on FP8 E4M3 weights and activations, with per-token and per-channel
scales (rowwise) or one scale per tensor (tensorwise, cuBLASLt), and the BF16 GEMM.

    python benchmarks/bench_torch.py --shape_n 8192 --shape_k 8192
"""

from collections.abc import Callable

import common
import torch
from common import DenseShape, Entry, PreparedCall, Shape, Unsupported


def fp8_prepare(rowwise: bool) -> Callable[[Shape, int], PreparedCall | Unsupported]:
    def prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
        assert isinstance(shape, DenseShape)
        torch.manual_seed(0)
        activation = torch.randn(shape.m, shape.k, device="cuda").to(torch.float8_e4m3fn)  # [M, K]
        weight = torch.randn(shape.n, shape.k, device="cuda").to(torch.float8_e4m3fn)  # [N, K]
        if rowwise:
            scale_a = torch.rand(shape.m, 1, device="cuda") + 0.5  # fp32 [M, 1]
            scale_b = torch.rand(1, shape.n, device="cuda") + 0.5  # fp32 [1, N]
        else:
            scale_a = torch.rand((), device="cuda") + 0.5
            scale_b = torch.rand((), device="cuda") + 0.5
        weight_copies = [weight] + [weight.clone() for _ in range(copies - 1)]

        def run(copy_index: int) -> torch.Tensor:
            return torch._scaled_mm(activation, weight_copies[copy_index].t(), scale_a=scale_a, scale_b=scale_b,
                                    out_dtype=torch.bfloat16)

        def check() -> tuple[bool, str]:
            reference = (activation.float() * scale_a) @ (weight.float() * scale_b.reshape(-1, 1)).t()
            return common.check_output(run(0), reference)

        return PreparedCall(run, check, 2.0 * shape.m * shape.n * shape.k, common.dense_traffic(shape, 1, 1), "")

    return prepare


def bf16_prepare(shape: Shape, copies: int) -> PreparedCall | Unsupported:
    assert isinstance(shape, DenseShape)
    torch.manual_seed(0)
    activation = torch.randn(shape.m, shape.k, device="cuda", dtype=torch.bfloat16)  # [M, K]
    weight = torch.randn(shape.n, shape.k, device="cuda", dtype=torch.bfloat16)  # [N, K]
    weight_copies = [weight] + [weight.clone() for _ in range(copies - 1)]

    def run(copy_index: int) -> torch.Tensor:
        return activation @ weight_copies[copy_index].t()

    def check() -> tuple[bool, str]:
        return common.check_output(run(0), activation.float() @ weight.float().t())

    return PreparedCall(run, check, 2.0 * shape.m * shape.n * shape.k, common.dense_traffic(shape, 2, 2), "")


def entries() -> list[Entry]:
    version = torch.__version__
    return [
        Entry("torch_fp8_rowwise", "torch", "w8a8_float8e4m3_rowwise", version, "fp8", 1, "dense", fp8_prepare(True)),
        Entry("torch_fp8_tensorwise", "torch", "w8a8_float8e4m3_tensorwise", version, "fp8", 1, "dense",
              fp8_prepare(False)),
        Entry("torch_bf16", "torch", "w16a16_bfloat16", version, "bf16", 2, "dense", bf16_prepare),
    ]


def main() -> None:
    args = common.bench_parser(__doc__).parse_args()
    common.run_bench(entries(), args, {})


if __name__ == "__main__":
    main()
