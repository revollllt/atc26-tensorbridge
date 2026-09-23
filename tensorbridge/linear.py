import torch

from tensorbridge.gemm import fp8_nvfp4_gemm
from tensorbridge.quant import fp8, nvfp4


class NVFP4Linear(torch.nn.Module):
    """Bias-free linear layer with NVFP4 weights and dynamically quantized FP8 activations."""

    def __init__(self, weight: nvfp4.NVFP4Weight):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        for name, tensor in vars(weight).items():
            self.register_buffer(name, tensor)

    @classmethod
    def from_float(cls, weight: torch.Tensor) -> "NVFP4Linear":
        return cls(nvfp4.prepare(*nvfp4.quantize(weight)))

    def forward(self, x: torch.Tensor, x_scale: torch.Tensor | None = None) -> torch.Tensor:
        """`x` is high precision, or already FP8 together with its per-token `x_scale`."""
        if x_scale is None:
            x, x_scale = fp8.quantize(x.view(-1, self.in_features))
        weight = nvfp4.NVFP4Weight(self.weight, self.scale, self.global_scale, self.locks)
        return fp8_nvfp4_gemm(x, x_scale, weight)
