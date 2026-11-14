import torch

from tensorbridge.gemm import fp8_nvfp4_gemm
from tensorbridge.jit.codegen import Bridge
from tensorbridge.quant import fp8, nvfp4


class NVFP4Linear(torch.nn.Module):
    """Bias-free linear layer with NVFP4 weights and dynamically quantized FP8 activations."""

    def __init__(self, weight: nvfp4.NVFP4Weight) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.weight)
        self.register_buffer("scale", weight.scale)
        self.register_buffer("global_scale", weight.global_scale)
        self.register_buffer("locks", weight.locks)
        self.bridge = weight.bridge  # not a tensor, so not in the state dict

    @classmethod
    def from_float(cls, weight: torch.Tensor, bridge: Bridge = "exact") -> "NVFP4Linear":
        return cls(nvfp4.prepare(*nvfp4.quantize(weight), bridge=bridge))

    def forward(self, a: torch.Tensor, a_scale: torch.Tensor | None = None) -> torch.Tensor:
        """`a` is high precision `[M, K]`, or already FP8 together with its per-token `a_scale` `[M, 1]`.

        Quantized here, the activations come from a kernel that lets the GEMM launch
        early, so the GEMM uses PDL; for FP8 input the previous kernel is unknown.
        """
        follows_quantize = a_scale is None
        if follows_quantize:
            a, a_scale = fp8.quantize(a.view(-1, self.in_features))
        weight = nvfp4.NVFP4Weight(self.weight, self.scale, self.global_scale, self.locks,
                                   (self.out_features, self.in_features), self.bridge)
        return fp8_nvfp4_gemm(a, a_scale, weight, enable_pdl=follows_quantize)
