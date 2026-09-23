"""Out-of-tree NVFP4 linear integration for the AE vLLM fork.

`TENSORBRIDGE_FPMA_ALPHA` sets the bridge's global-scale factor: a number (default 1)
or `lsq` for a per-layer least-squares fit (`nvfp4.lsq_alpha`).
"""

import os

import torch
from compressed_tensors.quantization import QuantizationArgs

from tensorbridge import fp8_nvfp4_gemm
from tensorbridge.quant import fp8, nvfp4
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
    CompressedTensorsW4A16Fp4,
)
from vllm.utils.torch_utils import direct_register_custom_op


def tensorbridge_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    locks: torch.Tensor,
    output: torch.Tensor,
) -> None:
    activations, activation_scale = fp8.quantize(x)
    weight = nvfp4.NVFP4Weight(packed_weight, weight_scale, global_scale, locks)
    fp8_nvfp4_gemm(activations, activation_scale, weight, out=output)


def tensorbridge_linear_fake(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    locks: torch.Tensor,
    output: torch.Tensor,
) -> None:
    return None


def _alpha(packed: torch.Tensor, scale: torch.Tensor) -> float:
    alpha = os.environ.get("TENSORBRIDGE_FPMA_ALPHA", str(nvfp4.FPMA_ALPHA))
    return nvfp4.lsq_alpha(packed, scale) if alpha == "lsq" else float(alpha)


class TensorBridgeScheme(CompressedTensorsW4A16Fp4):
    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        inverse_scales = layer.weight_global_scale.float()
        if not torch.equal(inverse_scales, inverse_scales[0].expand_as(inverse_scales)):
            raise ValueError("TensorBridge requires a shared global scale for fused linears")

        prepared = nvfp4.prepare(
            layer.weight_packed.data,
            layer.weight_scale.data,
            (1.0 / inverse_scales[0]).view(1),
            alpha=_alpha(layer.weight_packed.data, layer.weight_scale.data),
        )
        del layer.weight_packed, layer.weight_scale, layer.weight_global_scale
        layer.register_buffer("tensorbridge_weight", prepared.weight)
        layer.register_buffer("tensorbridge_scale", prepared.scale)
        layer.register_buffer("tensorbridge_global_scale", prepared.global_scale)
        layer.register_buffer("tensorbridge_locks", prepared.locks)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = torch.empty(
            (x.numel() // x.shape[-1], layer.output_size_per_partition),
            dtype=torch.bfloat16,
            device=x.device,
        )
        torch.ops.vllm.tensorbridge_linear(
            x.reshape(-1, x.shape[-1]),
            layer.tensorbridge_weight,
            layer.tensorbridge_scale,
            layer.tensorbridge_global_scale,
            layer.tensorbridge_locks,
            output,
        )
        output = output.reshape(*x.shape[:-1], layer.output_size_per_partition)
        if bias is not None:
            return output + bias
        else:
            return output


class TensorBridgeConfig(CompressedTensorsConfig):
    def get_name(self) -> str:
        return "tensorbridge"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, object], user_quant: str | None, hf_config: object = None
    ) -> str | None:
        if user_quant == "tensorbridge":
            return "tensorbridge"
        else:
            return None

    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> CompressedTensorsScheme | None:
        scheme_dict = self.get_scheme_dict(layer, layer_name)
        if scheme_dict is None:
            return super().get_scheme(layer, layer_name)
        else:
            weight_quant = scheme_dict.get("weights")
        if isinstance(weight_quant, QuantizationArgs) and self._is_nvfp4_format(weight_quant):
            self._check_scheme_supported(TensorBridgeScheme.get_min_capability())
            return TensorBridgeScheme()
        else:
            return super().get_scheme(layer, layer_name)


def register() -> None:
    direct_register_custom_op(
        op_name="tensorbridge_linear",
        op_func=tensorbridge_linear,
        mutates_args=["locks", "output"],
        fake_impl=tensorbridge_linear_fake,
    )
    register_quantization_config("tensorbridge")(TensorBridgeConfig)
