"""Load block-FP8 routed experts using vLLM's checkpoint loader and native FP4 experts."""

import logging
from typing import Literal

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts, modular_kernel
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.oracle import mxfp4, nvfp4
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kMxfp4Static,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter

from tensorbridge.quant import nvfp4 as weights
from tensorbridge.serving import share_global_scale
from tensorbridge.vllm.moe import keep_converted_weights_on_device

Baseline = Literal["marlin_nvfp4", "humming_nvfp4", "humming_mxfp4", "humming_mxfp4_g128"]
logger = logging.getLogger("vllm.tensorbridge")
converted_layers = 0


def convert_block_fp8(
    weight: torch.Tensor, scales: torch.Tensor, backend: Baseline, gated: bool, *, device: str = "cuda",
    nvfp4_quant_mode: weights.QuantMode = "mse_7ulp",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Checkpoint order [expert, N, K]. Stage one expert on GPU at a time."""
    experts, channels, width = weight.shape
    mx = backend in ("humming_mxfp4", "humming_mxfp4_g128")
    group = 32 if mx else 16
    packed = torch.empty((experts, channels, width // 2), device=device, dtype=torch.uint8)
    group_scales = torch.empty(
        (experts, channels, width // group), device=device, dtype=torch.uint8 if mx else torch.float8_e4m3fn
    )
    global_scales = None if mx else torch.empty(experts, device=device, dtype=torch.float32)
    midpoints = torch.tensor(weights.E2M1_MIDPOINTS, device=device, dtype=torch.float32)
    for expert in range(experts):
        fp8 = weight[expert : expert + 1].to(device)
        blocks = scales[expert : expert + 1].to(device)
        if mx:
            values = (
                fp8.float().view(channels // 128, 128, width // 128, 128)
                * blocks.view(channels // 128, 1, width // 128, 1)
            ).reshape(channels, width // group, group)
            # OCP MX shared exponent = floor(log2(amax)) - Emax(E2M1).
            amax = values.abs().amax(-1)
            exponent = torch.where(amax == 0, 0, (amax.log2().floor() - 2).clamp(-127, 127))
            group_scales[expert] = (exponent + 127).byte()
            normalized = values / exponent.exp2().unsqueeze(-1)
            del values, exponent, amax
            magnitude = normalized.abs()
            codes = torch.bucketize(magnitude, midpoints)
            # At an exact midpoint round to the even E2M1 significand.
            codes += (magnitude == midpoints[codes.clamp_max(6)]) & codes.bitwise_and(1).bool()
            del magnitude
            codes = (codes.byte() | (normalized.signbit().byte() << 3)).view(channels, width)
            del normalized
            packed[expert] = codes[:, ::2] | (codes[:, 1::2] << 4)
            del codes
        else:
            codes, local_scale, multiplier = weights.quantize_fp8_block_to_nvfp4_modelopt(
                fp8, blocks, is_gated_w13=gated, chunk_experts=1, mode=nvfp4_quant_mode, bridge=None
            )
            local_scale, multiplier = (
                share_global_scale(local_scale, multiplier, [channels // 2] * 2) if gated else (local_scale, multiplier)
            )
            assert global_scales is not None
            packed[expert], group_scales[expert], global_scales[expert] = codes[0], local_scale[0], multiplier[0]
            del codes, local_scale, multiplier
        del fp8, blocks
    return packed, group_scales, global_scales


class NativeFp4MoEMethod(FusedMoEMethodBase):
    def __init__(
        self, moe: FusedMoEConfig, name: str, config: Fp8Config, backend: Baseline,
        nvfp4_quant_mode: weights.QuantMode = "mse_7ulp",
    ) -> None:
        super().__init__(moe)
        if config.weight_block_size != [128, 128] or config.activation_scheme != "dynamic":
            raise ValueError(f"{name}: expected block-128 FP8 checkpoint with dynamic activations")
        if moe.hidden_dim % 128 or moe.intermediate_size_per_partition % 128:
            raise ValueError(f"{name}: expected local expert dimensions divisible by 128")
        if moe.has_bias or not moe.is_act_and_mul or moe.in_dtype != torch.bfloat16:
            raise ValueError(f"{name}: native FP4 adapter requires bias-free gated BF16 experts")
        self.nvfp4_quant_mode = nvfp4_quant_mode
        self.backend = backend
        self.layer_name = name
        self.quant_config = config
        self.weight_block_size = self.moe_block_shape = config.weight_block_size
        self.weight_scale_refine = None
        self.block_quant = True
        self.weight_scale_name = "weight_scale_inv"

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **attrs: object,
    ) -> None:
        with torch.device("cpu"):
            Fp8MoEMethod.create_weights(
                self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs
            )
        layer.params_dtype = params_dtype
        if get_current_vllm_config().load_config.load_format != "dummy":
            return
        layer.w13_weight.dummy_weight_value = layer.w2_weight.dummy_weight_value = 0.01

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        if self.backend == "marlin_nvfp4":
            return nvfp4.make_nvfp4_moe_quant_config(
                nvfp4.NvFp4MoeBackend.MARLIN,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w13_weight_scale_2,
                layer.w2_weight_scale_2,
                None,
                None,
                self.moe.swiglu_limit,
                self.moe.swiglu_alpha,
                self.moe.swiglu_beta,
                layer,
                use_a16=True,
            )
        from vllm.model_executor.layers.quantization.utils.humming import get_humming_moe_quant_config

        return get_humming_moe_quant_config(
            layer,
            gemm1_alpha=self.moe.swiglu_alpha,
            gemm1_beta=self.moe.swiglu_beta,
            gemm1_clamp_limit=self.moe.swiglu_limit,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        global converted_layers
        mx = self.backend in ("humming_mxfp4", "humming_mxfp4_g128")
        group_128 = self.backend == "humming_mxfp4_g128"
        activation_key = kFp8Dynamic128Sym if group_128 else kFp8DynamicTokenSym if mx else None
        # Replace each FP8 projection before converting the next. Only the layer
        # should own the checkpoint-format FP4 tensors when the backend repacks them.
        for prefix, gated in (("w13", True), ("w2", False)):
            weight_name = f"{prefix}_weight"
            scale_name = f"{weight_name}_scale_inv"
            packed, scales, multiplier = convert_block_fp8(
                layer.get_parameter(weight_name), layer.get_parameter(scale_name), self.backend, gated,
                nvfp4_quant_mode=self.nvfp4_quant_mode,
            )
            replace_parameter(layer, weight_name, packed)
            replace_parameter(layer, f"{weight_name}_scale", scales)
            del layer._parameters[scale_name], packed, scales
            if multiplier is None:
                continue
            # ModelOpt's stacked gate/up schema reads a multiplier per shard.
            if gated and self.backend == "humming_nvfp4":
                replace_parameter(layer, f"{weight_name}_scale_2", multiplier[:, None].expand(-1, 2).contiguous())
            else:
                replace_parameter(layer, f"{weight_name}_scale_2", multiplier)
            del multiplier

        if self.backend == "marlin_nvfp4":
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
                prepare_nvfp4_moe_layer_for_marlin,
            )

            names = (
                "w13_weight",
                "w13_weight_scale",
                "w13_weight_scale_2",
                "w2_weight",
                "w2_weight_scale",
                "w2_weight_scale_2",
            )
            repacked = prepare_nvfp4_moe_layer_for_marlin(layer, *(layer.get_parameter(name) for name in names), True)
            for name, tensor in zip(names, repacked, strict=True):
                replace_parameter(layer, name, tensor)
            del tensor, repacked
        else:
            recipe = {"quant_method": "mxfp4"} if mx else {"quant_method": "modelopt", "quant_algo": "nvfp4"}
            from vllm.model_executor.layers.quantization.utils.humming import (
                convert_to_humming_moe_kernel_format,
                quant_key_to_input_schema,
            )

            convert_to_humming_moe_kernel_format(
                layer,
                quant_config=recipe,
                input_schema=quant_key_to_input_schema(activation_key),
                allow_input_schema_fallback=False,
            )
        quant = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant
        fmt = (
            modular_kernel.FusedMoEActivationFormat.BatchedExperts
            if self.moe.moe_parallel_config.use_batched_activation_format
            else modular_kernel.FusedMoEActivationFormat.Standard
        )
        nv_backend = nvfp4.NvFp4MoeBackend.MARLIN if self.backend == "marlin_nvfp4" else nvfp4.NvFp4MoeBackend.HUMMING
        classes = (
            mxfp4.backend_to_kernel_cls(mxfp4.Mxfp4MoeBackend.HUMMING)
            if mx
            else nvfp4.backend_to_kernel_cls(nv_backend)
        )
        reasons = []
        for cls in classes:
            supported, reason = cls.is_supported_config(
                cls, self.moe, kMxfp4Static if mx else kNvfp4Static, activation_key, fmt
            )
            if supported:
                break
            reasons.append(reason)
        else:
            raise ValueError(f"{self.layer_name}: {self.backend} unavailable: {reasons}")
        tables = layer._expert_routing_tables()
        if self.backend == "humming_mxfp4" and self.moe.use_fi_nvl_one_sided_kernels:
            # Humming quantizes per-token AFP8 after dispatch. The one-sided buffer must reserve BF16 rows.
            from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize

            prepare = maybe_make_prepare_finalize(
                self.moe, quant, routing_tables=tables, allow_new_interface=True, input_dtype=self.moe.in_dtype
            )
            assert prepare is not None
            self.moe_kernel = modular_kernel.FusedMoEKernel(prepare, cls(moe_config=self.moe, quant_config=quant))
        elif mx:
            self.moe_kernel = mxfp4.make_mxfp4_moe_kernel(
                moe_quant_config=quant,
                moe_config=self.moe,
                mxfp4_backend=mxfp4.Mxfp4MoeBackend.HUMMING,
                experts_cls=cls,
                routing_tables=tables,
            )
        else:
            self.moe_kernel = nvfp4.make_nvfp4_moe_kernel(quant, self.moe, cls, nv_backend, routing_tables=tables)
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
        keep_converted_weights_on_device(layer)
        logger.info(
            "%s: MoE=%s experts=%s all2all=%s dispatch=%s activation=%s%s",
            self.layer_name,
            self.backend,
            cls.__name__,
            self.moe.moe_parallel_config.all2all_backend,
            "fp8" if group_128 else "bf16",
            "fp8_group_128" if group_128 else "fp8" if mx else "bf16",
            "" if mx else f" nvfp4_quant_mode={self.nvfp4_quant_mode}",
        )

        converted_layers += 1
        if converted_layers % 8:
            return
        torch.cuda.empty_cache()

    apply = ModelOptNvFp4FusedMoE.apply
