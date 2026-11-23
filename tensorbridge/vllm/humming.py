"""vLLM 0.32 Humming grouped pipeline with TensorBridge's two NVFP4 projections."""

import torch
from humming.schema.mxfp4 import Mxfp4WeightSchema
from humming.transform import prepare_layer_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingGroupedExperts
from vllm.model_executor.layers.quantization.utils.humming import quant_key_to_input_schema
from vllm.model_executor.layers.quantization.utils.humming.moe import make_humming_moe_quant_config
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, kFp8DynamicTokenSym

from tensorbridge.gemm import ContiguousRouting, fp8_nvfp4_grouped_gemm
from tensorbridge.serving import NVFP4MoE


def humming_pipeline_config(moe: FusedMoEConfig) -> FusedMoEQuantConfig:
    # Match the MXFP4/AFP8 baseline's activation and workspace metadata. Humming never owns expert weights here.
    input_schema = quant_key_to_input_schema(kFp8DynamicTokenSym)
    weight_schema = Mxfp4WeightSchema().to_humming_schema(moe.in_dtype)
    hidden, intermediate = moe.hidden_dim, moe.intermediate_size_per_partition
    configs = {
        name: prepare_layer_config(
            shape_n=channels,
            shape_k=width,
            weight_schema=weight_schema,
            input_schema=input_schema,
            num_experts=moe.num_local_experts,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=moe.in_dtype,
            device=torch.device(moe.device),
        )
        for name, channels, width in (("w13", 2 * intermediate, hidden), ("w2", hidden, intermediate))
    }
    config = make_humming_moe_quant_config(
        quant_dtype=str(input_schema.a_dtype),
        weight_dtype=str(weight_schema.b_dtype),
        weight_group_shape=GroupShape(row=weight_schema.weight_scale_group_size, col=1),
        humming_configs=configs,
    )
    config.w1_hadamard_block_size = config.w2_hadamard_block_size = weight_schema.hadamard_block_size
    return config


class TensorBridgeHummingGroupedExperts(HummingGroupedExperts):
    """Inherit prepare_buffers, process_input, apply, unpermute and reduction from Humming."""

    def __init__(self, moe: FusedMoEConfig, quant: FusedMoEQuantConfig, prepared: NVFP4MoE) -> None:
        super().__init__(moe, quant)
        self.projections = {"w13": prepared.w13, "w2": prepared.w2}

    def humming_forward(
        self,
        sublayer_name: str,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor | None,
        outputs: torch.Tensor,
        input_scale_2: torch.Tensor | None = None,
        *,
        valid_shape_m: int,
        expert_layout: torch.Tensor,
        compute_config: str,
        tuning_config: str,
    ) -> torch.Tensor:
        assert input_scale is not None and input_scale_2 is None
        return fp8_nvfp4_grouped_gemm(
            inputs,
            input_scale.view(-1, 1),
            self.projections[sublayer_name],
            ContiguousRouting(expert_layout),
            valid_shape_m,
            out=outputs,
        )
