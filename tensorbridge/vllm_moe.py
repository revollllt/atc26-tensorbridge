"""Opt-in vLLM 0.32 modular MoE experts (`additional_config.tensorbridge.moe`); vLLM owns routing and
communication. `vllm_plugin` chooses the layers they serve."""

import math
import warnings
from typing import TYPE_CHECKING, Literal, cast

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_masked_supported,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    get_moe_permute_scratch,
    moe_prepare_scatter,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config, ModelOptNvFp4FusedMoE
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.utils import replace_parameter
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from tensorbridge import serving
from tensorbridge.gemm import ContiguousRouting, IndexedRouting, MaskedRouting

if TYPE_CHECKING:
    from tensorbridge.vllm_plugin import Options  # the plugin imports this module

warned_moe_fallbacks: set[str] = set()


def uses_fused_silu(moe: FusedMoEConfig) -> bool:
    """Whether `moe` runs the fused `silu(gate) * up + quantize` fast path rather than vLLM's activation."""
    return (
        moe.activation == MoEActivation.SILU
        and moe.swiglu_limit is None
        and moe.swiglu_alpha is None
        and moe.swiglu_beta is None
    )


def unsupported_moe_reason(layer: RoutedExperts, quant_config: ModelOptNvFp4Config | None = None) -> str | None:
    """Why `layer` cannot run on TensorBridge's SM90 NVFP4 grouped kernels, or `None` if it can.

    Checked in `get_quant_method` before constructing `TensorBridgeMoEMethod` so unsupported layers
    fall back cleanly to vLLM's native MoE method with a single warning."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return "TensorBridge MoE requires Hopper (SM90)"
    moe = layer.moe_config
    current = get_current_vllm_config()
    parallel = moe.moe_parallel_config
    if moe.in_dtype != torch.bfloat16:
        return f"in_dtype={moe.in_dtype} (requires torch.bfloat16)"
    if moe.has_bias or moe.is_lora_enabled or parallel.enable_eplb:
        return "bias, LoRA and EPLB are not supported"
    if current.parallel_config.pipeline_parallel_size != 1:
        return "pipeline parallelism (PP > 1) is not supported"
    if layer.apply_router_weight_on_input:
        return "apply_router_weight_on_input is not supported"
    if parallel.use_ep and not (moe.use_deepep_ht_kernels or moe.use_deepep_ll_kernels):
        return f"EP backend {parallel.all2all_backend!r} is not supported (requires DeepEP HT/LL)"
    if quant_config is not None and quant_config.group_size != 16:
        return f"group_size={quant_config.group_size} (requires 16)"
    if not uses_fused_silu(moe):
        supported = (apply_moe_activation_masked_supported(moe.activation) if moe.use_deepep_ll_kernels
                     else apply_moe_activation_supported(moe.activation))
        if not supported:
            return f"activation {moe.activation.value!r} is not supported by vLLM's modular activation"
    hidden = moe.hidden_dim
    intermediate = moe.intermediate_size_per_partition
    if hidden <= 0 or intermediate <= 0 or hidden % 128 != 0 or intermediate % 128 != 0:
        return f"local MoE shape (H={hidden}, I_local={intermediate}) must be positive multiples of 128"
    return None


def warn_moe_fallback(prefix: str, reason: str) -> None:
    if reason not in warned_moe_fallbacks:
        warned_moe_fallbacks.add(reason)
        warnings.warn(
            f"TensorBridge MoE skipped {prefix} ({reason}); falling back to vLLM's native MoE method",
            stacklevel=3,
        )


class TensorBridgeMoEMethod(FusedMoEMethodBase):
    def __init__(
        self, moe: FusedMoEConfig, layer_name: str, options: "Options",
        quant_config: ModelOptNvFp4Config | None = None,
    ) -> None:
        super().__init__(moe)
        current = get_current_vllm_config()
        parallel = moe.moe_parallel_config
        if (moe.in_dtype != torch.bfloat16 or moe.has_bias or moe.is_lora_enabled
                or parallel.enable_eplb or current.parallel_config.pipeline_parallel_size != 1):
            raise ValueError(f"{layer_name}: TensorBridge requires BF16 without bias, LoRA, EPLB or PP")
        if parallel.use_ep and not (moe.use_deepep_ht_kernels or moe.use_deepep_ll_kernels):
            raise ValueError(f"{layer_name}: TensorBridge EP supports DeepEP HT/LL")
        if not uses_fused_silu(moe):
            supported = (apply_moe_activation_masked_supported(moe.activation) if moe.use_deepep_ll_kernels
                         else apply_moe_activation_supported(moe.activation))
            if not supported:
                raise ValueError(f"{layer_name}: unsupported MoE activation {moe.activation.value!r}")
        self.fp8_dispatch = parallel.use_ep and options.dispatch_dtype == "fp8"
        self.gate_up_quant = options.gate_up_quant or ("group_128" if self.fp8_dispatch else "per_token")
        if self.fp8_dispatch and self.gate_up_quant != "group_128":
            raise ValueError(f"{layer_name}: FP8 dispatch requires moe_gate_up_quant=group_128")
        if quant_config is not None and quant_config.group_size != 16:
            raise ValueError(f"{layer_name}: TensorBridge NVFP4 requires group-16 weights")
        self.options = options
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.group_size = 16  # consumed by the native compressed-tensors parameter declaration
        self.use_global_sf = False  # checkpoint activation scales are loaded locally, then discarded
        self.layout: Literal["indexed", "contiguous", "masked"] = (
            "masked" if moe.use_deepep_ll_kernels else "contiguous" if parallel.use_ep else "indexed"
        )

    @property
    def is_monolithic(self) -> bool:
        return False

    def uses_weight_scale_2_pattern(self) -> bool:
        return self.quant_config is not None

    def create_weights(
        self, layer: RoutedExperts, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
        params_dtype: torch.dtype, **extra_weight_attrs: object,
    ) -> None:
        # Reuse checkpoint declarations and loader attributes, without constructing a native compute backend.
        creator = (CompressedTensorsW4A4Nvfp4MoEMethod.create_weights if self.quant_config is None
                   else ModelOptNvFp4FusedMoE.create_weights)
        creator(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                **extra_weight_attrs)
        from tensorbridge.vllm_plugin import register_jit_warmup

        register_jit_warmup()

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        # This describes communication and SwiGLU clamp/alpha/beta parameters.
        # NVFP4 group-16 weights and independent projection activation modes live in NVFP4MoE;
        # LL 0.32 requires [128,128] to select FP8 dispatch.
        return FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn if self.fp8_dispatch else None,
            block_shape=[128, 128] if self.fp8_dispatch else None,
            weight_dtype=torch.uint8,
            is_scale_swizzled=False,
            gemm1_alpha=self.moe.swiglu_alpha,
            gemm1_beta=self.moe.swiglu_beta,
            gemm1_clamp_limit=self.moe.swiglu_limit,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if layer.apply_router_weight_on_input:
            raise ValueError(f"{self.layer_name}: TensorBridge MoE applies router weights to expert outputs")
        if self.quant_config is None:
            prepared = serving.prepare_moe(
                layer.w13_weight_packed, layer.w13_weight_scale, layer.w13_weight_global_scale,
                layer.w2_weight_packed, layer.w2_weight_scale, layer.w2_weight_global_scale,
                self.moe.experts_per_token, self.moe.num_experts, self.moe.ep_size, self.options.bridge,
                layout=self.layout, gate_up_quant=self.gate_up_quant, down_quant=self.options.down_quant,
                is_gated=self.moe.is_act_and_mul, layer_name=self.layer_name,
            )
            del layer.w13_weight_packed, layer.w2_weight_packed
            del layer.w13_weight_global_scale, layer.w2_weight_global_scale
            del layer.w13_input_global_scale, layer.w2_input_global_scale
        else:
            prepared = serving.prepare_moe(
                layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_scale_2,
                layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_scale_2,
                self.moe.experts_per_token, self.moe.num_experts, self.moe.ep_size, self.options.bridge,
                layout=self.layout, scale_convention="multiplier", gate_up_quant=self.gate_up_quant,
                down_quant=self.options.down_quant, is_gated=self.moe.is_act_and_mul, layer_name=self.layer_name,
            )
            del layer.w13_weight_scale_2, layer.w2_weight_scale_2
            del layer.w13_input_scale, layer.w2_input_scale
        replace_parameter(layer, "w13_weight", prepared.w13.weight)
        replace_parameter(layer, "w2_weight", prepared.w2.weight)
        replace_parameter(layer, "w13_weight_scale", prepared.w13.scale)
        replace_parameter(layer, "w2_weight_scale", prepared.w2.scale)
        layer.register_buffer("tensorbridge_w13_global_scale", prepared.w13.global_scale)
        layer.register_buffer("tensorbridge_w2_global_scale", prepared.w2.global_scale)
        layer.register_buffer("tensorbridge_w13_locks", prepared.w13.locks)
        layer.register_buffer("tensorbridge_w2_locks", prepared.w2.locks)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare_finalize = maybe_make_prepare_finalize(
            self.moe, self.moe_quant_config, routing_tables=layer._expert_routing_tables(), allow_new_interface=True,
        )
        assert prepare_finalize is not None
        experts = make_experts(self.moe, self.moe_quant_config, prepared,
                               prepare_finalize.max_num_tokens_per_rank(), prepare_finalize.num_dispatchers())
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
        # Inside GPUModelRunner.load_model (`JitWarmupRegistry._active` is set), defer precompile and
        # autotune to the single post-load `kernel_warmup` pass; in standalone tests run immediately.
        vllm_config = get_current_vllm_config()
        compilation = vllm_config.compilation_config
        if JitWarmupRegistry._active.get() is None or not vllm_config.kernel_config.enable_jit_warmup:
            if compilation.cudagraph_mode != CUDAGraphMode.NONE:
                serving.autotune_prepared(compilation.cudagraph_capture_sizes or [])

    def apply(
        self, layer: RoutedExperts, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None, shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids, activation=layer.activation,
            global_num_experts=layer.global_num_experts, expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts, shared_experts_input=shared_experts_input,
        )


# The checkpoint loader uses these method names to select its format-specific shard handling.
class TensorBridgeCompressedMoEMethod(TensorBridgeMoEMethod):
    pass


class TensorBridgeModelOptMoEMethod(TensorBridgeMoEMethod):
    pass


def make_experts(
    moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig, prepared: serving.NVFP4MoE,
    max_tokens: int | None, num_dispatchers: int,
) -> mk.FusedMoEExpertsModular:
    """Build the modular experts instance for the prepare/finalize activation layout."""
    fused_silu = uses_fused_silu(moe_config)
    unfused_act = not fused_silu

    # vLLM's required underscore hooks are defined inside this factory; project functions use public names.
    class TensorBridgeExperts(mk.FusedMoEExpertsModular):
        @staticmethod
        def activation_format() -> mk.FusedMoEActivationFormat:
            return (mk.FusedMoEActivationFormat.BatchedExperts if prepared.layout == "masked"
                    else mk.FusedMoEActivationFormat.Standard)

        @staticmethod
        def _supports_current_device() -> bool:
            return torch.cuda.get_device_capability() == (9, 0)

        @staticmethod
        def _supports_no_act_and_mul() -> bool:
            return True

        @staticmethod
        def _supports_quant_scheme(weight_key: QuantKey | None, activation_key: QuantKey | None) -> bool:
            return True  # the checkpoint format and communication descriptor were validated by the method

        @staticmethod
        def _supports_activation(activation: MoEActivation) -> bool:
            return (apply_moe_activation_masked_supported(activation) if prepared.layout == "masked"
                    else apply_moe_activation_supported(activation))

        @staticmethod
        def _supports_parallel_config(parallel: FusedMoEParallelConfig) -> bool:
            return not parallel.enable_eplb

        def moe_problem_size(
            self, a1: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_ids: torch.Tensor,
        ) -> tuple[int, int, int, int, int]:
            experts, channels, hidden = prepared.w13.shape
            return experts, a1.size(-2), channels, hidden, prepared.top_k

        def workspace_shapes(
            self, M: int, N: int, K: int, topk: int, global_num_experts: int, local_num_experts: int,
            expert_tokens_meta: mk.ExpertTokensMetadata | None, activation: MoEActivation,
        ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
            specs = serving.moe_buffer_specs(prepared, M, unfused_activation=unfused_act)
            sizes = {name: math.prod(shape) * dtype.itemsize for name, (shape, dtype) in specs.items()}
            output_shape = ((local_num_experts, M, K) if prepared.layout == "masked" else (M, K))
            output_bytes = math.prod(output_shape) * 2
            gate_up_and_act = sizes["gate_up"] + sizes.get("activated", 0)
            first_bytes = max(gate_up_and_act, output_bytes if prepared.layout == "masked"
                              else output_bytes + sizes["output"])
            second_bytes = max(sizes["input"] + sizes["input_scale"],
                               sizes["intermediate"] + sizes["intermediate_scale"])
            return ((first_bytes + 1) // 2,), ((second_bytes + 1) // 2,), output_shape

        def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
            if prepared.layout == "masked":
                return TopKWeightAndReduceDelegate()
            else:
                return TopKWeightAndReduceNoOP()

        def apply(
            self, output: torch.Tensor, hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
            topk_weights: torch.Tensor, topk_ids: torch.Tensor, activation: MoEActivation, global_num_experts: int,
            expert_map: torch.Tensor | None, a1q_scale: torch.Tensor | None, a2_scale: torch.Tensor | None,
            workspace13: torch.Tensor, workspace2: torch.Tensor, expert_tokens_meta: mk.ExpertTokensMetadata | None,
            apply_router_weight_on_input: bool,
        ) -> None:
            token_capacity = hidden_states.size(-2)
            hidden = hidden_states.view(-1, hidden_states.size(-1))
            use_vllm_act = unfused_act or activation != MoEActivation.SILU
            specs = serving.moe_buffer_specs(prepared, token_capacity, unfused_activation=use_vllm_act)
            buffers: dict[str, torch.Tensor] = {}
            second = workspace2.view(torch.uint8).flatten()
            for tensor_name, scale_name in (("input", "input_scale"), ("intermediate", "intermediate_scale")):
                tensor_shape, tensor_dtype = specs[tensor_name]
                scale_shape, scale_dtype = specs[scale_name]
                tensor_bytes = math.prod(tensor_shape) * tensor_dtype.itemsize
                scale_bytes = math.prod(scale_shape) * scale_dtype.itemsize
                buffers[tensor_name] = second[:tensor_bytes].view(tensor_dtype).view(tensor_shape)
                buffers[scale_name] = (second[tensor_bytes:tensor_bytes + scale_bytes]
                                       .view(scale_dtype).view(scale_shape))
            first = workspace13.flatten()
            gate_up_elems = math.prod(specs["gate_up"][0])
            buffers["gate_up"] = first[:gate_up_elems].view(specs["gate_up"][0])
            activated_buf: torch.Tensor | None = None
            if use_vllm_act:
                act_shape = specs["activated"][0]
                act_elems = math.prod(act_shape)
                if first.numel() >= gate_up_elems + act_elems:
                    activated_buf = first[gate_up_elems:gate_up_elems + act_elems].view(act_shape)
                else:
                    activated_buf = torch.empty(act_shape, dtype=torch.bfloat16, device=first.device)
            route_shape = specs["output"][0]
            buffers["output"] = (output.view(route_shape) if prepared.layout == "masked" else
                                 first[output.numel():output.numel() + math.prod(route_shape)].view(route_shape))
            moe_buffers = cast(serving.MoEBuffers, buffers)
            if prepared.layout == "masked":
                assert expert_tokens_meta is not None and expert_tokens_meta.expert_num_tokens is not None
                counts = expert_tokens_meta.expert_num_tokens
                moe_buffers["output"] = output.view(-1, output.size(-1))
                act_fn: serving.ActivationFn | None = None
                if use_vllm_act:
                    def _masked_act(act_out: torch.Tensor, act_in: torch.Tensor) -> None:
                        self.activation(activation, act_out, act_in, valid_token_counts=counts)
                    act_fn = _masked_act
                serving.moe_forward_masked(
                    prepared, hidden, MaskedRouting(counts), moe_buffers, a1q_scale,
                    topk_ids.numel() * prepared.dispatch_ranks,
                    activation_fn=act_fn, activated_buffer=activated_buf,
                )
                return
            elif prepared.layout == "indexed":
                rows = hidden.size(0) * prepared.top_k
                bm13, bm2 = serving.moe_block_sizes(prepared, rows)
                sorted13, experts13, padded13 = moe_align_block_size(
                    topk_ids, bm13, global_num_experts, expert_map, ignore_invalid_experts=True,
                )[:3]
                if bm13 == bm2:
                    sorted2, experts2, padded2 = sorted13, experts13, padded13
                else:
                    sorted2, experts2, padded2 = moe_align_block_size(
                        topk_ids, bm2, global_num_experts, expert_map, ignore_invalid_experts=True,
                    )[:3]
                act_fn = None
                if use_vllm_act:
                    def _indexed_act(act_out: torch.Tensor, act_in: torch.Tensor) -> None:
                        self.activation(activation, act_out, act_in, topk_ids=topk_ids, expert_map=expert_map)
                    act_fn = _indexed_act
                routed = serving.moe_forward_indexed(
                    prepared, hidden, IndexedRouting(sorted13, experts13, padded13, bm13, prepared.top_k),
                    IndexedRouting(sorted2, experts2, padded2, bm2, 1), moe_buffers, a1q_scale,
                    activation_fn=act_fn, activated_buffer=activated_buf,
                )
                moe_fused_mul_sum(routed.view(*topk_ids.shape, -1), topk_weights, outputs=output,
                                  topk_ids=topk_ids, expert_map=expert_map)
                return
            else:
                # Per-ubatch/lane scratch from vLLM's WorkspaceManager so DBO dual streams never collide.
                scratch = get_moe_permute_scratch(
                    num_experts=global_num_experts,
                    num_local_experts=prepared.w13.shape[0],
                    device=hidden.device,
                )
                offsets, scatter = moe_prepare_scatter(topk_ids, expert_map, scratch)
                inverse = scatter.flatten()
                input_scales_2d = (a1q_scale.view(hidden.size(0), -1) if a1q_scale is not None else None)
                local_rows_hint = serving.moe_post_dispatch_rows(prepared, topk_ids.size(0))
                act_fn = None
                if use_vllm_act:
                    valid_i32 = offsets[-1:].to(torch.int32)

                    def _contiguous_act(act_out: torch.Tensor, act_in: torch.Tensor) -> None:
                        if apply_moe_activation_masked_supported(activation):
                            self.activation(activation, act_out, act_in, valid_token_counts=valid_i32)
                        else:
                            self.activation(activation, act_out, act_in)
                    act_fn = _contiguous_act
                routed = serving.moe_forward_contiguous(
                    prepared, hidden, scatter, ContiguousRouting(offsets), moe_buffers, input_scales_2d,
                    local_rows_hint=local_rows_hint, activation_fn=act_fn, activated_buffer=activated_buf,
                )
                moe_unpermute(output, routed.view(*topk_ids.shape, -1), topk_weights, inverse, offsets)
                return

    return TensorBridgeExperts(
        moe_config, quant_config, max_tokens if prepared.layout == "masked" else None,
        num_dispatchers if prepared.layout == "masked" else None,
    )
