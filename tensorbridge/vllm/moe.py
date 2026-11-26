"""Opt-in vLLM 0.32 modular MoE experts (`additional_config.tensorbridge.moe`); vLLM owns routing and
communication. `tensorbridge.vllm.plugin` chooses the layers they serve."""

import logging
import math
from typing import TYPE_CHECKING, Literal, cast

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
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
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config, ModelOptNvFp4FusedMoE
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.utils import replace_parameter
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from tensorbridge import megamoe, serving
from tensorbridge.gemm import ContiguousRouting, IndexedRouting, MaskedRouting
from tensorbridge.nvlink_ep import NVLinkSymmetricEP
from tensorbridge.quant import nvfp4

if TYPE_CHECKING:
    from tensorbridge.vllm.plugin import Options  # the plugin imports this module

logger = logging.getLogger("vllm.tensorbridge")
_NVLINK_EP_CACHE: dict[tuple[int, int, int, int, int, int, int], NVLinkSymmetricEP] = {}
_NVLINK_PRECOMPILED: set[tuple[NVLinkSymmetricEP, str]] = set()
_FP8_CONVERTED_LAYERS = 0


def _get_or_create_nvlink_ep(
    local_e: int, hidden: int, intermediate: int, top_k: int, max_tokens_per_rank: int, device: int,
) -> NVLinkSymmetricEP:
    ep_group = get_ep_group().device_group
    ep_size = get_ep_group().world_size
    key = (device, ep_size, local_e, hidden, intermediate, top_k, max_tokens_per_rank)
    if (ep := _NVLINK_EP_CACHE.get(key)) is not None:
        return ep
    ep = NVLinkSymmetricEP(
        ep_group, local_e=local_e, hidden=hidden, intermediate=intermediate, top_k=top_k,
        max_tokens_per_rank=max_tokens_per_rank, device=device,
    )
    _NVLINK_EP_CACHE[key] = ep
    return ep


class _NvlinkEpExpertsStub:
    consumes_expert_mask: bool = False


class _NvlinkEpPrepareFinalizeStub:
    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32


class _NvlinkEpKernelStub:
    """Minimal FusedMoEKernel-compatible handle so vLLM's MoERunner and warmup passes recognize that
    NVLinkSymmetricEP handles internal dispatch/combine, produces reduced outputs, and overlaps shared_experts."""

    can_overlap_shared_experts: bool = True
    is_monolithic: bool = False

    def __init__(self) -> None:
        self.fused_experts = _NvlinkEpExpertsStub()
        self.prepare_finalize = _NvlinkEpPrepareFinalizeStub()
        self.impl = self

    def output_is_reduced(self) -> bool:
        return True


def uses_fused_silu(moe: FusedMoEConfig) -> bool:
    """Whether `moe` runs the fused `silu(gate) * up + quantize` fast path rather than vLLM's activation."""
    return (
        moe.activation == MoEActivation.SILU
        and moe.swiglu_limit is None
        and moe.swiglu_alpha is None
        and moe.swiglu_beta is None
    )


def unsupported_moe_reason(
    layer: RoutedExperts,
    quant_config: ModelOptNvFp4Config | None = None,
    options: "Options | None" = None,
) -> str | None:
    """Why `layer` cannot run on TensorBridge's SM90 NVFP4 grouped kernels, or `None` if it can.

    Checked in `get_quant_method` before constructing `TensorBridgeMoEMethod` so unsupported layers
    identify the unsupported configuration before kernel construction."""
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
    use_nvlink_ep = bool(options and options.moe_nvlink_ep and parallel.use_ep and parallel.use_all2all_kernels)
    if options and options.moe_nvlink_ep and not use_nvlink_ep:
        return "NVLinkSymmetricEP requires an active EP all-to-all group"
    if parallel.use_ep and parallel.use_all2all_kernels and not use_nvlink_ep and not (
        moe.use_deepep_ht_kernels or moe.use_deepep_ll_kernels
        or moe.use_ag_rs_all2all_kernels or moe.use_fi_nvl_one_sided_kernels
    ):
        return f"EP backend {parallel.all2all_backend!r} is not supported"
    if options and options.moe_backend_mode == "humming_grouped" and not (
        parallel.use_ep and parallel.use_all2all_kernels and moe.use_ag_rs_all2all_kernels
        and moe.is_act_and_mul and uses_fused_silu(moe)
    ):
        return "humming_grouped requires AG/RS expert parallelism and standard SwiGLU"
    if use_nvlink_ep:
        if parallel.ep_size > 8:
            return f"moe_nvlink_ep supports single-node ep_size <= 8, got {parallel.ep_size}"
        if moe.num_local_experts > 128:
            return f"moe_nvlink_ep requires local_e <= 128, got {moe.num_local_experts}"
        if not uses_fused_silu(moe) or not moe.is_act_and_mul:
            return "moe_nvlink_ep requires standard gated SiLU activation"
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


def warn_moe_fallback(prefix: str, reason: str, strict: bool) -> None:
    if strict:
        raise ValueError(f"{prefix}: requested MoE backend unavailable: {reason}")
    logger.warning("%s: using native vLLM MoE (%s)", prefix, reason)


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
        self.use_nvlink_ep = bool(options.moe_nvlink_ep and parallel.use_ep and parallel.use_all2all_kernels)
        if not uses_fused_silu(moe):
            supported = (apply_moe_activation_masked_supported(moe.activation) if moe.use_deepep_ll_kernels
                         else apply_moe_activation_supported(moe.activation))
            if not supported:
                raise ValueError(f"{layer_name}: unsupported MoE activation {moe.activation.value!r}")
        self.use_fused_swiglu = bool(
            (options.moe_backend_mode in ("fused_swiglu", "megamoe") or self.use_nvlink_ep)
            and uses_fused_silu(moe)
            and moe.is_act_and_mul
        )
        self.fp8_dispatch = (
            parallel.use_ep and parallel.use_all2all_kernels and (options.dispatch_dtype == "fp8" or self.use_nvlink_ep)
        )
        self.gate_up_quant = "group_128" if (self.use_fused_swiglu or self.use_nvlink_ep) else options.gate_up_quant
        self.down_quant = "group_128" if (self.use_fused_swiglu or self.use_nvlink_ep) else options.down_quant
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
            "masked" if (self.use_nvlink_ep or moe.use_deepep_ll_kernels)
            else "contiguous" if (parallel.use_ep or self.use_fused_swiglu) else "indexed"
        )
        self._nvlink_ep: NVLinkSymmetricEP | None = None
        self._prepared_moe: serving.NVFP4MoE | None = None

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        if self._nvlink_ep is not None:
            # When `shared_experts` has no TP collectives (SP+EP replicates weights with `disable_tp=True`,
            # or `tp_size == 1`), let vLLM's `SharedExperts.maybe_forward_async` (`MULTI_STREAM_OVERLAPPED`)
            # launch `shared_experts` on `aux_stream` BEFORE `gate(x) + topk + fp8_quant + MegaMoE`, hiding
            # `shared_experts` latency without risking TP all_reduce deadlocks. Keep `True`
            # (`MK_INTERNAL_OVERLAPPED`) only for TP-sharded `shared_experts`.
            if self.moe.moe_parallel_config.is_sequence_parallel or get_tensor_model_parallel_world_size() == 1:
                return False
            return True
        return super().mk_can_overlap_shared_experts

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.use_nvlink_ep:
            return torch.int32
        return super().topk_indices_dtype

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
        from tensorbridge.vllm.plugin import register_jit_warmup

        register_jit_warmup()

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        if self.options.moe_backend_mode == "humming_grouped":
            from tensorbridge.vllm.humming import humming_pipeline_config

            return humming_pipeline_config(self.moe)
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

    def _finalize_prepared_layer(
        self, layer: RoutedExperts, prepared: serving.NVFP4MoE, quant_mode: nvfp4.QuantMode | None = None,
    ) -> None:
        replace_parameter(layer, "w13_weight", prepared.w13.weight)
        replace_parameter(layer, "w2_weight", prepared.w2.weight)
        if "w13_weight_scale" in layer._parameters:
            replace_parameter(layer, "w13_weight_scale", prepared.w13.scale)
            replace_parameter(layer, "w2_weight_scale", prepared.w2.scale)
        else:
            layer.register_parameter("w13_weight_scale", torch.nn.Parameter(prepared.w13.scale, requires_grad=False))
            layer.register_parameter("w2_weight_scale", torch.nn.Parameter(prepared.w2.scale, requires_grad=False))
        layer.register_buffer("tensorbridge_w13_global_scale", prepared.w13.global_scale)
        layer.register_buffer("tensorbridge_w2_global_scale", prepared.w2.global_scale)
        layer.register_buffer("tensorbridge_w13_locks", prepared.w13.locks)
        layer.register_buffer("tensorbridge_w2_locks", prepared.w2.locks)
        self._prepared_moe = prepared
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        if self.use_nvlink_ep:
            local_e, _, hidden = prepared.w13.shape
            intermediate = prepared.w2.shape[2]
            dev = prepared.w13.weight.get_device()
            ep = _get_or_create_nvlink_ep(
                local_e, hidden, intermediate, self.moe.experts_per_token,
                self.options.moe_max_tokens_per_rank, dev,
            )
            self._nvlink_ep = ep
            self.moe_kernel = _NvlinkEpKernelStub()  # type: ignore[assignment]
            pre_key = (ep, prepared.w13.bridge)
            if pre_key not in _NVLINK_PRECOMPILED:
                capture_sizes = get_current_vllm_config().compilation_config.cudagraph_capture_sizes or ()
                # Include the final partial chunk of each captured shape as well as a full chunk.
                if ep.ep_group is not None and ep.ep_rank != 0:
                    torch.distributed.barrier(group=ep.ep_group)
                    ep.precompile(prepared.w13, prepared.w2, capture_sizes=capture_sizes)
                else:
                    ep.precompile(prepared.w13, prepared.w2, capture_sizes=capture_sizes)
                    if ep.ep_group is not None:
                        torch.distributed.barrier(group=ep.ep_group)
                _NVLINK_PRECOMPILED.add(pre_key)
        else:
            prepare_finalize = maybe_make_prepare_finalize(
                self.moe, self.moe_quant_config, routing_tables=layer._expert_routing_tables(),
                allow_new_interface=True,
            )
            assert prepare_finalize is not None
            if self.options.moe_backend_mode == "humming_grouped":
                from tensorbridge.vllm.humming import TensorBridgeHummingGroupedExperts

                experts = TensorBridgeHummingGroupedExperts(self.moe, self.moe_quant_config, prepared)
            else:
                experts = make_experts(
                    self.moe, self.moe_quant_config, prepared,
                    prepare_finalize.max_num_tokens_per_rank(), prepare_finalize.num_dispatchers(),
                )
            self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
            vllm_config = get_current_vllm_config()
            compilation = vllm_config.compilation_config
            if JitWarmupRegistry._active.get() is None or not vllm_config.kernel_config.enable_jit_warmup:
                if compilation.cudagraph_mode != CUDAGraphMode.NONE:
                    serving.autotune_prepared(compilation.cudagraph_capture_sizes or [])

        if self.use_nvlink_ep or self.options.moe_backend_mode == "humming_grouped":
            resolved = self.options.moe_backend_mode
        else:
            resolved = "modular_fused_swiglu" if self.use_fused_swiglu else "modular_standard"
        logger.info("%s: MoE=%s layout=%s all2all=%s dispatch=%s%s", self.layer_name,
                    resolved, self.layout,
                    ("NVLinkSymmetricEP" if self.use_nvlink_ep else self.moe.moe_parallel_config.all2all_backend),
                    "fp8" if self.fp8_dispatch else "bf16",
                    f" nvfp4_quant_mode={quant_mode}" if quant_mode else "")

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if layer.apply_router_weight_on_input:
            raise ValueError(f"{self.layer_name}: TensorBridge MoE applies router weights to expert outputs")
        if self.quant_config is None:
            prepared = serving.prepare_moe(
                layer.w13_weight_packed, layer.w13_weight_scale, layer.w13_weight_global_scale,
                layer.w2_weight_packed, layer.w2_weight_scale, layer.w2_weight_global_scale,
                self.moe.experts_per_token, self.moe.num_experts, self.moe.ep_size, self.options.bridge,
                layout=self.layout, gate_up_quant=self.gate_up_quant, down_quant=self.down_quant,
                is_gated=self.moe.is_act_and_mul, fuse_swiglu_fp8=self.use_fused_swiglu, layer_name=self.layer_name,
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
                down_quant=self.down_quant, is_gated=self.moe.is_act_and_mul,
                fuse_swiglu_fp8=self.use_fused_swiglu, layer_name=self.layer_name,
            )
            del layer.w13_weight_scale_2, layer.w2_weight_scale_2
            del layer.w13_input_scale, layer.w2_input_scale
        self._finalize_prepared_layer(layer, prepared)

    def apply(
        self, layer: RoutedExperts, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None, shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        if self._nvlink_ep is not None:
            assert self._prepared_moe is not None

            global_num_tokens: int | None = None
            if self.moe.moe_parallel_config.dp_size > 1:
                dp_meta = get_forward_context().dp_metadata
                if dp_meta is None:
                    raise RuntimeError(f"{self.layer_name}: NVLinkEP requires DP metadata for synchronized chunking")
                if dp_meta.local_sizes is not None:
                    global_num_tokens = max(dp_meta.local_sizes)
                elif dp_meta.num_tokens_across_dp_cpu is not None:
                    global_num_tokens = int(dp_meta.num_tokens_across_dp_cpu.max().item())
                else:
                    raise RuntimeError(f"{self.layer_name}: DP metadata contains no rank token counts")

            shared_hook = None
            use_internal_shared = self.mk_can_overlap_shared_experts
            if use_internal_shared and shared_experts is not None and shared_experts_input is not None:
                def _run_shared() -> None:
                    shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
                shared_hook = _run_shared

            prefill_permute = None
            if layer.expert_map is not None:
                def _prepare_scatter(all_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    scratch = get_moe_permute_scratch(
                        num_experts=layer.global_num_experts,
                        num_local_experts=self._prepared_moe.w13.shape[0],  # type: ignore[union-attr]
                        device=all_ids.device,
                    )
                    return moe_prepare_scatter(all_ids, layer.expert_map, scratch)

                prefill_permute = (_prepare_scatter, moe_unpermute)

            reserve_sms = (
                self.options.moe_shared_reserve_sms
                if (shared_experts is not None and not use_internal_shared)
                else 0
            )
            out = self._nvlink_ep.forward(
                x, topk_ids, topk_weights, self._prepared_moe.w13, self._prepared_moe.w2,
                shared_experts_hook=shared_hook, prefill_permute_fns=prefill_permute,
                global_num_tokens=global_num_tokens, reserve_sms=reserve_sms,
            )
            if use_internal_shared and shared_experts is not None:
                idx = shared_experts._output_idx
                if shared_experts._output[idx] is not None:
                    shared_experts._output[idx].record_stream(torch.cuda.current_stream(x.device))
            return out

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


def keep_converted_weights_on_device(layer: RoutedExperts) -> None:
    # vLLM's device_loading_context restores CPU parameters by *name* after conversion.
    # These packed tensors are inference buffers. The staging parameters have been consumed.
    for name in ("w13_weight", "w2_weight"):
        packed = layer.get_parameter(name)
        del layer._parameters[name]
        layer.register_buffer(name, packed)


class TensorBridgeFp8ToNvfp4MoEMethod(TensorBridgeMoEMethod):
    """Loads native block-quantized FP8 MoE weights (e.g. GLM-5.3 FP8) and converts them on-the-fly
    on GPU to ModelOpt-standard NVFP4 during `process_weights_after_loading` (zero extra disk)."""

    def __init__(
        self, moe: FusedMoEConfig, layer_name: str, options: "Options", fp8_config: Fp8Config,
    ) -> None:
        super().__init__(moe, layer_name, options, quant_config=None)
        self.fp8_quant_config = fp8_config
        self.weight_block_size = fp8_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = "weight_scale_inv" if self.block_quant else "weight_scale"
        self.moe_block_shape = self.weight_block_size
        self.weight_scale_refine: tuple[int, int] | None = None

    def create_weights(
        self, layer: RoutedExperts, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
        params_dtype: torch.dtype, **extra_weight_attrs: object,
    ) -> None:
        # Allocate temporary FP8 staging parameters on CPU so large models (e.g. 704GB GLM-5.3 FP8 on
        # 8xH20 DP8+EP8) never hit peak GPU OOM before per-layer on-the-fly NVFP4 conversion runs.
        saved_qc = self.quant_config
        self.quant_config = self.fp8_quant_config  # type: ignore[assignment]
        try:
            with torch.device("cpu"):
                Fp8MoEMethod.create_weights(
                    self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                    params_dtype, **extra_weight_attrs,
                )
        finally:
            self.quant_config = saved_qc
        if get_current_vllm_config().load_config.load_format == "dummy":
            # Avoid slow single-threaded CPU FP16 uniform_ over 84GB of dummy FP8 staging buffers.
            layer.w13_weight.dummy_weight_value = 0.01
            layer.w2_weight.dummy_weight_value = 0.01
        from tensorbridge.vllm.plugin import register_jit_warmup

        register_jit_warmup()

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        global _FP8_CONVERTED_LAYERS
        if layer.apply_router_weight_on_input:
            raise ValueError(f"{self.layer_name}: TensorBridge MoE applies router weights to expert outputs")
        block_size = tuple(self.weight_block_size) if self.weight_block_size is not None else (128, 128)
        scale_attr_13 = f"w13_{self.weight_scale_name}"
        scale_attr_2 = f"w2_{self.weight_scale_name}"
        dev = torch.device("cuda", torch.cuda.current_device())
        w13_fp8 = layer.w13_weight.data.to(dev, non_blocking=False)
        s13_fp8 = getattr(layer, scale_attr_13).data.to(dev, non_blocking=False)
        w13_packed, w13_scale, w13_gs = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(
            w13_fp8, s13_fp8, block_size=block_size, is_gated_w13=self.moe.is_act_and_mul, chunk_experts=4,
            mode=self.options.nvfp4_quant_mode, bridge=self.options.bridge,
        )
        del w13_fp8, s13_fp8

        w2_fp8 = layer.w2_weight.data.to(dev, non_blocking=False)
        s2_fp8 = getattr(layer, scale_attr_2).data.to(dev, non_blocking=False)
        w2_packed, w2_scale, w2_gs = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(
            w2_fp8, s2_fp8, block_size=block_size, is_gated_w13=False, chunk_experts=4,
            mode=self.options.nvfp4_quant_mode, bridge=self.options.bridge,
        )
        del w2_fp8, s2_fp8

        delattr(layer, scale_attr_13)
        delattr(layer, scale_attr_2)
        if getattr(layer, "w13_input_scale", None) is not None:
            delattr(layer, "w13_input_scale")
        if getattr(layer, "w2_input_scale", None) is not None:
            delattr(layer, "w2_input_scale")

        prepared = serving.prepare_moe(
            w13_packed, w13_scale, w13_gs,
            w2_packed, w2_scale, w2_gs,
            self.moe.experts_per_token, self.moe.num_experts, self.moe.ep_size, self.options.bridge,
            layout=self.layout, scale_convention="multiplier", gate_up_quant=self.gate_up_quant,
            down_quant=self.down_quant, is_gated=self.moe.is_act_and_mul,
            fuse_swiglu_fp8=self.use_fused_swiglu, layer_name=self.layer_name,
        )
        del w13_packed, w13_scale, w13_gs, w2_packed, w2_scale, w2_gs
        self._finalize_prepared_layer(layer, prepared, self.options.nvfp4_quant_mode)
        keep_converted_weights_on_device(layer)
        _FP8_CONVERTED_LAYERS += 1
        if _FP8_CONVERTED_LAYERS % 8 == 0:
            torch.cuda.empty_cache()


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
            inter_total = sizes["intermediate"] + sizes["intermediate_scale"]
            inter_aligned = ((inter_total + 127) // 128) * 128
            if prepared.fuse_swiglu_fp8:
                first_bytes = max(
                    gate_up_and_act,
                    (inter_aligned if prepared.layout == "masked"
                     else max(inter_aligned, output_bytes) + sizes["output"]),
                )
            else:
                first_bytes = max(gate_up_and_act, output_bytes if prepared.layout == "masked"
                                  else output_bytes + sizes["output"])
            second_bytes = max(sizes["input"] + sizes["input_scale"], inter_total)
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
            contiguous_rows_hint = 0
            fuse_swiglu_in_ws13 = False
            if prepared.layout == "contiguous":
                contiguous_rows_hint = (
                    serving.moe_post_dispatch_rows(prepared, topk_ids.size(0))
                    if (moe_config.use_deepep_ht_kernels or moe_config.use_fi_nvl_one_sided_kernels)
                    else serving.moe_rows(prepared, topk_ids.numel())
                )
                if prepared.fuse_swiglu_fp8 and not use_vllm_act:
                    e13, n13, k13 = prepared.w13.shape
                    fuse_swiglu_in_ws13 = (
                        prepared.configs[0].block_n == 256 if prepared.configs is not None
                        else megamoe.should_fuse_swiglu_contiguous(
                            contiguous_rows_hint, n13, k13, e13,
                            device=hidden.get_device(), bridge=prepared.w13.bridge,
                        )
                    )
            elif prepared.layout == "masked" and prepared.fuse_swiglu_fp8 and not use_vllm_act:
                fuse_swiglu_in_ws13 = True
            specs = serving.moe_buffer_specs(prepared, token_capacity, unfused_activation=use_vllm_act)
            buffers: dict[str, torch.Tensor] = {}
            second = workspace2.view(torch.uint8).flatten()
            first_bytes_view = workspace13.view(torch.uint8).flatten()
            for tensor_name, scale_name in (("input", "input_scale"), ("intermediate", "intermediate_scale")):
                tensor_shape, tensor_dtype = specs[tensor_name]
                scale_shape, scale_dtype = specs[scale_name]
                tensor_bytes = math.prod(tensor_shape) * tensor_dtype.itemsize
                scale_bytes = math.prod(scale_shape) * scale_dtype.itemsize
                # Fused SwiGLU+FP8 reads `input` and writes `intermediate` in one kernel; place
                # `intermediate` in unused `workspace13` (`gate_up` is never materialized) so it never
                # aliases `workspace2`. Unfused multi-token Prefill (`BN=128`) materializes `gate_up`
                # in `workspace13`, so keep `intermediate` in `workspace2` (disjoint from `gate_up` & `output`).
                backing = first_bytes_view if (fuse_swiglu_in_ws13 and tensor_name == "intermediate") else second
                buffers[tensor_name] = backing[:tensor_bytes].view(tensor_dtype).view(tensor_shape)
                buffers[scale_name] = (backing[tensor_bytes:tensor_bytes + scale_bytes]
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
            if prepared.layout == "masked":
                buffers["output"] = output.view(route_shape)
            elif fuse_swiglu_in_ws13:
                inter_bytes = (
                    math.prod(specs["intermediate"][0]) * specs["intermediate"][1].itemsize
                    + math.prod(specs["intermediate_scale"][0]) * specs["intermediate_scale"][1].itemsize
                )
                out_off_elems = max(output.numel(), ((inter_bytes + 127) // 128) * 64)
                buffers["output"] = first[out_off_elems:out_off_elems + math.prod(route_shape)].view(route_shape)
            else:
                buffers["output"] = first[output.numel():output.numel() + math.prod(route_shape)].view(route_shape)
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
                global_num_routes = hidden.size(0) * prepared.top_k
                local_rows_hint = serving.moe_rows(prepared, global_num_routes)
                bm13, bm2 = serving.moe_block_sizes(prepared, local_rows_hint)
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
                    local_rows_hint=local_rows_hint, activation_fn=act_fn, activated_buffer=activated_buf,
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
                local_rows_hint = contiguous_rows_hint
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
