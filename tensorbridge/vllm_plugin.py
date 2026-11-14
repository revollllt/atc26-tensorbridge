"""TensorBridge's vLLM 0.32 general plugin.

The checkpoint keeps its own quantization method (compressed-tensors, ModelOpt NVFP4 or a ModelOpt
mixed recipe), which vLLM detects as usual. `additional_config.tensorbridge` chooses the layers
TensorBridge serves, as vLLM's `--linear-backend` / `--moe-backend` choose kernels: `linear=true`
for NVFP4 linear layers, `moe=true` for NVFP4 routed experts, `linear_quant` (`per_token` or
`group_128`) for dense activation quantization, and `exact` for the exact (default) or FPMA-aware
weight bridge. Unsupported layers fall back cleanly to vLLM's native method with a single warning.
`vllm_moe` owns MoE expert assembly; vLLM keeps routing, communication, reduction and shared-expert
scheduling. See README.md for options.
"""

import dataclasses
import warnings
from collections.abc import Callable
from typing import Literal, cast

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptLinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry, VllmJitKernel
from vllm.utils.torch_utils import direct_register_custom_op

from tensorbridge import serving
from tensorbridge.jit.codegen import Bridge
from tensorbridge.quant import fp8, nvfp4
from tensorbridge.vllm_moe import (
    TensorBridgeCompressedMoEMethod,
    TensorBridgeModelOptMoEMethod,
    unsupported_moe_reason,
    warn_moe_fallback,
)

warned_linear_fallbacks: set[str] = set()


@dataclasses.dataclass(frozen=True)
class Options:
    """`additional_config["tensorbridge"]`."""

    linear: bool = False
    moe: bool = False
    bridge: Bridge = "exact"
    linear_quant: fp8.InputQuant = "per_token"
    dispatch_dtype: Literal["fp8", "bf16"] = "fp8"
    gate_up_quant: fp8.InputQuant | None = None
    down_quant: fp8.InputQuant = "per_token"


def read_options() -> Options:
    # vLLM 0.32.0 builds the quantization config in `VllmConfig.__post_init__` from the checkpoint's
    # `quantization_config` alone, and the layers under `set_current_vllm_config`, so the options are read as each
    # layer is built. `additional_config` is a dict or a `SupportsHash` object; the options are read from a dict
    # only, as vLLM's own layers read theirs.
    additional = get_current_vllm_config().additional_config
    options = additional.get("tensorbridge", {}) if isinstance(additional, dict) else {}
    allowed = {
        "linear", "moe", "exact", "linear_quant",
        "moe_dispatch_dtype", "moe_gate_up_quant", "moe_down_quant",
    }
    if not isinstance(options, dict) or not set(options) <= allowed:
        raise ValueError(f'additional_config["tensorbridge"] accepts {sorted(allowed)}; got {options!r}')
    for name in ("linear", "moe", "exact"):
        if name in options and not isinstance(options[name], bool):
            raise ValueError(f"tensorbridge.{name} must be a bool")
    linear_quant = options.get("linear_quant", "per_token")
    dispatch = options.get("moe_dispatch_dtype", "fp8")
    gate_up = options.get("moe_gate_up_quant")
    down = options.get("moe_down_quant", "per_token")
    if (linear_quant not in ("per_token", "group_128") or dispatch not in ("fp8", "bf16")
            or gate_up not in (None, "per_token", "group_128") or down not in ("per_token", "group_128")):
        raise ValueError("TensorBridge takes fp8/bf16 dispatch and per_token/group_128 quantization")
    return Options(
        options.get("linear", False), options.get("moe", False), "exact" if options.get("exact", True) else "fpma",
        cast(fp8.InputQuant, linear_quant), cast(Literal["fp8", "bf16"], dispatch),
        cast(fp8.InputQuant | None, gate_up), cast(fp8.InputQuant, down),
    )


class WarmupBatch(VllmJitKernel[VllmConfig]):
    """Single post-load precompile + autotune hook registered with `JitWarmupRegistry`."""

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[VllmConfig]:
        return [vllm_config]

    def compile(self, compile_key: VllmConfig) -> None:
        serving.precompile_prepared(compile_key.scheduler_config.max_num_batched_tokens)
        compilation = compile_key.compilation_config
        if compilation.cudagraph_mode != CUDAGraphMode.NONE:
            serving.autotune_prepared(compilation.cudagraph_capture_sizes or [])


jit_warmup_hook = WarmupBatch()


def register_jit_warmup() -> None:
    """Register with `GPUModelRunner.jit_warmup_registry` when active during `load_model`."""
    jit_warmup_hook.register_warmup()


def unsupported_linear_reason(layer: torch.nn.Module) -> str | None:
    """Why `layer` cannot run on TensorBridge's SM90 NVFP4 dense kernels, or `None` if it can."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return "TensorBridge requires Hopper (SM90)"
    if isinstance(layer, LinearBase):
        if layer.params_dtype != torch.bfloat16:
            return f"params_dtype={layer.params_dtype} (requires torch.bfloat16)"
        input_k = (layer.input_size // layer.tp_size if type(layer).__name__ == "RowParallelLinear"
                   else layer.input_size)
        output_n = sum(layer.output_partition_sizes)
        if output_n <= 0 or input_k <= 0 or output_n % 8 != 0 or input_k % nvfp4.GROUP_SIZE != 0:
            return f"local linear shape (N={output_n}, K={input_k}) requires N % 8 == 0 and K % 16 == 0"
    model_cfg = get_current_vllm_config().model_config
    if model_cfg is not None and model_cfg.dtype not in (torch.bfloat16, "bfloat16", "auto"):
        return f"model dtype={model_cfg.dtype} (requires bfloat16)"
    return None


def warn_linear_fallback(prefix: str | None, reason: str) -> None:
    if reason not in warned_linear_fallbacks:
        warned_linear_fallbacks.add(reason)
        label = prefix or "<linear>"
        warnings.warn(
            f"TensorBridge linear skipped {label} ({reason}); falling back to vLLM's native method",
            stacklevel=3,
        )


def tensorbridge_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    locks: torch.Tensor,
    output: torch.Tensor,
    exact_bridge: bool,
    group_128: bool = False,
) -> None:
    # Custom op schemas take bools, not Literals.
    bridge: Bridge = "exact" if exact_bridge else "fpma"
    input_quant: fp8.InputQuant = "group_128" if group_128 else "per_token"
    # The weight's shape is the activations' K and the output's N; its tensors are padded to multiples of 128.
    weight = nvfp4.NVFP4Weight(packed_weight, weight_scale, global_scale, locks, (output.size(1), x.size(1)), bridge)
    serving.linear(x, weight, output, input_quant=input_quant)


def tensorbridge_linear_fake(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    locks: torch.Tensor,
    output: torch.Tensor,
    exact_bridge: bool,
    group_128: bool = False,
) -> None:
    return None


def prepare_linear_layer(
    layer: torch.nn.Module, packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor,
    scale_convention: Literal["inverse", "multiplier"], bridge: Bridge,
    input_quant: fp8.InputQuant = "per_token",
) -> None:
    """Replace a loaded NVFP4 linear layer's checkpoint tensors, which the caller deletes, with the kernel
    layout."""
    prepared = serving.prepare_linear(
        packed, scale, global_scale, bridge, scale_convention, layer.logical_widths, input_quant=input_quant,
    )
    layer.register_buffer("tensorbridge_weight", prepared.weight)
    layer.register_buffer("tensorbridge_scale", prepared.scale)
    layer.register_buffer("tensorbridge_global_scale", prepared.global_scale)
    layer.register_buffer("tensorbridge_locks", prepared.locks)
    # Inside GPUModelRunner.load_model (`JitWarmupRegistry._active` is set), defer precompile and
    # autotune to the single post-load `kernel_warmup` pass; in standalone tests run immediately.
    vllm_config = get_current_vllm_config()
    compilation = vllm_config.compilation_config
    if JitWarmupRegistry._active.get() is None or not vllm_config.kernel_config.enable_jit_warmup:
        if compilation.cudagraph_mode != CUDAGraphMode.NONE:
            serving.autotune_prepared(compilation.cudagraph_capture_sizes or [])


def apply_linear_layer(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None, bridge: Bridge,
    input_quant: fp8.InputQuant = "per_token",
) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise ValueError(f"TensorBridge linear expects bfloat16 activations, got {x.dtype}")
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
        bridge == "exact",
        input_quant == "group_128",
    )
    output = output.reshape(*x.shape[:-1], layer.output_size_per_partition)
    if bias is not None:
        return output + bias
    else:
        return output


class TensorBridgeScheme(CompressedTensorsScheme):
    """A compressed-tensors NVFP4 linear layer (group 16, one global scale per shard), run with dynamic
    per-token or group-128 FP8 activations and the `bridge` table, whatever activations the checkpoint declares.

    A standard NVFP4 checkpoint, exported for Blackwell's FP4 activations, also stores `input_global_scale`;
    `checkpoint_has_input_scale` declares it so that vLLM's loader finds a parameter for it, and drops it once
    loaded: TensorBridge never uses it."""

    def __init__(
        self, bridge: Bridge, checkpoint_has_input_scale: bool = False,
        input_quant: fp8.InputQuant = "per_token",
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.checkpoint_has_input_scale = checkpoint_has_input_scale
        self.input_quant = input_quant

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs: object,
    ) -> None:
        # The checkpoint's names and layouts, as vLLM's own compressed-tensors NVFP4 scheme declares them.
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.register_parameter("weight_packed", ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition // 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        ))
        layer.register_parameter("weight_global_scale", PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        ))
        layer.register_parameter("weight_scale", GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // nvfp4.GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        ))
        if self.checkpoint_has_input_scale:
            layer.register_parameter("input_global_scale", PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            ))
        register_jit_warmup()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_linear_layer(
            layer, layer.weight_packed.data, layer.weight_scale.data, layer.weight_global_scale.data, "inverse",
            self.bridge, self.input_quant,
        )
        del layer.weight_packed, layer.weight_scale, layer.weight_global_scale
        if self.checkpoint_has_input_scale:
            del layer.input_global_scale

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return apply_linear_layer(layer, x, bias, self.bridge, self.input_quant)


@register_weight_loader_v2_supported_method
class TensorBridgeModelOptLinearMethod(LinearMethodBase):
    """A ModelOpt NVFP4 or W4A16 NVFP4 linear layer (group 16, one global scale per shard), run with dynamic
    per-token or group-128 FP8 activations and the `bridge` table, whatever activations the checkpoint declares.

    ModelOpt checkpoints store an `input_scale` for Blackwell's FP4 activations (older W4A16 exports too); it is
    declared so that vLLM's loader finds a parameter for it, and dropped once loaded: TensorBridge never uses it."""

    def __init__(self, bridge: Bridge, input_quant: fp8.InputQuant = "per_token") -> None:
        self.bridge = bridge
        self.input_quant = input_quant

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: object,
    ) -> None:
        # The checkpoint's names and layouts, as vLLM's own ModelOpt NVFP4 schemes declare them.
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.register_parameter("weight", ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition // 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        ))
        layer.register_parameter("weight_scale_2", PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        ))
        layer.register_parameter("weight_scale", ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // nvfp4.GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        ))
        layer.register_parameter("input_scale", PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        ))
        register_jit_warmup()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_linear_layer(
            layer, layer.weight.data, layer.weight_scale.data, layer.weight_scale_2.data, "multiplier",
            self.bridge, self.input_quant,
        )
        del layer.weight, layer.weight_scale, layer.weight_scale_2, layer.input_scale

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return apply_linear_layer(layer, x, bias, self.bridge, self.input_quant)


def modelopt_linear_method(
    native: QuantizeMethodBase | None, layer: torch.nn.Module, prefix: str, options: Options
) -> QuantizeMethodBase | None:
    """TensorBridge's method for a linear layer that vLLM's ModelOpt configs give an NVFP4 method (they resolve
    excluded layers and mixed recipes), else the native method."""
    if (not options.linear or not isinstance(layer, LinearBase) or not isinstance(native, ModelOptLinearMethod)
            or native.spec.weight != kNvfp4Static):
        return native
    if native.ctx.group_size != nvfp4.GROUP_SIZE:
        warn_linear_fallback(prefix, f"group_size={native.ctx.group_size} (requires 16)")
        return native
    if (reason := unsupported_linear_reason(layer)) is not None:
        warn_linear_fallback(prefix, reason)
        return native
    return TensorBridgeModelOptLinearMethod(options.bridge, options.linear_quant)


class TensorBridgeCompressedTensorsConfig(CompressedTensorsConfig):
    """compressed-tensors, with TensorBridge serving the NVFP4 layers `additional_config.tensorbridge` enables."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        options = read_options()
        if not isinstance(layer, RoutedExperts) or not options.moe:
            return super().get_quant_method(layer, prefix)
        self._add_fused_moe_to_target_scheme_map()
        projections = [
            projection for projection in (layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name, layer.ckpt_down_proj_name)
            if projection is not None
        ]
        schemes = [self.get_scheme_dict(layer, f"{prefix}.0.{projection}") for projection in projections]
        if not all(scheme == schemes[0] for scheme in schemes):
            warn_moe_fallback(prefix, "MoE projections do not share a quantization scheme")
            return super().get_quant_method(layer, prefix)
        scheme = schemes[0]
        weights = scheme.get("weights") if scheme is not None else None
        if isinstance(weights, QuantizationArgs) and self._is_nvfp4_format(weights):
            if (reason := unsupported_moe_reason(layer)) is not None:
                warn_moe_fallback(prefix, reason)
                return super().get_quant_method(layer, prefix)
            return TensorBridgeCompressedMoEMethod(layer.moe_config, prefix, options)
        else:
            return super().get_quant_method(layer, prefix)

    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> CompressedTensorsScheme | None:
        options = read_options()
        scheme_dict = self.get_scheme_dict(layer, layer_name) if options.linear else None
        weights = scheme_dict.get("weights") if scheme_dict is not None else None
        # `_is_nvfp4_format` and `_check_scheme_supported` are vLLM's (`CompressedTensorsConfig`, vLLM 0.32.0).
        if scheme_dict is not None and isinstance(weights, QuantizationArgs) and self._is_nvfp4_format(weights):
            if (reason := unsupported_linear_reason(layer)) is not None:
                warn_linear_fallback(layer_name, reason)
                return super().get_scheme(layer, layer_name)
            self._check_scheme_supported(TensorBridgeScheme.get_min_capability())
            # Standard NVFP4 checkpoints declare FP4 activations and store their global scale.
            inputs = scheme_dict.get("input_activations")
            return TensorBridgeScheme(
                options.bridge, isinstance(inputs, QuantizationArgs) and self._is_nvfp4_format(inputs),
                options.linear_quant,
            )
        else:
            return super().get_scheme(layer, layer_name)


class TensorBridgeModelOptConfig(ModelOptNvFp4Config):
    """ModelOpt NVFP4, with TensorBridge serving the layers `additional_config.tensorbridge` enables."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        options = read_options()
        if isinstance(layer, RoutedExperts) and options.moe and not self.is_layer_excluded(prefix):
            if (reason := unsupported_moe_reason(layer, self)) is not None:
                warn_moe_fallback(prefix, reason)
                return super().get_quant_method(layer, prefix)
            return TensorBridgeModelOptMoEMethod(layer.moe_config, prefix, options, self)
        else:
            return modelopt_linear_method(super().get_quant_method(layer, prefix), layer, prefix, options)


class TensorBridgeModelOptMixedConfig(ModelOptMixedPrecisionConfig):
    """A ModelOpt mixed recipe, with TensorBridge serving the NVFP4 layers `additional_config.tensorbridge`
    enables."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        options = read_options()
        if not isinstance(layer, RoutedExperts) or not options.moe or self.is_layer_excluded(prefix):
            return modelopt_linear_method(super().get_quant_method(layer, prefix), layer, prefix, options)
        algorithm = self._resolve_quant_algo(prefix)
        if algorithm in ("NVFP4", "W4A16_NVFP4"):
            config = self.nvfp4_config if algorithm == "NVFP4" else self.w4a16_nvfp4_config
            if (reason := unsupported_moe_reason(layer, config)) is not None:
                warn_moe_fallback(prefix, reason)
                return super().get_quant_method(layer, prefix)
            return TensorBridgeModelOptMoEMethod(layer.moe_config, prefix, options, config)
        else:
            return super().get_quant_method(layer, prefix)


def register() -> None:
    direct_register_custom_op(
        op_name="tensorbridge_linear",
        op_func=tensorbridge_linear,
        mutates_args=["locks", "output"],
        fake_impl=tensorbridge_linear_fake,
    )
    register_quantization_config("compressed-tensors")(TensorBridgeCompressedTensorsConfig)
    register_quantization_config("modelopt_fp4")(TensorBridgeModelOptConfig)
    register_quantization_config("modelopt_mixed")(TensorBridgeModelOptMixedConfig)
