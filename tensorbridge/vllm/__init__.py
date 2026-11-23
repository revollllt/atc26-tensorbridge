"""vLLM 0.32 integration for TensorBridge NVFP4 linear layers and routed MoE experts."""

from tensorbridge.vllm.presets import expand_moe_preset

__all__ = ["expand_moe_preset"]
