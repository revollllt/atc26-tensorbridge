"""vLLM general plugin entry point (`tensorbridge.vllm_plugin:register`)."""

from tensorbridge.vllm.plugin import *  # noqa: F401, F403
from tensorbridge.vllm.plugin import register

__all__ = ["register"]
