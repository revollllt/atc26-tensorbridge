"""Public TensorBridge MoE presets for FP8-checkpoint serving."""

RECIPES = {
    "tb_nvfp4_fp8": ("humming_grouped", "per_token", "bf16", False),
    "tb_nvfp4_fp8_g128": ("fused_swiglu", "group_128", "fp8", False),
    "tb_nvfp4_megamoe": ("megamoe", "group_128", "fp8", True),
}


def expand_moe_preset(options: dict[str, object], all2all_backend: str) -> dict[str, object]:
    name = options["preset"]
    if not isinstance(name, str) or name not in RECIPES:
        raise ValueError(f"Unknown TensorBridge MoE preset {name!r}")
    backend, quant, dispatch, nvlink = RECIPES[name]
    if not nvlink and all2all_backend != "allgather_reducescatter":
        raise ValueError(f"{name} requires allgather_reducescatter")
    recipe = dict(moe=True, exact=True, fp8_to_nvfp4=True, moe_backend_mode=backend, moe_gate_up_quant=quant,
                  moe_down_quant=quant, moe_dispatch_dtype=dispatch, moe_nvlink_ep=nvlink)
    for key, value in recipe.items():
        if key in options and options[key] != value:
            raise ValueError(f"{name} requires {key}={value!r}")
    return {**recipe, **options}
