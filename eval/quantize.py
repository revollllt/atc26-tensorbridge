"""Create the quantized checkpoints of Tables 3-4.

W4A8, NVFP4A8 and NVFP4A8_FPMA (for TensorBridge, see `fpma_rounding`) use GPTQ;
NVFP4 (W4A4) and MXFP4 use round-to-nearest.
Calibration: 128 fixed 2048-token WikiText2 train windows (`calibration.jsonl`).
"""

import argparse
import random
from pathlib import Path

import torch
from compressed_tensors.quantization.quant_scheme import preset_name_to_scheme
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import AutoConfig, AutoTokenizer

SCHEMES = ["W4A8", "NVFP4A8", "NVFP4A8_FPMA", "NVFP4", "MXFP4"]
IGNORE = ["lm_head", "re:.*embed_tokens$"]
# Qwen3.6 MoE: routers and the linear-attention layers stay BF16.
IGNORE_MOE = IGNORE + ["re:.*mlp\\.gate$", "re:.*shared_expert_gate$", "re:.*linear_attn.*",
                       "re:model\\.visual\\..*"]


def recipe(scheme, ignore):
    if scheme == "W4A8":
        return GPTQModifier(targets="Linear", scheme="W4A8", ignore=ignore)
    if scheme in ("NVFP4A8", "NVFP4A8_FPMA"):  # NVFP4 weights, dynamic per-token FP8 activations
        nvfp4a8 = preset_name_to_scheme("NVFP4A16", ["Linear"])
        nvfp4a8.input_activations = preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]).input_activations
        nvfp4a8.weights.actorder = "static"
        return GPTQModifier(config_groups={"group_0": nvfp4a8}, ignore=ignore)
    return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)


def fpma_rounding():
    """NVFP4A8_FPMA: GPTQ rounds each weight to the nearest value TensorBridge's bridge
    produces for its group's scale byte (SNC, alpha 1) instead of the nearest FP4 value,
    so its error feedback also absorbs the bridge's approximation. The checkpoint stays
    plain NVFP4: every bridge value is within 12.5% of its code, so compression stores it."""
    import llmcompressor.modifiers.gptq.gptq_quantize as gptq

    exact = gptq.fake_quantize
    addend = torch.tensor([0, 0, 2, 3, 4, 5, 6, 7], dtype=torch.int32) << 2  # SNC codes << 2

    def bridge_fake_quantize(x, scale, zero_point, args, global_scale=None, **kwargs):
        if not (args.type == "float" and args.num_bits == 4 and global_scale is not None):
            return exact(x, scale, zero_point, args, global_scale=global_scale, **kwargs)
        raw = scale.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32).reshape(x.shape)
        byte = (raw[..., None] - 0x1C).clamp(min=0) + addend.to(x.device)
        values = byte.to(torch.uint8).view(torch.float8_e4m3fn).float() * 6 / global_scale.float()
        values[..., 0] = 0
        values = values * (raw[..., None] >= 0x1C)  # the kernel zeroes groups below its range
        pick = (x.float().abs()[..., None] - values).abs().argmin(-1, keepdim=True)
        return (values.gather(-1, pick)[..., 0] * x.float().sign()).to(x.dtype)

    gptq.fake_quantize = bridge_fake_quantize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="base model")
    p.add_argument("--scheme", choices=SCHEMES, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    moe = "moe" in AutoConfig.from_pretrained(args.model, trust_remote_code=True).model_type
    calibration = Path(__file__).with_name("calibration.jsonl")
    if args.scheme == "NVFP4A8_FPMA":
        fpma_rounding()
    oneshot(
        model=args.model,
        processor=AutoTokenizer.from_pretrained(args.model),  # text only, also for Qwen3.6
        dataset=load_dataset("json", data_files=str(calibration), split="train"),
        text_column="text",
        concatenate_data=True,
        num_calibration_samples=128,
        max_seq_length=2048,
        precision="bfloat16",
        recipe=recipe(args.scheme, IGNORE_MOE if moe else IGNORE),
        output_dir=args.output,
        save_compressed=True,
        trust_remote_code_model=True,
        moe_calibrate_all_experts=moe,
    )


if __name__ == "__main__":
    main()
