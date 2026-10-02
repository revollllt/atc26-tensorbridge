"""WikiText2 perplexity (Table 3) or mean zero-shot accuracy (Table 4) of one method.

`ours` emulates the bridge's weights in Hugging Face; `ours_kernel` runs the
TensorBridge kernel through the vLLM plugin. Both apply `--alpha`.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from tensorbridge.quant.nvfp4 import FPMA_ALPHA

METHODS = ["bf16", "w4a8", "nvfp4a16", "nvfp4", "mxfp4", "nvfp4a8", "ours_nosnc", "ours",
           "ours_kernel"]
# lm-eval task -> reported metric
TASKS = {"arc_challenge": "acc_norm,none", "arc_easy": "acc_norm,none", "winogrande": "acc,none",
         "hellaswag": "acc_norm,none", "piqa": "acc_norm,none"}
BLOCK = 2048


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model", required=True, help="checkpoint, or the base model for bf16")
    p.add_argument("--tokenizer", required=True, help="base model")
    p.add_argument("--metric", choices=["ppl", "tasks"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=lambda v: v if v == "lsq" else float(v), default=FPMA_ALPHA,
                   help="global-scale factor of ours*: a number, or lsq (per layer)")
    p.add_argument("--batch-size", default="auto",
                   help="Hugging Face task batch size; auto probes the largest one up to 64")
    p.add_argument("--limit", type=int, help="short check: PPL blocks, or examples per task")
    return p.parse_args()


def load_hf(args):
    from transformers import AutoConfig, AutoModelForCausalLM, CompressedTensorsConfig

    import emulate
    import experts

    if args.method in ("ours", "ours_nosnc"):
        emulate.fpma(snc=args.method == "ours", alpha=args.alpha)
    elif args.method == "mxfp4":
        emulate.mxfp4()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {}
    if hasattr(config, "quantization_config"):
        # Decompress at load, so the hooks above apply; the compressed default ignores them.
        kwargs["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
    if args.method == "nvfp4a16":  # the NVFP4A8 checkpoint without activation quantization
        for group in config.quantization_config["config_groups"].values():
            if group.get("format") == "nvfp4-pack-quantized":  # FP8 layers keep theirs
                group["input_activations"] = None
    # Loading replaces the config's quantization dict, so keep the scheme first.
    group = dict(config.quantization_config["config_groups"]["group_0"]) if kwargs else None
    model = AutoModelForCausalLM.from_pretrained(  # on the GPU: 10x faster for the MoE
        args.model, config=config, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", **kwargs).eval()
    if group and experts.has_quantized_experts(args.model):  # MoE: see eval/experts.py
        experts.load(model, args.model, args.method, args.alpha, group)
    return model


def vllm_env(args):
    os.environ["TENSORBRIDGE_FPMA_ALPHA"] = str(args.alpha)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # CUDA is already initialized


def load_vllm(args, max_model_len):
    from vllm import LLM

    vllm_env(args)
    return LLM(args.model, tokenizer=args.tokenizer, quantization="tensorbridge",
               dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.85,
               enforce_eager=True)  # no compilation or CUDA graphs: same numerics, fewer moving parts


def perplexity(args):
    """QQQ convention: non-overlapping 2048-token blocks of the raw test split."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False,
                                              trust_remote_code=True)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                    split="test")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    blocks = ids.numel() // BLOCK
    blocks = min(blocks, args.limit or blocks)
    ids = ids[:, : blocks * BLOCK].view(blocks, BLOCK)

    if args.method == "ours_kernel":
        from vllm import SamplingParams

        llm = load_vllm(args, BLOCK + 1)  # the prompt plus one generated token
        params = SamplingParams(max_tokens=1, prompt_logprobs=1, detokenize=False)
        outputs = llm.generate([{"prompt_token_ids": row.tolist()} for row in ids], params)
        nll = [-sum(lp[t].logprob for lp, t in zip(out.prompt_logprobs[1:], row[1:].tolist()))
               / (BLOCK - 1) for out, row in zip(outputs, ids)]
        return {"ppl": math.exp(sum(nll) / blocks), "blocks": blocks}

    model = load_hf(args)
    losses = []
    with torch.no_grad():
        for row in ids:
            row = row.view(1, -1).cuda()
            logits = model(row, use_cache=False).logits[:, :-1]
            losses.append(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), row[:, 1:].reshape(-1)).float().cpu())
    return {"ppl": math.exp(torch.stack(losses).sum().item() / blocks), "blocks": blocks}


def zero_shot(args):
    import lm_eval

    if args.method == "ours_kernel":
        from lm_eval.models.vllm_causallms import VLLM

        vllm_env(args)
        lm = VLLM(pretrained=args.model, tokenizer=args.tokenizer, quantization="tensorbridge",
                  dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
                  enforce_eager=True, batch_size="auto")
    else:
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(pretrained=load_hf(args), tokenizer=args.tokenizer, batch_size=args.batch_size)
    results = lm_eval.simple_evaluate(model=lm, tasks=list(TASKS), num_fewshot=0,
                                      limit=args.limit)["results"]
    scores = {task: 100 * results[task][metric] for task, metric in TASKS.items()}
    return {**scores, "avg": sum(scores.values()) / len(scores)}


def main():
    args = parse_args()
    result = perplexity(args) if args.metric == "ppl" else zero_shot(args)
    result.update(method=args.method, model=args.model, limit=args.limit,
                  alpha=args.alpha if args.method.startswith("ours") else None,
                  torch=torch.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
