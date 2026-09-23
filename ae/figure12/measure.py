"""Fixed-token offline throughput with the AE vLLM fork."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="tensorbridge")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("ae/figure12/results"))
    args = parser.parse_args()

    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    vocab_size = int(model_config.vocab_size)
    excluded = {token for token in (model_config.eos_token_id, model_config.pad_token_id)
                if isinstance(token, int)}
    prompts = []
    for prompt_index in range(max(args.batch_sizes)):
        bos_token = model_config.bos_token_id
        token_ids = [bos_token] if isinstance(bos_token, int) else []
        base = 100 + (prompt_index * 997) % max(1, vocab_size - 200)
        while len(token_ids) < args.input_len:
            token = 3 + ((base + len(token_ids) * 37) % (vocab_size - 3))
            token = (token + 1) % vocab_size if token in excluded else token
            token_ids.append(token)
        prompts.append({"prompt_token_ids": token_ids[:args.input_len]})

    engine = LLM(
        model=args.model,
        quantization=None if args.quantization == "auto" else args.quantization,
        dtype="auto",
        gpu_memory_utilization=0.85,
        max_model_len=max(1536, args.input_len + args.output_len),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
        calculate_kv_scales=True,
    )
    sampling = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        engine.generate(prompts[:batch_size], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts[:batch_size], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output_tokens = sum(len(request.outputs[0].token_ids) for request in outputs)
        measurement = {
            "model": args.model,
            "quantization": args.quantization,
            "num_prompts": batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "elapsed": elapsed,
            "output_tokens": output_tokens,
            "output_tok_per_s": output_tokens / elapsed,
        }
        destination = args.output_dir / f"{args.quantization}-{batch_size}.json"
        destination.write_text(json.dumps(measurement, indent=2) + "\n")
        print(f"{destination}: {measurement['output_tok_per_s']:.2f} output token/s", flush=True)


if __name__ == "__main__":
    main()
