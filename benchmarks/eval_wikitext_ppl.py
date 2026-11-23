"""GPTQ/AWQ WikiText-2 raw test PPL for an offline tokenizer and a BF16 vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def read_wikitext(path: Path) -> str:
    """TXT contains the exact joined corpus. JSON preserves the original dataset rows."""
    text = path.read_bytes().decode("utf-8")
    if path.suffix.lower() not in (".json", ".jsonl"):
        return text
    # HF Dataset.to_json uses one record per line, including when the filename ends in .json.
    try:
        records = json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise
        records = [json.loads(line) for line in text.split("\n") if line.strip()]
    rows = records["text"] if isinstance(records, dict) else records
    if isinstance(rows, str) and isinstance(records, dict):
        return rows
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a text column or a list of test rows")
    return "\n\n".join(row["text"] if isinstance(row, dict) else row for row in rows)


def score_windows(
    input_ids: torch.Tensor, seqlen: int, concurrency: int, url: str, model: str
) -> dict[str, int | float]:
    if seqlen < 2 or concurrency < 1:
        raise ValueError("seqlen must be >= 2 and concurrency must be positive")
    total = input_ids.numel()
    nsamples = total // seqlen
    if not nsamples:
        raise ValueError(f"{total} tokens do not fill one {seqlen}-token window")
    windows = input_ids[: nsamples * seqlen].view(nsamples, seqlen)

    def score(window_index: int) -> list[float]:
        prompt = windows[window_index].tolist()
        payload = dict(
            model=model,
            prompt=prompt,
            add_special_tokens=False,
            return_token_ids=True,
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0,
        )
        request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            record = json.load(response)
        choice = record["choices"][0]
        values = choice["logprobs"]["token_logprobs"]
        if choice["prompt_token_ids"] != prompt or record["usage"]["prompt_tokens"] != seqlen:
            raise ValueError(f"WikiText window {window_index}: server changed prompt token IDs or length")
        if (
            len(values) != seqlen
            or values[0] is not None
            or not all(
                # vLLM's OpenAI serializer clamps -inf to -9999 before JSON encoding.
                isinstance(lp, (int, float)) and math.isfinite(lp) and lp > -9999.0
                for lp in values[1:]
            )
        ):
            raise ValueError(f"WikiText window {window_index}: incomplete, nonfinite or clipped prompt logprobs")
        return values[1:]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        logprobs = list(pool.map(score, range(nsamples)))  # preserve dataset order across concurrent requests
    losses = torch.stack([-torch.tensor(window, dtype=torch.float32).mean() * seqlen for window in logprobs])
    scored_tokens = nsamples * (seqlen - 1)
    mean_nll = math.fsum(-lp for window in logprobs for lp in window) / scored_tokens
    return dict(
        total_tokens=total,
        seqlen=seqlen,
        nsamples=nsamples,
        scored_tokens=scored_tokens,
        mean_nll_per_token=mean_nll,
        ppl_fp64=math.exp(mean_nll),
        ppl_gptq_fp32=(losses.sum() / (nsamples * seqlen)).exp().item(),
    )


def evaluate(path: Path, model_path: str, seqlen: int, concurrency: int, url: str, model: str) -> dict[str, object]:
    from transformers import AutoTokenizer

    start = time.perf_counter()
    joined = read_wikitext(path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    input_ids = tokenizer(joined, return_tensors="pt").input_ids[0]
    return {
        "dataset": "wikitext-2-raw-v1/test",
        "dataset_path": str(path),
        "dataset_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        "tokenizer": model_path,
        "served_model": model,
        "concurrency": concurrency,
        "precision_contract": {
            "model_dtype": "bfloat16",
            "lm_head_logits": "bfloat16",
            "log_softmax": "float32",
            "server_requirement": "vLLM 0.32 raw logprobs, BF16 model and head",
            "gptq_reduction": "float32 per-window mean times seqlen, float32 sum and exp",
            "fp64_reduction": "math.fsum over all scored tokens",
        },
        **score_windows(input_ids, seqlen, concurrency, url, model),
        "elapsed_s": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext-path", type=Path, required=True)
    parser.add_argument(
        "--model", default=os.environ.get("TENSORBRIDGE_MODEL_PATH", "GLM-5.3"), help="model or tokenizer path"
    )
    parser.add_argument("--served-model", default="glm53")
    parser.add_argument("--url", default="http://127.0.0.1:8123/v1/completions")
    parser.add_argument("--wikitext-seqlen", type=int, default=2048)
    parser.add_argument("--wikitext-concurrency", type=int, default=8)
    args = parser.parse_args()
    result = evaluate(
        args.wikitext_path, args.model, args.wikitext_seqlen, args.wikitext_concurrency, args.url, args.served_model
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
