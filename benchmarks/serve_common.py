"""Shared vLLM serving helpers for GLM-5.3 scale-up and PD-disaggregated benchmarks."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass

ADAPTERS = ("marlin_nvfp4", "humming_nvfp4", "humming_mxfp4", "humming_mxfp4_g128")
PRESETS = ("tb_nvfp4_fp8", "tb_nvfp4_fp8_g128", "tb_nvfp4_megamoe")
ALL_MODES = ("native_fp8", "fp8_deepgemm", *ADAPTERS, *PRESETS)

_METRIC_LINE_RE = re.compile(
    r'^(vllm:prefix_cache_(?:queries|hits))(?:_total)?(?:\{([^}]*)\})?\s+([0-9.eE+-]+)\s*$',
    re.MULTILINE,
)
_ENGINE_LABEL_RE = re.compile(r'\bengine="([^"]+)"')


@dataclass
class Completion:
    start: float
    arrivals: list[float]
    end: float
    token_ids: list[int]
    text: str
    tokens_per_frame: list[int]
    timeout_retries: int = 0
    prompt_tokens: int | None = None

    @property
    def coalesced_tokens(self) -> int:
        return max(0, len(self.token_ids) - len(self.arrivals))


def wait_for_health(port: int, proc: subprocess.Popen[bytes], timeout_s: int = 900) -> float:
    t0 = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited prematurely with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"vllm serve failed to become healthy within {timeout_s}s")


def stop_server(proc: subprocess.Popen[bytes], timeout_s: float = 30.0) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    else:
        proc.wait()


def read_stream(lines: Iterable[bytes], start: float, expected: int | None) -> Completion:
    arrivals, ids, pieces, tokens_per_frame = [], [], [], []
    usage = None
    for raw in lines:
        if not raw.startswith(b"data:"):
            continue
        event_time = time.perf_counter()
        body = raw[5:].strip()
        if body == b"[DONE]":
            break
        event = json.loads(body)
        if "error" in event:
            raise RuntimeError(event["error"])
        usage = event.get("usage") or usage
        before = len(ids)
        for choice in event["choices"]:
            pieces.append(choice["text"])
            ids.extend(choice.get("token_ids") or [])
        if len(ids) == before:
            # Includes heartbeats and detokenizer text flushes without a new token.
            continue
        arrivals.append(event_time)
        tokens_per_frame.append(len(ids) - before)
    else:
        raise ValueError("Incomplete SSE stream: missing [DONE]")
    tokens = usage.get("completion_tokens") if usage is not None else None
    if not ids or tokens != len(ids) or (expected is not None and len(ids) != expected):
        raise ValueError(f"Completion token mismatch: usage={usage}, streamed={len(ids)}, expected={expected}")
    return Completion(
        start,
        arrivals,
        time.perf_counter(),
        ids,
        "".join(pieces),
        tokens_per_frame,
        prompt_tokens=usage.get("prompt_tokens"),
    )


def stream_completion(
    port: int,
    prompt: list[int] | str,
    length: int,
    ignore_eos: bool = True,
    timeout_s: float = 600.0,
) -> Completion:
    payload = dict(
        model="glm53",
        prompt=prompt,
        max_tokens=length,
        ignore_eos=ignore_eos,
        temperature=0,
        stream=True,
        return_token_ids=True,
        stream_options={"include_usage": True},
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    timeout_retries = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                completion = read_stream(response, start, length if ignore_eos else None)
            completion.timeout_retries = timeout_retries
            return completion
        except (TimeoutError, urllib.error.URLError) as error:
            cause = error.reason if isinstance(error, urllib.error.URLError) else error
            if timeout_retries or not isinstance(cause, TimeoutError):
                raise
            timeout_retries += 1
            print(f"Retrying timed-out completion once: {error}", file=sys.stderr, flush=True)


def distribution(samples: list[float], percentiles: tuple[int, ...]) -> dict[str, float | None]:
    ordered = sorted(samples)
    metrics: dict[str, float | None] = {"mean": statistics.fmean(ordered) if ordered else None}
    for pct in percentiles:
        position = (len(ordered) - 1) * pct / 100
        lower = math.floor(position)
        metrics[f"p{pct}"] = (
            ordered[lower] + (position - lower) * (ordered[math.ceil(position)] - ordered[lower]) if ordered else None
        )
    return metrics


def kv_cache_calibration(startup: str, ready_port: int | None) -> dict[str, int | float | None]:
    # DP engines may report different capacities. Use the smallest observed values for calibration.
    blocks = re.findall(r"(?:after num_gpu_blocks is:|# GPU blocks:|KV cache blocks:)\s*([\d,]*\d)", startup)
    memory = re.findall(r"Available KV cache memory:\s*(-?[\d.]+)\s*GiB", startup)
    tokens = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", startup)
    if not blocks and ready_port is not None:
        # vLLM 0.32's per-engine block-count log is DEBUG; its metrics gauge exposes the resolved count at INFO too.
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ready_port}/metrics", timeout=5) as response:
                blocks = re.findall(
                    r'^vllm:cache_config_info\{[^}]*?\bnum_gpu_blocks="(\d+)"', response.read().decode(), re.MULTILINE
                )
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"KV block count unavailable from /metrics: {error}", file=sys.stderr, flush=True)
    return {
        "kv_cache_blocks": min((int(b.replace(",", "")) for b in blocks), default=None),
        "available_kv_cache_memory_gib": min(map(float, memory), default=None),
        "gpu_kv_cache_size_tokens": min((int(t.replace(",", "")) for t in tokens), default=None),
    }


def parse_prefix_cache_metrics(metrics_text: str) -> dict[str, dict[str, float]]:
    """Extract per-engine prefix cache query/hit counters from Prometheus `/metrics` text."""
    queries: dict[str, float] = {}
    hits: dict[str, float] = {}
    for match in _METRIC_LINE_RE.finditer(metrics_text):
        name, labels, raw_value = match.groups()
        engine_match = _ENGINE_LABEL_RE.search(labels or "")
        engine = engine_match.group(1) if engine_match else "0"
        target = queries if name == "vllm:prefix_cache_queries" else hits
        target[engine] = target.get(engine, 0.0) + float(raw_value)
    return {"queries": queries, "hits": hits}


def fetch_prefix_cache_metrics(port: int, timeout_s: float = 10.0) -> dict[str, dict[str, float]]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout_s) as response:
        return parse_prefix_cache_metrics(response.read().decode("utf-8", errors="replace"))


def diff_prefix_cache(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, object]:
    engines = sorted(set(before["queries"]) | set(after["queries"]) | set(before["hits"]) | set(after["hits"]))
    q_delta = {e: after["queries"].get(e, 0.0) - before["queries"].get(e, 0.0) for e in engines}
    h_delta = {e: after["hits"].get(e, 0.0) - before["hits"].get(e, 0.0) for e in engines}
    total_q = sum(q_delta.values())
    total_h = sum(h_delta.values())
    return {
        "queries": total_q,
        "hits": total_h,
        "hit_rate": (total_h / total_q) if total_q > 0 else None,
        "per_engine_queries": q_delta,
        "per_engine_hits": h_delta,
    }


def validate_backend_args(
    mode: str,
    *,
    all2all_backend: str,
    dispatch_dtype: str | None,
    tb_rule_only: bool,
    error_fn: Callable[[str], None],
) -> None:
    modular_preset = mode in PRESETS[:2]
    if mode in PRESETS:
        dispatch = "bf16" if mode == "tb_nvfp4_fp8" else "fp8"
        if dispatch_dtype not in (None, dispatch) or (
            modular_preset and all2all_backend != "allgather_reducescatter"
        ):
            error_fn(f"{mode} requires {dispatch} dispatch and allgather_reducescatter")
    if tb_rule_only and (not modular_preset or os.environ.get("TENSORBRIDGE_AUTOTUNE") == "1"):
        error_fn("--tb-rule-only requires a modular TensorBridge preset with autotune disabled")


def append_backend_args(
    cmd: list[str],
    mode: str,
    *,
    strict_moe: bool,
    nvfp4_quant_mode: str | None,
    shared_reserve_sms: int,
    max_tokens_per_rank: int,
) -> list[str]:
    if mode == "native_fp8":
        return cmd
    if mode == "fp8_deepgemm":
        return [*cmd, "--moe-backend", "deep_gemm"]
    options: dict[str, object] = {"strict_moe": strict_moe}
    if nvfp4_quant_mode is not None:
        options["nvfp4_quant_mode"] = nvfp4_quant_mode
    if mode in PRESETS:
        options.update(
            preset=mode,
            moe_shared_reserve_sms=shared_reserve_sms,
            moe_max_tokens_per_rank=max_tokens_per_rank,
        )
    elif mode in ADAPTERS:
        options["fp8_moe_baseline"] = mode
    else:
        raise ValueError(f"Unknown serving mode {mode!r}")
    return [*cmd, "--additional-config", json.dumps({"tensorbridge": options})]
