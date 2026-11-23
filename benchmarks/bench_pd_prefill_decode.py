"""PD-disaggregated Prefill (TTFT vs ISL & cache hit) and Decode (TPOT vs concurrency) benchmark for GLM-5.3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from serve_common import (  # noqa: E402
    ADAPTERS,
    ALL_MODES,
    PRESETS,
    Completion,
    append_backend_args,
    diff_prefix_cache,
    distribution,
    fetch_prefix_cache_metrics,
    kv_cache_calibration,
    parse_prefix_cache_metrics,
    stop_server,
    stream_completion,
    validate_backend_args,
    wait_for_health,
)

DEFAULT_PREFILL_ISLS = (50_000, 80_000, 160_000, 320_000)
DEFAULT_CACHE_HIT_RATES = (0.0, 0.9)
DEFAULT_DECODE_ISL = 80_000
DEFAULT_DECODE_OSL = 600
DEFAULT_DECODE_CONCURRENCIES = (1, 4, 8, 16, 32)
VOCAB_LOW = 1000
VOCAB_SPAN = 50_000


def align_up(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def make_unique_tokens(seed: int, length: int) -> list[int]:
    """Deterministic valid token IDs whose leading block uniquely identifies `seed`."""
    if length < 0:
        raise ValueError(f"Token length must be non-negative, got {length}")
    if length == 0:
        return []
    header = [
        VOCAB_LOW + (seed % VOCAB_SPAN),
        VOCAB_LOW + ((seed // VOCAB_SPAN + seed * 7 + 1) % VOCAB_SPAN),
        VOCAB_LOW + ((seed * 17 + 3) % VOCAB_SPAN),
        VOCAB_LOW + ((seed * 31 + 11) % VOCAB_SPAN),
    ]
    if length <= len(header):
        return header[:length]
    stride = 37 + (seed % 97) * 2
    offset = (seed * 131 + 19) % VOCAB_SPAN
    body = [VOCAB_LOW + ((i * stride + offset) % VOCAB_SPAN) for i in range(len(header), length)]
    return header + body


def compute_prefix_split(isl: int, hit_rate: float, block_size: int) -> tuple[int, int]:
    """Split `isl` into `(cached_prefix_len, uncached_suffix_len)` aligned to `block_size`."""
    if isl < 1:
        raise ValueError(f"ISL must be positive, got {isl}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if hit_rate < 0.0 or hit_rate >= 1.0:
        raise ValueError(f"cache hit rate must be in [0.0, 1.0), got {hit_rate}")
    if hit_rate == 0.0:
        return 0, isl
    prefix_len = (round(isl * hit_rate) // block_size) * block_size
    if prefix_len <= 0 or prefix_len >= isl:
        raise ValueError(
            f"ISL={isl} with hit_rate={hit_rate} and block_size={block_size} yields invalid prefix_len={prefix_len}"
        )
    return prefix_len, isl - prefix_len


def warm_prefix_cache(
    port: int,
    prefix: list[int],
    *,
    dp: int,
    block_size: int,
    seed_start: int,
    timeout_s: float,
) -> tuple[int, int]:
    """Populate `prefix` on every DP engine before timing cached prefill requests.

    Returns `(waves_used, next_seed)`.
    """
    expected_engines = [str(i) for i in range(dp)]
    before = fetch_prefix_cache_metrics(port)
    seed = seed_start
    max_waves = max(3, dp * 4)
    for wave in range(1, max_waves + 1):
        prompts = [prefix + make_unique_tokens(seed + i, block_size) for i in range(dp)]
        seed += dp
        with ThreadPoolExecutor(max_workers=dp) as pool:
            list(pool.map(lambda p: stream_completion(port, p, 1, True, timeout_s=timeout_s), prompts))
        after = fetch_prefix_cache_metrics(port)
        delta = diff_prefix_cache(before, after)
        per_engine = delta["per_engine_queries"]
        assert isinstance(per_engine, dict)
        # Every DP engine must have ingested at least one full copy of `prefix`.
        if all(per_engine.get(eng, 0.0) >= len(prefix) for eng in expected_engines):
            return wave, seed
    raise RuntimeError(
        f"Failed to warm prefix ({len(prefix)} tokens) on all {dp} DP engines within {max_waves} waves; "
        f"per-engine query deltas={per_engine}"
    )


def summarize_prefill_samples(
    samples: list[Completion],
    *,
    isl: int,
    target_hit_rate: float,
    prefix_len: int,
    uncached_len: int,
    concurrency: int,
    cache_delta: dict[str, object],
    warmup_waves: int,
) -> dict[str, object]:
    for c in samples:
        if c.prompt_tokens != isl:
            raise ValueError(f"Prefill prompt length mismatch: expected {isl}, got {c.prompt_tokens}")
        if len(c.token_ids) != 1:
            raise ValueError(f"Prefill expected 1 output token, got {len(c.token_ids)}")
    ttft_ms = [(c.arrivals[0] - c.start) * 1000.0 for c in samples]
    ttft_dist = distribution(ttft_ms, (50, 90, 95, 99))
    ttft_dist["min"] = min(ttft_ms)
    ttft_dist["max"] = max(ttft_ms)
    mean_s = float(ttft_dist["mean"]) / 1000.0  # type: ignore[arg-type]
    wall_s = max(c.end for c in samples) - min(c.start for c in samples)
    return {
        "isl": isl,
        "target_hit_rate": target_hit_rate,
        "aligned_hit_rate": prefix_len / isl,
        "prefix_len": prefix_len,
        "uncached_len": uncached_len,
        "concurrency": concurrency,
        "requests": len(samples),
        "warmup_waves": warmup_waves,
        "observed_hit_rate": cache_delta["hit_rate"],
        "prefix_cache_queries": cache_delta["queries"],
        "prefix_cache_hits": cache_delta["hits"],
        "ttft_ms": ttft_dist,
        "effective_prefill_tok_s": isl / mean_s if mean_s > 0 else None,
        "uncached_prefill_tok_s": uncached_len / mean_s if mean_s > 0 else None,
        "aggregate_prefill_tok_s": (isl * len(samples)) / wall_s if wall_s > 0 else None,
        "timeout_retries": sum(c.timeout_retries for c in samples),
        "samples": [
            {
                "ttft_ms": (c.arrivals[0] - c.start) * 1000.0,
                "e2e_ms": (c.end - c.start) * 1000.0,
                "prompt_tokens": c.prompt_tokens,
                "timeout_retries": c.timeout_retries,
            }
            for c in samples
        ],
    }


def _window_decode_metrics(samples: list[Completion], begin: float, end: float) -> dict[str, object] | None:
    if end <= begin:
        return None
    seconds = end - begin
    frame_sizes = [
        size
        for c in samples
        for stamp, size in zip(c.arrivals, c.tokens_per_frame, strict=True)
        if begin < stamp <= end
    ]
    token_count = sum(frame_sizes)
    if token_count == 0:
        return None
    coalesced = token_count - len(frame_sizes)
    itl = [
        (b - a) * 1000.0
        for c in samples
        for a, b in zip(c.arrivals, c.arrivals[1:])
        if begin < b <= end
    ]
    return {
        "seconds": seconds,
        "tokens": token_count,
        "decode_output_tok_s": token_count / seconds,
        "itl_ms": distribution(itl, (50, 90, 95, 99)),
        "itl_samples": len(itl),
        "coalesced_token_rate": coalesced / token_count,
    }


def summarize_decode_samples(
    samples: list[Completion],
    *,
    isl: int,
    osl: int,
    concurrency: int,
    rounds: int,
) -> dict[str, object]:
    for c in samples:
        if c.prompt_tokens != isl:
            raise ValueError(f"Decode prompt length mismatch: expected {isl}, got {c.prompt_tokens}")
        if len(c.token_ids) != osl:
            raise ValueError(f"Decode output length mismatch: expected {osl}, got {len(c.token_ids)}")

    # Subtract tokens already delivered in the first SSE frame so frame-0 coalescing never deflates TPOT.
    tpot_ms = [
        (c.end - c.arrivals[0]) * 1000.0 / max(1, len(c.token_ids) - c.tokens_per_frame[0])
        for c in samples
    ]
    ttft_ms = [(c.arrivals[0] - c.start) * 1000.0 for c in samples]
    tpot_dist = distribution(tpot_ms, (50, 90, 95, 99))
    tpot_dist["min"] = min(tpot_ms)
    tpot_dist["max"] = max(tpot_ms)

    overall_begin = min(c.arrivals[0] for c in samples)
    overall_end = max(c.end for c in samples)
    overall_win = _window_decode_metrics(samples, overall_begin, overall_end)

    # Overlap window where all `concurrency` requests in the first wave are simultaneously decoding.
    first_wave = samples[:concurrency]
    overlap_begin = max(c.arrivals[0] for c in first_wave)
    overlap_end = min(c.end for c in first_wave)
    overlap_win = _window_decode_metrics(first_wave, overlap_begin, overlap_end) if rounds == 1 else None
    overlap_tpot_ms = (
        (float(overlap_win["seconds"]) * 1000.0 * concurrency / int(overlap_win["tokens"]))
        if overlap_win is not None
        else None
    )
    if overlap_win is not None:
        overlap_win["tpot_ms"] = overlap_tpot_ms

    steady_win = None
    if rounds >= 3:
        stops = sorted(c.end for c in samples)
        steady_win = _window_decode_metrics(samples, stops[concurrency - 1], stops[-concurrency - 1])

    total_tokens = sum(len(c.token_ids) for c in samples)
    total_coalesced = sum(c.coalesced_tokens for c in samples)
    all_itl = [(b - a) * 1000.0 for c in samples for a, b in zip(c.arrivals, c.arrivals[1:])]
    wall_s = overall_end - min(c.start for c in samples)

    return {
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "rounds": rounds,
        "requests": len(samples),
        "verified_completion_tokens": total_tokens,
        "wall_s": wall_s,
        "tpot_ms": tpot_dist,
        "overlap_tpot_ms": overlap_tpot_ms,
        "ttft_ms": distribution(ttft_ms, (50, 95, 99)),
        "itl_ms": distribution(all_itl, (50, 90, 95, 99)),
        "itl_samples": len(all_itl),
        "decode_output_tok_s": overall_win["decode_output_tok_s"] if overall_win else total_tokens / wall_s,
        "overlap_decode_tok_s": overlap_win["decode_output_tok_s"] if overlap_win else None,
        "overlap": overlap_win,
        "steady": steady_win,
        "coalesced_tokens": total_coalesced,
        "coalesced_token_rate": total_coalesced / total_tokens if total_tokens else 0.0,
        "timeout_retries": sum(c.timeout_retries for c in samples),
    }


def resolve_defaults(args: argparse.Namespace) -> None:
    """Populate bench-specific defaults for DP/TP/DCP/KV/max-model-len."""
    if args.bench == "prefill":
        if args.dp is None:
            args.dp = 2
        if args.tp is None:
            args.tp = 4
        if args.dcp is None:
            args.dcp = 1
        if args.max_model_len is None:
            args.max_model_len = align_up(max(args.isl_list) + 1024, 1024)
        if args.max_num_seqs is None:
            args.max_num_seqs = max(16, args.prefill_concurrency * 2)
        if args.max_num_batched_tokens is None:
            args.max_num_batched_tokens = 8192
    elif args.bench == "decode":
        if args.dp is None:
            args.dp = 2
        if args.tp is None:
            args.tp = 4
        if args.dcp is None:
            args.dcp = args.tp if args.kv_cache_dtype in ("fp8", "fp8_ds_mla") else 1
        if args.max_model_len is None:
            args.max_model_len = align_up(args.decode_isl + args.decode_osl + 256, 1024)
        if args.max_num_seqs is None:
            args.max_num_seqs = max(64, max(args.concurrencies) * 2)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.out_json is None:
        parser.error("a benchmark run requires --out-json")
    if args.dp < 1 or args.tp < 1 or args.dcp < 1:
        parser.error("--dp, --tp, and --dcp must be positive")
    if args.dcp > args.tp or args.tp % args.dcp != 0:
        parser.error(f"--dcp ({args.dcp}) must divide --tp ({args.tp})")
    if args.dcp > 1 and args.kv_cache_dtype not in ("fp8", "fp8_ds_mla"):
        parser.error("DCP > 1 for GLM-5.3 sparse MLA requires --kv-cache-dtype fp8 (FLASHMLA_SPARSE)")
    if args.bench == "prefill":
        if min(args.isl_list) < args.block_size * 2 or args.repeats < 1 or args.prefill_concurrency < 1:
            parser.error("prefill requires valid ISLs, --repeats >= 1, and --prefill-concurrency >= 1")
        for isl in args.isl_list:
            if isl >= args.max_model_len:
                parser.error(f"ISL {isl} must be < --max-model-len {args.max_model_len}")
            for hit in args.cache_hit_rates:
                compute_prefix_split(isl, hit, args.block_size)
    else:
        if args.decode_isl < 1 or args.decode_osl < 2 or args.decode_rounds < 1 or min(args.concurrencies) < 1:
            parser.error("decode requires --decode-isl >= 1, --decode-osl >= 2, --decode-rounds >= 1")
        if args.decode_isl + args.decode_osl > args.max_model_len:
            parser.error(
                f"decode_isl + decode_osl ({args.decode_isl + args.decode_osl}) > --max-model-len {args.max_model_len}"
            )
    validate_backend_args(
        args.mode,
        all2all_backend=args.all2all_backend,
        dispatch_dtype=args.dispatch_dtype,
        tb_rule_only=args.tb_rule_only,
        error_fn=parser.error,
    )


def build_server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        "glm53",
        "--data-parallel-size",
        str(args.dp),
        "--tensor-parallel-size",
        str(args.tp),
        "--enable-expert-parallel",
        "--all2all-backend",
        args.all2all_backend,
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--block-size",
        str(args.block_size),
        "--stream-interval",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--load-format",
        args.load_format,
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.dcp > 1:
        cmd.extend(["--decode-context-parallel-size", str(args.dcp), "--dcp-comm-backend", "ag_rs"])
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.num_gpu_blocks_override is not None:
        cmd.extend(["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)])

    if args.bench == "prefill":
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")
        transfer = {
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0, "startup_fill": True},
        }
        cmd.extend(["--kv-transfer-config", json.dumps(transfer)])

    if args.enforce_eager:
        cmd.append("--enforce-eager")
    elif args.capture_sizes:
        cmd.extend(["--compilation-config", json.dumps({"cudagraph_capture_sizes": args.capture_sizes})])

    return append_backend_args(
        cmd,
        args.mode,
        strict_moe=args.strict_moe,
        nvfp4_quant_mode=args.nvfp4_quant_mode,
        shared_reserve_sms=args.shared_reserve_sms,
        max_tokens_per_rank=args.max_tokens_per_rank,
    )


def run_prefill_bench(args: argparse.Namespace, results: dict[str, object]) -> None:
    seed = 10_000
    for isl in args.isl_list:
        for hit_rate in args.cache_hit_rates:
            prefix_len, uncached_len = compute_prefix_split(isl, hit_rate, args.block_size)
            shared_prefix = make_unique_tokens(seed, prefix_len) if prefix_len > 0 else []
            seed += 1
            warmup_waves = 0
            if prefix_len > 0:
                warmup_waves, seed = warm_prefix_cache(
                    args.port,
                    shared_prefix,
                    dp=args.dp,
                    block_size=args.block_size,
                    seed_start=seed,
                    timeout_s=args.request_timeout,
                )

            total_reqs = args.repeats * args.prefill_concurrency
            prompts = [
                (shared_prefix + make_unique_tokens(seed + i, uncached_len))
                if prefix_len > 0
                else make_unique_tokens(seed + i, isl)
                for i in range(total_reqs)
            ]
            seed += total_reqs

            metrics_before = fetch_prefix_cache_metrics(args.port)
            samples: list[Completion] = []
            if args.prefill_concurrency == 1:
                for prompt in prompts:
                    samples.append(stream_completion(args.port, prompt, 1, True, timeout_s=args.request_timeout))
            else:
                with ThreadPoolExecutor(max_workers=args.prefill_concurrency) as pool:
                    for first in range(0, total_reqs, args.prefill_concurrency):
                        wave = prompts[first : first + args.prefill_concurrency]
                        samples.extend(
                            pool.map(
                                lambda p: stream_completion(args.port, p, 1, True, timeout_s=args.request_timeout),
                                wave,
                            )
                        )
            metrics_after = fetch_prefix_cache_metrics(args.port)
            cache_delta = diff_prefix_cache(metrics_before, metrics_after)

            point = summarize_prefill_samples(
                samples,
                isl=isl,
                target_hit_rate=hit_rate,
                prefix_len=prefix_len,
                uncached_len=uncached_len,
                concurrency=args.prefill_concurrency,
                cache_delta=cache_delta,
                warmup_waves=warmup_waves,
            )
            observed = point["observed_hit_rate"]
            expected_hit = prefix_len / isl
            if args.strict_cache_hit and (
                observed is None or abs(float(observed) - expected_hit) > args.cache_hit_tol
            ):
                raise RuntimeError(
                    f"Prefix cache hit mismatch at ISL={isl} target={hit_rate:.2f} (aligned={expected_hit:.3f}): "
                    f"observed={observed}, delta={cache_delta}"
                )
            results["points"].append(point)  # type: ignore[union-attr]
            args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
            print(
                f"[prefill] ISL={isl:>7d} hit={hit_rate * 100:4.0f}% (obs={float(observed or 0) * 100:5.1f}%) "
                f"TTFT mean={point['ttft_ms']['mean']:.1f}ms p50={point['ttft_ms']['p50']:.1f}ms "
                f"p99={point['ttft_ms']['p99']:.1f}ms eff={point['effective_prefill_tok_s']:.0f} tok/s",
                flush=True,
            )


def run_decode_bench(args: argparse.Namespace, results: dict[str, object]) -> None:
    seed = 50_000
    for concurrency in args.concurrencies:
        count = concurrency * args.decode_rounds
        prompts = [make_unique_tokens(seed + i, args.decode_isl) for i in range(count)]
        seed += count
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            samples = list(
                pool.map(
                    lambda p: stream_completion(args.port, p, args.decode_osl, True, timeout_s=args.request_timeout),
                    prompts,
                )
            )
        point = summarize_decode_samples(
            samples,
            isl=args.decode_isl,
            osl=args.decode_osl,
            concurrency=concurrency,
            rounds=args.decode_rounds,
        )
        results["points"].append(point)  # type: ignore[union-attr]
        args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        ov_tpot = point["overlap_tpot_ms"]
        ov_str = f" ov_TPOT={ov_tpot:.2f}ms" if ov_tpot is not None else ""
        print(
            f"[decode] ISL={args.decode_isl} OSL={args.decode_osl} C={concurrency:>3d} "
            f"TPOT mean={point['tpot_ms']['mean']:.2f}ms p50={point['tpot_ms']['p50']:.2f}ms "
            f"p99={point['tpot_ms']['p99']:.2f}ms{ov_str} ITL p50={point['itl_ms']['p50']:.2f}ms "
            f"tok/s={point['decode_output_tok_s']:.1f} coal={point['coalesced_token_rate'] * 100:.2f}%",
            flush=True,
        )


def summarize_files(paths: list[Path], out_csv: Path | None = None) -> str:
    """Format Prefill and Decode JSON results into Markdown tables (and optional CSV)."""
    prefill_rows: list[dict[str, object]] = []
    decode_rows: list[dict[str, object]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        bench = data.get("bench")
        mode = data["mode"]
        topo = f"DP{data['dp']}_TP{data['tp']}_DCP{data.get('dcp', 1)}_{data.get('kv_cache_dtype', 'auto')}"
        for pt in data.get("points", []):
            if bench == "prefill":
                prefill_rows.append(
                    {
                        "mode": mode,
                        "topo": topo,
                        "isl": pt["isl"],
                        "target_hit_pct": round(pt["target_hit_rate"] * 100, 1),
                        "obs_hit_pct": round((pt["observed_hit_rate"] or 0.0) * 100, 1),
                        "ttft_mean_ms": pt["ttft_ms"]["mean"],
                        "ttft_p50_ms": pt["ttft_ms"]["p50"],
                        "ttft_p99_ms": pt["ttft_ms"]["p99"],
                        "eff_tok_s": pt["effective_prefill_tok_s"],
                        "uncached_tok_s": pt["uncached_prefill_tok_s"],
                    }
                )
            elif bench == "decode":
                decode_rows.append(
                    {
                        "mode": mode,
                        "topo": topo,
                        "isl": pt["isl"],
                        "osl": pt["osl"],
                        "concurrency": pt["concurrency"],
                        "tpot_mean_ms": pt["tpot_ms"]["mean"],
                        "tpot_p50_ms": pt["tpot_ms"]["p50"],
                        "tpot_p99_ms": pt["tpot_ms"]["p99"],
                        "overlap_tpot_ms": pt.get("overlap_tpot_ms"),
                        "itl_p50_ms": pt["itl_ms"]["p50"],
                        "itl_p99_ms": pt["itl_ms"]["p99"],
                        "decode_tok_s": pt["decode_output_tok_s"],
                        "overlap_tok_s": pt["overlap_decode_tok_s"],
                        "coalesced_pct": pt["coalesced_token_rate"] * 100.0,
                    }
                )
            else:
                raise ValueError(f"{path}: unrecognized bench type {bench!r}")

    sections: list[str] = []
    if prefill_rows:
        prefill_rows.sort(key=lambda r: (int(r["isl"]), float(r["target_hit_pct"]), str(r["mode"])))
        lines = [
            "### Prefill Bench (TTFT)",
            "",
            "| Mode | Topo | ISL | Cache Hit (Target / Obs) | TTFT Mean (ms) | TTFT P50 (ms) | TTFT P99 (ms) | Eff. Prefill (tok/s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in prefill_rows:
            lines.append(
                f"| `{r['mode']}` | `{r['topo']}` | {int(r['isl']):,} | "
                f"{r['target_hit_pct']:.0f}% / {r['obs_hit_pct']:.1f}% | "
                f"{float(r['ttft_mean_ms']):.1f} | {float(r['ttft_p50_ms']):.1f} | {float(r['ttft_p99_ms']):.1f} | "
                f"{float(r['eff_tok_s']):,.0f} |"
            )
        sections.append("\n".join(lines))

    if decode_rows:
        decode_rows.sort(key=lambda r: (int(r["isl"]), int(r["osl"]), int(r["concurrency"]), str(r["mode"])))
        lines = [
            "### Decode Bench (TPOT)",
            "",
            "| Mode | Topo | ISL + OSL | Concurrency | TPOT Mean (ms) | TPOT P50 (ms) | TPOT P99 (ms) | Overlap TPOT (ms) | ITL P50 (ms) | Output (tok/s) | Coalesced |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in decode_rows:
            ov = f"{float(r['overlap_tpot_ms']):.2f}" if r["overlap_tpot_ms"] is not None else "-"
            lines.append(
                f"| `{r['mode']}` | `{r['topo']}` | {int(r['isl']):,}+{int(r['osl'])} | {int(r['concurrency'])} | "
                f"{float(r['tpot_mean_ms']):.2f} | {float(r['tpot_p50_ms']):.2f} | {float(r['tpot_p99_ms']):.2f} | "
                f"{ov} | {float(r['itl_p50_ms']):.2f} | {float(r['decode_tok_s']):.1f} | {float(r['coalesced_pct']):.2f}% |"
            )
        sections.append("\n".join(lines))

    if out_csv is not None:
        if prefill_rows and decode_rows:
            raise ValueError("--out-csv accepts either prefill or decode files in one call, not mixed")
        rows = prefill_rows or decode_rows
        if rows:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    return "\n\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--bench", choices=("prefill", "decode"))
    action.add_argument("--summarize", type=Path, nargs="+", help="summarize prefill/decode JSON results")

    parser.add_argument("--out-csv", type=Path, help="optional CSV path for --summarize")
    parser.add_argument("--mode", choices=ALL_MODES, default="tb_nvfp4_megamoe")
    parser.add_argument("--model", default=os.environ.get("TENSORBRIDGE_MODEL_PATH", "GLM-5.3"))
    parser.add_argument("--dp", type=int, help="data-parallel size (default: 2)")
    parser.add_argument("--tp", type=int, help="tensor-parallel size (default: 4)")
    parser.add_argument("--dcp", type=int, help="decode-context-parallel size (default: 1 prefill, tp decode w/ fp8)")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "bfloat16", "fp8", "fp8_ds_mla"), default="fp8")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--dispatch-dtype", choices=("bf16", "fp8"))
    parser.add_argument("--nvfp4-quant-mode", choices=("mse_7ulp", "four_over_six", "bridge_aware", "rtn"))
    parser.add_argument("--strict-moe", action="store_true")
    parser.add_argument("--tb-rule-only", action="store_true")
    parser.add_argument(
        "--shared-reserve-sms",
        type=int,
        default=0,
        help="SMs reserved for concurrent shared_experts alongside persistent MegaMoE (keeps ctas_per_sm intact)",
    )
    parser.add_argument("--max-tokens-per-rank", type=int, default=256)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--load-format", choices=("auto", "dummy"), default="auto")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    graph = parser.add_mutually_exclusive_group()
    graph.add_argument("--enforce-eager", action="store_true")
    graph.add_argument("--capture-sizes", nargs="+", type=int)

    # Prefill bench parameters
    parser.add_argument("--isl-list", nargs="+", type=int, default=list(DEFAULT_PREFILL_ISLS))
    parser.add_argument("--cache-hit-rates", nargs="+", type=float, default=list(DEFAULT_CACHE_HIT_RATES))
    parser.add_argument("--repeats", type=int, default=3, help="measured waves per (ISL, cache_hit_rate)")
    parser.add_argument("--prefill-concurrency", type=int, default=1)
    parser.add_argument("--cache-hit-tol", type=float, default=0.03)
    parser.add_argument("--no-strict-cache-hit", dest="strict_cache_hit", action="store_false")

    # Decode bench parameters
    parser.add_argument("--decode-isl", type=int, default=DEFAULT_DECODE_ISL)
    parser.add_argument("--decode-osl", type=int, default=DEFAULT_DECODE_OSL)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=list(DEFAULT_DECODE_CONCURRENCIES))
    parser.add_argument("--decode-rounds", type=int, default=1, help="closed-loop requests per concurrency slot")

    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--out-json", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize:
        print(summarize_files(args.summarize, args.out_csv))
        return

    resolve_defaults(args)
    validate_args(args, parser)
    cmd = build_server_command(args)
    log = args.server_log or args.out_json.with_suffix(".server.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "bench": args.bench,
        "mode": args.mode,
        "model": args.model,
        "dp": args.dp,
        "tp": args.tp,
        "dcp": args.dcp,
        "kv_cache_dtype": args.kv_cache_dtype,
        "block_size": args.block_size,
        "all2all_backend": args.all2all_backend,
        "nvfp4_quant_mode": args.nvfp4_quant_mode,
        "shared_reserve_sms": args.shared_reserve_sms,
        "max_tokens_per_rank": args.max_tokens_per_rank,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "eager": args.enforce_eager,
        "capture_sizes": args.capture_sizes,
        "argv": cmd,
        "server_log": str(log),
        "points": [],
    }

    with log.open("w", encoding="utf-8") as output:
        print(json.dumps(cmd), flush=True)
        env = {**os.environ, "TENSORBRIDGE_DISABLE_PROFILES": "1" if args.tb_rule_only else "0"}
        proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            try:
                results["startup_s"] = wait_for_health(args.port, proc, args.startup_timeout)
            except (TimeoutError, RuntimeError) as error:
                results["startup_error"] = str(error)
                raise
            finally:
                startup = log.read_text(errors="replace")
                calibration = kv_cache_calibration(startup, args.port if "startup_s" in results else None)
                results.update(calibration)
                args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps(calibration), flush=True)

            resolutions = sorted({line.split(": MoE=", 1)[1] for line in startup.splitlines() if ": MoE=" in line})
            results["resolved_moe"] = resolutions
            if args.strict_moe and args.mode in (*ADAPTERS, *PRESETS) and not resolutions:
                raise RuntimeError(
                    f"No per-layer TensorBridge resolution in {log}; check plugin registration and INFO logging"
                )

            # Warm up all DP engines (JIT + CUDA graphs) with tiny distinct requests before measurement.
            warmup_out = 1 if args.bench == "prefill" else 4
            with ThreadPoolExecutor(max_workers=args.dp) as pool:
                list(
                    pool.map(
                        lambda i: stream_completion(
                            args.port, make_unique_tokens(1 + i, 64), warmup_out, True, timeout_s=args.request_timeout
                        ),
                        range(args.dp),
                    )
                )

            if args.bench == "prefill":
                run_prefill_bench(args, results)
            else:
                run_decode_bench(args, results)
        finally:
            stop_server(proc)

    print("\n" + summarize_files([args.out_json]), flush=True)


if __name__ == "__main__":
    main()
