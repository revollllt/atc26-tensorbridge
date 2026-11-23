"""Single-node GLM-5.3 serving evaluation. All latency measurements use the client's monotonic clock."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import product
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
    distribution,
    kv_cache_calibration,
    read_stream,
    stop_server,
    stream_completion,
    validate_backend_args,
    wait_for_health,
)

QUALITY_PROMPTS = [
    "Explain the difference between Hopper WGMMA and Ampere MMA instructions concisely.",
    "Write a Python function to compute the longest increasing subsequence in O(n log n) time.",
    "Solve step by step: A train travels 180 km in 2 hours and 15 minutes. What is its average speed in m/s?",
    "请简要介绍 Mixture-of-Experts (MoE) 模型中 Expert Parallelism (EP) 的通信瓶颈及优化思路。",
    "Summarize the key trade-offs between FP8 and NVFP4 weight quantization for LLM inference.",
    "Write a C++17 lock-free single-producer single-consumer ring buffer skeleton.",
    "What is the time and space complexity of Tarjan's strongly connected components algorithm?",
    "请用简洁的语言解释 CUDA Green Context 与普通 CUDA Stream 的本质区别。",
]


def make_token_prompt(idx: int, prompt_len: int) -> list[int]:
    return [1000 + ((i * 37 + idx * 13) % 50000) for i in range(prompt_len)]


CORPUS = (
    "The database had acknowledged the transaction before the power failed. On restart, the page containing the "
    "new balance still held its old value. Recovery read the transaction log from the last checkpoint and found "
    "the missing update. The log record carried a sequence number, so replay could skip pages that already "
    "contained it. The test engineers cut power a second time during recovery and checked both account balances "
    "again. The sum of the balances stayed constant across both interruptions.",
    "Suppose a queue receives ninety requests per second, and each request spends an average of forty milliseconds "
    "in the system. Over a sufficiently long interval with stable arrivals and departures, the expected number in "
    "the system is 3.6. This includes requests receiving service. To estimate the number waiting for a worker, "
    "subtract the average service time before multiplying by the arrival rate. Averages from the startup burst "
    "would give a different answer because the queue is still filling.",
    "An atmospheric sensor sends a reading every ten seconds. Its clock drifts slowly, and several packets can "
    "arrive together when the radio link recovers. The archive keeps both the sampling time and the arrival time. "
    "A chart of temperature uses the former to put the readings in order. An alarm about the health of the link "
    "uses the latter. If a packet arrives twice, its sensor identifier and sequence number allow the archive to "
    "recognize the duplicate without discarding a legitimate repeated temperature.",
    "A matrix with six rows and four columns defines a linear map from four coordinates to six. If two columns "
    "are identical, subtracting the corresponding unit vectors gives a nonzero vector that maps to zero. Adding "
    "any multiple of that vector to a solution leaves the predicted measurements unchanged. A least squares "
    "solver can still return a particular solution by choosing the one with minimum Euclidean norm. More "
    "measurements help only if they distinguish the two coordinates.",
    "Consider a sorted array that contains several copies of the search key. The following loop returns the "
    "first position whose value is at least the key. It also returns a valid insertion position when the key "
    "is absent, including the position immediately after the final element. The upper bound is exclusive.\n"
    "lo, hi = 0, len(values)\nwhile lo < hi:\n    mid = (lo + hi) // 2\n    if values[mid] < key:\n"
    "        lo = mid + 1\n    else:\n        hi = mid\nreturn lo",
    "Two storage devices have the same advertised sequential read rate. One serves a directory tree much "
    "faster because opening many small files requires separate metadata lookups. To reproduce the observation, "
    "the experiment records the size distribution of the files, clears the application cache between trials, "
    "and alternates the order of the devices. A separate run repeatedly reads one large file. Combining the "
    "two results into a single average would obscure which part of the workload changed.",
    "A weather forecaster assigns a probability of rain to each day. Among days given a probability near "
    "one quarter, roughly one quarter should turn out wet if the forecast is calibrated. Accuracy after a "
    "threshold tells a different story. Moving the threshold changes the number of warnings even when every "
    "probability stays the same. For a farmer deciding whether to irrigate, the cost of watering unnecessarily "
    "and the cost of a missed shower determine which threshold makes sense.",
    "The compiler moved an addition out of a loop because neither operand changed between iterations. It kept "
    "a nearby load inside the loop. The pointer passed to a function in the loop could refer to the same "
    "location, and the function could write through it. The programmer then supplied an example in which the "
    "two pointers really did alias. Hoisting the load in that example produced the old value on the second "
    "iteration and changed the final sum.",
    "In a packet trace, a long gap between acknowledgments can have several causes. The receiver might be "
    "delaying a response, a packet might have been lost, or the sender might have run out of data. The analyst "
    "first compares sequence numbers on the two sides of the gap. A retransmission covering the missing bytes "
    "supports the loss explanation. If no bytes were outstanding during the interval, the same gap says little "
    "about the capacity of the network.",
    "To integrate a smooth function over an interval, divide the interval into equal pieces and approximate "
    "the area under each piece by a trapezoid. Repeating the calculation with half the step size gives a useful "
    "comparison. For a smooth curve with a nonzero leading error term, the error generally falls by a factor "
    "of four. A sharp corner can spoil this pattern. Splitting the interval at the corner lets the approximation "
    "treat the smooth portions separately.",
    "The laboratory labeled each specimen with a random code before sending it for measurement. The person "
    "operating the instrument could see the code but could not tell which treatment the specimen had received. "
    "A second person kept the mapping in a sealed file. After the measurements were recorded, the team discovered "
    "that all samples from one treatment had been measured on the same afternoon. A change in room temperature "
    "was therefore a plausible alternative explanation for the observed difference.",
    "The report needs one row per customer, including customers who placed no orders. A left join preserves "
    "those rows. Counting a column that is nonnull for every real order then gives zero for an unmatched "
    "customer. Counting every joined row would give one instead. The query groups by the stable identifier "
    "so that customers with the same display name remain distinct.\n"
    "SELECT c.id, COUNT(o.id) FROM customer AS c\n"
    "LEFT JOIN orders AS o ON o.customer_id = c.id GROUP BY c.id;",
    "The image decoder works on square tiles. A tile at the right edge may extend beyond the width of the "
    "image, so the decoder clips its final column before writing to the output buffer. Interior tiles follow "
    "the ordinary path. A useful test image has a width one pixel greater than a multiple of the tile size. "
    "It makes the last tile contain just one valid column and exposes an incorrect inclusive bound immediately.",
    "A distributed training job accumulated gradients for four microbatches before updating its parameters. "
    "Three microbatches contained the same number of examples, while the final one contained fewer. Averaging "
    "the four mean gradients with equal weights gave the small batch too much influence. The implementation "
    "instead accumulated the sum of the gradients and the number of examples. It divided once at the end. "
    "This matched the gradient from processing the concatenated examples under the same loss function.",
    "在一次索引重建中，工程师发现查询延迟突然升高。日志显示磁盘仍有空闲带宽，但读取一个页面需要等待更久。"
    "他们把前台查询和后台写入的队列分开统计，发现大量小写入正在排队。降低后台并发后，查询的尾延迟下降了，"
    "而索引完成的时间略有延长。报告同时保留这两个数字，以便值班人员按当天的业务量选择配置。",
    "地图上的两条道路长度相同，旅行时间却可能相差很大。一条道路经过三个红绿灯，另一条需要翻过山坡。"
    "规划路线时，可以把每个路口看作节点，把预计耗时作为边的权重。若早高峰改变了等待时间，就要更新相应的权重。"
    "记录出发时刻有助于解释同一对起点和终点为什么会得到不同的推荐路线。",
    "一段音频以固定频率采样。为了缩小文件，程序准备每隔一个样本保留一个值。直接丢弃样本会使部分高频成分"
    "在较低频率处出现。工程师先对输入做低通滤波，再进行抽取，并用一组已知频率的正弦波检查输出。"
    "靠近截止频率的信号尤其值得观察，因为滤波器在这一段的衰减会随频率变化。",
    "During a lunar eclipse, observers at different longitudes can watch the same shadow advance across the "
    "Moon. Their local clocks show different hours, so the observers convert each timestamp to a common time "
    "standard before comparing notes. One telescope produces a slightly softer boundary than the other. The "
    "team measures how much the recorded contact time changes when the brightness threshold is varied. That "
    "range gives an indication of the uncertainty introduced by the choice of boundary.",
)


def summarize(completions: list[Completion], begin: float, end: float, role: str = "colocated") -> dict[str, object]:
    # A completion contributes request metrics at completion; count token arrivals inside the same window.
    requests = [c for c in completions if begin < c.end <= end]
    ttft = [(c.arrivals[0] - c.start) * 1000 for c in requests]
    seconds = end - begin
    if role == "prefill":
        if any(c.prompt_tokens is None for c in requests):
            raise ValueError("Missing prompt usage for prefill throughput")
        verified = sum(c.prompt_tokens for c in requests if c.prompt_tokens is not None)
        return dict(seconds=seconds, requests=len(requests), req_s=len(requests) / seconds,
                    verified_prompt_tokens=verified, prefill_input_tok_s=verified / seconds,
                    ttft_ms=distribution(ttft, (50, 95, 99)))
    frame_sizes = [
        size
        for c in completions
        for stamp, size in zip(c.arrivals, c.tokens_per_frame, strict=True)
        if begin < stamp <= end
    ]
    token_count = sum(frame_sizes)
    coalesced_tokens = token_count - len(frame_sizes)
    itl = [(b - a) * 1000 for c in completions for a, b in zip(c.arrivals, c.arrivals[1:]) if begin < b <= end]
    tpot = [(c.end - c.arrivals[0]) * 1000 / max(1, len(c.token_ids) - 1) for c in requests]
    return {
        **({"decode_output_tok_s": token_count / seconds} if role == "decode" else {}),
        "seconds": seconds,
        "requests": len(requests),
        "verified_completion_tokens": sum(len(c.token_ids) for c in requests),
        "window_output_tokens": token_count,
        "output_tok_s": token_count / seconds,
        "ttft_ms": distribution(ttft, (99,)),
        "itl_ms": distribution(itl, (50, 95, 99)),
        "itl_samples": len(itl),
        "itl_basis": "observed inter-frame arrivals",
        "itl_coalesced_tokens": coalesced_tokens,
        "coalesced_token_rate": coalesced_tokens / token_count if token_count else 0.0,
        "coalesced_frame_rate": sum(size > 1 for size in frame_sizes) / len(frame_sizes) if frame_sizes else 0.0,
        "coalesced_requests": sum(c.coalesced_tokens > 0 for c in requests),
        "timeout_retries": sum(c.timeout_retries for c in requests),
        "tpot_ms": distribution(tpot, (99,)),
        "slo_goodput_req_s": {
            f"ttft2000_tpot{limit}": sum(t <= 2000 and p <= limit for t, p in zip(ttft, tpot)) / seconds
            for limit in (50, 80)
        },
    }


def read_quality_corpus(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8").strip("\n")
    passages = tuple(p for p in re.split(r"\n[ \t]*\n", text) if p.strip())
    if not passages:
        raise ValueError(f"{path}: quality corpus contains no passages")
    return passages


def quality(port: int, length: int, corpus: tuple[str, ...] = CORPUS) -> dict[str, object]:
    greedy = []
    for prompt in QUALITY_PROMPTS:
        c = stream_completion(port, prompt, length, False)
        greedy.append({"prompt": prompt, "text": c.text, "token_ids": c.token_ids})
    nll_sum, count = 0.0, 0
    for passage in corpus:
        payload = dict(model="glm53", prompt=passage, temperature=0, echo=True, logprobs=1, max_tokens=0)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            record = json.load(response)
        values = record["choices"][0]["logprobs"]["token_logprobs"]
        if len(values) != record["usage"]["prompt_tokens"] or not values or values[0] is not None:
            raise ValueError("Echo logprobs do not cover the prompt with an unscored first token")
        nll_sum -= math.fsum(values[1:])
        count += len(values) - 1
    if count == 0:
        raise ValueError("Quality corpus has no scoreable tokens after excluding each first token")
    return {
        "greedy": greedy,
        "corpus": list(corpus),
        "scored_tokens": count,
        "nll_sum": nll_sum,
        "nll_per_token": nll_sum / count,
        "ppl": math.exp(nll_sum / count),
    }


def server_command(args: argparse.Namespace) -> list[str]:
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
    if args.pd_role == "decode":
        transfer = dict(kv_connector="DecodeBenchConnector", kv_role="kv_both",
                        kv_connector_extra_config=dict(fill_mean=0.015, fill_std=0.0, startup_fill=True))
        cmd.extend(["--kv-transfer-config", json.dumps(transfer)])
    if args.wikitext_path is not None:
        cmd.extend(["--hf-overrides", '{"head_dtype":"model"}', "--logprobs-mode", "raw_logprobs"])
    for option in ("num_gpu_blocks_override", "max_num_batched_tokens"):
        value = vars(args)[option]
        if value is None:
            continue
        cmd.extend(["--" + option.replace("_", "-"), str(value)])
    cmd.extend(["--no-enable-prefix-caching"] if args.no_enable_prefix_caching else [])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    else:
        cmd.extend(
            ["--compilation-config", json.dumps({"cudagraph_capture_sizes": args.capture_sizes})]
            if args.capture_sizes
            else []
        )
    return append_backend_args(
        cmd,
        args.mode,
        strict_moe=args.strict_moe,
        nvfp4_quant_mode=args.nvfp4_quant_mode,
        shared_reserve_sms=args.shared_reserve_sms,
        max_tokens_per_rank=args.max_tokens_per_rank,
    )


def summarize_pd(paths: list[Path]) -> None:
    """Write one paired CSV row per regime, ISL, concurrency and timing window."""
    rows = {}
    for path in paths:
        result = json.loads(path.read_text())
        role = result["pd_role"]
        if role not in ("prefill", "decode"):
            raise ValueError(f"{path} has no isolated PD measurements")
        settings = json.dumps(result["pd_settings"], sort_keys=True)
        for point in result["sweep"]:
            for window in ("overall", "steady"):
                metrics = point[window]
                if metrics is None:
                    continue
                key = result["mode"], settings, point["prompt_len"], point["concurrency"], window
                pair = rows.setdefault(key, {})
                if role in pair:
                    raise ValueError(f"{path}: duplicate {role} measurement for {key}")
                pair[role] = metrics, result["decode_len"]
    if not rows:
        raise ValueError("No finished PD sweep points in the input files")
    writer = csv.writer(sys.stdout)
    writer.writerow(("mode", "settings", "prompt_len", "concurrency", "window", "decode_len",
                     "prefill_input_tok_s", "prefill_req_s",
                     "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
                     "decode_output_tok_s", "itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
                     "tpot_mean_ms", "tpot_p99_ms",
                     "coalesced_token_rate"))
    for key, pair in sorted(rows.items()):
        if set(pair) != {"prefill", "decode"}:
            raise ValueError(f"Missing matching PD role for {key}")
        prefill, _ = pair["prefill"]
        decode, length = pair["decode"]
        writer.writerow((*key, length, prefill["prefill_input_tok_s"], prefill["req_s"],
                         *(prefill["ttft_ms"][p] for p in ("mean", "p50", "p95", "p99")), decode["decode_output_tok_s"],
                         *(decode["itl_ms"][p] for p in ("mean", "p50", "p95", "p99")), decode["tpot_ms"]["mean"],
                         decode["tpot_ms"]["p99"], decode["coalesced_token_rate"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--mode", choices=ALL_MODES)
    action.add_argument("--summarize-pd", type=Path, nargs="+", help="write paired prefill/decode CSV to stdout")
    parser.add_argument("--pd-role", choices=("colocated", "prefill", "decode"), default="colocated")
    parser.add_argument("--model", default=os.environ.get("TENSORBRIDGE_MODEL_PATH", "GLM-5.3"))
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--dispatch-dtype", choices=("bf16", "fp8"), help="optional explicit check against preset dispatch")
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--no-enable-prefix-caching", action="store_true")
    graph = parser.add_mutually_exclusive_group()
    graph.add_argument("--enforce-eager", action="store_true")
    graph.add_argument("--capture-sizes", nargs="+", type=int)
    parser.add_argument("--nvfp4-quant-mode", choices=("mse_7ulp", "four_over_six", "bridge_aware", "rtn"))
    parser.add_argument("--strict-moe", action="store_true")
    parser.add_argument("--tb-rule-only", action="store_true", help="bypass saved tile profiles for modular TB")
    parser.add_argument(
        "--shared-reserve-sms",
        type=int,
        default=0,
        help="SMs reserved for concurrent shared_experts alongside persistent MegaMoE (keeps ctas_per_sm intact)",
    )
    parser.add_argument("--max-tokens-per-rank", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--load-format", choices=("auto", "dummy"), default="auto")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[8, 32, 64, 128])
    parser.add_argument("--waves", type=int, default=5, help="closed-loop requests per concurrency slot")
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 2048], help="cycle within each sweep")
    parser.add_argument("--decode-len", type=int)
    parser.add_argument("--wikitext-path", type=Path, help="WikiText-2 raw test JSON or joined text")
    parser.add_argument("--wikitext-seqlen", type=int, default=2048)
    parser.add_argument("--wikitext-concurrency", type=int, default=8)
    parser.add_argument("--check-quality", action="store_true")
    parser.add_argument("--quality-max-tokens", type=int, default=64)
    parser.add_argument(
        "--quality-corpus-file",
        type=Path,
        help="score UTF-8 text, one passage per blank-line-separated paragraph; enables quality checks",
    )
    parser.add_argument("--quality-reference", type=Path, help="compare greedy IDs and NLL against a previous result")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    if args.summarize_pd:
        summarize_pd(args.summarize_pd)
        return
    if args.out_json is None:
        parser.error("a serving run requires --out-json")
    if args.pd_role == "prefill" and args.decode_len not in (None, 1):
        parser.error("prefill emits exactly one token; use --decode-len 1")
    if args.decode_len is None:
        args.decode_len = {"colocated": 64, "prefill": 1, "decode": 128}[args.pd_role]
    if args.pd_role != "colocated":
        args.no_enable_prefix_caching = True
        if args.num_gpu_blocks_override is None:
            args.num_gpu_blocks_override = 1000
    if min(args.concurrencies + args.prompt_lens + [args.decode_len, args.waves]) < 1:
        parser.error("concurrency, prompt/decode lengths and waves must be positive")
    check_quality = args.check_quality or args.quality_reference is not None or args.quality_corpus_file is not None
    if args.pd_role == "decode" and (check_quality or args.wikitext_path is not None):
        parser.error("decode uses synthetic KV and cannot score model quality")
    validate_backend_args(
        args.mode,
        all2all_backend=args.all2all_backend,
        dispatch_dtype=args.dispatch_dtype,
        tb_rule_only=args.tb_rule_only,
        error_fn=parser.error,
    )
    if args.wikitext_path is not None and (
        not args.wikitext_path.is_file() or not 2 <= args.wikitext_seqlen < args.max_model_len
        or args.wikitext_concurrency < 1
    ):
        parser.error("WikiText requires a local file, 2 <= seqlen < max-model-len, and positive concurrency")
    if args.load_format == "dummy" and (check_quality or args.wikitext_path is not None):
        parser.error("quality evaluation requires real weights")
    corpus = CORPUS if args.quality_corpus_file is None else read_quality_corpus(args.quality_corpus_file)
    cmd = server_command(args)
    log = args.server_log or args.out_json.with_suffix(".server.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    results = dict(
        mode=args.mode,
        pd_role=args.pd_role,
        argv=cmd,
        server_log=str(log),
        eager=args.enforce_eager,
        tb_rule_only=args.tb_rule_only,
        capture_sizes=args.capture_sizes,
        waves=args.waves,
        prompt_lens=args.prompt_lens,
        decode_len=args.decode_len,
        sweep=[],
    )
    if args.pd_role != "colocated":
        results["pd_settings"] = {key: vars(args)[key] for key in (
            "model", "dp", "tp", "load_format", "num_gpu_blocks_override", "all2all_backend", "waves",
            "nvfp4_quant_mode", "tb_rule_only", "dispatch_dtype", "capture_sizes", "enforce_eager",
            "max_num_batched_tokens", "max_model_len", "max_num_seqs", "max_tokens_per_rank",
            "shared_reserve_sms",
        )}
    with log.open("w") as output:
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
                graph_sizes = re.findall(
                    r"\b(?:capture_sizes|cudagraph_capture_sizes)['\"]?\s*[:=]\s*(\[[\d, ]*\])", startup
                )
                results["resolved_capture_sizes"] = (
                    [] if args.enforce_eager else (json.loads(graph_sizes[-1]) if graph_sizes else None)
                )
                args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps(calibration), flush=True)
            results["backend_policy"] = (
                "vLLM deep_gemm oracle (Triton fallback for ineligible shapes)"
                if args.mode == "fp8_deepgemm"
                else args.mode
            )

            resolutions = sorted({line.split(": MoE=", 1)[1] for line in startup.splitlines() if ": MoE=" in line})
            results["resolved_moe"] = resolutions
            if args.strict_moe and args.mode in (*ADAPTERS, *PRESETS) and not resolutions:
                raise RuntimeError(
                    f"No per-layer TensorBridge resolution in {log}; check plugin registration and INFO logging"
                )
            warmup_tokens = 1 if args.pd_role == "prefill" else 4
            with ThreadPoolExecutor(max_workers=args.dp) as pool:
                list(pool.map(lambda i: stream_completion(args.port, make_token_prompt(i, 32), warmup_tokens),
                              range(args.dp)))
            if check_quality:
                measured = quality(args.port, args.quality_max_tokens, corpus)
                if args.quality_reference:
                    prior = json.loads(args.quality_reference.read_text())["quality"]
                    if measured["corpus"] != prior["corpus"] or prior["scored_tokens"] != measured["scored_tokens"]:
                        raise ValueError("Quality reference corpus or tokenization differs")
                    measured["delta_nll"] = measured["nll_per_token"] - prior["nll_per_token"]
                    measured["greedy_exact_matches"] = sum(
                        a == b for a, b in zip(measured["greedy"], prior["greedy"], strict=True)
                    )
                    measured["reference"] = str(args.quality_reference)
                else:
                    measured["reference"] = None
                results["quality"] = measured
            else:
                results["quality"] = None
            if args.wikitext_path is not None:
                from eval_wikitext_ppl import evaluate

                results["wikitext"] = evaluate(args.wikitext_path, args.model, args.wikitext_seqlen,
                                               args.wikitext_concurrency, f"http://127.0.0.1:{args.port}/v1/completions",
                                               "glm53")
                args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps(results["wikitext"]), flush=True)
            offset = args.dp
            lengths = [args.prompt_lens] if args.pd_role == "colocated" else [[n] for n in args.prompt_lens]
            for prompt_lengths, concurrency in product(lengths, args.concurrencies):
                count = concurrency * args.waves
                prompts = [
                    make_token_prompt(offset + i, prompt_lengths[i % len(prompt_lengths)]) for i in range(count)
                ]
                offset += count
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    # Prefill submits synchronized waves. Decode keeps each client slot occupied.
                    batch = concurrency if args.pd_role == "prefill" else count
                    samples = []
                    for first in range(0, count, batch):
                        samples.extend(pool.map(lambda p: stream_completion(args.port, p, args.decode_len),
                                                prompts[first:first + batch]))
                if args.pd_role != "colocated" and any(
                    c.prompt_tokens != len(p) for c, p in zip(samples, prompts, strict=True)
                ):
                    raise ValueError("PD prompt length differs from server usage.prompt_tokens")
                stops = sorted(c.end for c in samples)
                row = dict(
                    concurrency=concurrency,
                    overall=summarize(samples, min(c.start for c in samples), stops[-1], args.pd_role),
                    **({"prompt_len": prompt_lengths[0]} if args.pd_role != "colocated" else {}),
                )
                row["steady"] = (
                    summarize(samples, stops[concurrency - 1], stops[-concurrency - 1], args.pd_role)
                    if args.waves >= 3 else None
                )
                results["sweep"].append(row)
                print(json.dumps(row), flush=True)
                sweep_start = min(c.start for c in samples)
                row["traces"] = [
                    {
                        "prompt_len": len(prompt),
                        "start_ms": (c.start - sweep_start) * 1000,
                        "arrivals_ms": [(stamp - sweep_start) * 1000 for stamp in c.arrivals],
                        "end_ms": (c.end - sweep_start) * 1000,
                        "verified_tokens": len(c.token_ids),
                        "tokens_per_frame": c.tokens_per_frame,
                        "coalesced_tokens": c.coalesced_tokens,
                        "timeout_retries": c.timeout_retries,
                    }
                    for prompt, c in zip(prompts, samples, strict=True)
                ]
                args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
