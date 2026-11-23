"""CPU regression checks for GLM-5.3 request timing and teacher-forced scoring."""

import csv
import io
import itertools
import json
import math
import sys
import tempfile
import unittest
import urllib.error
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_vllm_serve_scaleup as bench  # noqa: E402
import serve_common  # noqa: E402


def frame(payload: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n"


class ServingMetricsTest(unittest.TestCase):
    def test_profile_measured_floor(self) -> None:
        # Import the pure-Python JIT package without initializing the CUDA tensorbridge package.
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "tensorbridge"), *sys.path]):
            from jit import profiles

        key = ("NVIDIA H20", 78, 6144, 2048, 32, "contiguous", False, "exact")
        with patch.object(profiles, "_registry", {}), patch.object(profiles, "_loaded_default_sources", True):
            profiles.load_profile_file(profiles.PROFILES_DIR / "h20_moe_ep8.json")
            measured = profiles.lookup(*key, 4096)
            self.assertEqual((measured.block_m, measured.block_n, measured.ctas_per_sm), (32, 256, 2))
            self.assertIsNone(profiles.lookup(*key, 4095))
            segments = profiles.compress_segments([(64, measured), (128, measured)], unbounded_tail=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "profile.json")
                profiles.save_profile_file(path, [(key, segments)])
                profiles.load_profile_file(path)
            self.assertIsNone(profiles.lookup(*key, 63))
            self.assertEqual(profiles.lookup(*key, 64), measured)
            self.assertEqual(profiles.lookup(*key, 1024), measured)

    def test_preset_precision(self) -> None:
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "tensorbridge/vllm"), *sys.path]):
            from presets import expand_moe_preset

        expected = {
            "tb_nvfp4_fp8": ("humming_grouped", "per_token", "bf16", False),
            "tb_nvfp4_fp8_g128": ("fused_swiglu", "group_128", "fp8", False),
            "tb_nvfp4_megamoe": ("megamoe", "group_128", "fp8", True),
        }
        for preset in bench.PRESETS:
            options = expand_moe_preset({"preset": preset}, "allgather_reducescatter")
            self.assertEqual((options["moe_backend_mode"], options["moe_down_quant"],
                              options["moe_dispatch_dtype"], options["moe_nvlink_ep"]), expected[preset])
            self.assertEqual(options["moe_gate_up_quant"], options["moe_down_quant"])
            self.assertTrue(options["fp8_to_nvfp4"] and options["exact"] and options["moe"])
        with self.assertRaisesRegex(ValueError, "moe_dispatch_dtype"):
            expand_moe_preset({"preset": "tb_nvfp4_fp8", "moe_dispatch_dtype": "fp8"}, "allgather_reducescatter")

    def test_pd_runs_and_presets(self) -> None:
        tick = itertools.count(1)

        def complete(port: int, prompt: list[int], length: int) -> bench.Completion:
            start = next(tick)
            arrivals = [start + .1 + i * .001 for i in range(length)]
            return bench.Completion(start, arrivals, arrivals[-1] + .001, list(range(length)), "", [1] * length,
                                    prompt_tokens=len(prompt))

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for preset in (*bench.PRESETS, "humming_mxfp4_g128"):
                paths = []
                for role in ("prefill", "decode"):
                    out = Path(directory, f"{preset}-{role}.json")
                    paths.append(out)
                    argv = ["bench", "--mode", preset, "--pd-role", role, "--dp", "1", "--concurrencies", "1",
                            "--waves", "3", "--prompt-lens", "128", "1024", "--out-json", str(out)]
                    with (patch.object(sys, "argv", argv), patch.object(bench.subprocess, "Popen"),
                          patch.object(bench, "wait_for_health", return_value=1),
                          patch.object(bench, "kv_cache_calibration", return_value={}),
                          patch.object(bench, "stream_completion", side_effect=complete)):
                        bench.main()
                    result = json.loads(out.read_text())
                    command = result["argv"]
                    options = json.loads(command[command.index("--additional-config") + 1])["tensorbridge"]
                    key = "fp8_moe_baseline" if preset == "humming_mxfp4_g128" else "preset"
                    self.assertEqual(options[key], preset)
                    self.assertNotIn("nvfp4_quant_mode", options)
                    self.assertIn("--no-enable-prefix-caching", command)
                    self.assertEqual(command[command.index("--num-gpu-blocks-override") + 1], "1000")
                    self.assertEqual([row["prompt_len"] for row in result["sweep"]], [128, 1024])
                    if role == "prefill":
                        self.assertNotIn("--kv-transfer-config", command)
                        self.assertEqual(result["decode_len"], 1)
                        for row in result["sweep"]:
                            for window, count in (("overall", 3), ("steady", 1)):
                                measurement = row[window]
                                self.assertEqual(measurement["verified_prompt_tokens"], count * row["prompt_len"])
                                self.assertAlmostEqual(measurement["prefill_input_tok_s"],
                                                       count * row["prompt_len"] / measurement["seconds"])
                                self.assertEqual(set(measurement["ttft_ms"]), {"mean", "p50", "p95", "p99"})
                    else:
                        self.assertEqual(result["decode_len"], 128)
                        self.assertEqual(json.loads(command[command.index("--kv-transfer-config") + 1]), {
                            "kv_connector": "DecodeBenchConnector", "kv_role": "kv_both",
                            "kv_connector_extra_config": {"fill_mean": .015, "fill_std": 0., "startup_fill": True},
                        })
                        self.assertEqual(result["sweep"][0]["overall"]["verified_completion_tokens"], 384)
                report = io.StringIO()
                with redirect_stdout(report):
                    bench.summarize_pd(paths)
                rows = list(csv.DictReader(io.StringIO(report.getvalue())))
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(float(r["prefill_input_tok_s"]) > 0 and float(r["decode_output_tok_s"]) > 0
                                    for r in rows))

    def test_token_arrivals(self) -> None:
        events = [
            frame({"choices": [{"text": text, "token_ids": ids}]})
            for text, ids in [("", []), ("a", [4]), ("", [5]), ("b", [6]), ("", [])]
        ]
        events.extend([frame({"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 128}}),
                       b"data: [DONE]"])
        with patch.object(serve_common.time, "perf_counter", side_effect=[1.6, 2, 2.01, 2.05, 2.055, 2.06, 2.065, 2.07]):
            completion = bench.read_stream(events, 1.5, 3)
        self.assertEqual(completion.arrivals, [2, 2.01, 2.05])
        self.assertEqual(completion.text, "ab")
        self.assertEqual(completion.prompt_tokens, 128)
        self.assertEqual(completion.coalesced_tokens, 0)
        later = bench.Completion(3.8, [4, 4.2], 4.5, [1, 2], "cd", [1, 1])
        result = bench.summarize([completion, later], 1.5, 6)
        self.assertEqual(result["verified_completion_tokens"], 5)
        self.assertAlmostEqual(result["ttft_ms"]["mean"], 350)
        self.assertAlmostEqual(result["itl_ms"]["mean"], 250 / 3)
        self.assertAlmostEqual(result["itl_ms"]["p50"], 40)
        self.assertAlmostEqual(result["tpot_ms"]["mean"], 267.5)
        self.assertAlmostEqual(result["output_tok_s"], 5 / 4.5)
        self.assertAlmostEqual(result["slo_goodput_req_s"]["ttft2000_tpot50"], 1 / 4.5)

    def test_coalesced_stream(self) -> None:
        events = [
            frame({"choices": [{"text": text, "token_ids": ids}]})
            for text, ids in [("", []), ("a", [1]), ("bc", [2, 3, 4]), ("d", [5])]
        ]
        events += [frame({"choices": [], "usage": {"completion_tokens": 5}}), b"data: [DONE]"]
        with patch.object(serve_common.time, "perf_counter", side_effect=[9.5, 10, 10.03, 10.10, 10.11, 10.115, 10.12]):
            c = bench.read_stream(events, 9, 5)
        self.assertEqual(c.token_ids, [1, 2, 3, 4, 5])
        self.assertEqual(c.arrivals, [10, 10.03, 10.10])
        self.assertEqual(c.tokens_per_frame, [1, 3, 1])
        self.assertEqual(c.coalesced_tokens, 2)
        whole = bench.summarize([c], 9, 11)
        self.assertEqual(whole["verified_completion_tokens"], 5)
        self.assertEqual(whole["output_tok_s"], 2.5)
        self.assertEqual(whole["ttft_ms"]["mean"], 1000)
        self.assertAlmostEqual(whole["tpot_ms"]["mean"], 30)
        self.assertAlmostEqual(whole["itl_ms"]["mean"], 50)
        self.assertEqual(whole["itl_samples"], 2)
        self.assertEqual(whole["itl_coalesced_tokens"], 2)
        self.assertEqual(whole["coalesced_requests"], 1)
        self.assertAlmostEqual(whole["coalesced_frame_rate"], 1 / 3)
        self.assertAlmostEqual(whole["coalesced_token_rate"], 2 / 5)
        # Only the three-token frame lands inside this steady window. Request metrics use completion time.
        window = bench.summarize([c], 10, 10.05)
        self.assertEqual(window["requests"], 0)
        self.assertEqual(window["window_output_tokens"], 3)
        self.assertAlmostEqual(window["output_tok_s"], 60)
        self.assertEqual(window["itl_samples"], 1)
        self.assertEqual(window["itl_coalesced_tokens"], 2)

    def test_transient_timeout(self) -> None:
        class TimedOutStream(io.BytesIO):
            def __iter__(self) -> Iterator[bytes]:
                yield frame({"choices": [{"text": "discard", "token_ids": [99]}]})
                raise TimeoutError("read timed out")

        stream = io.BytesIO(
            frame({"choices": [{"text": "ok", "token_ids": [1, 2]}]})
            + frame({"choices": [], "usage": {"completion_tokens": 2}})
            + b"data: [DONE]\n"
        )
        with (
            patch.object(serve_common.urllib.request, "urlopen", side_effect=[TimedOutStream(), stream]) as send,
            patch.object(serve_common.time, "perf_counter", side_effect=[1, 2, 3, 4, 5, 6]),
        ):
            c = bench.stream_completion(8123, "prompt", 2)
        self.assertEqual(send.call_count, 2)
        self.assertEqual((c.start, c.arrivals, c.token_ids, c.timeout_retries), (1, [3], [1, 2], 1))
        with (
            patch.object(serve_common.urllib.request, "urlopen", side_effect=urllib.error.URLError(TimeoutError())) as send,
            self.assertRaises(urllib.error.URLError),
        ):
            bench.stream_completion(8123, "prompt", 2)
        self.assertEqual(send.call_count, 2)

    def test_unverifiable_stream(self) -> None:
        for batches, usage, expected in [
            ([], 0, None),
            ([[1], [2]], 1, None),
            ([[1]], 2, None),
            ([[1]], None, None),
            ([[1, 2]], 2, 3),
        ]:
            events = [frame({"choices": [{"text": "x", "token_ids": ids}]}) for ids in batches]
            if usage is not None:
                events.append(frame({"choices": [], "usage": {"completion_tokens": usage}}))
            events.append(b"data: [DONE]")
            with self.subTest(batches=batches, usage=usage, expected=expected), self.assertRaises(ValueError):
                bench.read_stream(events, 0, expected)

    def test_capture_argv_and_timeout_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "smoke.json"

            corpus = Path(directory, "wikitext.txt")
            corpus.write_text("startup argument test")

            def timeout(*args: object) -> float:
                out.with_suffix(".server.log").write_text(
                    "MoE startup graph_mode=FULL capture_sizes=[1, 8, 32]\n"
                    "decoder_replay_cudagraph_capture_sizes=[]\n"
                    "Overriding num_gpu_blocks=9000 with num_gpu_blocks_override=8000\n"
                    "Available KV cache memory: 17.50 GiB\nAvailable KV cache memory: 16.25 GiB\n"
                    "GPU KV cache size: 100,352 tokens, Maximum concurrency for 4096 tokens per request: 24.5x\n"
                )
                raise TimeoutError("still capturing")

            with (
                patch.object(
                    bench.sys,
                    "argv",
                    ["bench", "--mode", "tb_nvfp4_fp8", "--capture-sizes", "1", "8", "32",
                     "--wikitext-path", str(corpus), "--nvfp4-quant-mode", "four_over_six", "--tb-rule-only",
                     "--out-json", str(out)],
                ),
                patch.object(bench, "server_command", wraps=bench.server_command) as build,
                patch.object(bench.subprocess, "Popen") as popen,
                patch.object(bench, "wait_for_health", side_effect=timeout),
                self.assertRaisesRegex(TimeoutError, "still capturing"),
            ):
                bench.main()
            args = build.call_args.args[0]
            args.mode = "native_fp8"
            self.assertNotIn("--additional-config", bench.server_command(args))
            argv = popen.call_args.args[0]
            self.assertEqual(popen.call_args.kwargs["env"]["TENSORBRIDGE_DISABLE_PROFILES"], "1")
            self.assertEqual(json.loads(argv[argv.index("--hf-overrides") + 1]), {"head_dtype": "model"})
            self.assertEqual(argv[argv.index("--logprobs-mode") + 1], "raw_logprobs")
            self.assertEqual(argv[argv.index("--dtype") + 1], "bfloat16")
            options = json.loads(argv[argv.index("--additional-config") + 1])["tensorbridge"]
            self.assertEqual(options["nvfp4_quant_mode"], "four_over_six")
            self.assertEqual(options["preset"], "tb_nvfp4_fp8")
            self.assertNotIn("moe_comm_sms", options)
            self.assertEqual(argv.count("--compilation-config"), 1)
            self.assertEqual(
                json.loads(argv[argv.index("--compilation-config") + 1]), {"cudagraph_capture_sizes": [1, 8, 32]}
            )
            result = json.loads(out.read_text())
            self.assertEqual(result["available_kv_cache_memory_gib"], 16.25)
            self.assertEqual(result["gpu_kv_cache_size_tokens"], 100352)
            self.assertIsNone(result["kv_cache_blocks"])
            self.assertIn("startup_error", result)
            self.assertEqual(result["resolved_capture_sizes"], [1, 8, 32])

    def test_resolved_kv_blocks(self) -> None:
        log = "8 Engines: vllm cache_config_info with initialization after num_gpu_blocks is: 3,072"
        self.assertEqual(bench.kv_cache_calibration(log, None)["kv_cache_blocks"], 3072)
        metrics = io.BytesIO(
            b"# HELP vllm:cache_config_info Metadata\n"
            b'vllm:cache_config_info{engine="0",num_gpu_blocks="2048",num_gpu_blocks_override="None"} 1.0\n'
            b'vllm:cache_config_info{num_gpu_blocks="2047",engine="1"} 1.0\n'
        )
        with patch.object(serve_common.urllib.request, "urlopen", return_value=metrics) as get:
            self.assertEqual(bench.kv_cache_calibration("", 8123)["kv_cache_blocks"], 2047)
        get.assert_called_once_with("http://127.0.0.1:8123/metrics", timeout=5)

    def test_teacher_forcing(self) -> None:
        logprobs = [[None, -1.0], [None, -2.0, -3.0], [None, -4.0]]
        responses = [
            io.BytesIO(
                json.dumps(
                    {"choices": [{"logprobs": {"token_logprobs": values}}], "usage": {"prompt_tokens": len(values)}}
                ).encode()
            )
            for values in logprobs
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.txt"
            path.write_text("sample A\n \n中文\n\nif ready:\n    run()\n", encoding="utf-8")
            corpus = bench.read_quality_corpus(path)
        self.assertEqual(corpus, ("sample A", "中文", "if ready:\n    run()"))
        with (
            patch.object(bench, "QUALITY_PROMPTS", []),
            patch.object(bench.urllib.request, "urlopen", side_effect=responses) as send,
        ):
            result = bench.quality(8123, 8, corpus)
        self.assertEqual(result["scored_tokens"], 4)
        self.assertEqual(result["nll_per_token"], 2.5)
        self.assertAlmostEqual(result["ppl"], math.exp(2.5))
        self.assertEqual(result["corpus"], list(corpus))
        for passage, call in zip(corpus, send.call_args_list, strict=True):
            payload = json.loads(call.args[0].data)
            self.assertEqual(payload["prompt"], passage)
            self.assertEqual((payload["echo"], payload["logprobs"], payload["max_tokens"]), (True, 1, 0))


if __name__ == "__main__":
    unittest.main()
