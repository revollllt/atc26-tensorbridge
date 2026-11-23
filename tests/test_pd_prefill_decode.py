"""Unit tests for benchmarks/bench_pd_prefill_decode.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_pd_prefill_decode as pd  # noqa: E402
from serve_common import Completion  # noqa: E402


class TestPdPrefillDecode(unittest.TestCase):
    def test_prefix_split_block_aligned(self) -> None:
        expected = {
            50_000: (44_992, 5_008),
            80_000: (72_000, 8_000),
            160_000: (144_000, 16_000),
            320_000: (288_000, 32_000),
        }
        for isl, (exp_prefix, exp_uncached) in expected.items():
            self.assertEqual(pd.compute_prefix_split(isl, 0.0, 64), (0, isl))
            prefix, uncached = pd.compute_prefix_split(isl, 0.9, 64)
            self.assertEqual((prefix, uncached), (exp_prefix, exp_uncached))
            self.assertEqual(prefix % 64, 0)
            self.assertEqual(prefix + uncached, isl)
        with self.assertRaises(ValueError):
            pd.compute_prefix_split(80_000, 1.0, 64)
        with self.assertRaises(ValueError):
            pd.compute_prefix_split(80_000, -0.1, 64)

    def test_make_unique_tokens_disjoint(self) -> None:
        a = pd.make_unique_tokens(100, 128)
        b = pd.make_unique_tokens(101, 128)
        self.assertEqual(len(a), 128)
        self.assertNotEqual(a[:64], b[:64])
        self.assertTrue(all(pd.VOCAB_LOW <= t < pd.VOCAB_LOW + pd.VOCAB_SPAN for t in a))

    def test_parse_and_diff_prefix_cache_metrics(self) -> None:
        before_txt = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries
vllm:prefix_cache_queries_total{engine="0",model_name="glm53"} 1000.0
vllm:prefix_cache_queries_total{engine="1",model_name="glm53"} 2000.0
vllm:prefix_cache_hits_total{engine="0",model_name="glm53"} 0.0
vllm:prefix_cache_hits_total{engine="1",model_name="glm53"} 0.0
"""
        after_txt = """
vllm:prefix_cache_queries_total{engine="0",model_name="glm53"} 81000.0
vllm:prefix_cache_queries_total{engine="1",model_name="glm53"} 82000.0
vllm:prefix_cache_hits_total{engine="0",model_name="glm53"} 72000.0
vllm:prefix_cache_hits_total{engine="1",model_name="glm53"} 72000.0
"""
        before = pd.parse_prefix_cache_metrics(before_txt)
        after = pd.parse_prefix_cache_metrics(after_txt)
        delta = pd.diff_prefix_cache(before, after)
        self.assertEqual(delta["queries"], 160_000.0)
        self.assertEqual(delta["hits"], 144_000.0)
        self.assertAlmostEqual(float(delta["hit_rate"]), 0.9)  # type: ignore[arg-type]

        # Single-engine without engine label or _total suffix
        single = pd.parse_prefix_cache_metrics("vllm:prefix_cache_queries 500\nvllm:prefix_cache_hits 450\n")
        self.assertEqual(single["queries"], {"0": 500.0})
        self.assertEqual(single["hits"], {"0": 450.0})

    def test_prefill_and_decode_server_commands(self) -> None:
        parser = pd.build_parser()
        p_args = parser.parse_args(
            ["--bench", "prefill", "--mode", "tb_nvfp4_megamoe", "--out-json", "/tmp/p.json"]
        )
        pd.resolve_defaults(p_args)
        pd.validate_args(p_args, parser)
        p_cmd = pd.build_server_command(p_args)
        self.assertIn("--enable-prefix-caching", p_cmd)
        self.assertNotIn("--kv-transfer-config", p_cmd)
        self.assertNotIn("--decode-context-parallel-size", p_cmd)
        self.assertEqual(p_args.dp, 2)
        self.assertEqual(p_args.tp, 4)
        self.assertEqual(p_args.dcp, 1)
        self.assertGreaterEqual(p_args.max_model_len, 321_024)

        d_args = parser.parse_args(
            ["--bench", "decode", "--mode", "tb_nvfp4_fp8_g128", "--shared-reserve-sms", "4", "--out-json", "/tmp/d.json"]
        )
        pd.resolve_defaults(d_args)
        pd.validate_args(d_args, parser)
        d_cmd = pd.build_server_command(d_args)
        tb_opts = json.loads(d_cmd[d_cmd.index("--additional-config") + 1])["tensorbridge"]
        self.assertEqual(tb_opts["moe_shared_reserve_sms"], 4)
        self.assertNotIn("moe_comm_sms", tb_opts)
        self.assertIn("--no-enable-prefix-caching", d_cmd)
        self.assertIn("--kv-transfer-config", d_cmd)
        self.assertIn("DecodeBenchConnector", d_cmd[d_cmd.index("--kv-transfer-config") + 1])
        self.assertIn("--decode-context-parallel-size", d_cmd)
        self.assertEqual(d_cmd[d_cmd.index("--decode-context-parallel-size") + 1], "4")
        self.assertIn("--dcp-comm-backend", d_cmd)
        self.assertEqual(d_cmd[d_cmd.index("--dcp-comm-backend") + 1], "ag_rs")
        self.assertEqual(d_args.decode_isl, 80_000)
        self.assertEqual(d_args.decode_osl, 600)

        bad = parser.parse_args(
            ["--bench", "decode", "--dcp", "4", "--kv-cache-dtype", "bfloat16", "--out-json", "/tmp/x.json"]
        )
        pd.resolve_defaults(bad)
        with self.assertRaises(SystemExit):
            pd.validate_args(bad, parser)

    def test_summarize_prefill_and_decode_and_markdown(self) -> None:
        p_samples = [
            Completion(start=10.0, arrivals=[11.2], end=11.21, token_ids=[42], text="x", tokens_per_frame=[1],
                       prompt_tokens=80_000),
            Completion(start=12.0, arrivals=[13.4], end=13.41, token_ids=[42], text="x", tokens_per_frame=[1],
                       prompt_tokens=80_000),
        ]
        p_pt = pd.summarize_prefill_samples(
            p_samples,
            isl=80_000,
            target_hit_rate=0.9,
            prefix_len=72_000,
            uncached_len=8_000,
            concurrency=1,
            cache_delta={"queries": 160_000.0, "hits": 144_000.0, "hit_rate": 0.9},
            warmup_waves=1,
        )
        self.assertAlmostEqual(float(p_pt["ttft_ms"]["mean"]), 1300.0)  # type: ignore[index]
        self.assertAlmostEqual(float(p_pt["observed_hit_rate"]), 0.9)  # type: ignore[arg-type]

        # 2 concurrent decode requests, OSL=6, 20 ms per token
        d_samples = [
            Completion(
                start=0.0,
                arrivals=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
                end=0.20,
                token_ids=[1, 2, 3, 4, 5, 6],
                text="abcdef",
                tokens_per_frame=[1] * 6,
                prompt_tokens=80_000,
            ),
            Completion(
                start=0.01,
                arrivals=[0.11, 0.13, 0.15, 0.17, 0.19, 0.21],
                end=0.21,
                token_ids=[1, 2, 3, 4, 5, 6],
                text="abcdef",
                tokens_per_frame=[1] * 6,
                prompt_tokens=80_000,
            ),
        ]
        d_pt = pd.summarize_decode_samples(d_samples, isl=80_000, osl=6, concurrency=2, rounds=1)
        self.assertAlmostEqual(float(d_pt["tpot_ms"]["mean"]), 20.0, places=5)  # type: ignore[index]
        self.assertAlmostEqual(float(d_pt["itl_ms"]["p50"]), 20.0, places=5)  # type: ignore[index]
        self.assertEqual(d_pt["coalesced_token_rate"], 0.0)
        self.assertIsNotNone(d_pt["overlap"])
        self.assertIsNotNone(d_pt["overlap_tpot_ms"])

        # Frame-0 coalescing (3 tokens in first SSE frame, then 3 single-token frames at 20ms each) must not deflate TPOT.
        coalesced_sample = Completion(
            start=0.0,
            arrivals=[0.10, 0.12, 0.14, 0.16],
            end=0.16,
            token_ids=[1, 2, 3, 4, 5, 6],
            text="abcdef",
            tokens_per_frame=[3, 1, 1, 1],
            prompt_tokens=80_000,
        )
        c_pt = pd.summarize_decode_samples([coalesced_sample], isl=80_000, osl=6, concurrency=1, rounds=1)
        self.assertAlmostEqual(float(c_pt["tpot_ms"]["mean"]), 20.0, places=5)  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp:
            p_json = Path(tmp) / "prefill.json"
            d_json = Path(tmp) / "decode.json"
            csv_path = Path(tmp) / "decode.csv"
            p_json.write_text(json.dumps({
                "bench": "prefill", "mode": "tb_nvfp4_fp8_g128", "dp": 2, "tp": 4, "dcp": 1,
                "kv_cache_dtype": "fp8", "points": [p_pt],
            }))
            d_json.write_text(json.dumps({
                "bench": "decode", "mode": "tb_nvfp4_fp8_g128", "dp": 2, "tp": 4, "dcp": 4,
                "kv_cache_dtype": "fp8", "points": [d_pt],
            }))
            md = pd.summarize_files([p_json, d_json])
            self.assertIn("### Prefill Bench (TTFT)", md)
            self.assertIn("### Decode Bench (TPOT)", md)
            self.assertIn("80,000+6", md)
            pd.summarize_files([d_json], out_csv=csv_path)
            self.assertTrue(csv_path.is_file())


if __name__ == "__main__":
    unittest.main()
