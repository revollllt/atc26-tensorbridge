"""CPU checks for the paired EP8 workload, distributed reference and report."""

import csv
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_moe_layer_ep8_compare as bench  # noqa: E402


def workload() -> bench.Case:
    return bench.Case(4, "uniform", 42, 8, 16, 128, 128, 2, 0, 3, 2, True, "H20", "test", "test")


def cpu_shape_worker(args: object, directory: Path, index: int) -> None:
    """Exercise the process boundary with one unsupported shape followed by a successful shape."""
    case = replace(workload(), tokens=index + 1)
    row = {"case": asdict(case), **asdict(bench.Result(bench.EXTERNAL[0], ""))}
    (directory / "case.json").write_text(json.dumps(row))
    if index == 0:
        (directory / "error-0.json").write_text(
            json.dumps({"rank": 0, "trace": "kernel rejected T=1", "status": "unsupported"})
        )
        return
    row["e2e_us"] = 17
    (directory / "result.jsonl").write_text(json.dumps(row))


class EP8Test(unittest.TestCase):
    def test_h20_contiguous_token_tiles(self) -> None:
        # Host-only: exercise the actual selector including its shared-memory and legal-config census.
        include = Path(__file__).resolve().parents[1] / "tensorbridge/csrc/heuristics"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "select.cpp")
            binary = source.with_suffix("")
            source.write_text("""
#include <iostream>
#include "sm90.hpp"
int main() {
    using namespace tensorbridge;
    const auto device = sm90::choose_device("NVIDIA H20", 78).device_class;
    for (const int t : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
        for (const auto [n, k] : {std::pair{4096, 6144}, std::pair{6144, 2048}}) {
            GemmDesc desc{8 * t, n, k, 78, 60 << 20, 227 * 1024, 32, GroupedLayout::Contiguous, false};
            std::cout << sm90::select_config(desc, *device, 0).block_m << ' ';
        }
    }
}
""")
            subprocess.run(["c++", "-std=c++17", "-I", str(include), str(source), "-o", str(binary)], check=True)
            selected = subprocess.check_output([str(binary)], text=True)
        expected = [8, 8, 8, 8, 8, 16, 24, 48, 64, 128]
        self.assertEqual(list(map(int, selected.split())), [bm for bm in expected for _ in range(2)])

    def test_all_backends(self) -> None:
        self.assertIn("humming_mxfp4_g128", bench.MODULAR)
        self.assertEqual(bench.MODULAR_TB, ("tb_nvfp4_fp8", "tb_nvfp4_fp8_g128"))
        self.assertEqual(bench.NVLINK, ("tb_nvfp4_megamoe",))
        for backend in bench.BACKENDS:
            args = bench.parse_args(["--backend", backend, "--deepgemm-path", "/path/to/deepgemm"])
            self.assertEqual(args.backend, backend)
            self.assertEqual(tuple(args.tokens), bench.TOKENS)
            self.assertEqual(args.max_tokens_per_rank, 256)
            self.assertEqual(args.shared_reserve_sms, 0)
            self.assertEqual(args.iters, 50)
        self.assertEqual(bench.parse_args(["--summarize", "a.jsonl"]).summarize, [Path("a.jsonl")])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
    def test_scatter_respects_local_destinations(self) -> None:
        from tensorbridge.quant import fp8

        # Only destinations 0 and 1 are local. Both come from the second four-row source tile.
        inputs = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        scatter = torch.tensor([*range(2, 14), 14, 0, 1, 15], device="cuda", dtype=torch.int32).view(8, 2)
        valid = torch.tensor([2], device="cuda", dtype=torch.int64)
        packed = torch.empty_like(inputs, dtype=torch.float8_e4m3fn)
        scales = torch.empty(8, 2, device="cuda")
        fp8.quantize_moe(inputs, "group_128", out=(packed, scales))
        for prequantized in (False, True):
            outputs = torch.zeros(16, 256, device="cuda").to(torch.float8_e4m3fn)
            output_scales = torch.full((16, 2), -1.0, device="cuda")
            if prequantized:
                fp8.scatter_fp8(packed, scales, scatter, out=(outputs, output_scales), num_valid_rows=valid)
            else:
                fp8.quantize_moe(inputs, "group_128", scatter_idx=scatter, num_valid_rows=valid,
                                 out=(outputs, output_scales))
            torch.testing.assert_close(outputs[:2].float(), packed[6:].float(), rtol=0, atol=0)
            torch.testing.assert_close(output_scales[:2], scales[6:])
            self.assertTrue((outputs[2:].float() == 0).all() and (output_scales[2:] == -1).all())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
    def test_interleaved_gate_up_quantize_moe_matches_standard(self) -> None:
        from tensorbridge.quant import fp8

        m, inter = 2048, 2048
        gate = torch.randn(m, inter, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(m, inter, device="cuda", dtype=torch.bfloat16)
        std_gu = torch.cat([gate, up], dim=1).contiguous()
        inter_gu = torch.cat(
            [gate.view(m, inter // 128, 128), up.view(m, inter // 128, 128)], dim=2,
        ).reshape(m, 2 * inter).contiguous()
        valid = torch.tensor([1537], device="cuda", dtype=torch.int64)

        out_std = torch.zeros(m, inter, device="cuda", dtype=torch.float8_e4m3fn)
        sf_std = torch.full((m, inter // 128), -1.0, device="cuda")
        out_int = torch.zeros_like(out_std)
        sf_int = torch.full_like(sf_std, -1.0)
        fp8.quantize_moe(std_gu, "group_128", activation=True, num_valid_rows=valid, out=(out_std, sf_std))
        fp8.quantize_moe(
            inter_gu, "group_128", activation=True, num_valid_rows=valid,
            out=(out_int, sf_int), interleaved_gate_up=True,
        )
        torch.testing.assert_close(out_int.float(), out_std.float(), rtol=0, atol=0)
        torch.testing.assert_close(sf_int, sf_std, rtol=0, atol=0)
        self.assertTrue((sf_int[1537:] == -1.0).all())

    def test_deterministic_routes(self) -> None:
        case = replace(workload(), tokens=800)
        cpu = torch.device("cpu")
        for routing in ("uniform", "zipf", "mixed_empty"):
            point = replace(case, routing=routing)
            x, ids, scores = bench.make_inputs(point, 0, cpu)
            torch.manual_seed(17)
            for actual, replay in zip((x, ids, scores), bench.make_inputs(point, 0, cpu), strict=True):
                self.assertTrue(torch.equal(actual, replay))
            self.assertTrue(torch.all(ids.sort(-1).values.diff() != 0))
            self.assertTrue(torch.allclose(scores.sum(-1), torch.ones(len(scores))))
            if routing == "zipf":
                self.assertGreater((ids == 0).sum(), 5 * (ids == 15).sum())
            else:
                self.assertGreater(ids.max().item(), 7)
            other = bench.make_inputs(point, 1, cpu)[0]
            self.assertEqual(len(other), 0 if routing == "mixed_empty" else case.tokens)
            self.assertFalse(torch.equal(x, other))

    def test_asymmetric_reference(self) -> None:
        case = replace(workload(), routing="mixed_empty")
        cpu = torch.device("cpu")
        sizes = bench.rank_sizes(case.tokens, case.routing, case.world)
        inputs = [bench.make_inputs(case, rank, cpu) for rank in range(8)]
        generator = torch.Generator().manual_seed(12)
        shards = [
            (bench.fp8_shard(2, 256, 128, generator, cpu), bench.fp8_shard(2, 128, 128, generator, cpu)) for _ in sizes
        ]
        # Compute the independent reference token by token across all 16 logical experts.
        experts = [(bench.dequant_expert(a, e), bench.dequant_expert(b, e)) for a, b in shards for e in range(2)]
        global_y = []
        for x, ids, weights in inputs:
            local_y = torch.zeros_like(x, dtype=torch.float)
            for token in range(len(x)):
                for slot, expert in enumerate(ids[token]):
                    gu, down = experts[expert]
                    gate, up = (gu @ x[token].float()).chunk(2)
                    local_y[token] += (down @ (torch.nn.functional.silu(gate) * up)) * weights[token, slot]
            global_y.append(local_y)
        contributions = []
        for rank, (gu, down) in enumerate(shards):
            columns = iter(zip(*inputs, strict=True))

            def gather(buffers: list[torch.Tensor], padded: torch.Tensor) -> None:
                for target, source in zip(buffers, next(columns), strict=True):
                    target.zero_()
                    target[: len(source)].copy_(source)
                torch.testing.assert_close(buffers[rank], padded)

            with (
                patch.object(bench.dist, "all_gather", side_effect=gather),
                patch.object(bench.dist, "all_reduce", side_effect=lambda values: contributions.append(values.clone())),
            ):
                local = bench.reference(inputs[rank], sizes, rank, gu, down, None)
                self.assertEqual(local.shape, inputs[rank][0].shape)
        torch.testing.assert_close(sum(contributions), torch.cat(global_y), rtol=1e-4, atol=1e-5)

    def test_rank_max_and_graph_fallback(self) -> None:
        case = workload()
        peer_times = iter([200.0, 50.0, 120.0])

        def rank_max(local: torch.Tensor, op: object, group: object) -> None:
            self.assertEqual(op, bench.dist.ReduceOp.MAX)
            local.fill_(max(local.item(), next(peer_times)))

        event = Mock()
        event.elapsed_time.side_effect = [0.1, 0.3, 0.2]  # milliseconds for two calls
        call = Mock(return_value=torch.ones(2))
        with (
            patch.object(torch.cuda, "Event", return_value=event),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "CUDAGraph", side_effect=RuntimeError("capture rejected")),
            patch.object(bench, "rank_errors", side_effect=lambda message, control: message),
            patch.object(bench.dist, "barrier"),
            patch.object(bench.dist, "all_reduce", side_effect=rank_max),
        ):
            rounds, mode, reason, actual = bench.measure(call, case, None)
        self.assertEqual(rounds, [200.0, 150.0, 120.0])
        self.assertEqual(statistics.median(rounds), 150.0)
        self.assertEqual(mode, "eager")
        self.assertIn("capture rejected", reason)
        self.assertIs(actual, call.return_value)

    def test_round_event_order(self) -> None:
        control = Mock(spec=bench.dist.ProcessGroup)
        for use_graph in (False, True):
            with self.subTest(graph=use_graph):
                case = replace(workload(), request_graph=use_graph, iters=50, rounds=2)
                log = []
                counts = torch.tensor([21, 5])
                routes = torch.tensor([7, 0])

                def launch() -> torch.Tensor:
                    log.append("launch")
                    return counts.add_(routes)

                def barrier(*, group: object, async_op: bool = False) -> Mock:
                    if group is control:
                        log.append("gloo")
                    else:
                        self.assertIs(group, bench.dist.group.WORLD)
                        self.assertTrue(async_op)
                        log.append("nccl")
                    return Mock(block_current_stream=lambda: log.append("stream wait"))

                start = Mock(record=lambda: log.append("start"), elapsed_time=lambda _: 1.0)
                stop = Mock(record=lambda: log.append("stop"), synchronize=lambda: log.append("stop sync"))
                with (
                    patch.object(torch.cuda, "Event", side_effect=[start, stop]),
                    patch.object(torch.cuda, "synchronize", side_effect=lambda: log.append("host sync")),
                    patch.object(torch.cuda, "CUDAGraph", return_value=Mock(replay=launch)),
                    patch.object(torch.cuda, "graph", return_value=nullcontext()),
                    patch.object(bench, "rank_errors", side_effect=lambda msg, _: msg),
                    patch.object(bench.dist, "barrier", side_effect=barrier),
                    patch.object(bench.dist, "all_reduce", side_effect=lambda *a, **kw: log.append("reduce")),
                ):
                    times, mode, _, _ = bench.measure(launch, case, control, recv_check=(counts, routes))
                one_round = [
                    "nccl",
                    "stream wait",
                    *(["launch"] * 3),
                    "start",
                    *(["launch"] * 50),
                    "stop",
                    "stop sync",
                    "reduce",
                ]
                self.assertEqual(log[log.index("nccl") :], one_round * 2)
                self.assertEqual(times, [20.0, 20.0])
                self.assertEqual(mode, "graph" if use_graph else "eager")

    def test_external_sweep_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep.jsonl"
            args = bench.parse_args(
                ["--backend", bench.EXTERNAL[0], "--deepgemm-path", tmp, "--tokens", "1", "2", "--output", str(out)]
            )
            with (
                patch.object(bench, "external_point", cpu_shape_worker),
                patch.object(bench.dist, "init_process_group"),
                patch.object(bench.dist, "get_rank", return_value=0),
                patch.object(bench.dist, "get_world_size", return_value=8),
                patch.object(bench.dist, "broadcast_object_list"),
                patch.object(bench.dist, "barrier"),
                patch.object(bench.dist, "all_reduce"),
            ):
                bench.external_sweep(args)
            first, second = [json.loads(row) for row in out.read_text().splitlines()]
            self.assertEqual(first["status"], "unsupported")
            self.assertIn("kernel rejected T=1", first["error"])
            self.assertEqual(second["e2e_us"], 17)

    def test_summary_alignment(self) -> None:
        case = workload()
        marlin = bench.Result("marlin_nvfp4", bench.COLLECTIVES[0], e2e_us=30, e2e_mode="graph")
        humming = bench.Result("humming_nvfp4", bench.COLLECTIVES[1], e2e_us=24, e2e_mode="graph")
        tb = bench.Result("tb_nvfp4_megamoe", "", e2e_us=10, e2e_mode="graph")
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "table"
            path = prefix.with_suffix(".jsonl")
            measurements = [
                (case, marlin),
                (case, replace(marlin, backend="humming_mxfp4_g128", e2e_us=22)),
                (case, humming),
                (case, tb),
                (replace(case, seed=43), replace(tb, e2e_us=7)),
                (case, replace(tb, backend="tb_nvfp4_fp8_g128", e2e_mode="eager")),
                (case, bench.Result(bench.EXTERNAL[0], "", status="unsupported", error="capacity")),
            ]
            path.write_text("".join(json.dumps({"case": asdict(c), **asdict(r)}) + "\n" for c, r in measurements))
            bench.summarize([path], prefix)
            with prefix.with_suffix(".csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertIn("kernel_only_us", prefix.with_suffix(".md").read_text())
            pairs = [r for r in rows if r["backend"] == "tb_nvfp4_megamoe" and r["seed"] == "42"]
            self.assertEqual([r["vs_humming_mxfp4_g128"] for r in pairs], ["2.2", ""])
            self.assertEqual([(r["vs_marlin_nvfp4"], r["vs_humming_nvfp4"]) for r in pairs], [("3.0", ""), ("", "2.4")])
            self.assertTrue(
                all(not r["vs_marlin_nvfp4"] for r in rows if r["seed"] == "43" or r["backend"] == "tb_nvfp4_fp8_g128")
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bench.summarize([path, path], prefix)


if __name__ == "__main__":
    unittest.main()
