#!/usr/bin/env python3
"""Two-stage offline full-space MoE config sweeper for model-specific profiles (Route A2).

Sweeps `gate_up` (group_128 / pgroup) and `down` (per_token / ptoken and/or group_128 / pgroup)
across our target MoE models (GLM 5.3, DeepSeek v4.1 Flash, Kimi K3.1 Pro, Muse-Spark-Open-816B-A42B)
and writes piecewise `(max_m, GemmConfig)` profiles consumable by `tensorbridge.jit.profiles`.

Example (full GPU SMs, then Green Context compute SM budgets):
    TENSORBRIDGE_NUM_SMS=132 python benchmarks/sweep_moe_profiles.py --models glm53 dsv41_flash kimi_k31_pro muse_spark_816b --ep 8 16 --output tensorbridge/profiles/h100_moe.json --explore-all
    TENSORBRIDGE_NUM_SMS=112 python benchmarks/sweep_moe_profiles.py --models glm53 dsv41_flash kimi_k31_pro muse_spark_816b --ep 8 16 --output tensorbridge/profiles/h100_moe.json --merge --explore-all
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    num_experts: int
    top_k: int
    gate_up_n: int
    gate_up_k: int
    down_n: int
    down_k: int


TARGET_MODELS: dict[str, ModelSpec] = {
    "glm53": ModelSpec("glm53", 256, 8, 4096, 6144, 6144, 2048),
    "dsv41_flash": ModelSpec("dsv41_flash", 384, 6, 4608, 5120, 5120, 2304),
    "kimi_k31_pro": ModelSpec("kimi_k31_pro", 896, 16, 6144, 3584, 3584, 3072),
    "muse_spark_816b": ModelSpec("muse_spark_816b", 256, 8, 8192, 4096, 4096, 4096),
}

DECODE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PREFILL_TOKENS = (512, 2048, 8192, 16384)


def _compile_candidate(
    task: tuple[int, int, int, str, list[int], str, bool, bool]
) -> tuple[list[int], bool, str]:
    """Worker: NVRTC-compile one candidate; optionally reject spills / serialized WGMMAs."""
    n, k, num_experts, layout, cfg_list, bridge, group_scales, require_clean = task
    if require_clean:
        os.environ["CUDA_CACHE_DISABLE"] = "1"
    from tensorbridge.jit import codegen, compiler

    cfg = codegen.GemmConfig.from_list(cfg_list)
    try:
        if group_scales:
            src = codegen.generate_pgroup(n, k, cfg, False, bridge, num_experts, layout)  # type: ignore[arg-type]
        else:
            src = codegen.generate(n, k, cfg, num_experts, layout, bridge)  # type: ignore[arg-type]
        compiled = compiler.build(src.source, src.name, compiler.INCLUDE_DIR)
    except Exception as exc:
        return cfg_list, False, str(exc)
    if require_clean and compiled.ptxas is not None:
        if compiled.ptxas.spill_bytes > 0 or compiled.ptxas.stack_bytes > 0 or compiled.ptxas.wgmma_serialized:
            return cfg_list, False, (
                f"unclean(spill={compiled.ptxas.spill_bytes},stack={compiled.ptxas.stack_bytes},"
                f"ser={compiled.ptxas.wgmma_serialized})"
            )
    return cfg_list, True, ""


def _filter_candidates_for_m(
    configs: list,
    avg_rows_per_expert: float,
    layout: str,
    max_candidates: int,
) -> list:
    """Prune obviously mismatched BM values so Stage 1 stays fast while preserving full CTA/BN/BK/stage freedom."""
    if len(configs) <= max_candidates:
        return configs
    # Keep small BM for Decode and larger BM for Prefill without pruning across BN/BK/CTA/WG boundaries.
    if layout == "masked" and avg_rows_per_expert <= 32:
        allowed_bm = {8, 16, 24, 32, 48, 64}
    elif layout == "masked":
        allowed_bm = {16, 32, 48, 64, 80, 96, 128}
    elif avg_rows_per_expert <= 128:
        allowed_bm = {32, 48, 64, 80, 96, 128, 160}
    else:
        allowed_bm = {64, 80, 96, 112, 128, 144, 160, 176, 192}
    filtered = [c for c in configs if c.block_m in allowed_bm]
    if len(filtered) > max_candidates:
        # Prefer stages in {4, 5, 6} first, then 3 and 7-8.
        filtered.sort(key=lambda c: (0 if c.num_stages in (4, 5, 6) else 1, c.m_fast_tile_order, c.block_m))
        filtered = filtered[:max_candidates]
    return filtered


def _prepare_moe_inputs(
    shape,
    weight_copies: list,
    layout: str,
    group_scales: bool,
):
    """Build inputs, scales, and routing tensors for one MoeShape (equal or zipf)."""
    import common
    import torch

    from tensorbridge.gemm import ContiguousRouting, MaskedRouting
    from tensorbridge.quant import fp8

    routes = common.route(shape)
    if routes.rows == 0:
        return None
    n, k, local_e = shape.n, shape.k, shape.local_experts
    torch.manual_seed(shape.seed + 17)
    if layout == "masked":
        cap = routes.capacity
        total_rows = local_e * cap
        x_hp = torch.randn(total_rows, k, device="cuda", dtype=torch.bfloat16)
        counts = routes.counts.to(device="cuda", dtype=torch.int32)
        if group_scales:
            a_buf = torch.empty((total_rows, k), device="cuda", dtype=torch.float8_e4m3fn)
            s_buf = torch.empty((total_rows, k // 128), device="cuda", dtype=torch.float32)
            a, a_scale = fp8.quantize_moe(x_hp, "group_128", expert_counts=counts, out=(a_buf, s_buf))
        else:
            a, a_scale = fp8.quantize(x_hp)
        routing = MaskedRouting(counts)
        out = torch.empty((total_rows, n), device="cuda", dtype=torch.bfloat16)
    else:
        total_rows = routes.rows
        x_hp = torch.randn(total_rows, k, device="cuda", dtype=torch.bfloat16)
        offsets = routes.bounds.to(device="cuda", dtype=torch.int64)
        if group_scales:
            a_buf = torch.empty((total_rows, k), device="cuda", dtype=torch.float8_e4m3fn)
            s_buf = torch.empty((total_rows, k // 128), device="cuda", dtype=torch.float32)
            a, a_scale = fp8.quantize_moe(x_hp, "group_128", out=(a_buf, s_buf))
        else:
            a, a_scale = fp8.quantize(x_hp)
        routing = ContiguousRouting(offsets)
        out = torch.empty((total_rows, n), device="cuda", dtype=torch.bfloat16)
    return a, a_scale, routing, out, routes.rows


def _prepare_weights(shape, bridge: str, group_scales: bool, copies: int) -> list:
    import common
    import torch

    from tensorbridge.quant import nvfp4

    n, k = shape.n, shape.k
    parts = []
    for chunk in common.moe_weight_chunks(shape):
        packed, scales, global_scale = nvfp4.quantize(chunk.reshape(-1, k))
        part = nvfp4.prepare_experts(
            packed.view(-1, n, k // 2),
            scales.view(-1, n, k // 16),
            global_scale.expand(chunk.size(0)),
            bridge=bridge,  # type: ignore[arg-type]
        )
        if group_scales:
            e_chunk = chunk.size(0)
            part = dataclasses.replace(
                part,
                scale=part.scale.view(e_chunk, k // 128, 8, n).permute(0, 1, 3, 2).contiguous(),
            )
        parts.append(part)
    base = nvfp4.NVFP4Experts(
        torch.cat([p.weight for p in parts]),
        torch.cat([p.scale for p in parts]),
        torch.cat([p.global_scale for p in parts]),
        torch.zeros_like(parts[0].locks),
        bridge=bridge,  # type: ignore[arg-type]
    )
    weight_copies = [base]
    for _ in range(copies - 1):
        weight_copies.append(dataclasses.replace(
            base,
            weight=base.weight.clone(),
            scale=base.scale.clone(),
            global_scale=base.global_scale.clone(),
            locks=torch.zeros_like(base.locks),
        ))
    return weight_copies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(TARGET_MODELS.keys()), choices=list(TARGET_MODELS.keys()))
    parser.add_argument("--projections", nargs="+", default=["gate_up", "down"], choices=["gate_up", "down"])
    parser.add_argument("--ep", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--layouts", nargs="+", default=["masked", "contiguous"], choices=["masked", "contiguous"])
    parser.add_argument("--down-quant", default="per_token", choices=["per_token", "group_128", "both"])
    parser.add_argument("--bridge", default="exact", choices=["exact", "fpma"])
    parser.add_argument("--explore-all", action="store_true",
                        help="Enumerate resource-legal configs (check_clean=False) and filter spills via NVRTC")
    parser.add_argument("--max-stage1", type=int, default=96, help="Max candidates per M bucket in Stage 1")
    parser.add_argument("--top-k", type=int, default=8, help="Candidates promoted from Stage 1 to Stage 2")
    parser.add_argument("--skew-weight", type=float, default=0.5, help="Weight on zipf distribution in Stage 2 (0..1)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--output", type=Path, required=True, help="Output profile JSON path")
    parser.add_argument("--merge", action="store_true", help="Merge into existing --output JSON file if present")
    args = parser.parse_args()

    import common
    import torch
    from common import MoeShape

    from tensorbridge.gemm import fp8_nvfp4_grouped_gemm
    from tensorbridge.jit import autotune, codegen, profiles, runtime

    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    num_sms = runtime.get_num_sms(device)
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    print(f"Device: {device_name} | active SMs: {num_sms} | L2: {l2_bytes / (1024**2):.1f} MB")

    # Disable loaded profiles during sweep so baseline comparison measures `sm90_rule.hpp` cleanly.
    profiles.clear_profiles(reload_defaults=False)

    existing_entries: dict[profiles.ProfileKey, list[profiles.ProfileSegment]] = {}
    if args.merge and args.output.is_file():
        raw = json.loads(args.output.read_text(encoding="utf-8"))
        for entry in raw.get("entries", []):
            key: profiles.ProfileKey = (
                entry["device_name"], int(entry["num_sms"]), int(entry["n"]), int(entry["k"]),
                int(entry["num_experts"]), entry["layout"], bool(entry["group_scales"]), entry["bridge"],
            )
            existing_entries[key] = [
                profiles.ProfileSegment(int(s["max_m"]), codegen.GemmConfig.from_list(s["config"]))
                for s in entry["segments"]
            ]

    compile_cache: dict[tuple[int, int, int, str, tuple[int, ...], bool], bool] = {}
    ctx = multiprocessing.get_context("spawn")
    summary_rows: list[str] = []

    for model_name in args.models:
        spec = TARGET_MODELS[model_name]
        for proj in args.projections:
            n = spec.gate_up_n if proj == "gate_up" else spec.down_n
            k = spec.gate_up_k if proj == "gate_up" else spec.down_k
            quant_modes = [True] if proj == "gate_up" else (
                [False] if args.down_quant == "per_token" else [True] if args.down_quant == "group_128" else [False, True]
            )
            for ep in args.ep:
                local_e = spec.num_experts // ep
                weight_bytes = local_e * n * k * 9 / 16
                copies = max(2, autotune.weight_copies(weight_bytes, l2_bytes))
                for group_scales in quant_modes:
                    # Prepare weight copies once per (model, proj, ep, group_scales).
                    dummy_shape = MoeShape(proj, 1, n, k, spec.num_experts, spec.top_k, 1, "equal", 0, ep, 0, "masked", 128)
                    weight_copies = _prepare_weights(dummy_shape, args.bridge, group_scales, copies)

                    for layout in args.layouts:
                        tokens_list = DECODE_TOKENS if layout == "masked" else PREFILL_TOKENS
                        cap = 128 * ep if layout == "masked" else 0
                        all_cfgs = runtime.all_legal_configs(
                            1, n, k, check_clean=not args.explore_all, device=device,
                            num_experts=local_e, layout=layout, group_scales=group_scales,
                        )
                        measured_points: list[tuple[int, codegen.GemmConfig]] = []
                        print(f"\n=== {model_name} {proj} EP{ep} (E_local={local_e}, N={n}, K={k}) "
                              f"{layout} g128={group_scales} ({len(all_cfgs)} legal cfgs) ===")

                        for tokens in tokens_list:
                            shape_eq = MoeShape(proj, 1, n, k, spec.num_experts, spec.top_k, tokens, "equal", 0, ep, 0, layout, cap)
                            shape_sk = MoeShape(proj, 1, n, k, spec.num_experts, spec.top_k, tokens, "zipf", 0, ep, 0, layout, cap)
                            inp_eq = _prepare_moe_inputs(shape_eq, weight_copies, layout, group_scales)
                            if inp_eq is None:
                                continue
                            a_eq, sfa_eq, rout_eq, out_eq, rows_eq = inp_eq
                            inp_sk = _prepare_moe_inputs(shape_sk, weight_copies, layout, group_scales)

                            rule_cfg = runtime.select_config(
                                rows_eq, n, k, local_e, layout, device=device,
                                bridge=args.bridge, group_scales=group_scales,
                            )
                            avg_rpe = rows_eq / max(local_e, 1)
                            stage1_cfgs = _filter_candidates_for_m(all_cfgs, avg_rpe, layout, args.max_stage1)
                            if rule_cfg not in stage1_cfgs:
                                stage1_cfgs = [rule_cfg] + stage1_cfgs

                            # Parallel NVRTC compile & spill filter.
                            to_compile = []
                            for c in stage1_cfgs:
                                ckey = (n, k, local_e, layout, tuple(c.to_list()), group_scales)
                                if ckey not in compile_cache:
                                    to_compile.append((n, k, local_e, layout, c.to_list(), args.bridge,
                                                       group_scales, args.explore_all))
                            if to_compile:
                                with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                                    for cfg_list, ok, _reason in pool.map(_compile_candidate, to_compile):
                                        ckey = (n, k, local_e, layout, tuple(cfg_list), group_scales)
                                        compile_cache[ckey] = ok

                            valid_cfgs = [
                                c for c in stage1_cfgs
                                if compile_cache.get((n, k, local_e, layout, tuple(c.to_list()), group_scales), False)
                            ]
                            if rule_cfg not in valid_cfgs:
                                valid_cfgs.insert(0, rule_cfg)

                            # Stage 1: fast screening on equal routing.
                            calls_s1 = {}
                            for c in valid_cfgs:
                                def _make_call(cfg=c):
                                    return lambda ci: fp8_nvfp4_grouped_gemm(
                                        a_eq, sfa_eq, weight_copies[ci], rout_eq, rows_eq,
                                        out=out_eq, config=cfg, group_scales=group_scales,
                                    )
                                calls_s1[c] = _make_call()
                            t_s1 = autotune.time_configs(calls_s1, copies, rounds=2, round_us=600)
                            ranked_s1 = sorted(valid_cfgs, key=lambda c: statistics.median(t_s1[c]))
                            top_cfgs = ranked_s1[:args.top_k]
                            if rule_cfg not in top_cfgs:
                                top_cfgs.append(rule_cfg)

                            # Stage 2: robust bidirectional timing on equal + zipf (skewed).
                            calls_eq = {c: calls_s1[c] for c in top_cfgs}
                            t_eq = autotune.time_configs(calls_eq, copies, rounds=5, round_us=1500)
                            med_eq = {c: statistics.median(t_eq[c]) for c in top_cfgs}

                            if inp_sk is not None and args.skew_weight > 0:
                                a_sk, sfa_sk, rout_sk, out_sk, rows_sk = inp_sk
                                calls_sk = {}
                                for c in top_cfgs:
                                    def _make_call_sk(cfg=c):
                                        return lambda ci: fp8_nvfp4_grouped_gemm(
                                            a_sk, sfa_sk, weight_copies[ci], rout_sk, rows_sk,
                                            out=out_sk, config=cfg, group_scales=group_scales,
                                        )
                                    calls_sk[c] = _make_call_sk()
                                t_sk = autotune.time_configs(calls_sk, copies, rounds=5, round_us=1500)
                                med_sk = {c: statistics.median(t_sk[c]) for c in top_cfgs}
                            else:
                                med_sk = med_eq

                            w_sk = args.skew_weight
                            score = {c: (1.0 - w_sk) * med_eq[c] + w_sk * med_sk[c] for c in top_cfgs}
                            best_cfg = min(top_cfgs, key=lambda c: score[c])
                            # Keep rule_cfg unless best_cfg improves by > 1%.
                            if score[best_cfg] >= score[rule_cfg] * 0.99:
                                best_cfg = rule_cfg

                            speedup = score[rule_cfg] / score[best_cfg]
                            measured_points.append((rows_eq, best_cfg))
                            line = (
                                f"  T={tokens:5d} rows={rows_eq:6d} | rule={score[rule_cfg]:7.2f}us "
                                f"best={score[best_cfg]:7.2f}us ({speedup:5.3f}x) | "
                                f"BM={best_cfg.block_m} BN={best_cfg.block_n} BK={best_cfg.block_k} "
                                f"S={best_cfg.num_stages} CTA={best_cfg.ctas_per_sm} WG={best_cfg.math_warpgroups} "
                                f"mfast={int(best_cfg.m_fast_tile_order)}"
                            )
                            print(line)
                            summary_rows.append(f"{model_name}/{proj}/EP{ep}/{layout}/g128={int(group_scales)}: {line.strip()}")

                        if measured_points:
                            pkey: profiles.ProfileKey = (
                                device_name, num_sms, codegen.pad_to_unit(n), codegen.pad_to_unit(k),
                                local_e, layout, group_scales, args.bridge,  # type: ignore[arg-type]
                            )
                            existing_entries[pkey] = profiles.compress_segments(
                                measured_points, unbounded_tail=(layout == "contiguous"),
                            )
                    del weight_copies
                    torch.cuda.empty_cache()

    profiles.save_profile_file(
        args.output,
        list(existing_entries.items()),
        metadata={
            "device_name": device_name,
            "num_sms": num_sms,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": args.models,
            "skew_weight": args.skew_weight,
        },
    )
    print(f"\nSaved {len(existing_entries)} profile entries to {args.output}")


if __name__ == "__main__":
    main()
