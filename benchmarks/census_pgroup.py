#!/usr/bin/env python3
"""NVRTC census for `impls/sm90_w4a8_pgroup.cuh`: finds clean (zero-spill, non-serialized WGMMA) token tiles.

Run on a Hopper GPU (H100/H20) with `CUDA_CACHE_DISABLE=1` (set automatically below for workers):

    python benchmarks/census_pgroup.py --layouts masked contiguous --workers 16
    python benchmarks/census_pgroup.py --layouts dense --workers 16

Prints ready-to-paste C++ bitmasks for `clean_block_ms` (kind == 3, 2 math warpgroups) and
`clean_pgroup_one_warpgroup_block_ms` (1 math warpgroup) in `tensorbridge/csrc/heuristics/sm90_common.hpp`,
and verifies `compiled.smem_bytes == config_smem_bytes(...)` for every legal candidate.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Ensure NVRTC runs ptxas and emits register/spill logs instead of hitting the driver compute cache.
os.environ["CUDA_CACHE_DISABLE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _compile_one(
    task: tuple[int, int, int, str, list[int], str]
) -> tuple[tuple[int, int, int, str, tuple[int, ...]], bool, int, str]:
    """Worker: compile one pgroup kernel and return (key, is_clean, smem_bytes, detail)."""
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    n, k, num_experts, layout, cfg_list, bridge = task
    from tensorbridge.jit import codegen, compiler

    cfg = codegen.GemmConfig.from_list(cfg_list)
    key = (n, k, num_experts, layout, tuple(cfg_list))
    try:
        src = codegen.generate_pgroup(n, k, cfg, False, bridge, num_experts, layout)  # type: ignore[arg-type]
        compiled = compiler.build(src.source, src.name, compiler.INCLUDE_DIR)
    except Exception as exc:
        return key, False, 0, f"compile_error: {exc}"
    if compiled.ptxas is None:
        return key, False, compiled.smem_bytes, "missing_ptxas_log"
    clean = compiled.ptxas.spill_bytes == 0 and compiled.ptxas.stack_bytes == 0 and not compiled.ptxas.wgmma_serialized
    detail = (
        f"regs={compiled.ptxas.registers} spill={compiled.ptxas.spill_bytes} "
        f"stack={compiled.ptxas.stack_bytes} ser={compiled.ptxas.wgmma_serialized}"
    )
    return key, clean, compiled.smem_bytes, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", default=["masked", "contiguous"],
                        choices=["dense", "indexed", "contiguous", "masked"])
    parser.add_argument("--shapes", nargs="+", default=["4096x4096x32", "4608x5120x112", "6144x3584x112", "3584x3072x112"],
                        help="Representative NxKxE_local shapes to AND together (covers K_blocks divisible by 3..8)")
    parser.add_argument("--bridge", default="exact", choices=["exact", "fpma"])
    parser.add_argument("--max-stages", type=int, default=8, help="Max ring depth to census (3..16)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = parser.parse_args()

    parsed_shapes: list[tuple[int, int, int]] = []
    for raw in args.shapes:
        parts = raw.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Expected NxKxE_local, got {raw}")
        parsed_shapes.append((int(parts[0]), int(parts[1]), int(parts[2])))

    import torch

    from tensorbridge.jit import codegen, runtime

    runtime.load_launcher()
    optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin

    tasks: list[tuple[int, int, int, str, list[int], str]] = []
    expected_smem: dict[tuple[int, int, int, str, tuple[int, ...]], int] = {}

    for layout in args.layouts:
        layout_id = codegen.LAYOUT_IDS[layout]  # type: ignore[index]
        sk_modes = [False, True] if layout == "dense" else [False]
        wg_ctas = [(2, 1), (2, 2), (1, 2), (1, 3), (1, 4)]
        shape_list = [(n, k, 0) for n, k, _ in parsed_shapes] if layout == "dense" else parsed_shapes
        # Deduplicate dense shapes if E_local was the only difference.
        shape_list = list(dict.fromkeys(shape_list))
        for n, k, experts in shape_list:
            for math_wgs, ctas in wg_ctas:
                bns = [128] if math_wgs == 1 else [128, 256]
                max_bm = 64 if math_wgs == 1 else 192
                max_stage = min(args.max_stages, 6 if math_wgs == 1 else 16)
                for bn in bns:
                    for stages in range(3, max_stage + 1):
                        for use_sk in sk_modes:
                            if math_wgs == 1 and use_sk:
                                continue
                            for bm in range(8, max_bm + 1, 8):
                                cfg = codegen.GemmConfig(
                                    block_m=bm, num_stages=stages, use_stream_k=use_sk,
                                    m_fast_tile_order=False, block_k=128, block_n=bn,
                                    ctas_per_sm=ctas, stream_k_aligned=False, math_warpgroups=math_wgs,
                                )
                                reason = torch.ops.tensorbridge.illegal_reason(
                                    n, k, optin, cfg.to_list(), False, experts, layout_id, True,
                                )
                                if reason:
                                    continue
                                cfg_list = cfg.to_list()
                                key = (n, k, experts, layout, tuple(cfg_list))
                                expected_smem[key] = int(torch.ops.tensorbridge.config_smem_bytes(
                                    cfg_list, experts, layout_id, True,
                                ))
                                tasks.append((n, k, experts, layout, cfg_list, args.bridge))

    print(f"Compiling {len(tasks)} resource-legal pgroup kernels across {parsed_shapes} with {args.workers} workers...")
    results: dict[tuple[int, int, int, str, tuple[int, ...]], bool] = {}
    smem_mismatches = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for idx, (key, clean, smem_bytes, detail) in enumerate(pool.map(_compile_one, tasks), 1):
            results[key] = clean
            if smem_bytes and smem_bytes != expected_smem[key]:
                smem_mismatches += 1
                print(f"[SMEM MISMATCH] {key}: cubin={smem_bytes} vs selector={expected_smem[key]} ({detail})")
            if idx % 100 == 0 or idx == len(tasks):
                print(f"  progress: {idx}/{len(tasks)}")

    if smem_mismatches:
        print(f"WARNING: {smem_mismatches} SMEM mismatches detected!")

    def mask_for(layout: str, math_wgs: int, ctas: int, bn: int, stages: int, use_sk: bool) -> int:
        shape_list = list(dict.fromkeys(
            [(n, k, 0) for n, k, _ in parsed_shapes] if layout == "dense" else parsed_shapes
        ))
        mask = 0
        for bm in range(8, 193, 8):
            cfg = codegen.GemmConfig(
                block_m=bm, num_stages=stages, use_stream_k=use_sk, m_fast_tile_order=False,
                block_k=128, block_n=bn, ctas_per_sm=ctas, stream_k_aligned=False, math_warpgroups=math_wgs,
            )
            cfg_t = tuple(cfg.to_list())
            if all(results.get((n, k, experts, layout, cfg_t), False) for n, k, experts in shape_list):
                mask |= 1 << (bm // 8 - 1)
        return mask

    # Combine contiguous & masked by bitwise AND (matching sm90_common.hpp convention).
    combined_layouts = [l for l in args.layouts if l in ("contiguous", "masked")]
    if combined_layouts:
        print("\n=== pgroup 2-WG (contiguous & masked ANDed) ===")
        for stages in range(3, args.max_stages + 1):
            row = []
            for ctas in (1, 2):
                bn_masks = []
                for bn in (128, 256):
                    m = 0xFFFFFFFF
                    for l in combined_layouts:
                        m &= mask_for(l, 2, ctas, bn, stages, False)
                    bn_masks.append(f"0x{m:08x}")
                row.append(f"ctas={ctas}: BN128={bn_masks[0]} BN256={bn_masks[1]}")
            print(f"  stages={stages:2d}: " + " | ".join(row))

        print("\n=== clean_pgroup_one_warpgroup_block_ms (1-WG, BN=128, contiguous & masked ANDed) ===")
        for ctas in (2, 3, 4):
            vals = []
            for stages in range(3, min(args.max_stages, 6) + 1):
                m = 0xFFFFFFFF
                for l in combined_layouts:
                    m &= mask_for(l, 1, ctas, 128, stages, False)
                vals.append(f"0x{m:08x}")
            print(f"  // {ctas} CTAs/SM: {{{', '.join(vals)}}},")

    if "dense" in args.layouts:
        print("\n=== pgroup 2-WG (dense: non-SK / SK) ===")
        for stages in range(3, args.max_stages + 1):
            parts = []
            for ctas in (1, 2):
                for bn in (128, 256):
                    m0 = mask_for("dense", 2, ctas, bn, stages, False)
                    m1 = mask_for("dense", 2, ctas, bn, stages, True)
                    parts.append(f"c{ctas}n{bn}=(0x{m0:08x}, 0x{m1:08x})")
            print(f"  stages={stages:2d}: " + " ".join(parts))


if __name__ == "__main__":
    main()
