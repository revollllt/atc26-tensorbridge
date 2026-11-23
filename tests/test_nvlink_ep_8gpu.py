"""8-GPU NVLink Symmetric Memory EP correctness, CUDA Graph, real-weight ModelOpt check & latency benchmark.

Run on 8xH20:
    torchrun --standalone --nproc_per_node=8 tests/test_nvlink_ep_8gpu.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _dequant_nvfp4_modelopt(packed: torch.Tensor, scale: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """Reference dequantization for ModelOpt NVFP4 `(packed [..., N, K//2], scale [..., N, K//16], gs)`."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device, dtype=torch.float32,
    )
    lo = lut[(packed & 0x0F).to(torch.int64)]
    hi = lut[((packed >> 4) & 0x0F).to(torch.int64)]
    vals = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    if gs.dim() == 2 and gs.size(1) == 2:
        half = scale.size(-2) // 2
        gs_exp = torch.cat([
            gs[:, 0:1].expand(-1, half),
            gs[:, 1:2].expand(-1, half),
        ], dim=1).unsqueeze(-1).to(torch.float32)
        s = scale.to(torch.float32) * gs_exp
    else:
        s = scale.to(torch.float32) * gs.reshape(*gs.shape, 1, 1).to(torch.float32)
    return (vals.reshape(*vals.shape[:-1], -1, 16) * s.unsqueeze(-1)).reshape(vals.shape)


def check_real_glm53_layer(ckpt_dir: Path | None, device: torch.device) -> None:
    """Load 4 real experts from layer 3 of a block-128 FP8 checkpoint (`TENSORBRIDGE_MODEL_PATH`),
    convert on-the-fly to ModelOpt NVFP4, and verify quantization fidelity & VRAM footprint."""
    if ckpt_dir is None:
        print("[real-weight] Skipping: set TENSORBRIDGE_MODEL_PATH to run real-weight check")
        return
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"[real-weight] Skipping: {index_path} not found")
        return
    from safetensors import safe_open

    from tensorbridge.quant import nvfp4

    weight_map = json.loads(index_path.read_text())["weight_map"]
    # Detect GLM-5.3 MoE expert parameter naming in layer 3
    prefix = "model.layers.3.mlp.experts.0."
    matches = [k for k in weight_map if k.startswith(prefix)]
    if not matches:
        sample = [k for k in weight_map if "layers.3.mlp" in k][:10]
        print(f"[real-weight] Unexpected layer 3 MLP keys: {sample}")
        return

    num_test_e = 4
    w_gate_list, s_gate_list = [], []
    w_up_list, s_up_list = [], []
    w_down_list, s_down_list = [], []

    def _load(key: str) -> torch.Tensor:
        shard = ckpt_dir / weight_map[key]
        with safe_open(str(shard), framework="pt", device=str(device)) as f:
            return f.get_tensor(key)

    for e in range(num_test_e):
        p = f"model.layers.3.mlp.experts.{e}."
        w_gate_list.append(_load(p + "gate_proj.weight"))
        s_gate_list.append(_load(p + "gate_proj.weight_scale_inv"))
        w_up_list.append(_load(p + "up_proj.weight"))
        s_up_list.append(_load(p + "up_proj.weight_scale_inv"))
        w_down_list.append(_load(p + "down_proj.weight"))
        s_down_list.append(_load(p + "down_proj.weight_scale_inv"))

    w_gate = torch.stack(w_gate_list, dim=0)
    s_gate = torch.stack(s_gate_list, dim=0)
    w_up = torch.stack(w_up_list, dim=0)
    s_up = torch.stack(s_up_list, dim=0)
    w13_fp8 = torch.cat([w_gate, w_up], dim=1)
    s13_fp8 = torch.cat([s_gate, s_up], dim=1)
    w2_fp8 = torch.stack(w_down_list, dim=0)
    s2_fp8 = torch.stack(s_down_list, dim=0)

    t0 = time.perf_counter()
    w13_packed, w13_scale, w13_gs = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(
        w13_fp8, s13_fp8, is_gated_w13=True, chunk_experts=2,
    )
    w2_packed, w2_scale, w2_gs = nvfp4.quantize_fp8_block_to_nvfp4_modelopt(
        w2_fp8, s2_fp8, is_gated_w13=False, chunk_experts=2,
    )
    torch.cuda.synchronize(device)
    dt_ms = (time.perf_counter() - t0) * 1e3

    # Reference FP8 -> FP32 dequant
    def _dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        e, n, k = w.shape
        return (w.float().view(e, n // 128, 128, k // 128, 128) * s.float()[:, :, None, :, None]).view(e, n, k)

    w13_ref = _dequant_fp8(w13_fp8, s13_fp8)
    w2_ref = _dequant_fp8(w2_fp8, s2_fp8)
    w13_nv = _dequant_nvfp4_modelopt(w13_packed, w13_scale, w13_gs)
    w2_nv = _dequant_nvfp4_modelopt(w2_packed, w2_scale, w2_gs)

    cos_w13 = F.cosine_similarity(w13_ref.flatten(), w13_nv.flatten(), dim=0).item()
    cos_w2 = F.cosine_similarity(w2_ref.flatten(), w2_nv.flatten(), dim=0).item()
    rel_w13 = (w13_ref - w13_nv).norm().item() / w13_ref.norm().item()
    rel_w2 = (w2_ref - w2_nv).norm().item() / w2_ref.norm().item()

    # Check end-to-end 1-expert MLP output cosine similarity
    x = torch.randn((16, w13_fp8.shape[2]), device=device, dtype=torch.float32) * 0.1
    def _mlp(x_in: torch.Tensor, w13_m: torch.Tensor, w2_m: torch.Tensor) -> torch.Tensor:
        gu = x_in @ w13_m[0].t()
        g, u = gu.chunk(2, dim=-1)
        return (F.silu(g) * u) @ w2_m[0].t()

    y_ref = _mlp(x, w13_ref, w2_ref)
    y_nv = _mlp(x, w13_nv, w2_nv)
    cos_y = F.cosine_similarity(y_ref.flatten(), y_nv.flatten(), dim=0).item()
    print(
        f"[real-weight GLM-5.3 L3] 4 experts FP8->ModelOpt NVFP4 in {dt_ms:.1f} ms | "
        f"w13 cos={cos_w13:.5f} rel={rel_w13:.4f} | w2 cos={cos_w2:.5f} rel={rel_w2:.4f} | "
        f"MLP output cos={cos_y:.5f}"
    )
    assert cos_w13 > 0.99 and cos_w2 > 0.99 and cos_y > 0.99, "ModelOpt NVFP4 fidelity too low!"


def main() -> None:
    from tensorbridge.nvlink_ep import NVLinkSymmetricEP
    from tensorbridge.quant import nvfp4

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device)
        print(f"=== 8-GPU NVLink Symmetric Memory EP Test (world_size={world_size}, GPU={gpu_name}) ===")
        model_path = os.environ.get("TENSORBRIDGE_MODEL_PATH")
        check_real_glm53_layer(Path(model_path) if model_path else None, device)
    dist.barrier()

    # GLM-5.3 EP8 dimensions
    num_global_e = 256
    assert num_global_e % world_size == 0
    local_e = num_global_e // world_size
    hidden = 6144
    intermediate = 2048
    top_k = 8
    max_tokens = 256

    torch.manual_seed(42 + rank)
    # Build ModelOpt NVFP4 weights for local experts and prepare for TensorBridge MegaMoE / Prefill
    w13_bf16 = torch.randn((local_e, 2 * intermediate, hidden), dtype=torch.bfloat16, device=device) * 0.02
    w2_bf16 = torch.randn((local_e, hidden, intermediate), dtype=torch.bfloat16, device=device) * 0.02
    w13_p, w13_s, w13_gs = nvfp4.quantize_modelopt(w13_bf16)
    w2_p, w2_s, w2_gs = nvfp4.quantize_modelopt(w2_bf16)
    # Convert to TensorBridge prepared grouped layout (w13 interleaved + group_128, w2 group_128)
    import dataclasses
    w13_raw = nvfp4.prepare_experts(w13_p, w13_s, w13_gs)
    w13_g128 = dataclasses.replace(
        w13_raw,
        scale=w13_raw.scale.view(local_e, hidden // 128, 8, 2 * intermediate).permute(0, 1, 3, 2).contiguous(),
    )
    w13 = nvfp4.interleave_gate_up_experts(w13_g128)
    w2_raw = nvfp4.prepare_experts(w2_p, w2_s, w2_gs)
    w2 = dataclasses.replace(
        w2_raw,
        scale=w2_raw.scale.view(local_e, intermediate // 128, 8, hidden).permute(0, 1, 3, 2).contiguous(),
    )
    w13_deq = _dequant_nvfp4_modelopt(w13_p, w13_s, w13_gs).to(torch.float32)
    w2_deq = _dequant_nvfp4_modelopt(w2_p, w2_s, w2_gs).to(torch.float32)
    del w13_bf16, w2_bf16, w13_raw, w13_g128, w2_raw, w13_p, w13_s, w13_gs, w2_p, w2_s, w2_gs

    ep_group = dist.group.WORLD
    ep = NVLinkSymmetricEP(
        ep_group, local_e=local_e, hidden=hidden, intermediate=intermediate, top_k=top_k,
        max_tokens_per_rank=max_tokens, device=device,
    )
    ep.precompile(w13, w2)
    dist.barrier()

    def _make_inputs(t: int, skewed: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=device).manual_seed(1000 + rank * 17 + t)
        x = torch.randn((t, hidden), dtype=torch.bfloat16, device=device, generator=gen) * 0.1
        logits = torch.randn((t, num_global_e), device=device, generator=gen)
        if skewed:
            logits[:, :16] += 3.0
        topk_w, topk_ids = torch.topk(logits, k=top_k, dim=-1)
        topk_w = torch.softmax(topk_w, dim=-1).to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        return x, topk_ids, topk_w

    def _pytorch_ep_ref(
        x_in: torch.Tensor, ids_in: torch.Tensor, w_in: torch.Tensor, global_t: int,
    ) -> torch.Tensor:
        t_local = x_in.size(0)
        x_pad = F.pad(x_in, (0, 0, 0, global_t - t_local))
        ids_pad = F.pad(ids_in, (0, 0, 0, global_t - t_local), value=-1)
        w_pad = F.pad(w_in, (0, 0, 0, global_t - t_local))
        all_x = [torch.empty_like(x_pad) for _ in range(world_size)]
        all_ids = [torch.empty_like(ids_pad) for _ in range(world_size)]
        all_w = [torch.empty_like(w_pad) for _ in range(world_size)]
        dist.all_gather(all_x, x_pad)
        dist.all_gather(all_ids, ids_pad)
        dist.all_gather(all_w, w_pad)
        x_cat = torch.cat(all_x, dim=0).float()
        ids_cat = torch.cat(all_ids, dim=0)
        w_cat = torch.cat(all_w, dim=0)
        partial = torch.zeros_like(x_cat)
        e_beg = rank * local_e
        for le in range(local_e):
            gid = e_beg + le
            mask = ids_cat == gid
            if not mask.any():
                continue
            tok_idx, k_idx = mask.nonzero(as_tuple=True)
            gu = x_cat[tok_idx] @ w13_deq[le].t()
            g, u = gu.chunk(2, dim=-1)
            dn = (F.silu(g) * u) @ w2_deq[le].t()
            partial.index_add_(0, tok_idx, dn * w_cat[tok_idx, k_idx].unsqueeze(-1))
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial[rank * global_t : rank * global_t + t_local]

    # 1. Numerical verification across MegaMoE (T <= 256), Contiguous Prefill (T > 256), skewed, and asymmetric shapes
    for t_val, skewed, label in [
        (1, False, "T=1 uniform"),
        (4, False, "T=4 uniform"),
        (16, False, "T=16 uniform"),
        (16, True, "T=16 skewed"),
        (32, False, "T=32 uniform"),
        (64, False, "T=64 uniform"),
        (128, False, "T=128 uniform"),
        (300, False, "T=300 contiguous-prefill"),
        (1024, False, "T=1024 contiguous-prefill"),
    ]:
        x, ids, w = _make_inputs(t_val, skewed=skewed)
        out = ep.forward(x, ids, w, w13, w2)
        py_ref = _pytorch_ep_ref(x, ids, w, t_val)
        cos_py = F.cosine_similarity(out.float().flatten(), py_ref.flatten(), dim=0).item()
        rel_py = (out.float() - py_ref).norm().item() / py_ref.norm().clamp_min(1e-6).item()
        if rank == 0:
            print(f"[correctness {label:26s}] cos={cos_py:.6f} rel_err={rel_py:.5f}")
        assert cos_py > 0.995 and rel_py < 0.08, f"Rank {rank} {label} mismatch: cos={cos_py}, rel={rel_py}"
        if t_val == 16 and not skewed:
            out_res = ep.forward(x, ids, w, w13, w2, reserve_sms=4)
            max_res_diff = (out_res.float() - out.float()).abs().max().item()
            if rank == 0:
                print(f"[reserve_sms=4 vs 0 T=16    ] max_abs={max_res_diff:.6f}")
            assert max_res_diff == 0.0, f"reserve_sms=4 diverged from reserve_sms=0: max_abs={max_res_diff}"

    # Asymmetric MegaMoE test: rank 0 has T=16, ranks 1..7 have T=0
    t_asym = 16 if rank == 0 else 0
    x_a, ids_a, w_a = _make_inputs(t_asym)
    out_a = ep.forward(x_a, ids_a, w_a, w13, w2, global_num_tokens=16)
    ref_a = _pytorch_ep_ref(x_a, ids_a, w_a, 16)
    if t_asym > 0:
        cos_a = F.cosine_similarity(out_a.float().flatten(), ref_a.flatten(), dim=0).item()
        rel_a = (out_a.float() - ref_a).norm().item() / ref_a.norm().clamp_min(1e-6).item()
        if rank == 0:
            print(f"[correctness asym T=16/0    ] cos={cos_a:.6f} rel_err={rel_a:.5f}")
        assert cos_a > 0.995 and rel_a < 0.08

    # Asymmetric Prefill test (mimics DP=2 TP=4 c=1 Prefill): ranks 0..3 have T=1024, ranks 4..7 have T=0
    t_pf = 1024 if rank < world_size // 2 else 0
    x_pf, ids_pf, w_pf = _make_inputs(t_pf)
    out_pf = ep.forward(x_pf, ids_pf, w_pf, w13, w2, global_num_tokens=1024)
    ref_pf = _pytorch_ep_ref(x_pf, ids_pf, w_pf, 1024)
    if t_pf > 0:
        cos_pf = F.cosine_similarity(out_pf.float().flatten(), ref_pf.flatten(), dim=0).item()
        rel_pf = (out_pf.float() - ref_pf).norm().item() / ref_pf.norm().clamp_min(1e-6).item()
        if rank == 0:
            print(f"[correctness asym T=1024/0  ] cos={cos_pf:.6f} rel_err={rel_pf:.5f}")
        assert cos_pf > 0.995 and rel_pf < 0.08

    del w13_deq, w2_deq

    # Eager Contiguous Prefill latency check (T=1024 and T=2048 per rank)
    for t_pf_bench in (1024, 2048):
        x_b, ids_b, w_b = _make_inputs(t_pf_bench)
        out_b = torch.empty_like(x_b)
        for _ in range(3):
            ep.forward(x_b, ids_b, w_b, w13, w2, out=out_b)
        torch.cuda.synchronize(device)
        dist.barrier()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(15):
            ep.forward(x_b, ids_b, w_b, w13, w2, out=out_b)
        ev1.record()
        torch.cuda.synchronize(device)
        pf_us = ev0.elapsed_time(ev1) * 1e3 / 15
        if rank == 0:
            print(f"[prefill T={t_pf_bench:4d}/rank] contiguous_ep={pf_us:7.1f}us")

    # 2. CUDA Graph capture + replay verification & latency benchmark on real 8xH20 NVLink
    dist.barrier()
    if rank == 0:
        print("\n=== 8xH20 Real NVLink Single-Layer MegaMoE Latency (CUDA Graph, GLM-5.3 EP8) ===")
        print(f"{'T/rank':>6s} | {'GlobalT':>7s} | {'MegaMoE':>10s} | {'Graph vs Eager':>14s}")
        print("-" * 48)

    for t_val in (1, 2, 4, 8, 16, 32, 64, 128):
        x, ids, w = _make_inputs(t_val)
        ref = ep.forward(x, ids, w, w13, w2).clone()
        out_buf = torch.empty_like(x)
        for _ in range(5):
            ep.forward(x, ids, w, w13, w2, out=out_buf)
        torch.cuda.synchronize(device)
        dist.barrier()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ep.forward(x, ids, w, w13, w2, out=out_buf)
        torch.cuda.synchronize(device)
        g.replay()
        torch.cuda.synchronize(device)
        max_abs = (out_buf.float() - ref.float()).abs().max().item()
        assert max_abs == 0.0, f"CUDAGraph replay mismatch for T={t_val}: max_abs={max_abs}"

        iters = 100
        dist.barrier()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            g.replay()
        ev1.record()
        torch.cuda.synchronize(device)
        lat_us = ev0.elapsed_time(ev1) * 1e3 / iters

        if rank == 0:
            print(f"{t_val:6d} | {t_val * world_size:7d} | {lat_us:8.1f}us | max_abs={max_abs:.6f}")

    ep.close()
    dist.destroy_process_group()
    if rank == 0:
        print("\nALL 8-GPU NVLINK SYMMETRIC EP TESTS PASSED!")


if __name__ == "__main__":
    main()
