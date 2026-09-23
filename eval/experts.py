"""Quantized experts for the fused MoE modules of transformers 5 (Qwen3.6-35B-A3B).

transformers keeps a layer's experts in two 3-D parameters (`gate_up_proj`,
`down_proj`) and cannot load the per-expert compressed Linears that llmcompressor
saves, so they would stay uninitialized. `load` dequantizes every saved expert
with the method's weight rule into those parameters, and routes the experts'
grouped matmuls through the same dynamic activation fake quantization that
compressed-tensors applies to Linear layers.
"""

import glob
import json
import os
import re

import torch
import transformers.integrations.moe as hf_moe
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from safetensors import safe_open

import emulate

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def has_quantized_experts(path):
    return any(re.search(r"\.experts\.\d+\.\w+_proj\.weight_(packed|scale)$", key)
               for key in _index(path))


def _index(path):
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        return {k: os.path.join(path, f) for k, f in json.load(open(index))["weight_map"].items()}
    keys = {}
    for file in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt") as f:
            keys.update(dict.fromkeys(f.keys(), file))
    return keys


def _fp4(packed):
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    values = E2M1.to(packed.device)[codes & 7]
    return torch.where(codes & 8 != 0, -values, values)


def _dequantize(t, fmt, method, alpha):
    """Stacked expert projections `[E, out, ...]` -> BF16 `[E, out, in]`."""
    if fmt == "int-quantized":  # W4A8: int4 values in int8, BF16 scale per 128
        w, s = t["weight"].float(), t["weight_scale"].float()
        return (w * s.repeat_interleave(w.shape[-1] // s.shape[-1], dim=-1)).bfloat16()
    packed, scale = t["weight_packed"], t["weight_scale"]
    if fmt == "mxfp4-pack-quantized":  # E8M0 scale per 32
        s = torch.exp2(scale.float() - 127)
        return (_fp4(packed) * s.repeat_interleave(32, dim=-1)).bfloat16()
    gscale = t["weight_global_scale"].float().view(-1, 1, 1)  # compressed-tensors: reciprocal
    if method in ("ours", "ours_nosnc"):
        experts, rows = packed.shape[:2]
        if alpha == "lsq":  # one least-squares factor per expert, as per layer for dense
            alpha = torch.tensor([emulate.lsq_alpha(p, s) for p, s in zip(packed, scale)],
                                 device=packed.device).view(-1, 1, 1)
        if torch.is_tensor(alpha):
            alpha = alpha.repeat_interleave(rows, dim=0)
        w = emulate.fpma_weight(packed.flatten(0, 1), scale.flatten(0, 1),
                                gscale.repeat_interleave(rows, dim=0), snc=method == "ours",
                                alpha=alpha)
        return w.view(experts, rows, -1)
    return (_fp4(packed) * (scale.float().repeat_interleave(16, dim=-1) / gscale)).bfloat16()


def load(model, path, method, alpha, group, chunk=64):
    """Fill every fused expert module of `model` from the checkpoint at `path`;
    `group` is its compressed-tensors scheme (`format`, `input_activations`).
    Experts are read stacked and dequantized `chunk` at a time."""
    files = _index(path)
    found = {}  # (layer, projection) -> {tensor name: {expert: key}}
    for key in files:
        m = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+_proj)\.(\w+)$", key)
        if m:
            found.setdefault((int(m[1]), m[3]), {}).setdefault(m[4], {})[int(m[2])] = key
    saved = {lp: {f: [k[e] for e in sorted(k)] for f, k in fields.items()}
             for lp, fields in found.items()}
    handles = {f: safe_open(f, "pt", device="cpu") for f in set(files.values())}
    scales = {}
    for name, experts in model.named_modules():
        if not (name.endswith("mlp.experts") and hasattr(experts, "gate_up_proj")):
            continue
        layer, device = int(re.search(r"layers\.(\d+)\.", name)[1]), experts.gate_up_proj.device
        n = experts.gate_up_proj.shape[0]
        out = {"gate_up": [], "down": []}
        for start in range(0, n, chunk):
            t = {p: {f: torch.stack([handles[files[k]].get_tensor(k)
                                     for k in keys[start:start + chunk]]).to(device)
                     for f, keys in saved[(layer, p)].items()}
                 for p in ("gate_proj", "up_proj", "down_proj")}
            out["gate_up"].append(torch.cat([_dequantize(t[p], group["format"], method, alpha)
                                             for p in ("gate_proj", "up_proj")], dim=1))
            out["down"].append(_dequantize(t["down_proj"], group["format"], method, alpha))
        for key, param, proj in (("gate_up", experts.gate_up_proj, "gate_proj"),
                                 ("down", experts.down_proj, "down_proj")):
            # Replace rather than copy: for int formats transformers stacked the raw int8 values.
            param.data = torch.cat(out[key])
            gs = saved[(layer, proj)].get("input_global_scale")
            scales[param.data_ptr()] = torch.stack(
                [handles[files[k]].get_tensor(k) for k in gs]).float().view(-1).to(device) if gs else None
    if group.get("input_activations"):
        _quantize_inputs(QuantizationArgs.model_validate(group["input_activations"]), scales)


def fake_quantize_rows(x, args, global_scale=None):
    """Dynamic fake quantization of `[rows, dim]` expert inputs, as compressed-tensors does
    for the `[batch, tokens, dim]` inputs of Linear layers (hence the leading axis).
    `global_scale` is a scalar or one value per row."""
    x = x[None]
    gs = None if global_scale is None else global_scale.reshape(1, -1, 1)  # vs [1, rows, groups]
    scale, zero_point = compute_dynamic_scales_and_zp(value=x, args=args, module=None,
                                                      global_scale=gs)
    return fake_quantize(x=x, scale=scale, zero_point=zero_point, args=args,
                         global_scale=None if gs is None else gs[..., None])[0]


def _quantize_inputs(args, scales):
    grouped_mm = hf_moe._grouped_mm

    def quantized_grouped_mm(input, weight, offs):
        gs = scales.get(weight.data_ptr())
        if gs is not None:  # per-expert input global scale, expanded to the rows
            counts = torch.diff(offs, prepend=offs.new_zeros(1)).long()
            gs = torch.repeat_interleave(gs, counts, output_size=input.shape[0])
        return grouped_mm(fake_quantize_rows(input, args, gs), weight, offs)

    hf_moe._grouped_mm = quantized_grouped_mm
