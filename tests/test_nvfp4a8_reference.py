"""TensorBridge NVFP4 W4A8 kernel vs the exact pure-torch reference.

The fused kernel uses the FPMA bridge (encoding-domain shift-and-add) plus SNC
zero/subnormal handling, so it is checked against the exact NVFP4A8 reference
within the FPMA + bf16 tolerance (paper Sec. 4.1: ULP-level per-byte error).
"""

import pytest
import torch

from nvfp4_reference import nvfp4_w4a8_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="The kernel targets Hopper (sm_90)",
)


def _rel_rms(out, ref):
    return (out - ref).pow(2).mean().sqrt().item() / ref.pow(2).mean().sqrt().clamp_min(1e-9).item()


def _kernel_and_reference(m, n, k, use_stream_k=None):
    import tensorbridge
    from tensorbridge.quant import fp8, nvfp4

    torch.manual_seed(0)
    weight_hp = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    inputs_hp = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")

    packed, scale, global_scale = nvfp4.quantize(weight_hp)
    config = tensorbridge.select_config(m, n, k, use_stream_k=use_stream_k)
    weight = nvfp4.prepare(packed, scale, global_scale)
    out = tensorbridge.fp8_nvfp4_gemm(*fp8.quantize(inputs_hp), weight, config=config)
    ref = nvfp4_w4a8_reference(
        inputs_hp, packed, scale, global_scale, group_size=16, out_dtype=torch.bfloat16
    )
    return out.float(), ref.float(), config


@pytest.mark.parametrize("m", [16, 128, 512])
def test_matches_reference(m):
    """Accuracy is measured as relative RMS (Frobenius) error, the standard metric
    for low-precision GEMM; the FPMA W4A8 bridge lands around 4%."""
    out, ref, _ = _kernel_and_reference(m, 4096, 4096)
    assert torch.isfinite(out).all()
    assert _rel_rms(out, ref) < 0.06


@pytest.mark.parametrize(
    "m,n,k", [(1, 128, 128), (7, 256, 384), (33, 1024, 2048), (300, 1024, 16384)]
)
def test_ragged_token_counts(m, n, k):
    """Token counts that do not fill the last tile, on the smallest legal N and K."""
    out, ref, _ = _kernel_and_reference(m, n, k)
    assert torch.isfinite(out).all()
    assert _rel_rms(out, ref) < 0.06


@pytest.mark.parametrize("m,n,k", [(512, 4096, 12288), (128, 6144, 4096)])
def test_stream_k_matches_data_parallel(m, n, k):
    """Each K slice of a tile is rounded to BF16 on its own and the slices are summed
    in BF16, so the two backends agree to BF16 rounding (2^-8 per slice), not bitwise."""
    tail, ref, tail_config = _kernel_and_reference(m, n, k, use_stream_k=True)
    dp, _, dp_config = _kernel_and_reference(m, n, k, use_stream_k=False)
    assert tail_config.use_stream_k and not dp_config.use_stream_k
    assert _rel_rms(tail, ref) < 0.06
    assert _rel_rms(tail, dp) < 0.01


@pytest.mark.parametrize("multicast", ["multicast_a", "multicast_b"])
def test_multicast_matches_independent_ctas(multicast):
    """A cluster of two CTAs sharing one tile load computes the same tiles, in the
    same order of operations, as two independent CTAs."""
    import dataclasses

    import tensorbridge
    from tensorbridge.quant import fp8, nvfp4

    torch.manual_seed(0)
    m, n, k = 840, 2048, 2048
    weight_hp = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    weight = nvfp4.prepare(*nvfp4.quantize(weight_hp))
    a, sfa = fp8.quantize(torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0"))

    independent = tensorbridge.select_config(m, n, k, use_stream_k=False)
    clustered = dataclasses.replace(independent, **{multicast: 2})
    assert torch.equal(
        tensorbridge.fp8_nvfp4_gemm(a, sfa, weight, config=clustered),
        tensorbridge.fp8_nvfp4_gemm(a, sfa, weight, config=independent),
    )


def test_linear_layer():
    import tensorbridge

    torch.manual_seed(0)
    weight = torch.randn((1024, 2048), dtype=torch.bfloat16, device="cuda:0")
    x = torch.randn((64, 2048), dtype=torch.bfloat16, device="cuda:0")
    layer = tensorbridge.NVFP4Linear.from_float(weight)
    assert _rel_rms(layer(x).float(), x.float() @ weight.float().t()) < 0.15
