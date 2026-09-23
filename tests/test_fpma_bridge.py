"""FPMA bridge checks (paper Sec. 4.1).

(1) Byte-level scalar checks of the FPMA encoding-domain construction
    byte = prefolded_scale + (sign<<7 | mag<<2), with FP4-zero lanes cleared.
(2) GPU three-way error decomposition: the kernel matches the bridge-accurate
    reference far more tightly than the exact reference, showing the residual is
    the FPMA approximation itself (not a kernel bug).
"""

import pytest
import torch

from fpma_bridge_reference import (
    fpma_bridge_fp8_bytes,
    nvfp4_w4a8_bridge_reference,
    prefold_scale_and_global,
)
from nvfp4_reference import nvfp4_w4a8_reference, unpack_fp4_packed_to_codes


def _hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


# --- (1) Scalar byte-level checks (CPU) -------------------------------------

def _bridge_one(code, raw_scale_byte, snc=False):
    codes = torch.tensor([[code]], dtype=torch.uint8)
    raw = torch.tensor([[raw_scale_byte]], dtype=torch.uint8).view(torch.float8_e4m3fn)
    pref, _ = prefold_scale_and_global(raw, None)
    b = fpma_bridge_fp8_bytes(codes, pref, group_size=1, snc=snc)
    return int(b.view(torch.uint8).item())


def test_bridge_byte_known_values():
    # FP4 code 4 = +2.0 (mag=4); raw scale 0x38 = fp8 1.0 -> prefold 0x1C.
    # addend = (0<<7)|(4<<2) = 0x10; byte = 0x1C + 0x10 = 0x2C.
    assert _bridge_one(0x4, 0x38) == 0x2C
    # FP4 code 0 (+0.0) and code 8 (-0.0, mag==0) -> cleared to fp8 +0.
    assert _bridge_one(0x0, 0x38) == 0x00
    assert _bridge_one(0x8, 0x38) == 0x00
    # Sign propagates to bit7: code 0xC = -2.0 -> 0x80 | 0x2C = 0xAC.
    assert _bridge_one(0xC, 0x38) == 0xAC


def test_bridge_byte_matches_fp8_decode_value():
    # 0x2C as fp8e4m3 = 2^(5-7)*(1+4/8) = 0.375; *global(1)*6 = 2.25.
    b = torch.tensor([[0x2C]], dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    assert abs(b.item() - 0.375) < 1e-6
    assert abs(b.item() * 6.0 - 2.25) < 1e-6


def test_snc_keeps_the_subnormal_in_proportion():
    # E2M1's 0.5 is a subnormal: exponent field 0, mantissa bit set. Shifted into
    # the E4M3 layout, that bit lands on the mantissa half instead of lowering the
    # exponent, so 0.5 would decode to 2/3 of 1.0. SNC stores it as magnitude 0,
    # one exponent step below 1.0 (magnitude 2), which is exactly half.
    def decode(byte):
        return torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()

    one = decode(_bridge_one(0x2, 0x38))
    assert decode(_bridge_one(0x1, 0x38)) / one == pytest.approx(2 / 3)
    assert decode(_bridge_one(0x1, 0x38, snc=True)) / one == 0.5


# --- (2) Offline preparation (CPU) -------------------------------------------

def test_thread_order_feeds_each_thread_its_fragment():
    """Replay the consumer side in Python: the 16-byte vector thread `t` reads for
    a call pair, split the way `fpma::dequant_word` splits it, must be the SNC
    codes of channel rows (r, r + 8) x K columns (c .. c + 3, c + 16 .. c + 19)."""
    from tensorbridge.quant import nvfp4

    torch.manual_seed(0)
    packed = torch.randint(0, 256, (128, 64), dtype=torch.uint8)
    codes = unpack_fp4_packed_to_codes(nvfp4.recode_snc(packed))  # [128, 128]
    logical = nvfp4.to_thread_order(nvfp4.recode_snc(packed)).flatten()

    # What lands in shared memory: TMA applies the 64B swizzle to the tile.
    address = torch.arange(128 * 64)
    smem = logical[address ^ (((address >> 7) & 0x3) << 4)]

    for thread in (0, 5, 37, 130, 255):
        warp, lane = divmod(thread, 32)
        row = lane // 4 + (warp % 4) * 16 + (warp // 4) * 64
        col = (lane % 4) * 4
        for pair in range(2):
            vector = smem[pair * 4096 + thread * 16:][:16].view(2, 2, 4)  # [call, row, byte]
            for call in range(2):
                k = (pair * 2 + call) * 32 + col
                for r in range(2):
                    assert torch.equal(vector[call, r] & 0xF, codes[row + 8 * r, k:k + 4])
                    assert torch.equal(vector[call, r] >> 4, codes[row + 8 * r, k + 16:k + 20])


# --- (3) GPU three-way error decomposition ----------------------------------

@pytest.mark.skipif(not _hopper(), reason="The kernel targets Hopper (sm_90)")
def test_kernel_matches_bridge_ref_tighter_than_exact():
    import tensorbridge
    from tensorbridge.quant import fp8, nvfp4

    torch.manual_seed(0)
    n, k, m, g = 4096, 4096, 256, 16
    weight_hp = torch.randn((n, k), dtype=torch.bfloat16, device="cuda:0")
    inputs_hp = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0")

    packed, scale, gscale = nvfp4.quantize(weight_hp)
    weight = nvfp4.prepare(packed, scale, gscale, alpha=1.0)  # the reference omits alpha
    out = tensorbridge.fp8_nvfp4_gemm(*fp8.quantize(inputs_hp), weight).float()
    codes = unpack_fp4_packed_to_codes(packed)

    args = (inputs_hp, codes, scale, gscale, g, torch.bfloat16)
    bridge = nvfp4_w4a8_bridge_reference(*args).float()
    bridge_snc = nvfp4_w4a8_bridge_reference(*args, snc=True).float()
    exact = nvfp4_w4a8_reference(inputs_hp, packed, scale, gscale, g, torch.bfloat16).float()

    def rel_rms(a, b):
        return (a - b).pow(2).mean().sqrt().item() / b.pow(2).mean().sqrt().clamp_min(1e-9).item()

    # The kernel reproduces the FPMA bridge math, so it is closer to the
    # bridge-accurate reference than to the exact reference, and closer still
    # once the reference models the SNC recoding. What then remains is the
    # activation rounding of the two FP8 quantizers and BF16 rounding.
    print(f"kernel vs exact {rel_rms(out, exact):.5f}, vs bridge {rel_rms(out, bridge):.5f}, "
          f"vs bridge+SNC {rel_rms(out, bridge_snc):.5f}")
    assert rel_rms(out, bridge) < rel_rms(out, exact)
    assert rel_rms(out, bridge) < 0.045
    assert rel_rms(out, bridge_snc) < 0.01
