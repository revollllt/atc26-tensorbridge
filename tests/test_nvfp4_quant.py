"""CPU checks for the on-load ModelOpt quantizer. The decoder is independent of the encoder."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tensorbridge"))
from quant import modelopt  # noqa: E402

MODES = ("rtn", "bridge_aware", "four_over_six", "mse_7ulp")


def unpack(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()


def decode(codes: torch.Tensor, scale: torch.Tensor, multiplier: torch.Tensor, bridge: str | None) -> torch.Tensor:
    code = codes & 7
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[code]
    if bridge is None:
        decoded = values * scale.float() * multiplier
    elif bridge == "exact":
        decoded = (values / 6 * scale.float()).to(torch.float8_e4m3fn).float() * (6 * multiplier)
    else:
        raw = scale.view(torch.uint8).int()
        bits = (raw - 28 + torch.tensor([0, 0, 8, 12, 16, 20, 24, 28])[code]).byte()
        decoded = bits.masked_fill((code == 0) | (raw < 28), 0).view(torch.float8_e4m3fn).float() * (6 * multiplier)
    return torch.where(codes < 8, decoded, -decoded)


class QuantizationTest(unittest.TestCase):
    def test_gaussian_and_zero_groups(self) -> None:
        weight = torch.randn(32, 128, generator=torch.Generator().manual_seed(42))
        weight[0, :16] = -0.0
        weight[0, 16:32] *= 1e-7
        for bridge in (None, "exact", "fpma"):
            errors = {}
            for mode in MODES:
                self.assertEqual(modelopt.parse_quant_mode(mode), mode)
                packed, scale, gs = modelopt.quantize_modelopt(weight, mode=mode, bridge=bridge)
                raw = scale.view(torch.uint8)
                self.assertEqual(int(raw[0, 0]), 0)
                self.assertFalse(packed[0, :8].any())
                self.assertLessEqual(int(raw.max()), 0x7E)
                if mode == "mse_7ulp" and bridge is not None:
                    self.assertGreaterEqual(int(raw[0, 1]), 0x1C)
                values = decode(unpack(packed).view(32, 8, 16), scale.unsqueeze(-1), gs, bridge)
                errors[mode] = (values.flatten(1) - weight).square().mean().item()
            self.assertLess(errors["mse_7ulp"], errors["four_over_six"])
            self.assertLess(errors["four_over_six"], errors["rtn"])
        with self.assertRaisesRegex(ValueError, "nvfp4_quant_mode"):
            modelopt.parse_quant_mode("unknown")

    def test_rtn_original_bytes(self) -> None:
        weight = torch.randn(3, 7, 64, generator=torch.Generator().manual_seed(9))
        weight[1] *= 0.1
        groups = weight.reshape(3, 7, 4, 16)
        peak = groups.abs().amax(-1)
        gs = (peak.amax((1, 2), keepdim=True) / (448 * 6)).clamp_min(1e-30)
        scales = (peak / (6 * gs)).clamp(0, 448).to(torch.float8_e4m3fn)
        normalized = groups / (scales.float() * gs).clamp_min(1e-30).unsqueeze(-1)
        expected = torch.bucketize(normalized.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]))
        expected |= torch.signbit(normalized).long() << 3
        packed, actual, global_scale = modelopt.quantize_modelopt(weight, mode="rtn", bridge="exact")
        self.assertTrue(torch.equal(expected.flatten(-2), unpack(packed)))
        self.assertTrue(torch.equal(scales.view(torch.uint8), actual.view(torch.uint8)))
        self.assertTrue(torch.equal(gs.flatten(), global_scale))

    def test_search_matches_exhaustive_decoding(self) -> None:
        weight = torch.randn(3, 32, generator=torch.Generator().manual_seed(513))
        groups = weight.view(-1, 16)
        for bridge in (None, "exact", "fpma"):
            for mode in MODES[1:]:
                packed, scales, gs = modelopt.quantize_modelopt(weight, mode=mode, bridge=bridge)
                peak = groups.abs().amax(-1)
                divisor = 6 if mode == "bridge_aware" else 4
                torch.testing.assert_close(gs, (peak.max() / (448 * divisor)).view(1), rtol=0, atol=0)
                base = (peak / (6 * gs)).clamp(max=448).to(torch.float8_e4m3fn)
                candidates = [base]
                if mode == "four_over_six":
                    candidates.append((peak / (4 * gs)).clamp(max=448).to(torch.float8_e4m3fn))
                elif mode == "mse_7ulp":
                    candidates = [
                        (base.view(torch.uint8).int() + offset)
                        .clamp(0 if bridge is None else 28, 126)
                        .byte()
                        .view(torch.float8_e4m3fn)
                        for offset in range(-2, 5)
                    ]
                losses = []
                for scale in candidates:
                    levels = decode(torch.arange(8), scale[:, None], gs, bridge)
                    losses.append((groups.abs()[:, :, None] - levels[:, None, :]).square().amin(-1).sum(-1))
                values = decode(unpack(packed).view_as(groups), scales.reshape(-1, 1), gs, bridge)
                torch.testing.assert_close((values - groups).square().sum(-1), torch.stack(losses).amin(0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA allocator required")
    def test_glm_four_expert_peak(self) -> None:
        # Peak above the resident checkpoint includes both the returned outputs and scratch space.
        source = torch.zeros(4, 4096, 6144, device="cuda", dtype=torch.float8_e4m3fn)
        blocks = torch.ones(4, 32, 48, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        modelopt.quantize_fp8_block_to_nvfp4_modelopt(source, blocks, is_gated_w13=True, bridge="exact")
        self.assertLess(torch.cuda.max_memory_allocated() - resident, 512 * 1024**2)

    def test_fp8_gated_block_boundaries(self) -> None:
        generator = torch.Generator().manual_seed(17)
        codes = torch.randn(5, 10, 32, generator=generator).to(torch.float8_e4m3fn)
        blocks = torch.rand(5, 3, 2, generator=generator)
        dequantized = codes.float() * blocks.repeat_interleave(4, 1)[:, :10].repeat_interleave(16, 2)
        for mode in MODES:
            packed, scales, gs = modelopt.quantize_fp8_block_to_nvfp4_modelopt(
                codes, blocks, (4, 16), is_gated_w13=True, chunk_experts=4, mode=mode, bridge="exact"
            )
            for shard, rows in enumerate((slice(0, 5), slice(5, 10))):
                expected = modelopt.quantize_modelopt(dequantized[:, rows], mode=mode, bridge="exact")
                self.assertTrue(torch.equal(packed[:, rows], expected[0]))
                self.assertTrue(torch.equal(scales[:, rows].view(torch.uint8), expected[1].view(torch.uint8)))
                self.assertTrue(torch.equal(gs[:, shard], expected[2]))
        default = modelopt.quantize_fp8_block_to_nvfp4_modelopt(codes[0], blocks[0], (4, 16))
        explicit = modelopt.quantize_modelopt(dequantized[0], mode="mse_7ulp")
        self.assertTrue(
            all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(default, explicit, strict=True))
        )


if __name__ == "__main__":
    unittest.main()
