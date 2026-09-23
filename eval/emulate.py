"""Weight decompression hooks for compressed-tensors checkpoints.

`fpma` makes Hugging Face inference see exactly the FP8 weights that the
TensorBridge bridge (`tensorbridge/include/tensorbridge/dequant/fpma.cuh`)
feeds to the tensor core, times the global scale folded by `nvfp4.prepare`.
`mxfp4` supplies the MXFP4 decompression that compressed-tensors lacks.
"""

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization.lifecycle.forward import dequantize

from tensorbridge.quant.nvfp4 import E2M1_MAX, PREFOLD_DELTA, RAW_SCALE_MIN, lsq_alpha


def fpma_weight(packed, scale, global_scale, snc=True, alpha=1.0):
    """`[N, K/2]` packed E2M1, `[N, K/16]` E4M3 scales -> `[N, K]` BF16 bridge weights.
    `alpha` is a number or "lsq", as in the kernel's vLLM plugin."""
    if isinstance(alpha, str):  # "lsq"
        alpha = lsq_alpha(packed, scale)
    n, groups = scale.shape
    raw = scale.view(torch.uint8).to(torch.int16).view(n, groups, 1)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).view(n, groups, 16).to(torch.int16)
    magnitude, sign = codes & 0x7, codes & 0x8
    nonzero = magnitude != 0
    if snc:  # 0.5 is stored as magnitude 0; zero takes the freed code and is masked
        magnitude = torch.where(magnitude == 1, 0, torch.where(nonzero, magnitude, 1))
    byte = (raw - PREFOLD_DELTA).clamp(min=0) + (sign << 4) + (magnitude << 2)
    # Scales below the bridge's range zero their group, as in `nvfp4.prepare`.
    byte = torch.where(nonzero & (raw >= RAW_SCALE_MIN), byte, 0).to(torch.uint8)
    value = byte.view(torch.float8_e4m3fn).float()
    # compressed-tensors stores the reciprocal of the global scale.
    value = value * ((E2M1_MAX * alpha) / global_scale.float())
    return value.view(n, groups * 16).to(torch.bfloat16)


def fpma(snc=True, alpha=1.0):
    @classmethod
    @torch.no_grad()
    def decompress(cls, state_dict, scheme):
        state_dict = state_dict.copy()
        scale = state_dict["weight_scale"]
        state_dict["weight"] = fpma_weight(
            state_dict.pop("weight_packed"), scale, state_dict["weight_global_scale"], snc, alpha
        )
        state_dict["weight_scale"] = torch.nn.Parameter(scale.to(torch.bfloat16), requires_grad=False)
        return state_dict

    NVFP4PackedCompressor.decompress = decompress


def mxfp4():
    @classmethod
    @torch.no_grad()
    def decompress(cls, state_dict, scheme):
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        values = unpack_fp4_from_uint8(packed, packed.shape[0], packed.shape[1] * 2)
        # E8M0 scales: 2 ** (byte - 127)
        scale = torch.exp2(state_dict["weight_scale"].float() - 127).to(values.dtype)
        state_dict["weight"] = dequantize(x_q=values, scale=scale, args=scheme.weights,
                                          dtype=values.dtype)
        state_dict["weight_scale"] = torch.nn.Parameter(scale, requires_grad=False)
        return state_dict

    MXFP4PackedCompressor.decompress = decompress
