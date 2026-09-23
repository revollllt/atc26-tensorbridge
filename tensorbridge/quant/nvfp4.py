"""NVFP4 weights: quantization, and the offline preparation for the kernel.

Checkpoint format (what `quantize` returns and `prepare` accepts):

    packed        uint8  [N, K / 2]   E2M1 codes; byte b = code[2b] | code[2b + 1] << 4
    scale         fp8    [N, K / 16]  one E4M3 scale per 16 weights
    global_scale  fp32   [1]          weight = decode(code) * scale * global_scale

Everything the kernel would otherwise pay for per launch is moved into
`prepare`, which runs once per layer:

    1. scale prefold       raw E4M3 byte - 0x1C, and global_scale * 6 * alpha
    2. SNC recoding        0.5 -> magnitude 0, zero -> magnitude 1
    3. thread-order repack one 16-byte vector per consumer thread

Steps 1 and 2 serve the FPMA bridge (`include/tensorbridge/dequant/fpma.cuh`),
step 3 the register loads (`include/tensorbridge/memory/s2r.cuh`).
"""

import dataclasses

import torch

GROUP_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0

# The bridge adds the scale byte to `sign << 7 | magnitude << 2` without carry
# between byte lanes, which bounds the raw E4M3 scale byte to this range.
PREFOLD_DELTA = 0x1C
RAW_SCALE_MIN, RAW_SCALE_MAX = 0x1C, 0x7E

# Optional factor folded into the global scale. 1 keeps the bridge's values; the
# first-order debias of its byte add, 1 - (5/16 eligible cases) * (1/8 E4M3 binade
# ULP) ~= 0.961, lowers zero-shot error but raises perplexity.
FPMA_ALPHA = 1.0

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Rounding boundaries between the E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclasses.dataclass
class NVFP4Weight:
    """Device layout consumed by `tensorbridge.fp8_nvfp4_gemm`."""

    weight: torch.Tensor  # int32 [N, K / 8], SNC-recoded, in thread order
    scale: torch.Tensor  # fp8 [K / 16, N], prefolded
    global_scale: torch.Tensor  # fp32 [1], times 6 * alpha
    # Lock words ordering the Stream-K slices of an output tile; all zero between launches.
    locks: torch.Tensor  # int32 [1024]

    @property
    def shape(self) -> tuple[int, int]:
        return self.scale.size(1), self.scale.size(0) * GROUP_SIZE


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-to-nearest NVFP4 quantization of `[N, K]` weights (ties toward zero)."""
    n, k = weight.shape
    assert k % GROUP_SIZE == 0
    groups = weight.float().view(n, k // GROUP_SIZE, GROUP_SIZE)
    group_scale = groups.abs().amax(dim=-1).clamp_min(1e-30) / E2M1_MAX

    scaled = groups * (1 / group_scale).unsqueeze(-1)
    midpoints = torch.tensor(_E2M1_MIDPOINTS, device=weight.device)
    codes = torch.bucketize(scaled.abs(), midpoints) | (torch.signbit(scaled) << 3)
    codes = codes.to(torch.uint8).view(n, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)

    # The group scales are themselves stored in E4M3, relative to a global
    # scale that maps the largest of them onto the E4M3 maximum. This keeps the
    # spread within the range the bridge supports.
    global_scale = (group_scale.max() / E4M3_MAX).view(1)
    scale = (group_scale / global_scale).to(torch.float8_e4m3fn)
    return packed.contiguous(), scale, global_scale


def prepare(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    clamp_scale: bool = False,
    alpha: float = FPMA_ALPHA,
) -> NVFP4Weight:
    """Checkpoint format -> kernel layout. `N` and `K` must be multiples of 128.

    Groups whose scale byte is below the bridge's range are zeroed, as their
    weights are negligible. Larger scale bytes raise, unless `clamp_scale` is
    set, which changes the affected groups' values and is meant for benchmarking
    checkpoints that were not quantized for this kernel. `alpha` is folded into
    the global scale.
    """
    n, half_k = packed.shape
    k = half_k * 2
    assert packed.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
    assert scale.shape == (n, k // GROUP_SIZE)
    assert n % 128 == 0 and k % 128 == 0, "N and K must be multiples of 128"

    raw_scale = scale.view(torch.uint8).to(torch.int16)
    # These groups must yield zeros, but the bridge would still emit `0 + value`.
    underflow = raw_scale < RAW_SCALE_MIN
    groups = packed.view(n, k // GROUP_SIZE, GROUP_SIZE // 2)
    packed = groups.masked_fill(underflow.unsqueeze(-1), 0).view(n, half_k)
    raw_scale = raw_scale.masked_fill(underflow, 0)

    if clamp_scale:
        raw_scale = raw_scale.clamp(max=RAW_SCALE_MAX)
    if (raw_scale > RAW_SCALE_MAX).any():
        raise ValueError(
            f"E4M3 weight scales must not exceed {RAW_SCALE_MAX:#x}; see `clamp_scale`"
        )

    prefolded = (raw_scale - PREFOLD_DELTA).clamp(0, 255).to(torch.uint8).view(torch.float8_e4m3fn)
    if global_scale is None:
        global_scale = torch.ones(1, device=packed.device)
    return NVFP4Weight(
        weight=to_thread_order(recode_snc(packed)).view(torch.int32),
        scale=prefolded.t().contiguous(),
        global_scale=global_scale.float().view(1) * (E2M1_MAX * alpha),
        locks=torch.zeros(1024, dtype=torch.int32, device=packed.device),
    )


def lsq_alpha(packed: torch.Tensor, scale: torch.Tensor) -> float:
    """Per-layer `alpha` with the least squared error between the bridge's weights and
    the exact NVFP4 dequantization; computed offline from the checkpoint."""
    raw = scale.view(torch.uint8).to(torch.int32).unsqueeze(-1)
    codes = torch.stack((packed & 0xF, packed >> 4), -1).view(*scale.shape, 16).to(torch.int32)
    magnitude = codes & 0x7
    kept = (magnitude != 0) & (raw >= RAW_SCALE_MIN)
    values = torch.tensor(_E2M1_VALUES, device=packed.device)
    exact = values[magnitude] * scale.float().unsqueeze(-1) / E2M1_MAX
    byte = (raw - PREFOLD_DELTA).clamp(min=0) + (torch.where(magnitude == 1, 0, magnitude) << 2)
    bridge = byte.to(torch.uint8).view(torch.float8_e4m3fn).float()
    exact, bridge = exact * kept, bridge * kept
    return ((exact * bridge).sum() / (bridge * bridge).sum()).item()


def recode_snc(packed: torch.Tensor) -> torch.Tensor:
    """Stored-nonzero convention: 0.5 (magnitude 1) -> 0, and zero -> the freed code 1.

    The sign of a zero is dropped, so both zeros become nibble 0x1.
    """

    def recode(nibble: torch.Tensor) -> torch.Tensor:
        magnitude = nibble & 0x7
        swapped = torch.where(magnitude == 1, nibble & 0x8, nibble)
        return torch.where(magnitude == 0, torch.ones_like(nibble), swapped)

    return recode(packed & 0xF) | (recode(packed >> 4) << 4)


def to_thread_order(packed: torch.Tensor) -> torch.Tensor:
    """Permute each 128-channel x 128-deep tile into the order the threads read it.

    Within a tile, consumer thread `t = warp * 32 + lane` owns channel rows
    `r = lane / 4 + (warp % 4) * 16 + (warp / 4) * 64` and `r + 8`, and in every
    k32 call the K columns `c .. c + 3` and `c + 16 .. c + 19`, `c = (lane % 4) * 4`.
    Its 16-byte vector for the call pair `(i, i + 1)` is four words

        [call i, row r] [call i, row r + 8] [call i + 1, row r] [call i + 1, row r + 8]

    where each word interleaves its K-low and K-high nibbles (low nibble of a
    byte = K-low), exactly as `fpma::dequant_word` splits them. The vectors are
    stored at `pair * 4096 + t * 16`, as seen *through* the 64-byte swizzle that
    TMA applies to the weight tile, so the permutation includes its inverse.
    """
    n, half_k = packed.shape
    device = packed.device
    warp, lane = torch.meshgrid(
        torch.arange(8, device=device), torch.arange(32, device=device), indexing="ij"
    )
    row = (lane // 4 + (warp % 4) * 16 + (warp // 4) * 64).flatten()  # [256]
    col = ((lane % 4) * 4).flatten()

    # Source (row, column) of each of the 32 nibbles of a thread's vector.
    call = torch.arange(4, device=device).view(2, 2, 1, 1)  # [pair, call in pair, 1, 1]
    row_offset = torch.tensor([0, 8], device=device).view(1, 1, 2, 1)
    nibble = torch.arange(8, device=device).view(1, 1, 1, 8)
    k_offset = call * 32 + nibble // 2 + (nibble % 2) * 16  # even nibbles K-low, odd K-high
    src_row = (row.view(1, -1, 1, 1, 1) + row_offset.unsqueeze(1)).expand(2, 256, 2, 2, 8)
    src_col = (col.view(1, -1, 1, 1, 1) + k_offset.unsqueeze(1)).expand(2, 256, 2, 2, 8)

    # [N tiles, K tiles, 128 channels, 64 bytes]
    tiles = packed.view(n // 128, 128, half_k // 64, 64).permute(0, 2, 1, 3)
    codes = torch.stack([tiles & 0xF, tiles >> 4], dim=-1).flatten(-2)  # [.., 128, 128] nibbles
    gathered = codes[:, :, src_row, src_col]  # [.., pair, thread, call, row, nibble]
    vectors = gathered[..., 0::2] | (gathered[..., 1::2] << 4)  # [.., 2, 256, 2, 2, 4] bytes
    physical = vectors.reshape(*tiles.shape[:2], 128 * 64)

    # 64B swizzle: the index of a 16-byte group within its 64-byte row is XORed
    # with bits 7-8 of the address. It is its own inverse.
    address = torch.arange(128 * 64, device=device)
    logical = address ^ (((address >> 7) & 0x3) << 4)
    out = torch.empty_like(physical)
    out[..., logical] = physical
    return out.view(*tiles.shape[:2], 128, 64).permute(0, 2, 1, 3).reshape(n, half_k).contiguous()
