"""NVFP4 weights: quantization, and the offline preparation for the kernel.

Checkpoint format (what `quantize` returns and `prepare` accepts):

    packed        uint8  [N, K / 2]   E2M1 codes; byte b = code[2b] | code[2b + 1] << 4
    scale         fp8    [N, K / 16]  one E4M3 scale per 16 weights
    global_scale  fp32   [1]          weight = decode(code) * scale * global_scale

Everything the kernel would otherwise pay for per launch is moved into
`prepare`, which runs once per layer:

    0. padding           zero codes and zero scales up to N and K multiples of 128 (`codegen.pad_to_unit`),
                         which the bridge turns into weights of exactly 0
    1. scale prefold       raw E4M3 byte - 0x1C (`bridge="exact"`: q = raw - 0x1C - d for the
                           correction d, stored as 2q + d, `prefold_exact`), and global_scale * 6
    2. SNC recoding        0.5 -> magnitude 0, zero -> magnitude 1
    3. thread-order repack one 16-byte vector per consumer thread
    4. magnitude grouping  per word, the K-low magnitudes in the low half, K-high in the high half

Steps 1, 2 and 4 serve the FPMA bridge's table lookup (`include/tensorbridge/dequant/fpma.cuh`),
step 3 the register loads (`fpma::load_weights_pair` in the same header).
"""

import dataclasses

import torch
import triton
import triton.language as tl

from tensorbridge.jit.codegen import Bridge, pad_to_unit
from tensorbridge.quant.modelopt import (
    E2M1_MAX,
    E2M1_MIDPOINTS,
    E4M3_MAX,
    GROUP_SIZE,
    RAW_SCALE_MAX,
    RAW_SCALE_MIN,
)
from tensorbridge.quant.modelopt import (
    QuantMode as QuantMode,
)
from tensorbridge.quant.modelopt import (
    quantize_fp8_block_to_nvfp4_modelopt as quantize_fp8_block_to_nvfp4_modelopt,
)
from tensorbridge.quant.modelopt import (
    quantize_modelopt as quantize_modelopt,
)

PREFOLD_DELTA = 0x1C


@dataclasses.dataclass
class NVFP4Weight:
    """Device layout consumed by `tensorbridge.fp8_nvfp4_gemm`: the weight zero-padded to `N_pad` and `K_pad`, N and
    K rounded up to multiples of 128 (`codegen.pad_to_unit`)."""

    weight: torch.Tensor  # int32 [N_pad, K_pad / 8], SNC-recoded, in thread order
    scale: torch.Tensor  # fp8 [K_pad / 16, N_pad], prefolded (2q + d for `bridge="exact"`)
    global_scale: torch.Tensor  # fp32 [1], times 6
    # Lock words ordering the Stream-K slices of an output tile; all zero between launches.
    locks: torch.Tensor  # int32 [1024]
    # `(N, K)` of the checkpoint: the activations are [M, K] and the output [M, N].
    shape: tuple[int, int]
    # The bridge the scales are prepared for, which selects the kernel (`fp8_nvfp4_gemm`).
    bridge: Bridge = "exact"


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-to-nearest NVFP4 quantization of `[N, K]` weights (ties toward zero)."""
    n, k = weight.shape
    if k % GROUP_SIZE:
        raise ValueError(f"K must be a multiple of {GROUP_SIZE}, got {k}")
    groups = weight.float().view(n, k // GROUP_SIZE, GROUP_SIZE)
    group_scale = groups.abs().amax(dim=-1).clamp_min(1e-30) / E2M1_MAX

    scaled = groups * (1 / group_scale).unsqueeze(-1)
    midpoints = torch.tensor(E2M1_MIDPOINTS, device=weight.device)
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
    bridge: Bridge = "exact",
) -> NVFP4Weight:
    """Checkpoint format -> kernel layout, zero-padded to `N` and `K` multiples of 128; `global_scale` defaults to
    1. `N` must be a multiple of 8 and `K` of 16 (`K` is one already, the scale group): the activations' and the
    output's rows are TMA strides, multiples of 16 bytes.

    `bridge="exact"` (the default) feeds the tensor cores the RNE E4M3 rounding of every weight
    times its scale (raw scales 0x23 to 0x7E), for standard NVFP4 checkpoints, at a few percent more
    time; `bridge="fpma"` is for FPMA-aware checkpoints, whose quantization accounts for the FPMA
    approximation.

    Groups whose scale byte is below the bridge's range are zeroed, as their
    weights are negligible. Larger scale bytes raise, unless `clamp_scale` is
    set, which changes the affected groups' values and is meant for benchmarking
    checkpoints that were not quantized for this kernel.
    """
    n, half_k = packed.shape
    k = half_k * 2
    if packed.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"packed must be uint8 and scale float8_e4m3fn, got {packed.dtype} and {scale.dtype}")
    if scale.shape != (n, k // GROUP_SIZE):
        raise ValueError(f"scale must have shape {(n, k // GROUP_SIZE)}, got {tuple(scale.shape)}")
    if n <= 0 or k <= 0 or n % 8 or k % GROUP_SIZE:
        raise ValueError(f"N must be a positive multiple of 8 and K of {GROUP_SIZE}, got N = {n}, K = {k}")

    # Zero scale bytes are below the bridge's range, so the padding's groups are zeroed with the others below.
    padded_n, padded_k = pad_to_unit(n), pad_to_unit(k)
    packed = torch.nn.functional.pad(packed, (0, (padded_k - k) // 2, 0, padded_n - n))
    raw_scale = torch.nn.functional.pad(scale.view(torch.uint8), (0, (padded_k - k) // GROUP_SIZE, 0, padded_n - n))
    raw_scale = raw_scale.to(torch.int16)
    # These groups must yield zeros, but the bridge would still emit `0 + value`.
    underflow = raw_scale < RAW_SCALE_MIN
    groups = packed.view(padded_n, padded_k // GROUP_SIZE, GROUP_SIZE // 2)
    packed = groups.masked_fill(underflow.unsqueeze(-1), 0).view(padded_n, padded_k // 2)
    raw_scale = raw_scale.masked_fill(underflow, 0)

    if clamp_scale:
        raw_scale = raw_scale.clamp(max=RAW_SCALE_MAX)
    if (raw_scale > RAW_SCALE_MAX).any():
        raise ValueError(
            f"E4M3 weight scales must not exceed {RAW_SCALE_MAX:#x}; see `clamp_scale`"
        )

    if bridge == "exact":
        prefolded = prefold_exact(raw_scale).view(torch.float8_e4m3fn)
    else:
        prefolded = (raw_scale - PREFOLD_DELTA).clamp(0, 255).to(torch.uint8).view(torch.float8_e4m3fn)
    if global_scale is None:
        global_scale = torch.ones(1, device=packed.device)
    return NVFP4Weight(
        weight=group_magnitudes(to_thread_order(recode_snc(packed)).view(torch.int32)),
        scale=prefolded.t().contiguous(),
        global_scale=global_scale.float().view(1) * E2M1_MAX,
        locks=torch.zeros(1024, dtype=torch.int32, device=packed.device),
        shape=(n, k),
        bridge=bridge,
    )


@dataclasses.dataclass
class NVFP4Experts:
    """Device layout of `E` stacked experts, consumed by `tensorbridge.fp8_nvfp4_grouped_gemm`:
    expert `e` is laid out exactly as `prepare` lays out its weights alone."""

    weight: torch.Tensor  # int32 [E, N, K / 8], SNC-recoded, in thread order
    scale: torch.Tensor  # fp8 prefolded: [E, K/16, N], or [E, K/128, N, 8] with group-128 input
    global_scale: torch.Tensor  # fp32 [E], times 6
    locks: torch.Tensor  # int32 [1024]
    bridge: Bridge = "exact"

    @property
    def shape(self) -> tuple[int, int, int]:
        """`(E, N, K)`."""
        return self.weight.size(0), self.weight.size(1), self.weight.size(2) * 8

    def expert(self, e: int) -> NVFP4Weight:
        """Views, not copies, of expert `e`'s tensors: what `prepare` returns for it alone."""
        return NVFP4Weight(self.weight[e], self.scale[e], self.global_scale[e:e + 1], self.locks, self.shape[1:],
                           self.bridge)


def prepare_experts(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    clamp_scale: bool = False,
    bridge: Bridge = "exact",
) -> NVFP4Experts:
    """Checkpoint format of `E` experts -> the grouped kernel's layout: `packed` uint8
    `[E, N, K / 2]`, `scale` E4M3 `[E, N, K / 16]` and `global_scale` `[E]` (default: ones), as `prepare`
    takes them per expert, with the same scale rules (`clamp_scale`, zeroed groups) and `bridge`. `N` and `K` must be
    multiples of 128: the grouped kernels write their output rows without the padding's column bound
    (`impls/sm90_w4a8_ptoken.cuh`).

    The transform of `prepare` works per 128 x 128 tile and per element, so it runs once on the
    `[E * N, K / 2]` view; only the scale transpose is per expert.
    """
    num_experts, n, half_k = packed.shape
    k = half_k * 2
    if n % 128 or k % 128:
        raise ValueError(f"N and K must be multiples of 128, got N = {n}, K = {k}")
    if not clamp_scale:
        raw_scale = scale.view(torch.uint8)
        too_large = raw_scale > RAW_SCALE_MAX
        if too_large.any():
            expert = int(too_large.flatten(1).any(1).nonzero()[0])
            byte = int(raw_scale[expert][too_large[expert]].max())
            raise ValueError(
                f"expert {expert}: E4M3 weight scale byte {byte:#x} exceeds {RAW_SCALE_MAX:#x}; see `clamp_scale`"
            )
    stacked = prepare(
        packed.reshape(num_experts * n, half_k), scale.reshape(num_experts * n, k // GROUP_SIZE),
        clamp_scale=clamp_scale, bridge=bridge,
    )
    return NVFP4Experts(
        weight=stacked.weight.view(num_experts, n, k // 8),
        scale=stacked.scale.view(k // GROUP_SIZE, num_experts, n).permute(1, 0, 2).contiguous(),
        global_scale=(torch.ones(num_experts, device=packed.device) if global_scale is None
                      else global_scale.float().view(num_experts)) * E2M1_MAX,
        locks=stacked.locks,
        bridge=bridge,
    )


def interleave_gate_up_experts(experts: NVFP4Experts) -> NVFP4Experts:
    """Offline pack `[E, 2*I, K]` gate-then-up `NVFP4Experts` into 128-channel interleaved `(gate_128, up_128)` pairs.

    Because `prepare_experts` operates strictly within each 128-channel tile, interleaving 128-channel blocks on
    already-prepared weights and scales is exact and zero-loss, enabling `BLOCK_N=256` fused
    `SwiGLU + group-128 FP8` epilogue with zero cross-CTA reduction.
    """
    num_experts, two_i, k = experts.shape
    if two_i % 256 != 0:
        raise ValueError(f"interleave_gate_up_experts requires N = 2*I to be a multiple of 256, got {two_i}")
    num_groups = two_i // 256
    weight = (
        experts.weight.reshape(num_experts, 2, num_groups, 128, k // 8)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_experts, two_i, k // 8)
        .contiguous()
    )
    if experts.scale.dim() == 4:
        # group-128 layout: [E, K/128, 2*I, 8]
        k_blocks = experts.scale.size(1)
        scale = (
            experts.scale.reshape(num_experts, k_blocks, 2, num_groups, 128, 8)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(num_experts, k_blocks, two_i, 8)
            .contiguous()
        )
    elif experts.scale.dim() == 3:
        # per-token layout: [E, K/16, 2*I]
        k_groups = experts.scale.size(1)
        scale = (
            experts.scale.reshape(num_experts, k_groups, 2, num_groups, 128)
            .permute(0, 1, 3, 2, 4)
            .reshape(num_experts, k_groups, two_i)
            .contiguous()
        )
    else:
        raise ValueError(f"Expected 3D or 4D expert scale tensor, got dim={experts.scale.dim()}")
    return NVFP4Experts(
        weight=weight,
        scale=scale,
        global_scale=experts.global_scale,
        locks=experts.locks,
        bridge=experts.bridge,
    )


def prefold_exact(raw_scale: torch.Tensor) -> torch.Tensor:
    """Raw E4M3 scale bytes (int16, in range, or 0 for zeroed groups) -> the prefolded bytes of the
    exact table (`dequant/fpma.cuh`): `Q = 2q + d` with `q = raw - 0x1C - d`, where `d = 1` when the
    raw mantissa is 0, 1, 2, 6 or 7, the scales on which the encoding-domain add rounds the even
    magnitudes one E4M3 step high. Zeroed groups keep `d = 0` and prefold to 0, as in `fpma`."""
    mantissa = raw_scale & 0x7
    flag = (((mantissa <= 2) | (mantissa >= 6)) & (raw_scale >= RAW_SCALE_MIN)).to(raw_scale.dtype)
    prefolded = (raw_scale - PREFOLD_DELTA - flag).clamp(0, 0x7F)
    return (2 * prefolded + flag).to(torch.uint8)


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
    byte = K-low); `group_magnitudes` then regroups each word for the bridge. The vectors are
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


@triton.jit
def compact_magnitudes(magnitudes):
    """Bits 8i .. 8i + 2 -> 4i .. 4i + 2 of a uint32, for i = 0 .. 3."""
    return (magnitudes & 0x7) | ((magnitudes >> 4) & 0x70) | ((magnitudes >> 8) & 0x700) | ((magnitudes >> 12) & 0x7000)


@triton.jit
def group_magnitudes_kernel(words_ptr, grouped_ptr, num_words, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offsets < num_words
    word = tl.load(words_ptr + offsets, mask=in_range).to(tl.uint32, bitcast=True)
    low = compact_magnitudes(word & 0x07070707)
    high = compact_magnitudes((word >> 4) & 0x07070707)
    grouped = (word & 0x88888888) | low | (high << 16)
    tl.store(grouped_ptr + offsets, grouped.to(tl.int32, bitcast=True), mask=in_range)


def group_magnitudes(words: torch.Tensor) -> torch.Tensor:
    """Regroup each int32 word of `to_thread_order` for the table lookup of `fpma::dequant`.

    In, byte `i` of a word is `lo_i | hi_i << 4`: the SNC codes of the `i`-th K-low and K-high
    value. Out, bits `4i .. 4i + 2` hold the magnitude of `lo_i` and bits `16 + 4i .. 16 + 4i + 2`
    that of `hi_i`, so each half is four PRMT selectors; the sign bits `8i + 3` (`lo_i`) and
    `8i + 7` (`hi_i`) stay where they are. One fused pass on the GPU: as separate tensor
    operations it would cost most of `prepare`.
    """
    assert words.is_cuda and words.dtype == torch.int32
    words = words.contiguous()
    grouped = torch.empty_like(words)
    block = 1024
    group_magnitudes_kernel[(triton.cdiv(words.numel(), block),)](words, grouped, words.numel(), block)
    return grouped
