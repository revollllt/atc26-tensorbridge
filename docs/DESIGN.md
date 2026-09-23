# TensorBridge design → code map

`tensorbridge/` holds one kernel: `D = A @ B.T` with FP8 E4M3 activations and
NVFP4 weights, for Hopper (sm_90). The layout follows the path of a call.

```text
tensorbridge/
  gemm.py, linear.py          Python entry points
  quant/nvfp4.py              NVFP4 quantization + offline weight preparation
  quant/fp8.py                per-token FP8 activation quantization (Triton)
  jit/codegen.py              generated translation unit: Config + WGMMA instruction
  jit/compiler.py             NVRTC + on-disk cubin cache
  jit/runtime.py              builds the launcher, registers compiled kernels
  csrc/heuristics/            shape -> GemmConfig (C++, no profiling)
  csrc/launcher/              TMA descriptors + driver-API launch (torch library)
  include/tensorbridge/
    impls/sm90_fp8_nvfp4_gemm.cuh   the kernel: producer loop, consumer mainloop, epilogue
    dequant/fpma.cuh                FPMA bridge
    mma/wgmma.cuh                   WGMMA issue
    memory/storage.cuh              shared-memory layout (stage ring, barriers)
    memory/g2s.cuh                  global -> shared    (producer: TMA + async copies)
    memory/s2r.cuh                  shared -> register  (consumer: weights, scales)
    memory/r2s.cuh                  register -> shared  (epilogue: scale, BF16, transpose)
    memory/s2g.cuh                  shared -> global    (epilogue: TMA store / Stream-K reduce)
    scheduler/gemm.cuh              data-parallel tiles + Stream-K tail
    common/, ptx/                   tile geometry, PTX instruction wrappers
```

Read in this order: `common/tile.cuh` (who owns which rows and columns),
`impls/sm90_fp8_nvfp4_gemm.cuh` (the whole dataflow on two pages), then the
file of whichever arrow you care about.

The mechanisms of the paper (Sec. 4), then the host side:

## 1. FPMA dequantization bridge

NVFP4 weights are reconstructed in the FP8 domain without BF16 intermediates.
For each packed E2M1 nibble the bridge forms an FP8-compatible byte by an
encoding-domain add `byte = prefolded_scale + (sign<<7 | mag<<2)`, then clears
true-zero lanes.

- Device bridge, both mask variants: `include/tensorbridge/dequant/fpma.cuh`.
- Overflow-safe scale prefolding (`raw_e4m3 - 0x1C`, global `*= 6`, optionally
  times a debias factor `alpha`) and offline
  SNC recoding (0.5 subnormal + zero marker): `quant/nvfp4.py` (`prepare`,
  `recode_snc`).
- Python reproduction + byte-level checks: `tests/fpma_bridge_reference.py`,
  `tests/test_fpma_bridge.py`.

## 2. WGMMA-oriented tiling and overlap pipeline

The CTA evaluates the transposed product `Y^T = W X^T` (operand swap) so the
register-constructed weight fragment goes on the register-capable WGMMA operand
and the token dimension maps to the flexible WGMMA `n`. The consumer warps run
the issue-first lagged schedule: S2R two k32 calls ahead, FPMA one ahead.

- Tile geometry and thread ownership: `include/tensorbridge/common/tile.cuh`.
- Producer / consumer mainloop (lagged, two-buffer and Stream-K-slice
  schedules): `include/tensorbridge/impls/sm90_fp8_nvfp4_gemm.cuh`.
- Stage ring and its barriers: `memory/storage.cuh`, `memory/g2s.cuh`,
  `memory/s2r.cuh`.
- Epilogue (STSM transposed store and scalar fallback): `memory/r2s.cuh`,
  `memory/s2g.cuh`.

## 3. Dual-MMA preinterleaved weight layout

Packed weights are permuted offline so a single 16-byte shared-memory read
yields, per thread, the fragments of two consecutive `k=32` calls, already in
WGMMA order and with K-low/K-high nibbles interleaved for the bridge.

- Offline layout: `quant/nvfp4.py` (`to_thread_order`).
- Device load: `memory/s2r.cuh` (`load_weights_pair`).
- Layout replayed thread by thread in Python:
  `tests/test_fpma_bridge.py::test_thread_order_feeds_each_thread_its_fragment`.

## 4. Workload-aware backend (data-parallel vs Stream-K tail)

A shape-only predictor fixes `B_N = B_K = 128`, picks the tile height `B_M`, and
decides whether to split only the underfilled MN tail with Stream-K, using the
wave-fill ratio `f` and an estimated reduction budget.

- Model: `csrc/heuristics/sm90.hpp` (`select_block_m`, `select_stream_k`,
  `select_config`).
- Measured per-shape overrides, kept out of the model:
  `csrc/heuristics/sm90_tuned.hpp`.
- Device side of the same split: `include/tensorbridge/scheduler/gemm.cuh`;
  ordered reduction of the slices: `memory/s2g.cuh`.

## JIT and launch

- One kernel per `(N, K, GemmConfig)`, compiled on first use and cached on disk:
  `jit/`.
- TMA descriptors and the driver-API launch on the current CUDA stream (so calls
  can be graph-captured): `csrc/launcher/`.

## Scope and known limitations

- Dense GEMM, `N` and `K` multiples of 128, BF16 output, Hopper (sm_90) only.
- Weight scales must fit the FPMA bridge: raw E4M3 bytes at most `0x7E`
  (`quant/nvfp4.py`; `prepare(..., clamp_scale=True)` for foreign checkpoints).
  Groups with a byte below `0x1C` are zeroed.
- Stream-K outputs are not bitwise reproducible run to run once a tile is cut
  into three or more slices: every slice is rounded to BF16 on its own and the
  slices after the first are added in whatever order their CTAs take the tile's
  lock, and BF16 addition is not associative. The spread is one or two BF16
  ULPs. Data-parallel configs are deterministic.
- Multicast clusters are data parallel only. Combined with Stream-K, the
  members of a cluster sporadically stored stale shared memory in the tail of
  their output tile; the kernel rejects that combination at compile time and
  the heuristics never produce it.
