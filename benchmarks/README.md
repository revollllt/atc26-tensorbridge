# Benchmarks

Kernel timing. One script per library, `bench_<library>.py`, measures one workload (a dense weight, or one MoE
projection) over a range of M and writes `results/<gpu>/<workload>/<library>_<recipe>.json`. `compare.py` runs
TensorBridge, humming, cuBLAS FP8, DeepGEMM and Marlin on fixed shape sets in one process, alternating them round
after round. The speedups it prints are the ones to compare.

```text
benchmarks/
├── common.py               workloads, the call protocol, timing, result files
├── bench_tensorbridge.py   NVFP4 x FP8, FPMA bridge: dense; MoE indexed / contiguous / masked
├── bench_humming.py        humming MXFP4 x FP8: dense; MoE indexed / contiguous / masked
├── bench_torch.py          torch._scaled_mm FP8 rowwise / tensorwise (cuBLAS), and BF16
├── bench_deepgemm.py       DeepGEMM FP8 x FP8: dense; MoE contiguous / masked
├── bench_marlin.py         Marlin NVFP4 x BF16: dense; prearranged MoE contiguous / masked
├── compare.py              same-process comparison on the shape sets
└── shapes/                 fixed shape sets
```

`pip install -e ".[cu13,bench]"` installs humming-kernels 0.1.17 and vLLM 0.32.0 (DeepGEMM and Marlin).

## Shape sets

| Set | Contents | Calls |
| --- | --- | ---: |
| `dense8.csv` | the dense GEMMs of eight MoE models at TP 1 (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, Qwen3.8-2.4T-A95B, DeepSeek-V4.1-Flash, DeepSeek-V4-Pro, GLM-5.3, GLM-5.3-Flash, Kimi-K3.1-Pro), 77 weights x M 1 to 128 (decode) and 512 to 16384 (prefill) | 924 |
| `dense8_ffn.csv` | the FFN and MoE linears of `dense8`: dense MLPs, shared experts and Kimi's latent MoE projections (24 weights x the same M) | 288 |
| `ep_<model>_<projection>.csv` | one rank of DeepEP-style expert parallelism (EP 4, 8, 16, 32) on one GPU, per model and projection (`gate_up`, `down`): masked (low-latency dispatch) at T = 1 to 128 tokens per rank, contiguous (high-throughput dispatch) at T = 512 to 16384; routes spread evenly over the rank's local experts | 48 each, 16 sets |

`dense8_ffn` is the dense set the README tables report; `dense8` also has attention, indexer and other projections.
Columns: dense sets `model, module, M, N, K`; EP sets `model, projection, ep, rank, N, K, num_experts,
local_experts, top_k, tokens, distribution, seed, layout, capacity`. A CSV in either format can be passed by path.

## One library

```bash
python benchmarks/bench_tensorbridge.py --shape_n 8192 --shape_k 8192
python benchmarks/bench_humming.py --shape_n 8192 --shape_k 8192 --shape_m_list 1 16 128
python benchmarks/bench_torch.py --shape_n 8192 --shape_k 8192
python benchmarks/bench_deepgemm.py --shape_n 8192 --shape_k 8192
python benchmarks/bench_marlin.py --shape_n 8192 --shape_k 8192
python benchmarks/bench_tensorbridge.py --shape_n 512 --shape_k 7168 --num_experts 256 --top_k 8
python benchmarks/bench_deepgemm.py --shape_n 2048 --shape_k 512 --num_experts 256 --top_k 8 --projection down
```

`--num_experts` other than 0 makes an MoE workload (`--top_k`, `--projection gate_up|down`, `--distribution
balanced|zipf`, `--ep` and `--rank` for one rank's local experts); its M is the token count. TensorBridge's
`--layouts` limits the MoE layouts. A result file holds, per M, the time of a call (ms), the memory bandwidth
(minimal bytes over time) and the compute throughput (`2 M N K` over time, routed rows for MoE). Each file is
measured on its own; compare libraries with `compare.py`.

TensorBridge times the FPMA kernel (`bridge="fpma"`). It does not compare the output to a reference. The other
entries check an output of their own format before timing (relative RMS at most 1%).

## Same-process comparison

```bash
python benchmarks/compare.py --suites dense8_ffn
python benchmarks/compare.py --suites dense8_ffn ep_glm53_gate_up ep_glm53_down
python benchmarks/compare.py --suites ep_glm53_gate_up ep_glm53_down --rounds 3
```

The README aggregate tables used `--rounds 3`. The default is 40 calls and 6 rounds, which is the protocol of the
TFLOPS figures.

- **Entries.** Dense shapes: TensorBridge, humming MXFP4 x FP8, cuBLAS FP8 rowwise, cuBLAS FP8 tensorwise, DeepGEMM
  FP8, Marlin NVFP4 x BF16. MoE shapes: TensorBridge, humming and, where that library has the layout, DeepGEMM and
  Marlin. A row's `layout` selects the entries (`masked` or `contiguous` in the EP sets). `bench_torch.py` also
  times BF16; `compare.py` does not.
- **Measurement.** Each entry's calls are captured into one CUDA graph, call `i` on weight copy `i mod copies`,
  and the graphs are replayed in alternating order. The median is the result. Activation quantization, MoE
  routing, padding and scale-layout conversion are not timed.
- **Streamed weights.** The weight copies together exceed twice the L2, so every call reads its weights from HBM
  (`--l2 warm` keeps one copy). A copy count limited by `--memory-budget-gib` is marked `stream_limited`.
- **Marlin MoE.** Native `moe_wna16_marlin_gemm` on BF16 activations, with row ids prearranged into the requested
  layout. Alignment, quantization, reduction and communication are outside the timed region.
- **Indexed EP.** TensorBridge and humming align a rank's routes with vLLM's `moe_align_block_size`, an expert map
  and `ignore_invalid_experts`.
- **Output.** `benchmarks/runs/<date>-<commit>/` (not in git): `results.csv` and `meta.json`. The process prints,
  per suite and layout, the geometric-mean speedup of TensorBridge over each baseline (baseline time /
  TensorBridge time) and the worst shape. A call a baseline cannot run is left out of its ratio.
- **Known limit.** humming's MoE path fails on weights of more than 2^31 elements per rank (some EP 4 calls of the
  largest models); those calls are `unsupported`.

Measurements on different machines, processes or times are not comparable: the same shape can differ by several
percent between nodes.
