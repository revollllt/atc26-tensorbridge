# Benchmarks & Evaluation

```text
benchmarks/
├── common.py                      Level 1 workload definitions, timing protocol, result serialization
├── compare.py                     Level 1 same-process GEMM comparison across all backends
├── bench_tensorbridge.py          Level 1: TensorBridge NVFP4 x FP8 (dense; indexed / contiguous / masked)
├── bench_humming.py               Level 1: Humming MXFP4 x FP8 & NVFP4 x BF16
├── bench_marlin.py                Level 1: Marlin NVFP4 x BF16
├── bench_deepgemm.py              Level 1: DeepGEMM FP8 x FP8
├── bench_torch.py                 Level 1: torch._scaled_mm FP8 (cuBLAS) and BF16
├── shapes/                        Fixed dense8, dense8_ffn, and EP MoE shape CSVs
├── bench_moe_layer_ep8_compare.py Level 2: Complete 8-GPU EP8 MoE layer comparison (incl. external MegaMoE)
├── bench_moe_mlp_e2e.py           Level 2: Single-GPU MoE MLP breakdown & native FP4 adapter verification
├── serve_common.py                Shared Level 3 vLLM server lifecycle, SSE stream client & Prometheus helpers
├── bench_vllm_serve_scaleup.py    Level 3: vLLM serving (colocated & 2-node PD-disaggregated Prefill/Decode)
├── bench_pd_prefill_decode.py     Level 3: Long-context PD Prefill (TTFT vs ISL/cache hit) & Decode (TPOT vs C)
├── eval_wikitext_ppl.py           Accuracy: Standard GPTQ/AWQ WikiText-2 (seqlen=2048) perplexity evaluator
└── sweep_moe_profiles.py          Offline H20/H100 MoE tile profile generator

tests/
├── test_nvfp4_quant.py            ModelOpt NVFP4 quantizer modes, zero-group & <512 MiB CUDA peak checks
├── test_wikitext_ppl.py           WikiText-2 windowing & FP32 GPTQ loss arithmetic checks
├── test_serve_scaleup.py          Preset expansion, profile floor, PD summary, SSE coalescing checks
├── test_pd_prefill_decode.py      PD Prefill cache-hit split/metrics & Decode TPOT/DCP command checks
├── test_moe_layer_ep8_compare.py  EP8 reference, SM90 H20 heuristic tile selection, scatter-bound checks
└── test_nvlink_ep_8gpu.py         8-GPU NVLinkSymmetricEP correctness, CUDA Graph & real-weight check
```

Install benchmark dependencies:

```bash
pip install -e ".[cu13,bench]"   # installs vllm==0.32.0 and humming-kernels==0.1.17
```

---

## Tests

Run CPU + single-GPU unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

Run the 8-GPU NVLink Symmetric EP test on an 8×H20 node:

```bash
torchrun --standalone --nproc-per-node=8 tests/test_nvlink_ep_8gpu.py
```

---

## Level 1: Single-GPU GEMM Microbenchmarks

`compare.py` captures CUDA graphs with HBM-streamed weight copies (>2× L2) and replays backends in
alternating rounds, reporting geometric-mean speedups and worst shapes.

```bash
# Dense FFN shapes
python benchmarks/compare.py --suites dense8_ffn --rounds 3

# GLM-5.3 MoE EP shapes (per-token exact bridge)
python benchmarks/compare.py --suites ep_glm53_gate_up ep_glm53_down --bridge exact --rounds 3

# Matched group-128 activation comparison (TensorBridge g128 vs Humming MXFP4 g128)
python benchmarks/compare.py --suites ep_glm53_gate_up ep_glm53_down --bridge exact --group-128 --rounds 3

# Single library timing
python benchmarks/bench_tensorbridge.py --shape_n 8192 --shape_k 8192
python benchmarks/bench_humming.py --shape_n 512 --shape_k 512 --num_experts 8 --top_k 2 --shape_m_list 64
```

Shape sets under `benchmarks/shapes/`:
- `dense8.csv` / `dense8_ffn.csv`: dense projections from 8 MoE models (TP=1, decode M=1..128 & prefill M=512..16384).
- `ep_<model>_<projection>.csv`: per-rank EP4/8/16/32 `gate_up` and `down` shapes (`masked` T=1..128, `contiguous` T=512..16384).

---

## Level 2: Complete 8-GPU EP8 MoE Layer

`bench_moe_layer_ep8_compare.py` times a full 8-rank EP8 GLM-5.3 MoE layer (`E=256`, `local_E=32`,
`top_k=8`, `H=6144`, `I=2048`) with CUDA-Graph 50-replay GPU events after an NCCL barrier, starting
every backend and the distributed FP32 reference from identical block-128 FP8 weights.

```bash
for backend in tb_nvfp4_fp8 tb_nvfp4_fp8_g128 tb_nvfp4_megamoe \
               humming_mxfp4 humming_mxfp4_g128 humming_nvfp4 marlin_nvfp4; do
  torchrun --standalone --nproc-per-node=8 benchmarks/bench_moe_layer_ep8_compare.py \
    --backend "$backend" --all2all-backend allgather_reducescatter \
    --routing uniform --tokens 1 2 4 8 16 32 64 128 256 512 \
    --rounds 5 --iters 50 --check-correctness \
    --output "results/ep8/$backend.jsonl"
done
python benchmarks/bench_moe_layer_ep8_compare.py --summarize results/ep8/*.jsonl --output results/ep8/comparison
```

Supported flags:
- `--routing uniform|zipf|mixed_empty`
- `--with-shared-mlp --shared-intermediate 2048` (measures shared-expert overlap)
- `--all2all-backend allgather_reducescatter|flashinfer_nvlink_one_sided`
- `--backend dg_upstream_sm90_fp8_megamoe|sgl_pr53_sm90_fp8_megamoe|sgl_pr53_sm90_mxfp4_megamoe --deepgemm-path PATH`
- `--profile` (Nsight `ep8_layer` NVTX capture for per-kernel attribution)

Single-GPU adapter & kernel breakdown check:

```bash
python benchmarks/bench_moe_mlp_e2e.py --check-adapters
python benchmarks/bench_moe_mlp_e2e.py --tokens 32 128 --kernel-breakdown
```

---

## Level 3: Full-Model vLLM Serving & Standard WikiText-2 PPL

`bench_vllm_serve_scaleup.py` launches vLLM 0.32 on a block-128 FP8 MoE checkpoint (e.g., GLM-5.3) with strict MoE
enforcement (`--strict-moe`), BF16 model/head + FP32 `raw_logprobs`, locked KV cache
(`--num-gpu-blocks-override`), verified token counts, and SSE coalescing diagnostics. Allocate ample
host CPUs (e.g. `--cpus-per-task=64 --mem=0` on Slurm) to prevent SSE stream coalescing.

### Colocated DP8+EP8 Serving + WikiText-2 (`seqlen=2048`)

```bash
python benchmarks/bench_vllm_serve_scaleup.py \
  --mode tb_nvfp4_fp8_g128 --model /path/to/GLM-5.3 \
  --dp 8 --tp 1 --all2all-backend allgather_reducescatter \
  --strict-moe --no-enable-prefix-caching \
  --num-gpu-blocks-override 1000 \
  --capture-sizes 1 2 4 8 16 32 64 128 256 \
  --max-model-len 4096 --max-num-seqs 256 --max-num-batched-tokens 4096 \
  --wikitext-path /path/to/wikitext-2-raw-v1-test.json \
  --wikitext-seqlen 2048 --wikitext-concurrency 8 \
  --prompt-lens 128 --decode-len 64 --concurrencies 8 32 64 128 256 --waves 3 \
  --out-json results/tb_nvfp4_fp8_g128.json
```

Pass `--nvfp4-quant-mode {mse_7ulp,four_over_six,bridge_aware,rtn}` to select the on-the-fly
FP8→NVFP4 quantizer (`mse_7ulp` is the default). Standalone WikiText-2 scoring against a running
BF16/`raw_logprobs` server:

```bash
python benchmarks/eval_wikitext_ppl.py --wikitext-path /path/to/wikitext-2-raw-v1-test.txt
```

### Two-Node Isolated Prefill & Decode Capacity (PD-Disaggregated)

Runs one independent DP8+EP8 server per node: Node 0 measures isolated Prefill (`--pd-role prefill --decode-len 1`),
and Node 1 measures isolated Decode using vLLM 0.32's native `DecodeBenchConnector` (`--pd-role decode --decode-len 128`).

```bash
# On Node 0 (Prefill):
python benchmarks/bench_vllm_serve_scaleup.py --mode tb_nvfp4_fp8 --dp 8 --tp 1 \
  --strict-moe --all2all-backend allgather_reducescatter --num-gpu-blocks-override 1000 \
  --no-enable-prefix-caching --max-model-len 4096 \
  --capture-sizes 1 2 4 8 16 32 64 128 256 --prompt-lens 128 1024 \
  --concurrencies 8 32 64 128 256 --waves 3 \
  --pd-role prefill --decode-len 1 --out-json results/pd/prefill.json

# On Node 1 (Decode):
python benchmarks/bench_vllm_serve_scaleup.py --mode tb_nvfp4_fp8 --dp 8 --tp 1 \
  --strict-moe --all2all-backend allgather_reducescatter --num-gpu-blocks-override 1000 \
  --no-enable-prefix-caching --max-model-len 4096 \
  --capture-sizes 1 2 4 8 16 32 64 128 256 --prompt-lens 128 1024 \
  --concurrencies 8 32 64 128 256 --waves 3 \
  --pd-role decode --decode-len 128 --out-json results/pd/decode.json

# Paired CSV summary:
python benchmarks/bench_vllm_serve_scaleup.py --summarize-pd results/pd/prefill.json results/pd/decode.json
```

### Long-Context PD Prefill (TTFT) & Decode (TPOT) Benchmark

`bench_pd_prefill_decode.py` runs P6KD-style isolated Prefill and Decode sweeps on a single 8×H20 node using
`DP=2, TP=4, EP=8` + `FLASHMLA_SPARSE` FP8 KV cache (`--kv-cache-dtype fp8`), which fits up to 320K context
for Prefill (~600K tokens/DP-rank) and high-concurrency 80K+600 Decode via `DCP=4` (~4.6M aggregate KV tokens)
while exercising the full EP8 all-to-all path across all backends:

```bash
# Prefill bench: 50K / 80K / 160K / 320K, 0% & 90% verified prefix cache hit -> TTFT
python benchmarks/bench_pd_prefill_decode.py \
  --bench prefill --mode tb_nvfp4_megamoe --strict-moe \
  --isl-list 50000 80000 160000 320000 --cache-hit-rates 0.0 0.9 --repeats 3 \
  --out-json results/pd_long/prefill_tb_nvlink.json

# Decode bench: 80K context + 600 output tokens, concurrency sweep -> TPOT
python benchmarks/bench_pd_prefill_decode.py \
  --bench decode --mode tb_nvfp4_megamoe --strict-moe \
  --decode-isl 80000 --decode-osl 600 --concurrencies 1 4 8 16 32 \
  --out-json results/pd_long/decode_tb_nvlink.json

# Markdown + CSV comparison table across backends:
python benchmarks/bench_pd_prefill_decode.py --summarize results/pd_long/prefill_*.json --out-csv results/pd_long/prefill.csv
python benchmarks/bench_pd_prefill_decode.py --summarize results/pd_long/decode_*.json --out-csv results/pd_long/decode.csv
```
