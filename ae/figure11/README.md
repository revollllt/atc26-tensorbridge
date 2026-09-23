# Figure 11: GEMM kernel speedup

Figure 11 compares TensorBridge with eight GEMM kernels on 40 layer shapes from
Qwen3-8B, Llama3-70B and DeepSeek-V3, at M = 16, 128, 512 and 4096 tokens.

## Replot from our data (CPU only)

```bash
pip install -r ae/requirements.txt
bash ae/figure11/reproduce.sh
```

This reads `data/latencies.md` and writes `ae/figure11/results/figure11.png` and
`.pdf`, which should match `reference.pdf`. The table holds the median time per
call in microseconds for each shape and kernel. The paper's geometric-mean
speedups of TensorBridge (Section 5.3) come from this data:

| Baseline | Speedup |
| --- | ---: |
| cuBLAS BF16 | 1.91x |
| cuBLAS FP8 | 1.30x |
| vLLM CUTLASS W4A8 | 1.15x |
| QServe W4A8 | 2.26x |
| TensorRT-LLM FP8 | 1.37x |
| TensorRT-LLM W4A8 | 1.45x |
| TensorRT-LLM W4A16 | 1.86x |
| NVFP4A16 Marlin | 2.62x |

## Measure on an H100

With the environment from the [top-level README](../../README.md):

```bash
bash ae/figure11/measure.sh     # about 5 minutes, including kernel compilation
```

The script times TensorBridge, vLLM NVFP4A16 Marlin, vLLM CUTLASS W4A8, and
cuBLAS FP8 and BF16 (through PyTorch) on random data. Each kernel runs as an
isolated GEMM, replayed in a CUDA graph, and each cell is the median over six
rounds. At the end it prints the geometric-mean speedup of TensorBridge over
each baseline, for example:

```text
| baseline | speedup |
| --- | ---: |
| NVFP4A16 Marlin | 2.61 |
| vLLM CUTLASS W4A8 | 1.15 |
| Torch FP8 | 1.31 |
| Torch BF16 | 1.91 |
```

These should be within a few percent of the table above. The measured
latencies are in `ae/figure11/results/measured.md` and the plot in
`measured.png`; the plot leaves out the TensorRT-LLM and QServe bars, which
need their own environments.

TensorBridge picks a tuned kernel configuration for each shape from
`tensorbridge/tuned/NVIDIA_H100_80GB_HBM3.json`. On other GPUs it falls back to
its built-in heuristics. To tune for your GPU (about 12 minutes for the 40
shapes), pass the shapes as `MxNxK`:

```bash
python -m tensorbridge.autotune 16x6144x4096 128x6144x4096 ...
```
