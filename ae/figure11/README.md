# Figure 11: GEMM kernel speedup

Figure 11 compares TensorBridge with eight GEMM kernels on 40 GEMM shapes. The shapes are
ten linear layers of Qwen3-8B, Llama3-70B and DeepSeek-V3 (`ae/figure11/data/shapes.csv`),
each at M = 16, 128, 512 and 4096 tokens.

## Replot from our data (CPU only)

```bash
bash ae/figure11/reproduce.sh
```

This reads `ae/figure11/data/latencies.md`, which holds the median time per call in
microseconds for each shape and kernel. It writes `ae/figure11/results/paper-data/figure11.png`
and `.pdf`. Compare with `ae/figure11/reference.pdf`.

## Measure on an H100 (about 5 minutes)

This needs the environment from the [top-level README](../../README.md).

```bash
bash ae/figure11/measure.sh     # includes kernel compilation
```

The script times TensorBridge, vLLM NVFP4A16 Marlin, vLLM CUTLASS W4A8, and cuBLAS FP8 and
BF16 on random data. cuBLAS runs through PyTorch and is labelled `Torch FP8` and `Torch BF16`
in the output. Each kernel runs as an isolated GEMM, replayed in a CUDA graph, and each cell is
the median over six rounds. Outputs:

- `ae/figure11/results/measured.md`: the latencies and the geometric-mean speedups, which are also printed;
- `ae/figure11/results/figure11.png` and `.pdf`: Figure 11. The QServe and TensorRT-LLM bars come
  from `ae/figure11/data/latencies.md`, because those libraries need their own environments.

## Expected results

Geometric-mean speedup of TensorBridge across the 40 shapes (Section 5.3):

| Baseline (`measured.md` label) | Speedup |
| --- | ---: |
| NVFP4A16 Marlin | 2.62x |
| vLLM CUTLASS W4A8 | 1.15x |
| Torch FP8 (cuBLAS FP8) | 1.30x |
| Torch BF16 (cuBLAS BF16) | 1.91x |
| *From `data/latencies.md` only:* | |
| QServe W4A8 | 2.26x |
| TensorRT-LLM FP8 | 1.37x |
| TensorRT-LLM W4A8 | 1.45x |
| TensorRT-LLM W4A16 | 1.86x |

## Other Hopper GPUs

The tuned kernel table `tensorbridge/tuned/NVIDIA_H100_80GB_HBM3.json` applies only to that
device name. Other GPUs, including H100 PCIe and H100 NVL, use the shape-based heuristics.

To tune for your GPU (about 12 minutes), run the command below. It writes
`tensorbridge/tuned/<GPU name>.json`; on an H100 SXM, it updates the shipped table.

```bash
python -m tensorbridge.autotune $(tail -n +2 ae/figure11/data/shapes.csv | awk -F, '{print $3"x"$4"x"$5}')
```
