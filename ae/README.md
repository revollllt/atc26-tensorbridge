# Figures and tables

Each folder here reproduces one figure or table of the evaluation (Section 5).
Run every command from the repository root. Outputs go to the `results/`
folder inside each figure or table folder.

## Quick start (CPU only)

This replots Figures 11-13 from the data we measured for the paper:

```bash
pip install -r ae/requirements.txt
bash ae/reproduce_all.sh
```

It writes `ae/figure11/results/figure11.pdf`, `ae/figure12/results/figure12.pdf`
and `ae/figure13/results/figure13.pdf`. Compare them with `reference.pdf` in
the same folder.

## What each folder covers

| Folder | Paper result | Replot from our data | Rerun on H100 | Rerun needs |
| --- | --- | --- | --- | --- |
| [figure11](figure11) | GEMM kernel speedup | `reproduce.sh` | `measure.sh`, about 5 min | nothing else |
| [figure12](figure12) | offline serving throughput | `reproduce.sh` | `measure.sh`, a few min per model and method | checkpoints |
| [figure13](figure13) | online serving throughput | `reproduce.sh` | `measure.sh`, about 30 min per model and method | checkpoints, ShareGPT |
| [table3](table3) | WikiText2 perplexity | - | `run.sh`, about 30 min per model | checkpoints |
| [table4](table4) | zero-shot accuracy | - | `run.sh`, 10-20 min per model and method | checkpoints |

The H100 runs need the environment from the [top-level README](../README.md).
The serving figures and the tables use quantized checkpoints; [`eval/`](../eval)
explains how to create them.

Figure 11 measures TensorBridge, vLLM's NVFP4A16 Marlin and CUTLASS W4A8 kernels,
and cuBLAS FP8 and BF16. The TensorRT-LLM and QServe columns come from our
paper runs, since those libraries need their own environments.
