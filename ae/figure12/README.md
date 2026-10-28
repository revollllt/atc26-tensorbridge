# Figure 12: offline serving throughput

Figure 12 compares the output throughput of vLLM with TensorBridge against BF16, FP8,
W4A8 (CUTLASS) and NVFP4A16 (Marlin). It covers Llama-3-8B, Qwen3-8B, Qwen3-14B and
Llama-3.3-70B. Each request has 1024 input and 512 output tokens, and the batch sizes are
1, 4, 16, 64 and 256. All runs use an FP8 KV cache.

## Replot from our data (CPU only)

```bash
bash ae/figure12/reproduce.sh
```

This reads `ae/figure12/data/measurements.csv` (4 models x 5 methods x 5 batch sizes) and
writes `ae/figure12/results/paper-data/figure12.png` and `.pdf`. Compare with
`ae/figure12/reference.pdf`.

## Measure on an H100 (a few minutes per model and method)

This needs:

- the environment from the [top-level README](../../README.md);
- the base model and its `NVFP4A8`, `NVFP4A16` and `W4A8` checkpoints, from
  [`eval/README.md`](../../eval/README.md#checkpoints).

`NAME` is the panel (`llama3_8b`, `qwen3_8b`, `qwen3_14b` or `llama33_70b`). `METHOD` is the
line. FP8 and BF16 load the base model, which vLLM quantizes to FP8 at load time for the FP8 line.
`BASE` is the model folder from [`eval/README.md`](../../eval/README.md#models).

```bash
export MODEL_ROOT=/path/to/models
export NAME=qwen3_8b BASE=$MODEL_ROOT/Qwen/Qwen3-8B
METHOD=tensorbridge MODEL=models/$NAME/NVFP4A8  bash ae/figure12/measure.sh
METHOD=nvfp4a16     MODEL=models/$NAME/NVFP4A16 bash ae/figure12/measure.sh
METHOD=w4a8         MODEL=models/$NAME/W4A8     bash ae/figure12/measure.sh
METHOD=fp8          MODEL=$BASE                 bash ae/figure12/measure.sh
METHOD=bf16         MODEL=$BASE                 bash ae/figure12/measure.sh
```

For Llama-3.3-70B, add `TP=2` to use two GPUs. Outputs:

- `ae/figure12/results/<NAME>/<METHOD>/batch-<B>.json`: one file per batch size;
- `ae/figure12/results/measurements.csv` and `ae/figure12/results/figure12.png` and `.pdf`:
  every model and method measured so far. A panel without data shows "not measured".

## Expected results

From Section 5.4.1 and `ae/figure12/data/measurements.csv`:

- Geometric-mean throughput of TensorBridge across the four models and five batch sizes:
  1.64x BF16, 1.09x FP8, 1.13x W4A8 and 1.35x NVFP4A16 Marlin.
- Per model, the geometric-mean gain over BF16 rises from 1.40x on Llama-3-8B to 2.33x on Llama-3.3-70B.
- Peak TensorBridge throughput: 10.3k, 9.7k, 6.7k and 3.2k output tokens/s on Llama-3-8B,
  Qwen3-8B, Qwen3-14B and Llama-3.3-70B.

Compare your measurement through `ae/figure12/results/measurements.csv`.
