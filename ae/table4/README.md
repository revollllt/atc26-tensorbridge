# Table 4: zero-shot accuracy

Table 4 reports the mean zero-shot accuracy over ARC-Challenge, ARC-Easy, WinoGrande,
HellaSwag and PIQA (lm-eval 0.4.11; `acc` for WinoGrande, `acc_norm` for the others). It covers
five models under BF16 and seven quantization methods. The script also runs Ours with the
TensorBridge kernel inside vLLM (`ours_kernel`, dense models only).

## Run (per method: 10-20 minutes on a dense model, about 45 minutes on Qwen3.6-35B-A3B)

This needs the checkpoints from [`eval/README.md`](../../eval/README.md). On one H100:

```bash
export MODEL_ROOT=/path/to/models
bash ae/table4/run.sh                           # all models and methods
MODELS=qwen3_8b bash ae/table4/run.sh ours      # one model, one method
```

## Output

Each run writes `ae/table4/results/<model>/<method>-tasks.json`, with the per-task scores and
their mean. The script then prints the table and saves it to `ae/table4/results/table4.md`.
The paper's value is in parentheses next to each measurement, and `*` marks a `LIMIT` run.

## Expected results

These are the paper values (Table 4). `Avg.` is the mean over the five models. On the
`ours_kernel` row, the output puts the paper's Ours value in parentheses; our run of
`ours_kernel` gave the values below. Recreated checkpoints and lm-eval batch sizes change these
means by up to 0.6 points. The paper's comparison is the five-model average: Ours 71.98 vs
W4A8 71.39.

| Method | Llama2-7B | Llama3-8B | Qwen3-8B | Qwen3-14B | Qwen3.6-35B-A3B | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| `bf16` (BF16) | 68.62 | 72.80 | 71.56 | 74.97 | 73.34 | 72.26 |
| `w4a8` (W4A8) | 68.38 | 71.25 | 69.77 | 74.39 | 73.18 | 71.39 |
| `nvfp4a16` (NVFP4A16) | 68.21 | 71.61 | 71.31 | 74.20 | 73.44 | 71.75 |
| `nvfp4` (NVFP4) | 67.59 | 70.67 | 70.86 | 72.91 | 72.58 | 70.92 |
| `mxfp4` (MXFP4) | 66.20 | 69.38 | 67.09 | 72.15 | 73.31 | 69.63 |
| `nvfp4a8` (NVFP4A8 Exact) | 68.39 | 71.63 | 71.19 | 74.37 | 72.90 | 71.70 |
| `ours_nosnc` (Ours w/o SNC) | 67.82 | 71.85 | 71.74 | 74.18 | 73.48 | 71.81 |
| `ours` (Ours) | 68.29 | 71.90 | 71.68 | 74.40 | 73.62 | 71.98 |
| `ours_kernel` | 68.05 | 71.98 | 71.45 | 74.67 | - | - |
