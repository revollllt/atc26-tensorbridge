# Table 4: zero-shot accuracy

Table 4 reports the mean zero-shot accuracy over ARC-Challenge, ARC-Easy,
WinoGrande, HellaSwag and PIQA (lm-eval 0.4.11; `acc` for WinoGrande,
`acc_norm` for the others) for five models under BF16 and eight quantization
methods.

## Run

Set up the accuracy environment and create the checkpoints as described in
[`eval/README.md`](../../eval/README.md). Then, on one H100:

```bash
export MODEL_ROOT=/path/to/models
bash ae/table4/run.sh                           # all 5 models and 9 methods
MODELS=qwen3_8b bash ae/table4/run.sh ours      # one model, one method
```

Each method takes 10-20 minutes on a dense model and about 45 minutes on
Qwen3.6-35B-A3B. `ours_kernel` runs the TensorBridge kernel inside vLLM and
supports the dense models only.

## Output

Each run writes `ae/table4/results/<model>/<method>-tasks.json` with the
per-task scores and their mean. The script ends by printing the table and
saving it to `ae/table4/results/table4.md`, with the paper's value in
parentheses next to each measurement.

## Expected results

These are the paper's values, except the last column, which is not in the paper
and should be close to Ours. A model's five-task mean moves by up to about 0.5
between runs and batch sizes.

| Model | BF16 | W4A8 | NVFP4A16 | NVFP4 | MXFP4 | NVFP4A8 (Exact) | Ours w/o SNC | Ours | Ours (kernel) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama2-7B | 68.62 | 67.91 | 68.21 | 67.59 | 66.20 | 68.39 | 67.82 | 68.29 | 68.05 |
| Llama3-8B | 72.80 | 71.48 | 71.61 | 70.67 | 69.38 | 71.63 | 71.85 | 71.90 | 71.98 |
| Qwen3-8B | 71.56 | 70.33 | 71.31 | 70.86 | 67.09 | 71.19 | 71.74 | 71.68 | 71.45 |
| Qwen3-14B | 74.97 | 74.78 | 74.20 | 72.91 | 72.15 | 74.37 | 74.18 | 74.40 | 74.67 |
| Qwen3.6-35B-A3B | 73.34 | 73.58 | 73.44 | 72.58 | 73.31 | 72.90 | 73.48 | 73.62 | - |
| Average | 72.26 | 71.62 | 71.75 | 70.92 | 69.63 | 71.70 | 71.81 | 71.98 | - |
