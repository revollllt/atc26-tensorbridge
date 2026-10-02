# Table 3: WikiText2 perplexity

Table 3 reports the WikiText2 perplexity of five models under BF16 and eight
quantization methods. Table 2's measured Llama columns (BF16, W4A8/GPTQ, Ours)
come from the same runs.

## Run

Set up the accuracy environment and create the checkpoints as described in
[`eval/README.md`](../../eval/README.md). Then, on one H100:

```bash
export MODEL_ROOT=/path/to/models
bash ae/table3/run.sh                           # all 5 models and 9 methods
MODELS=qwen3_8b bash ae/table3/run.sh ours      # one model, one method
```

All methods take about 30 minutes per dense model and a few hours for
Qwen3.6-35B-A3B. The method names are explained in
[`eval/README.md`](../../eval/README.md); `ours_kernel` runs the TensorBridge
kernel inside vLLM and supports the dense models only.

## Output

Each run writes `ae/table3/results/<model>/<method>-ppl.json`. The script ends
by printing the table and saving it to `ae/table3/results/table3.md`, with the
paper's value in parentheses next to each measurement.

## Expected results

These are the paper's values, except the last column, which is not in the paper
and should be close to Ours. Reruns should agree to within about 0.01.

| Model | BF16 | W4A8 | NVFP4A16 | NVFP4 | MXFP4 | NVFP4A8 (Exact) | Ours w/o SNC | Ours | Ours (kernel) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama2-7B | 5.47 | 5.64 | 5.56 | 5.78 | 6.15 | 5.57 | 5.64 | 5.56 | 5.57 |
| Llama3-8B | 6.14 | 6.65 | 6.41 | 9.74 | 7.74 | 6.44 | 6.64 | 6.42 | 6.43 |
| Qwen3-8B | 9.73 | 9.99 | 9.85 | 10.06 | 10.90 | 9.87 | 9.78 | 9.80 | 9.81 |
| Qwen3-14B | 8.65 | 8.82 | 8.77 | 8.91 | 9.95 | 8.80 | 9.20 | 8.79 | 8.82 |
| Qwen3.6-35B-A3B | 6.76 | 6.91 | 6.81 | 7.04 | 7.24 | 6.82 | 6.88 | 6.83 | - |
