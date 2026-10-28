# Table 3: WikiText2 perplexity

Table 3 reports the WikiText2 perplexity of five models under BF16 and seven quantization
methods. The script also runs Ours with the TensorBridge kernel inside vLLM (`ours_kernel`,
dense models only). Table 2's measured Llama columns (BF16, W4A8/GPTQ, Ours) come from the
same runs.

## Run (all methods: about 30 minutes per dense model, a few hours for Qwen3.6-35B-A3B)

This needs the checkpoints from [`eval/README.md`](../../eval/README.md). On one H100:

```bash
export MODEL_ROOT=/path/to/models
bash ae/table3/run.sh                           # all models and methods
MODELS=qwen3_8b bash ae/table3/run.sh ours      # one model, one method
```

The method names are explained in [`eval/README.md`](../../eval/README.md#methods).

## Output

Each run writes `ae/table3/results/<model>/<method>-ppl.json`. The script then prints the
table and saves it to `ae/table3/results/table3.md`. The paper's value is in parentheses next
to each measurement, and `*` marks a `LIMIT` run.

## Expected results

These are the paper values (Table 3). On the `ours_kernel` row, the output puts the paper's Ours
value in parentheses. Our run of `ours_kernel` gave the values below, within 0.03 of Ours
(Section 5.2). Checkpoints recreated with GPTQ change these values by up to 0.07; BF16 is
unchanged.

| Method | Llama2-7B | Llama3-8B | Qwen3-8B | Qwen3-14B | Qwen3.6-35B-A3B |
| --- | --- | --- | --- | --- | --- |
| `bf16` (BF16) | 5.47 | 6.14 | 9.73 | 8.65 | 6.76 |
| `w4a8` (W4A8) | 5.64 | 6.66 | 9.92 | 8.87 | 6.89 |
| `nvfp4a16` (NVFP4A16) | 5.56 | 6.41 | 9.85 | 8.77 | 6.81 |
| `nvfp4` (NVFP4) | 5.78 | 9.74 | 10.06 | 8.91 | 7.04 |
| `mxfp4` (MXFP4) | 6.15 | 7.74 | 10.90 | 9.95 | 7.24 |
| `nvfp4a8` (NVFP4A8 Exact) | 5.57 | 6.44 | 9.87 | 8.80 | 6.82 |
| `ours_nosnc` (Ours w/o SNC) | 5.64 | 6.64 | 9.78 | 9.20 | 6.88 |
| `ours` (Ours) | 5.56 | 6.42 | 9.80 | 8.79 | 6.83 |
| `ours_kernel` | 5.57 | 6.43 | 9.81 | 8.82 | - |
