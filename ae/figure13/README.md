# Figure 13: online serving throughput

Figure 13 replays the ShareGPT trace against a vLLM server at increasing request
rates and reports the sustained request throughput. It compares TensorBridge
with BF16, FP8, W4A8 (CUTLASS) and NVFP4A16 (Marlin) on four models.

## Replot from our data (CPU only)

```bash
pip install -r ae/requirements.txt
bash ae/figure13/reproduce.sh
```

This reads `data/measurements.csv` and writes `ae/figure13/results/figure13.png`
and `.pdf`, which should match `reference.pdf`. TensorBridge saturates at about
73.1, 66.7, 44.6 and 19.5 requests/s on Llama-3-8B, Qwen3-8B, Qwen3-14B and
Llama-3.3-70B (Section 5.4.2).

## Measure on an H100

You need the environment from the [top-level README](../../README.md), an NVFP4
checkpoint (see [Figure 12](../figure12/README.md)), and the ShareGPT trace
`ShareGPT_V3_unfiltered_cleaned_split.json` from the
`anon8231489123/ShareGPT_Vicuna_unfiltered` dataset on Hugging Face.

```bash
export MODEL=models/qwen3_8b/NVFP4A8
export SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
export REQUEST_RATE_LIST=8,16,24,28,48,56,64,68,72,76,80,84,88,92,96
bash ae/figure13/measure.sh
```

The script starts a vLLM server with TensorBridge, sends 4708 requests at each
rate, and writes one JSON file per rate to `ae/figure13/results/tensorbridge/`.
A full sweep takes about 30 minutes. The rates we used for each model are:

| Model | `REQUEST_RATE_LIST` |
| --- | --- |
| Llama-3-8B | 8,16,24,28,80,88,96,104,112,120,128,136,144,152,160 |
| Qwen3-8B | 8,16,24,28,48,56,64,68,72,76,80,84,88,92,96 |
| Qwen3-14B | 8,16,24,28,32,36,40,42,44,46,48,50,52,56,60 |
| Llama-3.3-70B | 4,6,8,10,12,14,16,18,20,22,24,26 (with `TP=2`) |

The baselines use the same `MODEL` and `QUANTIZATION` settings as in
[Figure 12](../figure12/README.md), each with its own `OUTDIR`, for example
`MODEL=models/qwen3_8b/NVFP4A16 QUANTIZATION=auto OUTDIR=ae/figure13/results/marlin`.
