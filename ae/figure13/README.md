# Figure 13: online serving throughput

Figure 13 replays the ShareGPT trace against a vLLM server at increasing request rates and
reports the sustained request throughput. It compares TensorBridge with BF16, FP8, W4A8
(CUTLASS) and NVFP4A16 (Marlin) on the four models of [Figure 12](../figure12/README.md).

## Replot from our data (CPU only)

```bash
bash ae/figure13/reproduce.sh
```

This reads `ae/figure13/data/measurements.csv` and writes
`ae/figure13/results/paper-data/figure13.png` and `.pdf`. Compare with
`ae/figure13/reference.pdf`.

## Measure on an H100

A method takes about 35-60 minutes on the 8B and 14B models and 1.5-2 hours on Llama-3.3-70B.

This needs the checkpoints of [Figure 12](../figure12/README.md) and the ShareGPT trace from
[`eval/README.md`](../../eval/README.md#models). Set `NAME`, `METHOD`, `MODEL` and `TP` as in
Figure 12, and take `REQUEST_RATE_LIST` from the table below.

```bash
export NAME=qwen3_8b METHOD=tensorbridge MODEL=models/qwen3_8b/NVFP4A8
export SHAREGPT_PATH=/path/to/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
export REQUEST_RATE_LIST=8,16,24,28,48,56,64,68,72,76,80,84,88,92,96
bash ae/figure13/measure.sh
```

| Model | `REQUEST_RATE_LIST` |
| --- | --- |
| Llama-3-8B | 8,16,24,28,80,88,96,104,112,120,128,136,144,152,160 |
| Qwen3-8B | 8,16,24,28,48,56,64,68,72,76,80,84,88,92,96 |
| Qwen3-14B | 8,16,24,28,32,36,40,42,44,46,48,50,52,56,60 |
| Llama-3.3-70B | 4,6,8,10,12,14,16,18,20,22,24,26 (with `TP=2`) |

The script starts a vLLM server and sends 4708 requests at each rate. Outputs:

- `ae/figure13/results/<NAME>/<METHOD>/rate-<R>.json` (one file per rate) and `server.log`;
- `ae/figure13/results/measurements.csv` and `ae/figure13/results/figure13.png` and `.pdf`:
  every model and method measured so far.

## Expected results

From Section 5.4.2 and `ae/figure13/data/measurements.csv`:

- TensorBridge saturates at 73.1, 66.7, 44.6 and 19.5 requests/s on Llama-3-8B, Qwen3-8B,
  Qwen3-14B and Llama-3.3-70B.
- Its peak is 1.16-1.63x that of BF16, 1.82-2.18x that of NVFP4A16 Marlin, and within
  2.6-6.5% of W4A8.

A method's peak is its largest `request_throughput` in `ae/figure13/results/measurements.csv`.
