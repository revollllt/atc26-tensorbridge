# Figure 12: offline serving throughput

Figure 12 compares the output throughput of vLLM with TensorBridge against BF16,
FP8, W4A8 (CUTLASS) and NVFP4A16 (Marlin). Each request has 1024 input and 512
output tokens; the batch size goes from 1 to 256. All runs use an FP8 KV cache.

## Replot from our data (CPU only)

```bash
pip install -r ae/requirements.txt
bash ae/figure12/reproduce.sh
```

This reads `data/measurements.csv` (4 models x 5 methods x 5 batch sizes) and
writes `ae/figure12/results/figure12.png` and `.pdf`, which should match
`reference.pdf`. Across the four models and five batch sizes, TensorBridge's
geometric-mean throughput is 1.64x that of BF16, 1.09x FP8, 1.13x W4A8 and 1.34x
NVFP4A16 Marlin (Section 5.4.1).

## Measure on an H100

You need the environment from the [top-level README](../../README.md) and an
NVFP4 checkpoint of the model. Create it in the checkpoint environment described
in [`eval/README.md`](../../eval/README.md), with the `NVFP4A8` scheme:

```bash
.venv-quantize/bin/python eval/quantize.py --model /path/to/Qwen3-8B \
  --scheme NVFP4A8 --output models/qwen3_8b/NVFP4A8
```

Then measure TensorBridge at the five batch sizes (a few minutes per model):

```bash
MODEL=models/qwen3_8b/NVFP4A8 bash ae/figure12/measure.sh
```

This writes one JSON file per batch size to `ae/figure12/results/`, for example
`tensorbridge-256.json`, and prints the output throughput of each. For
Llama-3.3-70B, add `--tensor-parallel-size 2`.

The baselines run through the same script with vLLM's own kernels. Give each one its own output folder:

```bash
# NVFP4A16 Marlin: same NVFP4 checkpoint
MODEL=models/qwen3_8b/NVFP4A8 QUANTIZATION=auto bash ae/figure12/measure.sh \
  --output-dir ae/figure12/results/marlin
# W4A8 CUTLASS: the W4A8 checkpoint from eval/quantize.py
MODEL=models/qwen3_8b/W4A8 QUANTIZATION=auto bash ae/figure12/measure.sh \
  --output-dir ae/figure12/results/w4a8
# FP8: the original model, quantized by vLLM when it loads
MODEL=/path/to/Qwen3-8B QUANTIZATION=fp8 bash ae/figure12/measure.sh \
  --output-dir ae/figure12/results/fp8
# BF16: the original model
MODEL=/path/to/Qwen3-8B QUANTIZATION=auto bash ae/figure12/measure.sh \
  --output-dir ae/figure12/results/bf16
```

Our peak throughputs for TensorBridge are about 10.3k, 9.7k, 6.7k and 3.2k
tokens/s for Llama-3-8B, Qwen3-8B, Qwen3-14B and Llama-3.3-70B. Your numbers
depend on the GPU clock and driver, so compare the ratios between methods
rather than the absolute values.
