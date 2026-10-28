# Models, checkpoints and accuracy evaluation

This folder downloads the models, creates the quantized checkpoints used by Tables 3-4 and
Figures 12-13, and holds the accuracy harness behind [Table 3](../ae/table3) and
[Table 4](../ae/table4). The environments are in the [top-level README](../README.md).

## Models

```bash
export MODEL_ROOT=/path/to/models
hf download meta-llama/Llama-2-7b-hf --exclude "*.bin" --local-dir $MODEL_ROOT/meta-llama/Llama-2-7b-hf   # llama2_7b
hf download meta-llama/Meta-Llama-3-8B --exclude "original/*" --local-dir $MODEL_ROOT/meta-llama/Llama-3-8B   # llama3_8b
hf download Qwen/Qwen3-8B --local-dir $MODEL_ROOT/Qwen/Qwen3-8B                               # qwen3_8b
hf download Qwen/Qwen3-14B --local-dir $MODEL_ROOT/Qwen/Qwen3-14B                             # qwen3_14b
hf download Qwen/Qwen3.6-35B-A3B --local-dir $MODEL_ROOT/Qwen/Qwen3.6-35B-A3B                 # qwen36_35b_a3b, tables only
hf download meta-llama/Llama-3.3-70B-Instruct --exclude "original/*" \
  --local-dir $MODEL_ROOT/meta-llama/Llama-3.3-70B-Instruct                                   # llama33_70b, Figures 12-13 only
hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
  --repo-type dataset --local-dir /path/to/sharegpt                                           # Figure 13 only
```

WikiText2 and the zero-shot tasks are downloaded from Hugging Face on first use.

## Checkpoints

`W4A8`, `NVFP4A8` and `NVFP4A8_FPMA` use GPTQ, and `NVFP4` and `MXFP4` use round-to-nearest.
All of them are calibrated on the 128 WikiText2 windows in `eval/calibration.jsonl`, on one H100.
A dense model takes 10-20 minutes per scheme. Llama-3.3-70B needs only `W4A8` and `NVFP4A8`.
Figures 12-13 use only `W4A8`, `NVFP4A8` and `NVFP4A16`.

```bash
export MODEL_ROOT=/path/to/models
for name in llama2_7b llama3_8b qwen3_8b qwen3_14b qwen36_35b_a3b llama33_70b; do
  schemes="W4A8 NVFP4A8 NVFP4A8_FPMA NVFP4 MXFP4"
  case $name in
    llama2_7b)      base=$MODEL_ROOT/meta-llama/Llama-2-7b-hf ;;
    llama3_8b)      base=$MODEL_ROOT/meta-llama/Llama-3-8B ;;
    qwen3_8b)       base=$MODEL_ROOT/Qwen/Qwen3-8B ;;
    qwen3_14b)      base=$MODEL_ROOT/Qwen/Qwen3-14B ;;
    qwen36_35b_a3b) base=$MODEL_ROOT/Qwen/Qwen3.6-35B-A3B ;;
    llama33_70b)    base=$MODEL_ROOT/meta-llama/Llama-3.3-70B-Instruct; schemes="W4A8 NVFP4A8" ;;
  esac
  for scheme in $schemes; do
    .venv-quantize/bin/python eval/quantize.py --model $base --scheme $scheme \
      --output models/$name/$scheme
  done
done
for name in llama3_8b qwen3_8b qwen3_14b llama33_70b; do    # NVFP4A16 (Marlin), Figures 12-13
  python ae/figure12/nvfp4a16.py models/$name/NVFP4A8 models/$name/NVFP4A16
done
```

`NVFP4A16` reuses the `NVFP4A8` weights without activation quantization. The script writes a new
config and symlinks the weights. `eval/run.sh` reads checkpoints from `models/`, or from
`CKPT_ROOT` if it is set.

## Methods

| Method | Row in the paper | Checkpoint | What runs |
| --- | --- | --- | --- |
| `bf16` | BF16 | original model | BF16 |
| `w4a8` | W4A8 | `W4A8` | INT4 weights, FP8 activations |
| `nvfp4a16` | NVFP4A16 | `NVFP4A8` | NVFP4 weights, BF16 activations |
| `nvfp4` | NVFP4 | `NVFP4` | NVFP4 weights and activations |
| `mxfp4` | MXFP4 | `MXFP4` | MXFP4 weights and activations |
| `nvfp4a8` | NVFP4A8 (Exact) | `NVFP4A8` | exact NVFP4 weights, FP8 activations |
| `ours_nosnc` | Ours w/o SNC | `NVFP4A8_FPMA` | FPMA bridge without SNC |
| `ours` | Ours | `NVFP4A8_FPMA` | FPMA bridge with SNC |
| `ours_kernel` | - (compared with Ours) | `NVFP4A8_FPMA` | TensorBridge kernel inside vLLM (dense models) |

All methods except `ours_kernel` run in Hugging Face Transformers. Their weights are dequantized
to BF16 at load, and quantized activations are simulated. `NVFP4A8_FPMA` is a standard NVFP4
checkpoint whose GPTQ rounds each weight to a value the FPMA bridge reproduces, so GPTQ also
absorbs the bridge's error.

## Evaluating one model

`ae/table3/run.sh` and `ae/table4/run.sh` run all models. `eval/run.sh` measures both metrics of one model:

```bash
bash eval/run.sh qwen3_8b                     # all methods
bash eval/run.sh qwen3_8b ours ours_kernel    # selected methods
LIMIT=20 bash eval/run.sh qwen3_8b ours       # quick check, not a paper number
python eval/summarize.py                      # print Tables 3 and 4
```

Results go to `ae/table3/results/` and `ae/table4/results/`. Finished runs are skipped, so an
interrupted run continues where it stopped. A `LIMIT` run replaces that method's result, is
marked `*`, and is run again by the next full run.
