# Accuracy evaluation (Tables 3 and 4)

This folder creates the quantized checkpoints and measures WikiText2 perplexity
(Table 3) and mean zero-shot accuracy (Table 4) for Llama2-7B, Llama3-8B,
Qwen3-8B, Qwen3-14B and Qwen3.6-35B-A3B. Run all commands from the repository
root. The tables have their own entry points in [`ae/table3`](../ae/table3) and
[`ae/table4`](../ae/table4).

## Methods

| Method | Row in the paper | Checkpoint | What runs |
| --- | --- | --- | --- |
| `bf16` | BF16 | original model | Hugging Face |
| `w4a8` | W4A8 | `W4A8` | INT4 weights, FP8 activations |
| `nvfp4a16` | NVFP4A16 | `NVFP4A8` | NVFP4 weights, BF16 activations |
| `nvfp4` | NVFP4 | `NVFP4` | NVFP4 weights and activations |
| `mxfp4` | MXFP4 | `MXFP4` | MXFP4 weights and activations |
| `nvfp4a8` | NVFP4A8 (Exact) | `NVFP4A8` | exact NVFP4 weights, FP8 activations |
| `ours_nosnc` | Ours w/o SNC | `NVFP4A8_FPMA` | FPMA bridge without SNC |
| `ours` | Ours | `NVFP4A8_FPMA` | FPMA bridge with SNC |
| `ours_kernel` | Ours | `NVFP4A8_FPMA` | TensorBridge kernel inside vLLM (dense models) |

All methods except `ours_kernel` simulate the quantization in PyTorch.
`NVFP4A8_FPMA` is a standard NVFP4 checkpoint whose GPTQ rounds each weight to
a value the FPMA bridge reproduces, so GPTQ also absorbs the bridge's error.

## Environment

The evaluation runs in the environment from the
[top-level README](../README.md), with a few more packages:

```bash
pip install -r eval/requirements.txt
```

Creating the checkpoints needs a separate environment, because llmcompressor
requires a newer compressed-tensors than vLLM 0.20.2:

```bash
python3.12 -m venv .venv-quantize
.venv-quantize/bin/pip install -r eval/requirements-quantize.txt
```

Download the models into one folder. The Llama models require accepting their
license on Hugging Face and logging in with `hf auth login`.

```bash
export MODEL_ROOT=/path/to/models
hf download meta-llama/Llama-2-7b-hf --local-dir $MODEL_ROOT/meta-llama/Llama-2-7b-hf
hf download meta-llama/Meta-Llama-3-8B --local-dir $MODEL_ROOT/meta-llama/Llama-3-8B
hf download Qwen/Qwen3-8B --local-dir $MODEL_ROOT/Qwen/Qwen3-8B
hf download Qwen/Qwen3-14B --local-dir $MODEL_ROOT/Qwen/Qwen3-14B
hf download Qwen/Qwen3.6-35B-A3B --local-dir $MODEL_ROOT/Qwen/Qwen3.6-35B-A3B
```

WikiText2 and the zero-shot tasks are downloaded from Hugging Face on first use.

## Step 1: create the checkpoints

Each model needs five checkpoints. GPTQ calibrates on the 128 WikiText2 samples
in `eval/calibration.jsonl`, on one H100. A dense model takes 10-20 minutes per
scheme; Qwen3.6-35B-A3B takes about 4 hours.

```bash
for scheme in W4A8 NVFP4A8 NVFP4A8_FPMA NVFP4 MXFP4; do
  .venv-quantize/bin/python eval/quantize.py --model $MODEL_ROOT/Qwen/Qwen3-8B \
    --scheme $scheme --output models/qwen3_8b/$scheme
done
```

Repeat for the other models, using `llama2_7b`, `llama3_8b`, `qwen3_14b` and
`qwen36_35b_a3b` as folder names. The scripts look for checkpoints in
`models/`; set `CKPT_ROOT` if you put them elsewhere.

## Step 2: evaluate

`eval/run.sh` measures one model with the given methods, or with all of them:

```bash
bash eval/run.sh qwen3_8b                     # all methods, both metrics
bash eval/run.sh qwen3_8b ours ours_kernel    # selected methods
LIMIT=20 bash eval/run.sh qwen3_8b ours       # quick check, not a paper number
python eval/summarize.py                      # print Tables 3 and 4
```

Perplexity results go to `ae/table3/results/` and zero-shot results to
`ae/table4/results/`. Finished runs are skipped, so an interrupted run picks up
where it stopped.

`ALPHA=0.961 bash eval/run.sh ...` scales the global scale of the three FPMA
methods (`ours`, `ours_nosnc`, `ours_kernel`) by 0.961, which slightly lowers the zero-shot error but raises
perplexity. The paper uses the default, 1.
