# TensorBridge (ATC '26) artifact

Code and scripts for "TensorBridge: Bridging Microscaled FP4 Weights to FP8 Tensor
Cores for LLM Serving" (ACM SIGOPS ATC '26). TensorBridge is a GEMM kernel for Hopper
GPUs that keeps weights in NVFP4 and runs them on the FP8 Tensor Cores.

## Repository layout

```text
atc26-tensorbridge/
├── ae/                  one folder per figure and table, each with a README
│   ├── figure11/        GEMM kernel speedup
│   ├── figure12/        offline serving throughput
│   ├── figure13/        online serving throughput
│   ├── table3/          WikiText2 perplexity
│   └── table4/          zero-shot accuracy
├── eval/                models, quantized checkpoints, accuracy harness
├── tensorbridge/        kernel, JIT compiler, autotuner, vLLM plugin
├── vllm/                vLLM 0.20.2 fork (submodule)
├── tests/               kernel correctness tests
├── examples/            standalone GEMM check and timing
├── docs/DESIGN.md       how the kernel code is organized
└── integrations/        optional: TensorBridge in DeepGEMM and FlashInfer (not used for the paper results)
```

## Requirements

- Replotting Figures 11-13 from our data: Python 3.10 or newer, on any machine.
- Everything else: NVIDIA Hopper GPUs with 80 GB of memory. Serving Llama-3.3-70B needs two.
- The Llama models are gated on Hugging Face: accept their licenses, then run `hf auth login`.

We used H100 80GB HBM3 (SXM) GPUs, an Intel Xeon 6530P, Rocky Linux 8.10, CUDA 12.8,
GCC 11.4, Python 3.12 and PyTorch 2.11.0+cu128. All times below, and the tuned kernel
table, are for this GPU.

## Installation

```bash
git clone --recursive https://github.com/revollllt/atc26-tensorbridge.git
cd atc26-tensorbridge
conda create -n tensorbridge python=3.12 -y
conda activate tensorbridge
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
pip install -r vllm/requirements/build/cuda.txt
pip install --no-build-isolation -e vllm     # about 30 minutes
pip install --no-build-isolation -e .
pip install -r ae/requirements.txt -r eval/requirements.txt
```

Creating checkpoints needs a second environment, because llmcompressor requires a newer
compressed-tensors than vLLM 0.20.2:

```bash
python3.12 -m venv .venv-quantize
.venv-quantize/bin/pip install -r eval/requirements-quantize.txt
```

TensorBridge compiles its kernels on first use. This needs:

- the CUDA toolkit (set `CUDA_HOME` if it is not `/usr/local/cuda`);
- GCC 11 or newer (otherwise set `CC` and `CXX`);
- on a machine without a GPU driver, `$CUDA_HOME/lib64/stubs` added to `LIBRARY_PATH`.

Test the kernel against its reference. All tests pass, and the first run compiles the kernels.
Without an sm_90 GPU, the kernel tests are skipped.

```bash
pip install pytest
pytest tests
```

## Evaluation

Run every command from the repository root. Each `ae/` folder writes to its own `results/`:
replots of our data go to `results/paper-data/`, and your own measurements go to `results/`.
The folder READMEs give the arguments, outputs and expected results.

| Step | What | Command | Time |
| --- | --- | --- | --- |
| 1 | Replot Figures 11-13 from our data (CPU) | `bash ae/reproduce_all.sh` | 1 min |
| 2 | [Figure 11](ae/figure11), GEMM speedup | `bash ae/figure11/measure.sh` | 5 min |
| 3 | Models and checkpoints | see [eval](eval/README.md) | - |
| 4 | [Table 3](ae/table3), perplexity | `bash ae/table3/run.sh` | 4-5 h |
| 5 | [Table 4](ae/table4), zero-shot accuracy | `bash ae/table4/run.sh` | 12-18 h |
| 6 | [Figure 12](ae/figure12), offline serving | `bash ae/figure12/measure.sh` | 1-2 h |
| 7 | [Figure 13](ae/figure13), online serving | `bash ae/figure13/measure.sh` | 19 h |

Steps 4-7 need step 3, and can run in any order.

Step 1 writes `ae/figure11/results/paper-data/figure11.pdf`,
`ae/figure12/results/paper-data/figure12.pdf` and
`ae/figure13/results/paper-data/figure13.pdf`. Compare each with the `reference.pdf` of its
figure folder, for example `ae/figure11/reference.pdf`.
On a machine without the environment above, run step 1 in a temporary venv:

```bash
(python3 -m venv .venv && . .venv/bin/activate && pip install -r ae/requirements.txt && bash ae/reproduce_all.sh)
```

[`docs/DESIGN.md`](docs/DESIGN.md) maps the paper's mechanisms to the kernel code.
