# TensorBridge: Artifact Evaluation

This repository contains the code and scripts for the ATC '26 paper
"TensorBridge: Bridging Microscaled FP4 Weights to FP8 Tensor Cores for LLM
Serving". TensorBridge is a GEMM kernel for Hopper GPUs that keeps weights in
NVFP4 and runs them on the FP8 Tensor Cores.

- [`tensorbridge/`](tensorbridge) is the kernel and its vLLM plugin.
- [`ae/`](ae) has one folder per figure and table of the evaluation, each with
  its data, scripts, and a README. **Start here.**
- [`eval/`](eval) creates the quantized checkpoints and runs the accuracy
  evaluation behind Tables 3 and 4.
- [`vllm/`](vllm) is our fork of vLLM 0.20.2, used for the serving experiments.

## Project structure

```text
atc26-tensorbridge/
├── README.md
├── ae/                  figures and tables of the evaluation
│   ├── README.md
│   ├── figure11/        GEMM kernel speedup
│   ├── figure12/        offline serving throughput
│   ├── figure13/        online serving throughput
│   ├── table3/          WikiText2 perplexity
│   └── table4/          zero-shot accuracy
├── eval/                checkpoint creation and accuracy evaluation
├── tensorbridge/        kernel, JIT compiler and vLLM plugin
├── vllm/                vLLM fork (submodule)
├── docs/DESIGN.md       how the kernel code is organized
├── tests/               kernel correctness tests
└── integrations/        optional: TensorBridge inside DeepGEMM and FlashInfer
```

`integrations/` is not needed to reproduce the paper.

## Hardware and software

We ran every experiment on the following machine:

- 1x NVIDIA H100 80GB HBM3 (SXM); Llama-3.3-70B serving uses 2 GPUs
- Intel Xeon 6530P
- Rocky Linux 8.10, CUDA 12.8, GCC 11.4, Python 3.12
- PyTorch 2.11.0+cu128

The kernel needs a Hopper GPU (sm_90). Replotting the figures from our data
needs only a CPU.

## Installation

Clone the repository with its submodules:

```bash
git clone --recursive https://github.com/revollllt/atc26-tensorbridge.git
cd atc26-tensorbridge
```

Create a Python 3.12 environment and install PyTorch, the vLLM fork, and
TensorBridge, in this order. Building vLLM takes about 30 minutes.

```bash
conda create -n tensorbridge python=3.12 -y
conda activate tensorbridge
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
pip install -r vllm/requirements/build/cuda.txt
pip install --no-build-isolation -e vllm
pip install --no-build-isolation -e .
pip install -r ae/requirements.txt
```

TensorBridge compiles its kernels on first use, so the CUDA toolkit must be
installed (set `CUDA_HOME` if it is not under `/usr/local/cuda`) and GCC must
be version 11 or newer (set `CC` and `CXX` otherwise). If you build on a
machine without a GPU driver, add `$CUDA_HOME/lib64/stubs` to `LIBRARY_PATH`.

Check the kernel against its reference:

```bash
pip install pytest
pytest tests
```

## Reproducing the results

See [`ae/README.md`](ae/README.md). In short:

```bash
bash ae/reproduce_all.sh        # replot Figures 11-13 from our data, CPU only
bash ae/figure11/measure.sh     # measure Figure 11 on an H100, about 5 minutes
bash ae/table3/run.sh           # Table 3, needs the checkpoints from eval/
```
