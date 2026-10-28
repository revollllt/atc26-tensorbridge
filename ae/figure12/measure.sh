#!/usr/bin/env bash
# Measure one model and method of Figure 12 on an H100, then replot Figure 12
# from everything measured so far (ae/figure12/results/<NAME>/<METHOD>/).
# Usage: NAME=qwen3_8b METHOD=tensorbridge MODEL=models/qwen3_8b/NVFP4A8 \
#          bash ae/figure12/measure.sh [extra measure.py arguments]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL:?Set MODEL to the checkpoint (or the original model for fp8 and bf16)}"
: "${NAME:?Set NAME to the model name used in the figure}"
: "${METHOD:?Set METHOD to tensorbridge, nvfp4a16, w4a8, fp8 or bf16}"
case "$NAME" in
  llama3_8b|qwen3_8b|qwen3_14b|llama33_70b) ;;
  *) echo "NAME must be llama3_8b, qwen3_8b, qwen3_14b or llama33_70b" >&2; exit 2 ;;
esac
case "$METHOD" in
  tensorbridge) quantization=tensorbridge ;;
  fp8) quantization=fp8 ;;  # vLLM quantizes the BF16 model when it loads
  nvfp4a16|w4a8|bf16) quantization=auto ;;  # as the checkpoint says
  *) echo "METHOD must be tensorbridge, nvfp4a16, w4a8, fp8 or bf16" >&2; exit 2 ;;
esac
export VLLM_PLUGINS=tensorbridge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=${AE_PYTHON:-python3}
"$python_bin" ae/figure12/measure.py --model "$MODEL" --quantization "$quantization" \
  --tensor-parallel-size "${TP:-1}" --output-dir "ae/figure12/results/$NAME/$METHOD" "$@"
"$python_bin" ae/figure12/plot.py --figure 12 --results ae/figure12/results \
  --output ae/figure12/results/figure12.png
