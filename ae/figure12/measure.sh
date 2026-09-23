#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL:?Set MODEL to a local checkpoint path}"
export VLLM_PLUGINS=tensorbridge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${AE_PYTHON:-python3}" ae/figure12/measure.py \
  --model "$MODEL" --quantization "${QUANTIZATION:-tensorbridge}" "$@"
