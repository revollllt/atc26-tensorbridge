#!/usr/bin/env bash
# Measure the Figure 11 kernels on one H100 and plot them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python ae/figure11/measure.py "$@"
python ae/figure11/plot.py --input ae/figure11/results/measured.md \
  --output ae/figure11/results/measured.png
