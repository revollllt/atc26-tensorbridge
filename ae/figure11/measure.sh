#!/usr/bin/env bash
# Measure the Figure 11 kernels on one H100 and plot Figure 11 from the measurement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${AE_PYTHON:-python3}" ae/figure11/measure.py "$@"
"${AE_PYTHON:-python3}" ae/figure11/plot.py --input ae/figure11/results/measured.md \
  --output ae/figure11/results/figure11.png
