#!/usr/bin/env bash
# Replot Figure 13 from data/measurements.csv (CPU only).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python ae/figure12/plot.py --figure 13 --output ae/figure13/results/figure13.png
