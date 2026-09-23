#!/usr/bin/env bash
# Replot Figure 11 from data/latencies.md (CPU only).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python ae/figure11/plot.py
