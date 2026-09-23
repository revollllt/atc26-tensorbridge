#!/usr/bin/env bash
# Replot Figures 11-13 from the data in each figure folder (CPU only).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for figure in figure11 figure12 figure13; do
  bash "ae/$figure/reproduce.sh"
done
