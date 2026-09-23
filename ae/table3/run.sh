#!/usr/bin/env bash
# Table 3: WikiText2 perplexity of every method on every model (setup: eval/README.md).
# Usage: [MODELS="qwen3_8b ..."] bash ae/table3/run.sh [METHOD...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for model in ${MODELS:-llama2_7b llama3_8b qwen3_8b qwen3_14b qwen36_35b_a3b}; do
  METRICS=ppl bash eval/run.sh "$model" "$@"
done
"${AE_PYTHON:-python}" eval/summarize.py --table 3
