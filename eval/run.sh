#!/usr/bin/env bash
# Usage: bash eval/run.sh MODEL [METHOD...]   (see eval/README.md)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model=${1:?Usage: bash eval/run.sh MODEL [METHOD...]}
shift
: "${MODEL_ROOT:?Set MODEL_ROOT to the directory holding meta-llama/ and Qwen/}"
ckpt=${CKPT_ROOT:-$PWD/models}/$model
case "$model" in
  llama2_7b) base=$MODEL_ROOT/meta-llama/Llama-2-7b-hf ;;
  llama3_8b) base=$MODEL_ROOT/meta-llama/Llama-3-8B ;;
  qwen3_8b) base=$MODEL_ROOT/Qwen/Qwen3-8B ;;
  qwen3_14b) base=$MODEL_ROOT/Qwen/Qwen3-14B ;;
  qwen36_35b_a3b) base=$MODEL_ROOT/Qwen/Qwen3.6-35B-A3B ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
methods=("$@")
if (( ${#methods[@]} == 0 )); then
  methods=(bf16 w4a8 nvfp4a16 nvfp4 mxfp4 nvfp4a8 ours_nosnc ours ours_kernel)
  [[ $model == qwen36_35b_a3b ]] && unset 'methods[8]'  # the kernel plugin covers dense layers only
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
for method in "${methods[@]}"; do
  case "$method" in
    bf16) checkpoint=$base ;;
    w4a8) checkpoint=$ckpt/W4A8 ;;
    nvfp4) checkpoint=$ckpt/NVFP4 ;;
    mxfp4) checkpoint=$ckpt/MXFP4 ;;
    ours*) checkpoint=$ckpt/NVFP4A8_FPMA ;;
    *) checkpoint=$ckpt/NVFP4A8 ;;
  esac
  for metric in ${METRICS:-ppl tasks}; do
    table=$([[ $metric == ppl ]] && echo 3 || echo 4)
    out=${RESULT_ROOT:-ae/table$table/results}/$model/$method-$metric.json
    if [[ -z ${LIMIT:-} ]] && grep -qs '"limit": null' "$out"; then echo "Keep $out"; continue; fi
    "${AE_PYTHON:-python}" eval/accuracy.py --method "$method" --metric "$metric" \
      --model "$checkpoint" --tokenizer "$base" --output "$out" \
      ${LIMIT:+--limit "$LIMIT"} ${ALPHA:+--alpha "$ALPHA"}
  done
done
