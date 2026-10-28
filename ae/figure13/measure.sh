#!/usr/bin/env bash
# Measure one model and method of Figure 13 on an H100, then replot Figure 13
# from everything measured so far (ae/figure13/results/<NAME>/<METHOD>/).
# Usage: NAME=qwen3_8b METHOD=tensorbridge MODEL=models/qwen3_8b/NVFP4A8 \
#          SHAREGPT_PATH=... REQUEST_RATE_LIST=8,16,... bash ae/figure13/measure.sh
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
: "${SHAREGPT_PATH:?Set SHAREGPT_PATH to the ShareGPT trace}"
: "${REQUEST_RATE_LIST:?Set comma-separated request rates}"
export VLLM_PLUGINS=tensorbridge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=${AE_PYTHON:-python3}
output_dir=$PWD/ae/figure13/results/$NAME/$METHOD
port=${AE_PORT:-19080}
mkdir -p "$output_dir"

if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
  printf 'Port %s already serves a model; set AE_PORT.\n' "$port" >&2
  exit 2
fi

if [[ "$quantization" == auto ]]; then
  quantization_args=()
else
  quantization_args=(--quantization "$quantization")
fi
# -P: do not put the repository root on sys.path, where the vllm/ submodule folder
# would shadow the (editable) vllm package.
"$python_bin" -P -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port "$port" --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --max-num-seqs 512 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --calculate-kv-scales --tensor-parallel-size "${TP:-1}" \
  "${quantization_args[@]}" \
  > "$output_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

for attempt in {1..180}; do
  if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    printf 'vLLM exited; see %s/server.log\n' "$output_dir" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null

IFS=, read -r -a rates <<< "$REQUEST_RATE_LIST"
for rate in "${rates[@]}"; do
  "$python_bin" -P -m vllm.entrypoints.cli.main bench serve \
    --backend openai --host 127.0.0.1 --port "$port" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" \
    --num-prompts 4708 --request-rate "$rate" --max-concurrency 512 \
    --num-warmups 2 --temperature 0 --ignore-eos \
    --goodput ttft:2000 tpot:50 \
    --save-result --save-detailed --result-dir "$output_dir" \
    --result-filename "rate-${rate}.json" "$@"
done
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
"$python_bin" ae/figure12/plot.py --figure 13 --results ae/figure13/results \
  --output ae/figure13/results/figure13.png
