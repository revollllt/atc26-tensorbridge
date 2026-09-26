#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL:?Set MODEL to a local checkpoint path}"
: "${SHAREGPT_PATH:?Set SHAREGPT_PATH to the ShareGPT trace}"
: "${REQUEST_RATE_LIST:?Set comma-separated request rates}"
export VLLM_PLUGINS=tensorbridge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=${AE_PYTHON:-python3}
# -P: do not put the repository root on sys.path, where the vllm/ submodule folder
# would shadow the (editable) vllm package.
quantization=${QUANTIZATION:-tensorbridge}
output_dir=${OUTDIR:-$PWD/ae/figure13/results/$quantization}
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
