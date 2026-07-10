#!/usr/bin/env bash
set -uo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
ENV_FILE="${1:-/tmp/pcv32_doctor_suite_env_20260707_0330}"
MODEL="${2:-gpt-4.1-mini}"
RUN_DIR="${3:-outputs_llm_patient_realizer_rubric_v3_2_final_verified_cache_20260708}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"

MODEL_SAFE="${MODEL//\//_}"
REQUEST_PATH="$RUN_DIR/mdd5k_patient_realizer_rubric_judge_requests.jsonl"
OUTPUT_PATH="$RUN_DIR/closed_llm_rubric_judge_outputs_${MODEL_SAFE}.jsonl"
REPORT_DIR="$RUN_DIR/closed_llm_rubric_judge_report_${MODEL_SAFE}_full_20260708_final"
SHARD_DIR="$RUN_DIR/rubric_sharded_accel_${MODEL_SAFE}_$(date +%Y%m%d_%H%M%S)"

if [[ ! -s "$ENV_FILE" ]]; then
  echo "Missing readable env file: $ENV_FILE" >&2
  exit 2
fi
if [[ ! -s "$REQUEST_PATH" ]]; then
  echo "Missing request path: $REQUEST_PATH" >&2
  exit 3
fi

echo "=== sharded rubric accel start $(date) ==="
echo "model=$MODEL"
echo "num_shards=$NUM_SHARDS"
echo "request_path=$REQUEST_PATH"
echo "output_path=$OUTPUT_PATH"
echo "shard_dir=$SHARD_DIR"

"$PY" scripts/prepare_and_merge_rubric_shards_v1.py \
  --mode prepare \
  --request-path "$REQUEST_PATH" \
  --output-path "$OUTPUT_PATH" \
  --shard-dir "$SHARD_DIR" \
  --num-shards "$NUM_SHARDS"

declare -a pids=()
declare -a statuses=()
for idx in $(seq 0 $((NUM_SHARDS - 1))); do
  request_shard="$SHARD_DIR/requests_shard_${idx}.jsonl"
  output_shard="$SHARD_DIR/outputs_shard_${idx}.jsonl"
  log_shard="$SHARD_DIR/worker_${idx}.log"
  if [[ ! -s "$request_shard" ]]; then
    echo "[$(date)] shard ${idx} empty"
    continue
  fi
  echo "[$(date)] launch shard ${idx}: $(wc -l < "$request_shard") requests"
  "$PY" scripts/call_closed_llm_for_patient_realizer.py \
    --env-file "$ENV_FILE" \
    --pending-path "$request_shard" \
    --output-path "$output_shard" \
    --model "$MODEL" \
    --limit 0 \
    --max-output-tokens 700 \
    --temperature 0.0 \
    --timeout-seconds 180 \
    --max-retries 2 \
    --sleep-seconds 0.1 >"$log_shard" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if wait "$pid"; then
    statuses+=("0")
  else
    statuses+=("$?")
  fi
done
echo "[$(date)] shard_statuses=${statuses[*]:-none}"

for idx in $(seq 0 $((NUM_SHARDS - 1))); do
  request_shard="$SHARD_DIR/requests_shard_${idx}.jsonl"
  output_shard="$SHARD_DIR/outputs_shard_${idx}.jsonl"
  req_count=0
  out_count=0
  [[ -f "$request_shard" ]] && req_count=$(wc -l < "$request_shard")
  [[ -f "$output_shard" ]] && out_count=$(wc -l < "$output_shard")
  echo "[$(date)] shard ${idx}: outputs=${out_count}/${req_count}"
done

"$PY" scripts/prepare_and_merge_rubric_shards_v1.py \
  --mode merge \
  --request-path "$REQUEST_PATH" \
  --output-path "$OUTPUT_PATH" \
  --shard-dir "$SHARD_DIR" \
  --num-shards "$NUM_SHARDS"

"$PY" scripts/summarize_patient_realizer_rubric_judge_outputs_v1.py \
  --request-path "$REQUEST_PATH" \
  --output-path "$OUTPUT_PATH" \
  --report-dir "$REPORT_DIR"

echo "=== sharded rubric accel end $(date) ==="
