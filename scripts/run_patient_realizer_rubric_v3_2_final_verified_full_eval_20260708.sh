#!/usr/bin/env bash
set -uo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
ENV_FILE="${1:-/tmp/pcv32_doctor_suite_env_20260707_0330}"
MODEL="${2:-gpt-4o-mini}"
RUN_DIR="${3:-outputs_llm_patient_realizer_rubric_v3_2_final_verified_cache_20260708}"
BATCH="${BATCH:-100}"

cd "$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"

MODEL_SAFE="${MODEL//\//_}"
REQUEST_PATH="$RUN_DIR/mdd5k_patient_realizer_rubric_judge_requests.jsonl"
OUTPUT_PATH="$RUN_DIR/closed_llm_rubric_judge_outputs_${MODEL_SAFE}.jsonl"
REPORT_DIR="$RUN_DIR/closed_llm_rubric_judge_report_${MODEL_SAFE}_full_20260708_final"

if [[ ! -s "$ENV_FILE" ]]; then
  echo "Missing readable env file: $ENV_FILE" >&2
  exit 2
fi
if [[ ! -s "$REQUEST_PATH" ]]; then
  echo "Missing request path: $REQUEST_PATH" >&2
  exit 3
fi

echo "=== final verified patient rubric eval start $(date) ==="
echo "run_dir=$RUN_DIR"
echo "model=$MODEL"
echo "batch=$BATCH"
echo "request_path=$REQUEST_PATH"
echo "output_path=$OUTPUT_PATH"

while true; do
  total=$(wc -l < "$REQUEST_PATH")
  if [[ -f "$OUTPUT_PATH" ]]; then
    done_count=$(wc -l < "$OUTPUT_PATH")
  else
    done_count=0
  fi
  echo "[$(date)] progress=${done_count}/${total} batch=${BATCH}"
  if [[ "$done_count" -ge "$total" ]]; then
    echo "[$(date)] all rubric judge outputs completed"
    "$PY" scripts/summarize_patient_realizer_rubric_judge_outputs_v1.py \
      --request-path "$REQUEST_PATH" \
      --output-path "$OUTPUT_PATH" \
      --report-dir "$REPORT_DIR" || true
    break
  fi

  "$PY" scripts/call_closed_llm_for_patient_realizer.py \
    --env-file "$ENV_FILE" \
    --pending-path "$REQUEST_PATH" \
    --output-path "$OUTPUT_PATH" \
    --model "$MODEL" \
    --limit "$BATCH" \
    --max-output-tokens 700 \
    --temperature 0.0 \
    --timeout-seconds 180 \
    --max-retries 2 \
    --sleep-seconds 0.1
  status=$?
  echo "[$(date)] call_status=$status"

  "$PY" scripts/summarize_patient_realizer_rubric_judge_outputs_v1.py \
    --request-path "$REQUEST_PATH" \
    --output-path "$OUTPUT_PATH" \
    --report-dir "$REPORT_DIR" || true

  if [[ "$status" -ne 0 ]]; then
    echo "[$(date)] batch failed; sleeping before resume"
    sleep 60
  else
    sleep 5
  fi
done

echo "=== final verified patient rubric eval end $(date) ==="
