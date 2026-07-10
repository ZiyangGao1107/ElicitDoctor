#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PYTHON_BIN="${AR_GRPO_PYTHON:-python}"
ENV_FILE="${1:-/tmp/pcv32_doctor_suite_env_20260707_0330}"
MODEL="${2:-gpt-4o-mini}"
RUN_DIR="${3:-outputs_llm_patient_realizer_rubric_v3_2_qwenrealizer_verified_sample_20260707}"

cd "$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"

REQUEST_PATH="$RUN_DIR/mdd5k_patient_realizer_rubric_judge_requests.jsonl"
OUTPUT_PATH="$RUN_DIR/closed_llm_rubric_judge_outputs_${MODEL//\//_}.jsonl"
REPORT_DIR="$RUN_DIR/closed_llm_rubric_judge_report_${MODEL//\//_}"
CALL_LOG="logs/patient_realizer_rubric_v3_2_qwenrealizer_verified_sample_${MODEL//\//_}_call_20260707.log"
SUMMARY_LOG="logs/patient_realizer_rubric_v3_2_qwenrealizer_verified_sample_${MODEL//\//_}_summary_20260707.log"

if [[ ! -s "$ENV_FILE" ]]; then
  echo "Missing readable env file: $ENV_FILE" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/call_closed_llm_for_patient_realizer.py \
  --env-file "$ENV_FILE" \
  --pending-path "$REQUEST_PATH" \
  --output-path "$OUTPUT_PATH" \
  --model "$MODEL" \
  --limit 0 \
  --max-output-tokens 700 \
  --temperature 0.0 \
  --sleep-seconds 0.1 \
  >"$CALL_LOG" 2>&1

"$PYTHON_BIN" scripts/summarize_patient_realizer_rubric_judge_outputs_v1.py \
  --request-path "$REQUEST_PATH" \
  --output-path "$OUTPUT_PATH" \
  --report-dir "$REPORT_DIR" \
  >"$SUMMARY_LOG" 2>&1
