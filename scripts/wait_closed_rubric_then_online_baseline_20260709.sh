#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
POLL_SECONDS="${POLL_SECONDS:-600}"

REQ_PATH="${REQ_PATH:-outputs_llm_patient_realizer_rubric_v3_2_final_verified_cache_20260708/mdd5k_patient_realizer_rubric_judge_requests.jsonl}"
RUBRIC_OUT="${RUBRIC_OUT:-outputs_llm_patient_realizer_rubric_v3_2_final_verified_cache_20260708/closed_llm_rubric_judge_outputs_gpt-4.1-mini.jsonl}"
RUBRIC_REPORT_DIR="${RUBRIC_REPORT_DIR:-outputs_llm_patient_realizer_rubric_v3_2_final_verified_cache_20260708/closed_llm_rubric_judge_report_gpt-4.1-mini_full_20260708_final}"
RUBRIC_SUMMARY="$RUBRIC_REPORT_DIR/mdd5k_patient_realizer_rubric_judge_summary.json"

CACHE_PATH="${CACHE_PATH:-outputs_pcv32_full_patient_realizer_hardened_repair_20260708_2105_12_verified_cache_repair12/mdd5k_verified_patient_response_cache_repair_include_warned.jsonl}"
CACHE_SUMMARY="${CACHE_SUMMARY:-outputs_pcv32_full_patient_realizer_hardened_repair_20260708_2105_12_verified_cache_repair12/mdd5k_verified_patient_response_cache_repair_summary_include_warned.json}"
PRIMARY_VERIFY_SUMMARY="${PRIMARY_VERIFY_SUMMARY:-outputs_pcv32_full_patient_realizer_hardened_repair_20260708_2105_03_primary_verify/mdd5k_patient_realizer_verification_summary_llm_outputs.json}"
REPAIR_VERIFY_SUMMARY="${REPAIR_VERIFY_SUMMARY:-outputs_pcv32_full_patient_realizer_hardened_repair_20260708_2105_06_repair_verify/mdd5k_patient_realizer_verification_summary_llm_outputs.json}"
REPAIR2_VERIFY_SUMMARY="${REPAIR2_VERIFY_SUMMARY:-outputs_pcv32_full_patient_realizer_hardened_repair_20260708_2105_10_repair2_verify/mdd5k_patient_realizer_verification_summary_llm_outputs.json}"
FREEZE_DIR="${FREEZE_DIR:-outputs_final_patient_freeze_report_pcv32_qwen_realizer_20260709_final}"

BASELINE_MODELS="${BASELINE_MODELS:-closed_evidence,qwen_base,qwen_sft_r16,qwen_grpo_v6_300,qwen_grpo_v6_full1500,qwen_valueaug_full1500,qwen_grpo_rfv2_ckpt1600}"
CLOSED_ENV_FILE="${CLOSED_ENV_FILE:-/tmp/pcv32_doctor_suite_env_20260707_0330}"
CLOSED_MODEL="${CLOSED_MODEL:-gpt-4.1-mini}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"
REALIZER_BATCH_SIZE="${REALIZER_BATCH_SIZE:-4}"

cd "$PHASE_DIR"
mkdir -p logs
LOG="logs/wait_closed_rubric_then_online_baseline_20260709.log"
exec > >(tee -a "$LOG") 2>&1

count_lines() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo 0
    return
  fi
  wc -l < "$path"
}

echo "=== wait closed rubric then online baseline start $(date) ==="
echo "req_path=$REQ_PATH"
echo "rubric_out=$RUBRIC_OUT"
echo "freezer_dir=$FREEZE_DIR"
echo "baseline_models=$BASELINE_MODELS"

REQ_N=$(count_lines "$REQ_PATH")
if [[ "$REQ_N" -le 0 ]]; then
  echo "No rubric requests found: $REQ_PATH" >&2
  exit 2
fi

while true; do
  OUT_N=$(count_lines "$RUBRIC_OUT")
  echo "[$(date)] closed_rubric_progress=${OUT_N}/${REQ_N}"
  if [[ "$OUT_N" -ge "$REQ_N" ]]; then
    break
  fi
  if ! pgrep -af 'run_patient_realizer_rubric_v3_2_final|call_closed_llm_for_patient_realizer' >/dev/null; then
    echo "Closed rubric process not running and incomplete: ${OUT_N}/${REQ_N}" >&2
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

echo "=== closed rubric complete; refresh summary $(date) ==="
"$PY" scripts/summarize_patient_realizer_rubric_judge_outputs_v1.py \
  --request-path "$REQ_PATH" \
  --output-path "$RUBRIC_OUT" \
  --report-dir "$RUBRIC_REPORT_DIR"

echo "=== build final freeze report $(date) ==="
"$PY" scripts/build_final_patient_freeze_report_v1.py \
  --request-path "$REQ_PATH" \
  --closed-rubric-output-path "$RUBRIC_OUT" \
  --closed-rubric-summary-path "$RUBRIC_SUMMARY" \
  --cache-path "$CACHE_PATH" \
  --cache-summary-path "$CACHE_SUMMARY" \
  --primary-verify-summary-path "$PRIMARY_VERIFY_SUMMARY" \
  --repair-verify-summary-path "$REPAIR_VERIFY_SUMMARY" \
  --repair2-verify-summary-path "$REPAIR2_VERIFY_SUMMARY" \
  --output-dir "$FREEZE_DIR"

FREEZE_PASS=$("$PY" - "$FREEZE_DIR/final_patient_freeze_report.json" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
print("true" if obj.get("freeze_pass") else "false")
PY
)
echo "freeze_pass=$FREEZE_PASS"
if [[ "$FREEZE_PASS" != "true" ]]; then
  echo "Final patient freeze failed; baseline will not start. See $FREEZE_DIR" >&2
  exit 4
fi

wait_gpu_idle() {
  while true; do
    local line
    line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1)
    local mem util
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    if [[ "${mem:-99999}" -lt 1000 && "${util:-100}" -lt 10 ]] && ! pgrep -af 'run_pcv32_online_final_patient|call_qwen3|train_qwen3|run_llm_doctor_online_replay' >/dev/null; then
      echo "GPU idle: mem=${mem}MiB util=${util}%"
      break
    fi
    echo "[$(date)] waiting GPU idle: mem=${mem}MiB util=${util}%"
    sleep 300
  done
}

echo "=== wait GPU idle before turn24 baseline $(date) ==="
wait_gpu_idle
echo "=== launch turn24 baseline $(date) ==="
MODELS_CSV="$BASELINE_MODELS" \
CLOSED_ENV_FILE="$CLOSED_ENV_FILE" \
CLOSED_MODEL="$CLOSED_MODEL" \
REPLAY_BATCH_SIZE="$REPLAY_BATCH_SIZE" \
REALIZER_BATCH_SIZE="$REALIZER_BATCH_SIZE" \
  bash scripts/run_pcv32_online_final_patient_doctor_eval_suite_20260709.sh \
  pcv32_online_final_patient_baseline_turn24_20260709_after_freeze 24

echo "=== wait GPU idle before turn32 baseline $(date) ==="
wait_gpu_idle
echo "=== launch turn32 baseline $(date) ==="
MODELS_CSV="$BASELINE_MODELS" \
CLOSED_ENV_FILE="$CLOSED_ENV_FILE" \
CLOSED_MODEL="$CLOSED_MODEL" \
REPLAY_BATCH_SIZE="$REPLAY_BATCH_SIZE" \
REALIZER_BATCH_SIZE="$REALIZER_BATCH_SIZE" \
  bash scripts/run_pcv32_online_final_patient_doctor_eval_suite_20260709.sh \
  pcv32_online_final_patient_baseline_turn32_20260709_after_freeze 32

echo "=== wait closed rubric then online baseline done $(date) ==="
