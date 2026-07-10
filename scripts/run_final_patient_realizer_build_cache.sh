#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT/cache/qwen3-8b-hf-remote-code}"
TAG="${1:-final_patient_realizer_cache}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-${2:-}}"

cd "$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"

ROOT="outputs_${TAG}"
REQ_DIR="${ROOT}_01_requests"
PRIMARY_OUT_DIR="${ROOT}_02_primary_qwen_outputs"
PRIMARY_VERIFY_DIR="${ROOT}_03_primary_verify"
REPAIR_REQ_DIR="${ROOT}_04_repair_requests"
REPAIR_OUT_DIR="${ROOT}_05_repair_qwen_outputs"
REPAIR_VERIFY_DIR="${ROOT}_06_repair_verify"
CACHE_DIR="${ROOT}_07_verified_cache"
mkdir -p "$ROOT" "$PRIMARY_OUT_DIR" logs
LOG="logs/${TAG}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== final patient realizer cache build start $(date) tag=$TAG ==="
echo "trajectory=$TRAJECTORY_PATH"
echo "model_path=$MODEL_PATH"

if [[ ! -s "$TRAJECTORY_PATH" ]]; then
  echo "missing trajectory. Set TRAJECTORY_PATH or pass it as the second argument." >&2
  exit 10
fi

echo "--- step1 prepare hardened v3 realizer requests ---"
"$PY" scripts/prepare_patient_realizer_requests.py \
  --trajectory-path "$TRAJECTORY_PATH" \
  --output-dir "$REQ_DIR" \
  --max-requests 0 \
  --max-requests-per-cell 0 \
  --sample-seed 17

REQ_PATH="$REQ_DIR/mdd5k_llm_patient_realizer_requests.jsonl"
REQ_SUMMARY="$REQ_DIR/mdd5k_llm_patient_realizer_request_summary.json"
cat "$REQ_SUMMARY"
REQ_N=$(wc -l < "$REQ_PATH")
echo "request_count=$REQ_N"

echo "--- step2 call Qwen3-8B primary realizer ---"
PRIMARY_OUT="$PRIMARY_OUT_DIR/qwen3_patient_realizer_outputs.jsonl"
"$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
  --input-path "$REQ_PATH" \
  --output-path "$PRIMARY_OUT" \
  --model-path "$MODEL_PATH" \
  --no-adapter \
  --provider-tag remote_qwen3_8b_pcv32_hardened_full_primary \
  --model-tag Qwen3-8B-Patient-Realizer-PCV32-Hardened-Full-Primary \
  --limit 0 \
  --max-new-tokens 220 \
  --temperature 0.0 \
  --dtype bf16 \
  --batch-size 4 \
  --flush-every 32
wc -l "$PRIMARY_OUT"

echo "--- step3 verify primary outputs ---"
"$PY" scripts/verify_patient_realizer_outputs.py \
  --request-path "$REQ_PATH" \
  --output-path "$PRIMARY_OUT" \
  --report-dir "$PRIMARY_VERIFY_DIR" \
  --leak-threshold 0.72 \
  --allowed-threshold 0.45 \
  --reference-min-coverage 0.30 \
  --severe-max-coverage 0.45

PRIMARY_VERIFY_RECORDS="$PRIMARY_VERIFY_DIR/mdd5k_patient_realizer_verification_records_llm_outputs.jsonl"
PRIMARY_VERIFY_SUMMARY="$PRIMARY_VERIFY_DIR/mdd5k_patient_realizer_verification_summary_llm_outputs.json"
cat "$PRIMARY_VERIFY_SUMMARY"

echo "--- step4 prepare repair requests ---"
"$PY" scripts/prepare_patient_realizer_repair_requests.py \
  --request-path "$REQ_PATH" \
  --verification-records "$PRIMARY_VERIFY_RECORDS" \
  --output-dir "$REPAIR_REQ_DIR"

REPAIR_REQ_PATH="$REPAIR_REQ_DIR/mdd5k_llm_patient_realizer_repair_requests.jsonl"
REPAIR_REQ_SUMMARY="$REPAIR_REQ_DIR/mdd5k_llm_patient_realizer_repair_request_summary.json"
cat "$REPAIR_REQ_SUMMARY"
REPAIR_N=$(wc -l < "$REPAIR_REQ_PATH")
echo "repair_request_count=$REPAIR_N"

if [[ "$REPAIR_N" -gt 0 ]]; then
  echo "--- step5 call Qwen3-8B repair realizer ---"
  mkdir -p "$REPAIR_OUT_DIR"
  REPAIR_OUT="$REPAIR_OUT_DIR/qwen3_patient_realizer_repair_outputs.jsonl"
  "$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
    --input-path "$REPAIR_REQ_PATH" \
    --output-path "$REPAIR_OUT" \
    --model-path "$MODEL_PATH" \
    --no-adapter \
    --provider-tag remote_qwen3_8b_pcv32_hardened_full_repair \
    --model-tag Qwen3-8B-Patient-Realizer-PCV32-Hardened-Full-Repair \
    --limit 0 \
    --max-new-tokens 180 \
    --temperature 0.0 \
    --dtype bf16 \
    --batch-size 4 \
    --flush-every 32
  wc -l "$REPAIR_OUT"

  echo "--- step6 verify repair outputs ---"
  "$PY" scripts/verify_patient_realizer_outputs.py \
    --request-path "$REPAIR_REQ_PATH" \
    --output-path "$REPAIR_OUT" \
    --report-dir "$REPAIR_VERIFY_DIR" \
    --leak-threshold 0.72 \
    --allowed-threshold 0.45 \
    --reference-min-coverage 0.30 \
    --severe-max-coverage 0.45
  REPAIR_VERIFY_RECORDS="$REPAIR_VERIFY_DIR/mdd5k_patient_realizer_verification_records_llm_outputs.jsonl"
  REPAIR_VERIFY_SUMMARY="$REPAIR_VERIFY_DIR/mdd5k_patient_realizer_verification_summary_llm_outputs.json"
  cat "$REPAIR_VERIFY_SUMMARY"
  REPAIR_ARGS=(--repair-request-path "$REPAIR_REQ_PATH" --repair-verification-records "$REPAIR_VERIFY_RECORDS")
else
  echo "--- step5/6 no repair needed ---"
  REPAIR_ARGS=()
fi

echo "--- step7 build verified cache, no rule fallback ---"
"$PY" scripts/build_verified_patient_realizer_cache.py \
  --primary-request-path "$REQ_PATH" \
  --primary-verification-records "$PRIMARY_VERIFY_RECORDS" \
  "${REPAIR_ARGS[@]}" \
  --output-dir "$CACHE_DIR" \
  --include-warned

CACHE_PATH="$CACHE_DIR/mdd5k_verified_patient_response_cache_repair_include_warned.jsonl"
CACHE_SUMMARY="$CACHE_DIR/mdd5k_verified_patient_response_cache_repair_summary_include_warned.json"
cat "$CACHE_SUMMARY"
CACHE_N=$(wc -l < "$CACHE_PATH")
if [[ "$CACHE_N" -lt "$REQ_N" ]]; then
  echo "FULL_REALIZER_INCOMPLETE cache=$CACHE_N requests=$REQ_N; remaining failures require another repair pass or prompt/verifier update." >&2
  exit 20
fi

echo "=== final patient realizer cache build PASS $(date) tag=$TAG ==="
