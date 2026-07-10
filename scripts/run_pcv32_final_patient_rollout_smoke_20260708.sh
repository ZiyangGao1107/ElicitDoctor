#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
TAG="${1:-pcv32_final_patient_rollout_smoke_20260708_1905}"
MAX_GROUPS="${MAX_GROUPS:-3}"
MAX_PROFILES="${MAX_PROFILES:-3}"
MAX_TURNS="${MAX_TURNS:-4}"
MAX_REQUESTS="${MAX_REQUESTS:-24}"
MODEL_PATH="${MODEL_PATH:-$PROJECT/cache/qwen3-8b-hf-remote-code}"

cd "$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"

ROOT="outputs_${TAG}"
RULE_DIR="${ROOT}_01_rule_source"
REQ_DIR="${ROOT}_02_realizer_requests"
OUT_DIR="${ROOT}_03_qwen_outputs"
VERIFY_DIR="${ROOT}_04_verify"
REPAIR_REQ_DIR="${ROOT}_05_repair_requests"
REPAIR_OUT_DIR="${ROOT}_06_repair_qwen_outputs"
REPAIR_VERIFY_DIR="${ROOT}_07_repair_verify"
CACHE_DIR="${ROOT}_08_cache"
FINAL_DIR="${ROOT}_09_verified_replay"
mkdir -p "$ROOT" "$OUT_DIR" logs
LOG="logs/${TAG}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== PCV3.2 final-patient rollout smoke start $(date) tag=$TAG ==="
echo "settings: max_groups=$MAX_GROUPS max_profiles=$MAX_PROFILES max_turns=$MAX_TURNS max_requests=$MAX_REQUESTS"

echo "--- step1 rule source trajectory ---"
"$PY" scripts/run_llm_doctor_online_replay_v1.py \
  --output-dir "$RULE_DIR" \
  --patient-controller-version v3_2 \
  --patient-realizer-mode rule \
  --provider scripted \
  --missing-output-policy scripted \
  --max-groups "$MAX_GROUPS" \
  --max-per-slot 1 \
  --max-profiles "$MAX_PROFILES" \
  --max-turns "$MAX_TURNS" \
  --policies closed_llm_general

RULE_RECORDS="$RULE_DIR/mdd5k_llm_doctor_online_replay_records.jsonl"
if [[ ! -s "$RULE_RECORDS" ]]; then
  echo "missing rule source records: $RULE_RECORDS" >&2
  exit 10
fi
wc -l "$RULE_RECORDS"

echo "--- step2 prepare LLM realizer requests ---"
"$PY" scripts/prepare_llm_patient_realizer_requests_v2.py \
  --trajectory-path "$RULE_RECORDS" \
  --output-dir "$REQ_DIR" \
  --max-requests "$MAX_REQUESTS" \
  --max-requests-per-cell 4 \
  --sample-seed 808

REQ_PATH="$REQ_DIR/mdd5k_llm_patient_realizer_requests.jsonl"
wc -l "$REQ_PATH"

echo "--- step3 call Qwen3-8B patient realizer ---"
QWEN_OUT="$OUT_DIR/qwen3_patient_realizer_outputs.jsonl"
"$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
  --input-path "$REQ_PATH" \
  --output-path "$QWEN_OUT" \
  --model-path "$MODEL_PATH" \
  --no-adapter \
  --provider-tag remote_qwen3_8b_final_patient_smoke \
  --model-tag Qwen3-8B-Patient-Realizer-PCV32-Smoke \
  --limit 0 \
  --max-new-tokens 220 \
  --temperature 0.0 \
  --dtype bf16 \
  --batch-size 4 \
  --flush-every 4
wc -l "$QWEN_OUT"

echo "--- step4 verify LLM realizer outputs ---"
"$PY" scripts/verify_llm_patient_realizer_outputs_v1.py \
  --request-path "$REQ_PATH" \
  --output-path "$QWEN_OUT" \
  --report-dir "$VERIFY_DIR" \
  --leak-threshold 0.72 \
  --allowed-threshold 0.45 \
  --reference-min-coverage 0.30 \
  --severe-max-coverage 0.45

VERIFY_RECORDS="$VERIFY_DIR/mdd5k_patient_realizer_verification_records_llm_outputs.jsonl"
VERIFY_SUMMARY="$VERIFY_DIR/mdd5k_patient_realizer_verification_summary_llm_outputs.json"
cat "$VERIFY_SUMMARY"

echo "--- step5 build verified cache, no rule fallback ---"
echo "--- step5 prepare repair requests for verifier failures ---"
"$PY" scripts/prepare_llm_patient_realizer_repair_requests_v1.py \
  --request-path "$REQ_PATH" \
  --verification-records "$VERIFY_RECORDS" \
  --output-dir "$REPAIR_REQ_DIR"

REPAIR_REQ_PATH="$REPAIR_REQ_DIR/mdd5k_llm_patient_realizer_repair_requests.jsonl"
REPAIR_SUMMARY="$REPAIR_REQ_DIR/mdd5k_llm_patient_realizer_repair_request_summary.json"
cat "$REPAIR_SUMMARY"

REPAIR_N=$(wc -l < "$REPAIR_REQ_PATH")
if [[ "$REPAIR_N" -gt 0 ]]; then
  echo "--- step6 call Qwen3-8B patient realizer repair ---"
  mkdir -p "$REPAIR_OUT_DIR"
  REPAIR_QWEN_OUT="$REPAIR_OUT_DIR/qwen3_patient_realizer_repair_outputs.jsonl"
  "$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
    --input-path "$REPAIR_REQ_PATH" \
    --output-path "$REPAIR_QWEN_OUT" \
    --model-path "$MODEL_PATH" \
    --no-adapter \
    --provider-tag remote_qwen3_8b_final_patient_smoke_repair \
    --model-tag Qwen3-8B-Patient-Realizer-PCV32-Smoke-Repair \
    --limit 0 \
    --max-new-tokens 180 \
    --temperature 0.0 \
    --dtype bf16 \
    --batch-size 4 \
    --flush-every 4
  wc -l "$REPAIR_QWEN_OUT"

  echo "--- step7 verify repair outputs ---"
  "$PY" scripts/verify_llm_patient_realizer_outputs_v1.py \
    --request-path "$REPAIR_REQ_PATH" \
    --output-path "$REPAIR_QWEN_OUT" \
    --report-dir "$REPAIR_VERIFY_DIR" \
    --leak-threshold 0.72 \
    --allowed-threshold 0.45 \
    --reference-min-coverage 0.30 \
    --severe-max-coverage 0.45
  REPAIR_VERIFY_RECORDS="$REPAIR_VERIFY_DIR/mdd5k_patient_realizer_verification_records_llm_outputs.jsonl"
  REPAIR_VERIFY_SUMMARY="$REPAIR_VERIFY_DIR/mdd5k_patient_realizer_verification_summary_llm_outputs.json"
  cat "$REPAIR_VERIFY_SUMMARY"
else
  echo "--- step6/7 no repair needed ---"
  REPAIR_VERIFY_RECORDS=""
fi

echo "--- step8 build verified cache from primary + repair, no rule fallback ---"
if [[ "$REPAIR_N" -gt 0 ]]; then
  REPAIR_ARGS=(--repair-request-path "$REPAIR_REQ_PATH" --repair-verification-records "$REPAIR_VERIFY_RECORDS")
else
  REPAIR_ARGS=()
fi
"$PY" scripts/build_verified_patient_realizer_cache_with_repair_v1.py \
  --primary-request-path "$REQ_PATH" \
  --primary-verification-records "$VERIFY_RECORDS" \
  "${REPAIR_ARGS[@]}" \
  --output-dir "$CACHE_DIR" \
  --include-warned

CACHE_PATH="$CACHE_DIR/mdd5k_verified_patient_response_cache_repair_include_warned.jsonl"
CACHE_SUMMARY="$CACHE_DIR/mdd5k_verified_patient_response_cache_repair_summary_include_warned.json"
cat "$CACHE_SUMMARY"

REQ_N=$(wc -l < "$REQ_PATH")
CACHE_N=$(wc -l < "$CACHE_PATH")
if [[ "$CACHE_N" -lt "$REQ_N" ]]; then
  echo "SMOKE_FAIL insufficient verified cache: cache=$CACHE_N requests=$REQ_N. Do not train with fallback hidden." >&2
  exit 20
fi

echo "--- step9 rerun same scripted trajectory with verified_cache + error ---"
"$PY" scripts/run_llm_doctor_online_replay_v1.py \
  --output-dir "$FINAL_DIR" \
  --patient-controller-version v3_2 \
  --patient-realizer-mode verified_cache \
  --patient-realizer-cache-path "$CACHE_PATH" \
  --patient-realizer-cache-policy error \
  --provider scripted \
  --missing-output-policy scripted \
  --max-groups "$MAX_GROUPS" \
  --max-per-slot 1 \
  --max-profiles "$MAX_PROFILES" \
  --max-turns "$MAX_TURNS" \
  --policies closed_llm_general

FINAL_RECORDS="$FINAL_DIR/mdd5k_llm_doctor_online_replay_records.jsonl"
"$PY" - "$FINAL_RECORDS" <<'PY'
import json, sys, collections
path=sys.argv[1]
c=collections.Counter(); n=0
with open(path, encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r=json.loads(line); n+=1
        c[str(r.get('patient_realizer_mode'))]+=1
print(json.dumps({'records': n, 'patient_realizer_mode_counts': dict(c)}, ensure_ascii=False, indent=2))
if not c or any(k != 'verified_llm_cache' for k in c):
    raise SystemExit('SMOKE_FAIL non-verified patient_realizer_mode present')
PY

echo "=== PCV3.2 final-patient rollout smoke PASS $(date) ==="
