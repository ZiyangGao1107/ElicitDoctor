#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT/cache/qwen3-8b-hf-remote-code}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-$PHASE_DIR/outputs_qwen3_doctor_sft_lora_f32_f41_2ksteps_bf16/final_lora_adapter}"
STATE_BUILDER_SCRIPT="${STATE_BUILDER_SCRIPT:-scripts/build_guided_exploration_state_bank_v1.py}"
TEMPLATE_CANDIDATE_SCRIPT="${TEMPLATE_CANDIDATE_SCRIPT:-scripts/build_guided_exploration_template_candidate_outputs_v1.py}"

OUT="${1:?output dir required}"
shift
if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 OUTPUT_DIR LABEL=SOURCE_OUTPUT_DIR [LABEL=SOURCE_OUTPUT_DIR ...]" >&2
  exit 2
fi

MAX_STATES="${MAX_STATES:-180}"
MAX_TURN_INDEX="${MAX_TURN_INDEX:-18}"
STATE_SHARD_INDEX="${STATE_SHARD_INDEX:-0}"
STATE_SHARD_COUNT="${STATE_SHARD_COUNT:-1}"
STRATEGIES="${STRATEGIES:-direct,gentle_low_burden,bridge_low_sensitivity,permission_before_sensitive,repair_after_refusal,choice_based_concrete}"
VARIANTS_PER_STRATEGY="${VARIANTS_PER_STRATEGY:-1}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-4}"
REALIZER_BATCH_SIZE="${REALIZER_BATCH_SIZE:-4}"
CANDIDATE_TEMPERATURE="${CANDIDATE_TEMPERATURE:-0.35}"
REALIZER_TEMPERATURE="${REALIZER_TEMPERATURE:-0.0}"
REQUIRE_FULL_VERIFIED="${REQUIRE_FULL_VERIFIED:-0}"
CANDIDATE_GENERATOR_MODE="${CANDIDATE_GENERATOR_MODE:-template}"
DATASET_PREFIX="${DATASET_PREFIX:-mdd5k}"
LANGUAGE="${LANGUAGE:-zh}"
PROFILE_PATH="${PROFILE_PATH:-}"
SCHEMA_PATH="${SCHEMA_PATH:-}"

if [[ -z "$PROFILE_PATH" ]]; then
  if [[ "$DATASET_PREFIX" == "daic" ]]; then
    PROFILE_PATH="data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl"
  else
    PROFILE_PATH="data/patient_profiles/mdd5k_dialogue_derived_patient_profiles.jsonl"
  fi
fi
if [[ -z "$SCHEMA_PATH" ]]; then
  if [[ "$DATASET_PREFIX" == "daic" ]]; then
    SCHEMA_PATH="schemas/daic_symptom_slot_schema.json"
  else
    SCHEMA_PATH="schemas/mdd5k_symptom_slot_schema.json"
  fi
fi

export ACTIVE_REASONING_PROJECT="$PROJECT"
export HF_HOME="$PROJECT/cache/hf_home"
export HF_MODULES_CACHE="$PROJECT/cache/hf_modules"
export TRANSFORMERS_CACHE="$PROJECT/cache/transformers"
export MODELSCOPE_CACHE="$PROJECT/cache/modelscope"
export TOKENIZERS_PARALLELISM=false

cd "$PHASE_DIR"
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
mkdir -p "$OUT" logs
MAIN_LOG="$OUT/guided_exploration_first_turn_main.log"

count_file() {
  local path="$1"
  if [[ -s "$path" ]]; then
    "$PY" - "$path" <<'PY'
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(sum(1 for line in f if line.strip()))
PY
  else
    echo 0
  fi
}

echo "=== guided exploration first-turn start $(date) out=$OUT ===" | tee "$MAIN_LOG"
echo "max_states=$MAX_STATES max_turn_index=$MAX_TURN_INDEX shard=$STATE_SHARD_INDEX/$STATE_SHARD_COUNT strategies=$STRATEGIES variants=$VARIANTS_PER_STRATEGY candidate_generator=$CANDIDATE_GENERATOR_MODE" | tee -a "$MAIN_LOG"
echo "dataset_prefix=$DATASET_PREFIX language=$LANGUAGE profile_path=$PROFILE_PATH schema_path=$SCHEMA_PATH" | tee -a "$MAIN_LOG"

STATE_DIR="$OUT/01_state_bank"
STATE_PATH="$STATE_DIR/guided_exploration_state_bank.jsonl"
REQUEST_PATH="$STATE_DIR/guided_exploration_doctor_candidate_requests.jsonl"
BUILD_ARGS=(
  --output-dir "$STATE_DIR"
  --max-states "$MAX_STATES"
  --max-turn-index "$MAX_TURN_INDEX"
  --state-shard-index "$STATE_SHARD_INDEX"
  --state-shard-count "$STATE_SHARD_COUNT"
  --strategies "$STRATEGIES"
  --variants-per-strategy "$VARIANTS_PER_STRATEGY"
  --language "$LANGUAGE"
)
for source in "$@"; do
  BUILD_ARGS+=(--source "$source")
done
"$PY" "$STATE_BUILDER_SCRIPT" "${BUILD_ARGS[@]}" >"$OUT/01_state_bank.log" 2>&1
tail -40 "$OUT/01_state_bank.log" | tee -a "$MAIN_LOG" || true

REQUEST_N=$(count_file "$REQUEST_PATH")
if [[ "$REQUEST_N" -eq 0 ]]; then
  echo "no candidate requests built" >&2
  exit 10
fi

CAND_DIR="$OUT/02_doctor_candidate_outputs"
CAND_OUT="$CAND_DIR/qwen3_guided_exploration_doctor_candidate_outputs.jsonl"
mkdir -p "$CAND_DIR"
echo "=== generate guided doctor candidates $(date) requests=$REQUEST_N ===" | tee -a "$MAIN_LOG"
if [[ "$CANDIDATE_GENERATOR_MODE" == "template" ]]; then
  "$PY" "$TEMPLATE_CANDIDATE_SCRIPT" \
    --request-path "$REQUEST_PATH" \
    --output-path "$CAND_OUT" \
    --language "$LANGUAGE" \
    >"$OUT/02_doctor_candidate_outputs.log" 2>&1
elif [[ "$CANDIDATE_GENERATOR_MODE" == "qwen_sft" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_lora_for_pending_requests.py \
    --input-path "$REQUEST_PATH" \
    --output-path "$CAND_OUT" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$SFT_ADAPTER_PATH" \
    --provider-tag remote_qwen3_8b_guided_exploration_sft_candidate \
    --model-tag Qwen3-8B-SFT-GuidedExplorationCandidate \
    --limit 0 \
    --max-new-tokens 96 \
    --temperature "$CANDIDATE_TEMPERATURE" \
    --top-p 0.92 \
    --dtype bf16 \
    --batch-size "$CANDIDATE_BATCH_SIZE" \
    --flush-every 10 \
    >"$OUT/02_doctor_candidate_outputs.log" 2>&1
else
  echo "unsupported CANDIDATE_GENERATOR_MODE=$CANDIDATE_GENERATOR_MODE" >&2
  exit 11
fi
tail -40 "$OUT/02_doctor_candidate_outputs.log" | tee -a "$MAIN_LOG" || true

RULE_DIR="$OUT/03_rule_rollout"
echo "=== rule controller probe rollout $(date) ===" | tee -a "$MAIN_LOG"
"$PY" scripts/build_final_patient_candidate_rollout.py \
  --state-bank "$STATE_PATH" \
  --candidate-requests "$REQUEST_PATH" \
  --candidate-outputs "$CAND_OUT" \
  --output-dir "$RULE_DIR" \
  --profiles "$PROFILE_PATH" \
  --schema "$SCHEMA_PATH" \
  --require-all-outputs \
  >"$OUT/03_rule_rollout.log" 2>&1
tail -40 "$OUT/03_rule_rollout.log" | tee -a "$MAIN_LOG" || true
RULE_RECORDS="$RULE_DIR/final_patient_candidate_rule_rollout_records.jsonl"
RULE_N=$(count_file "$RULE_RECORDS")

WORK="$OUT/04_patient_realizer_work"
REQ_DIR="$WORK/01_requests"
PATIENT_REQ="$REQ_DIR/${DATASET_PREFIX}_llm_patient_realizer_requests.jsonl"
echo "=== prepare patient realizer requests $(date) rule_records=$RULE_N ===" | tee -a "$MAIN_LOG"
"$PY" scripts/prepare_patient_realizer_requests.py \
  --trajectory-path "$RULE_RECORDS" \
  --output-dir "$REQ_DIR" \
  --dataset-prefix "$DATASET_PREFIX" \
  --language "$LANGUAGE" \
  --max-requests 0 \
  --max-requests-per-cell 0 \
  --sample-seed 909 \
  >"$WORK.prepare_requests.log" 2>&1
tail -40 "$WORK.prepare_requests.log" | tee -a "$MAIN_LOG" || true

PRIMARY_DIR="$WORK/02_primary_qwen_outputs"
PRIMARY_OUT="$PRIMARY_DIR/qwen3_patient_realizer_outputs.jsonl"
mkdir -p "$PRIMARY_DIR"
PATIENT_REQ_N=$(count_file "$PATIENT_REQ")
echo "=== generate primary patient realizer $(date) requests=$PATIENT_REQ_N ===" | tee -a "$MAIN_LOG"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
  --input-path "$PATIENT_REQ" \
  --output-path "$PRIMARY_OUT" \
  --model-path "$MODEL_PATH" \
  --no-adapter \
  --provider-tag remote_qwen3_8b_guided_exploration_patient_primary \
  --model-tag Qwen3-8B-Patient-Realizer-GuidedExploration \
  --limit 0 \
  --max-new-tokens 220 \
  --temperature "$REALIZER_TEMPERATURE" \
  --dtype bf16 \
  --batch-size "$REALIZER_BATCH_SIZE" \
  --flush-every 8 \
  >"$WORK/02_primary_qwen_outputs.log" 2>&1
tail -40 "$WORK/02_primary_qwen_outputs.log" | tee -a "$MAIN_LOG" || true

PRIMARY_VERIFY_DIR="$WORK/03_primary_verify"
"$PY" scripts/verify_patient_realizer_outputs.py \
  --request-path "$PATIENT_REQ" \
  --output-path "$PRIMARY_OUT" \
  --report-dir "$PRIMARY_VERIFY_DIR" \
  --dataset-prefix "$DATASET_PREFIX" \
  --leak-threshold 0.72 \
  --allowed-threshold 0.45 \
  --reference-min-coverage 0.30 \
  --severe-max-coverage 0.45 \
  >"$WORK/03_primary_verify.log" 2>&1
tail -40 "$WORK/03_primary_verify.log" | tee -a "$MAIN_LOG" || true

REPAIR_REQ_FILES=()
REPAIR_VERIFY_FILES=()
SOURCE_REQ="$PATIENT_REQ"
SOURCE_VERIFY="$PRIMARY_VERIFY_DIR/${DATASET_PREFIX}_patient_realizer_verification_records_llm_outputs.jsonl"
for repair_round in 1 2; do
  REPAIR_REQ_DIR="$WORK/04_repair${repair_round}_requests"
  "$PY" scripts/prepare_patient_realizer_repair_requests.py \
    --request-path "$SOURCE_REQ" \
    --verification-records "$SOURCE_VERIFY" \
    --output-dir "$REPAIR_REQ_DIR" \
    --dataset-prefix "$DATASET_PREFIX" \
    >"$WORK/04_repair${repair_round}_requests.log" 2>&1
  REPAIR_REQ="$REPAIR_REQ_DIR/${DATASET_PREFIX}_llm_patient_realizer_repair_requests.jsonl"
  REPAIR_N=$(count_file "$REPAIR_REQ")
  echo "repair_round_${repair_round}_requests=$REPAIR_N" | tee -a "$MAIN_LOG"
  if [[ "$REPAIR_N" -eq 0 ]]; then
    break
  fi
  REPAIR_OUT_DIR="$WORK/05_repair${repair_round}_qwen_outputs"
  REPAIR_OUT="$REPAIR_OUT_DIR/qwen3_patient_realizer_repair_outputs.jsonl"
  mkdir -p "$REPAIR_OUT_DIR"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
    --input-path "$REPAIR_REQ" \
    --output-path "$REPAIR_OUT" \
    --model-path "$MODEL_PATH" \
    --no-adapter \
    --provider-tag "remote_qwen3_8b_guided_exploration_patient_repair${repair_round}" \
    --model-tag "Qwen3-8B-Patient-Realizer-GuidedExploration-Repair${repair_round}" \
    --limit 0 \
    --max-new-tokens 180 \
    --temperature "$REALIZER_TEMPERATURE" \
    --dtype bf16 \
    --batch-size "$REALIZER_BATCH_SIZE" \
    --flush-every 8 \
    >"$WORK/05_repair${repair_round}_qwen_outputs.log" 2>&1
  REPAIR_VERIFY_DIR="$WORK/06_repair${repair_round}_verify"
  "$PY" scripts/verify_patient_realizer_outputs.py \
    --request-path "$REPAIR_REQ" \
    --output-path "$REPAIR_OUT" \
    --report-dir "$REPAIR_VERIFY_DIR" \
    --dataset-prefix "$DATASET_PREFIX" \
    --leak-threshold 0.72 \
    --allowed-threshold 0.45 \
    --reference-min-coverage 0.30 \
    --severe-max-coverage 0.45 \
    >"$WORK/06_repair${repair_round}_verify.log" 2>&1
  REPAIR_REQ_FILES+=("$REPAIR_REQ")
  REPAIR_VERIFY_FILES+=("$REPAIR_VERIFY_DIR/${DATASET_PREFIX}_patient_realizer_verification_records_llm_outputs.jsonl")
  SOURCE_REQ="$REPAIR_REQ"
  SOURCE_VERIFY="$REPAIR_VERIFY_DIR/${DATASET_PREFIX}_patient_realizer_verification_records_llm_outputs.jsonl"
done

BUILD_CACHE_DIR="$WORK/07_verified_cache"
REPAIR_ARGS=()
if [[ "${#REPAIR_REQ_FILES[@]}" -gt 0 ]]; then
  MERGED_REPAIR_REQ="$WORK/merged_repair_requests.jsonl"
  MERGED_REPAIR_VERIFY="$WORK/merged_repair_verify.jsonl"
  "$PY" - "$MERGED_REPAIR_REQ" "${REPAIR_REQ_FILES[@]}" <<'PY'
import sys
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as out:
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line)
PY
  "$PY" - "$MERGED_REPAIR_VERIFY" "${REPAIR_VERIFY_FILES[@]}" <<'PY'
import sys
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as out:
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line)
PY
  REPAIR_ARGS=(--repair-request-path "$MERGED_REPAIR_REQ" --repair-verification-records "$MERGED_REPAIR_VERIFY")
fi
"$PY" scripts/build_verified_patient_realizer_cache.py \
  --primary-request-path "$PATIENT_REQ" \
  --primary-verification-records "$PRIMARY_VERIFY_DIR/${DATASET_PREFIX}_patient_realizer_verification_records_llm_outputs.jsonl" \
  "${REPAIR_ARGS[@]}" \
  --output-dir "$BUILD_CACHE_DIR" \
  --dataset-prefix "$DATASET_PREFIX" \
  --include-warned \
  >"$WORK/07_verified_cache.log" 2>&1
tail -40 "$WORK/07_verified_cache.log" | tee -a "$MAIN_LOG" || true

CACHE_PATH="$BUILD_CACHE_DIR/${DATASET_PREFIX}_verified_patient_response_cache_repair_include_warned.jsonl"
VERIFIED_DIR="$OUT/05_verified_rollout"
"$PY" scripts/apply_verified_patient_cache_to_candidate_rollout.py \
  --records "$RULE_RECORDS" \
  --cache "$CACHE_PATH" \
  --output-dir "$VERIFIED_DIR" \
  --drop-hard-errors \
  >"$OUT/05_apply_verified_cache.log" 2>&1
tail -40 "$OUT/05_apply_verified_cache.log" | tee -a "$MAIN_LOG" || true

VERIFIED_RECORDS="$VERIFIED_DIR/final_patient_candidate_verified_rollout_records.jsonl"
VERIFIED_N=$(count_file "$VERIFIED_RECORDS")
if [[ "$REQUIRE_FULL_VERIFIED" == "1" && "$VERIFIED_N" -lt "$RULE_N" ]]; then
  echo "verified coverage incomplete: verified=$VERIFIED_N rule=$RULE_N" >&2
  exit 20
fi

"$PY" - "$OUT" "$REQUEST_N" "$RULE_N" "$PATIENT_REQ_N" "$VERIFIED_N" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
summary = {
    "output_dir": str(out),
    "candidate_requests": int(sys.argv[2]),
    "rule_records": int(sys.argv[3]),
    "patient_realizer_requests": int(sys.argv[4]),
    "verified_records": int(sys.argv[5]),
    "verified_coverage": round(int(sys.argv[5]) / int(sys.argv[3]), 6) if int(sys.argv[3]) else 0.0,
    "stage": "first_turn_guided_exploration_verified_rollout",
}
(out / "GUIDED_EXPLORATION_FIRST_TURN_READY.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== guided exploration first-turn done $(date) out=$OUT ===" | tee -a "$MAIN_LOG"
