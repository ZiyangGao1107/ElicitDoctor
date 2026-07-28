#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
PY="${AR_GRPO_PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT/cache/qwen3-8b-hf-remote-code}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

DATASET_PREFIX="daic"
LANGUAGE="en"
PROFILE_PATH="${PROFILE_PATH:-data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl}"
SCHEMA_PATH="${SCHEMA_PATH:-schemas/daic_symptom_slot_schema.json}"
GROUP_DIR="${GROUP_DIR:-data/daic/profile_split}"
CANONICAL_DIR="${CANONICAL_DIR:-data/daic/canonical_evidence}"
CANONICAL_PREFIX="${CANONICAL_PREFIX:-daic}"

MAX_TURNS="${MAX_TURNS:-24}"
TRAIN_MAX_PROFILES="${TRAIN_MAX_PROFILES:-107}"
TRAIN_MAX_GROUPS="${TRAIN_MAX_GROUPS:-856}"
TEST_MAX_PROFILES="${TEST_MAX_PROFILES:-219}"
TEST_MAX_GROUPS="${TEST_MAX_GROUPS:-1752}"
GUIDED_MAX_STATES="${GUIDED_MAX_STATES:-180}"
GUIDED_MAX_TURN_INDEX="${GUIDED_MAX_TURN_INDEX:-18}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-200}"
TRAIN_MAX_CANDIDATES="${TRAIN_MAX_CANDIDATES:-6}"
TRAIN_EVAL_GROUPS="${TRAIN_EVAL_GROUPS:-64}"

SOURCE_OUT="${SOURCE_OUT:-outputs_daic_qwen8b_base_low_info_train_source_${RUN_TAG}}"
GUIDED_OUT="${GUIDED_OUT:-outputs_daic_qwen8b_only_short_term_guided_train_${RUN_TAG}}"
BELIEF_DIR="${BELIEF_DIR:-outputs_daic_qwen8b_only_short_term_belief_train_${RUN_TAG}}"
REWARD_DIR="${REWARD_DIR:-outputs_daic_qwen8b_only_short_term_reward_train_${RUN_TAG}}"
TRAIN_OUT="${TRAIN_OUT:-outputs_daic_qwen8b_only_short_term_lora_${RUN_TAG}}"
TEST_OUT="${TEST_OUT:-outputs_daic_qwen8b_only_short_term_low_info_test_${RUN_TAG}}"

export ACTIVE_REASONING_PROJECT="$PROJECT"
export ACTIVE_REASONING_PHASE_DIR="$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"
export HF_HOME="$PROJECT/cache/hf_home"
export HF_MODULES_CACHE="$PROJECT/cache/hf_modules"
export TRANSFORMERS_CACHE="$PROJECT/cache/transformers"
export MODELSCOPE_CACHE="$PROJECT/cache/modelscope"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PHASE_DIR"
mkdir -p logs "$BELIEF_DIR"
LOG="logs/daic_qwen8b_only_short_term_train_eval_${RUN_TAG}.log"
exec > >(tee -a "$LOG") 2>&1

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

echo "=== DAIC Qwen-8B only-short train/eval start $(date) tag=$RUN_TAG ==="
echo "model_path=$MODEL_PATH"
echo "source_out=$SOURCE_OUT guided_out=$GUIDED_OUT belief_dir=$BELIEF_DIR reward_dir=$REWARD_DIR train_out=$TRAIN_OUT test_out=$TEST_OUT"

SOURCE_RECORDS="$SOURCE_OUT/${DATASET_PREFIX}_llm_doctor_online_replay_records.jsonl"
if [[ ! -s "$SOURCE_RECORDS" ]]; then
  echo "=== step 1/6: build DAIC train source replay $(date) ==="
  DATASET_PREFIX="$DATASET_PREFIX" \
  LANGUAGE="$LANGUAGE" \
  EVAL_SPLITS="train" \
  GROUP_DIR="$GROUP_DIR" \
  PROFILE_PATH="$PROFILE_PATH" \
  SCHEMA_PATH="$SCHEMA_PATH" \
  CANONICAL_DIR="$CANONICAL_DIR" \
  CANONICAL_PREFIX="$CANONICAL_PREFIX" \
  MAX_PROFILES="$TRAIN_MAX_PROFILES" \
  MAX_GROUPS="$TRAIN_MAX_GROUPS" \
  MAX_PER_SLOT=999 \
  MODEL_PATH="$MODEL_PATH" \
  DOCTOR_MODEL_PATH="$MODEL_PATH" \
  PATIENT_MODEL_PATH="$MODEL_PATH" \
  PATIENT_REALIZER_FALLBACK_TO_RULE=1 \
  "$PHASE_DIR/scripts/run_final_patient_doctor_eval_one.sh" qwen_base "$SOURCE_OUT" "$MAX_TURNS"
else
  echo "=== step 1/6: source replay exists, skip: $SOURCE_RECORDS lines=$(count_file "$SOURCE_RECORDS") ==="
fi

GUIDED_READY="$GUIDED_OUT/GUIDED_EXPLORATION_FIRST_TURN_READY.json"
if [[ ! -s "$GUIDED_READY" ]]; then
  echo "=== step 2/6: build guided same-state candidate rollout $(date) ==="
  DATASET_PREFIX="$DATASET_PREFIX" \
  LANGUAGE="$LANGUAGE" \
  PROFILE_PATH="$PROFILE_PATH" \
  SCHEMA_PATH="$SCHEMA_PATH" \
  MODEL_PATH="$MODEL_PATH" \
  MAX_STATES="$GUIDED_MAX_STATES" \
  MAX_TURN_INDEX="$GUIDED_MAX_TURN_INDEX" \
  CANDIDATE_GENERATOR_MODE=template \
  REQUIRE_FULL_VERIFIED=0 \
  "$PHASE_DIR/scripts/run_guided_exploration_first_turn_v1.sh" "$GUIDED_OUT" "qwen_base_train=$SOURCE_OUT"
else
  echo "=== step 2/6: guided rollout exists, skip: $GUIDED_READY ==="
fi

VERIFIED_RECORDS="$GUIDED_OUT/05_verified_rollout/final_patient_candidate_verified_rollout_records.jsonl"
BELIEF_REQ="$BELIEF_DIR/belief_guided_same_state_belief_requests.jsonl"
if [[ ! -s "$BELIEF_REQ" ]]; then
  echo "=== step 3/6: prepare belief reward requests $(date) ==="
  "$PY" scripts/build_belief_guided_same_state_reward_data.py prepare \
    --records "$VERIFIED_RECORDS" \
    --output-dir "$BELIEF_DIR" \
    --max-turn-index "$GUIDED_MAX_TURN_INDEX" \
    --min-candidates 2
else
  echo "=== step 3/6: belief requests exist, skip: $BELIEF_REQ lines=$(count_file "$BELIEF_REQ") ==="
fi

BELIEF_OUT="$BELIEF_DIR/qwen3_belief_outputs.jsonl"
BELIEF_REQ_N=$(count_file "$BELIEF_REQ")
BELIEF_OUT_N=$(count_file "$BELIEF_OUT")
if [[ "$BELIEF_OUT_N" -lt "$BELIEF_REQ_N" ]]; then
  echo "=== step 4/6: generate belief evaluator outputs $(date) requests=$BELIEF_REQ_N existing=$BELIEF_OUT_N ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_lora_for_pending_requests.py \
    --input-path "$BELIEF_REQ" \
    --output-path "$BELIEF_OUT" \
    --model-path "$MODEL_PATH" \
    --no-adapter \
    --provider-tag remote_qwen3_8b_daic_short_belief_eval \
    --model-tag Qwen3-8B-DAIC-ShortBeliefEval \
    --limit 0 \
    --max-new-tokens 512 \
    --temperature 0.0 \
    --dtype bf16 \
    --batch-size 4 \
    --flush-every 10
else
  echo "=== step 4/6: belief outputs complete, skip: $BELIEF_OUT lines=$BELIEF_OUT_N ==="
fi

GROUP_FILE="$REWARD_DIR/belief_guided_same_state_grpo_groups.jsonl"
if [[ ! -s "$GROUP_FILE" ]]; then
  echo "=== step 5/6a: score short-term belief rewards $(date) ==="
  "$PY" scripts/build_belief_guided_same_state_reward_data.py score \
    --records "$VERIFIED_RECORDS" \
    --belief-outputs "$BELIEF_OUT" \
    --output-dir "$REWARD_DIR" \
    --max-turn-index "$GUIDED_MAX_TURN_INDEX" \
    --min-candidates 2 \
    --language "$LANGUAGE"
else
  echo "=== step 5/6a: GRPO groups exist, skip: $GROUP_FILE lines=$(count_file "$GROUP_FILE") ==="
fi

FINAL_ADAPTER="$TRAIN_OUT/final_lora_adapter"
if [[ ! -s "$FINAL_ADAPTER/adapter_config.json" ]]; then
  echo "=== step 5/6b: train Qwen3-8B only-short LoRA $(date) ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_qwen3_grpo_from_v6_groups.py \
    --model-path "$MODEL_PATH" \
    --sft-adapter-path "$PHASE_DIR/outputs_missing_sft_adapter_for_base_lora" \
    --group-data "$GROUP_FILE" \
    --output-dir "$TRAIN_OUT" \
    --max-steps "$TRAIN_MAX_STEPS" \
    --max-groups 0 \
    --eval-groups "$TRAIN_EVAL_GROUPS" \
    --max-candidates "$TRAIN_MAX_CANDIDATES" \
    --max-length 768 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 6e-6 \
    --warmup-ratio 0.03 \
    --logging-steps 10 \
    --eval-steps 50 \
    --save-steps 0 \
    --save-milestones 50,100,150,200 \
    --advantage-mode zscore \
    --reward-clip 3.0 \
    --kl-coef 0.03 \
    --length-normalize \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --bf16 \
    --gradient-checkpointing
else
  echo "=== step 5/6b: final adapter exists, skip: $FINAL_ADAPTER ==="
fi

TEST_RECORDS="$TEST_OUT/${DATASET_PREFIX}_llm_doctor_online_replay_records.jsonl"
if [[ ! -s "$TEST_RECORDS" ]]; then
  echo "=== step 6/6: evaluate trained only-short LoRA on Extended-DAIC test $(date) ==="
  DATASET_PREFIX="$DATASET_PREFIX" \
  LANGUAGE="$LANGUAGE" \
  EVAL_SPLITS="test" \
  GROUP_DIR="$GROUP_DIR" \
  PROFILE_PATH="$PROFILE_PATH" \
  SCHEMA_PATH="$SCHEMA_PATH" \
  CANONICAL_DIR="$CANONICAL_DIR" \
  CANONICAL_PREFIX="$CANONICAL_PREFIX" \
  MAX_PROFILES="$TEST_MAX_PROFILES" \
  MAX_GROUPS="$TEST_MAX_GROUPS" \
  MAX_PER_SLOT=999 \
  MODEL_PATH="$MODEL_PATH" \
  DOCTOR_MODEL_PATH="$MODEL_PATH" \
  PATIENT_MODEL_PATH="$MODEL_PATH" \
  CUSTOM_ADAPTER_PATH="$FINAL_ADAPTER" \
  CUSTOM_MODEL_ID="qwen8b_only_short_term_daic_${RUN_TAG}" \
  CUSTOM_MODEL_OUTPUT_FILENAME="qwen8b_only_short_term_daic_pcv32_online_doctor_outputs.jsonl" \
  CUSTOM_PROVIDER_TAG="remote_qwen3_8b_only_short_term_daic_pcv32_online" \
  CUSTOM_MODEL_TAG="Qwen3-8B-OnlyShortTerm-DAIC-PCV32-OnlineFinalPatient" \
  PATIENT_REALIZER_FALLBACK_TO_RULE=1 \
  "$PHASE_DIR/scripts/run_final_patient_doctor_eval_one.sh" qwen_lora_custom "$TEST_OUT" "$MAX_TURNS"
else
  echo "=== step 6/6: test replay exists, skip: $TEST_RECORDS lines=$(count_file "$TEST_RECORDS") ==="
fi

"$PY" - "$SOURCE_RECORDS" "$VERIFIED_RECORDS" "$GROUP_FILE" "$FINAL_ADAPTER" "$TEST_OUT" <<'PY'
import json
import sys
from pathlib import Path

source_records, verified_records, group_file, final_adapter, test_out = map(Path, sys.argv[1:])
summary = {
    "source_records": str(source_records),
    "source_record_count": sum(1 for line in source_records.open(encoding="utf-8") if line.strip()) if source_records.exists() else 0,
    "verified_candidate_records": str(verified_records),
    "verified_candidate_record_count": sum(1 for line in verified_records.open(encoding="utf-8") if line.strip()) if verified_records.exists() else 0,
    "grpo_groups": str(group_file),
    "grpo_group_count": sum(1 for line in group_file.open(encoding="utf-8") if line.strip()) if group_file.exists() else 0,
    "final_adapter": str(final_adapter),
    "test_output_dir": str(test_out),
}
Path(test_out).mkdir(parents=True, exist_ok=True)
(Path(test_out) / "DAIC_QWEN8B_ONLY_SHORT_TERM_PIPELINE_SUMMARY.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== DAIC Qwen-8B only-short train/eval done $(date) tag=$RUN_TAG ==="
