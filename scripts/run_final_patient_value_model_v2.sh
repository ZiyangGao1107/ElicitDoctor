#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
ENV_PATH="${AR_GRPO_ENV:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VERIFIED_CANDIDATE_ROLLOUT_RECORDS [RUN_TAG]" >&2
  exit 2
fi

CANDIDATE_RECORDS="$1"
RUN_TAG="${2:-final_patient_value_model_v2_$(date +%Y%m%d_%H%M%S)}"

METRIC_NAME="${METRIC_NAME:-keyword_supported_only}"
DATASET_PREFIX="${DATASET_PREFIX:-mdd5k}"
CANONICAL_DIR="${CANONICAL_DIR:-$PHASE_DIR/data/tree_aligned_canonical_evidence}"
TARGET_MODE="${TARGET_MODE:-action_value_total}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.5}"
PAIR_MIN_MARGIN="${PAIR_MIN_MARGIN:-0.01}"
EPOCHS="${EPOCHS:-12}"
REWARD_SOURCE="${REWARD_SOURCE:-immediate_delta}"
BASE_REWARD_WEIGHT="${BASE_REWARD_WEIGHT:-1.0}"
VALUE_WEIGHT="${VALUE_WEIGHT:-0.5}"

ACTION_VALUE_DIR="${ACTION_VALUE_DIR:-$PHASE_DIR/outputs_final_patient_action_value_data_${RUN_TAG}}"
VALUE_MODEL_DIR="${VALUE_MODEL_DIR:-$PHASE_DIR/outputs_final_patient_action_value_model_${RUN_TAG}}"
VALUE_SCORE_DIR="${VALUE_SCORE_DIR:-$PHASE_DIR/outputs_final_patient_action_value_scores_${RUN_TAG}}"
VALUE_GROUP_DIR="${VALUE_GROUP_DIR:-$PHASE_DIR/outputs_final_patient_valueaug_grpo_groups_${RUN_TAG}}"

if [[ ! -f "$CANDIDATE_RECORDS" ]]; then
  echo "Missing candidate rollout records: $CANDIDATE_RECORDS" >&2
  exit 2
fi
if [[ "$DATASET_PREFIX" == "daic" && "$CANONICAL_DIR" == "$PHASE_DIR/data/tree_aligned_canonical_evidence" ]]; then
  CANONICAL_DIR="$PHASE_DIR/data/daic/canonical_evidence"
fi

export ACTIVE_REASONING_PROJECT="$PROJECT"
export PYTHONUNBUFFERED=1
mkdir -p "$PHASE_DIR/logs"

if [[ -n "${ENV_PATH:-}" ]]; then source "$ENV_PATH/bin/activate"; fi
cd "$PHASE_DIR"

python scripts/build_final_patient_action_value_data.py \
  --records "$CANDIDATE_RECORDS" \
  --output-dir "$ACTION_VALUE_DIR" \
  --canonical-dir "$CANONICAL_DIR" \
  --dataset-prefix "$DATASET_PREFIX" \
  --metric-name "$METRIC_NAME"

python scripts/train_final_patient_rfv_value_model.py \
  --record-path "$ACTION_VALUE_DIR/final_patient_action_value_records.jsonl" \
  --output-dir "$VALUE_MODEL_DIR" \
  --target-mode "$TARGET_MODE" \
  --pairwise-weight "$PAIRWISE_WEIGHT" \
  --pair-min-margin "$PAIR_MIN_MARGIN" \
  --epochs "$EPOCHS"

python scripts/score_final_patient_value_model.py \
  --record-path "$ACTION_VALUE_DIR/final_patient_action_value_records.jsonl" \
  --model-dir "$VALUE_MODEL_DIR" \
  --output-dir "$VALUE_SCORE_DIR" \
  --target-mode "$TARGET_MODE" \
  --pair-min-margin "$PAIR_MIN_MARGIN"

python scripts/build_final_patient_grpo_groups.py \
  --records "$CANDIDATE_RECORDS" \
  --output-dir "$VALUE_GROUP_DIR" \
  --reward-source "$REWARD_SOURCE" \
  --value-predictions "$VALUE_SCORE_DIR/final_patient_value_model_predictions.jsonl" \
  --base-reward-weight "$BASE_REWARD_WEIGHT" \
  --value-weight "$VALUE_WEIGHT" \
  --require-value-predictions

cat <<EOF
Value Model V2 pipeline complete.
Action-value data: $ACTION_VALUE_DIR/final_patient_action_value_records.jsonl
Value model:       $VALUE_MODEL_DIR/final_patient_rfv_value_model_numpy.npz
Predictions:       $VALUE_SCORE_DIR/final_patient_value_model_predictions.jsonl
GRPO groups:       $VALUE_GROUP_DIR/final_patient_candidate_grpo_groups.jsonl
EOF
