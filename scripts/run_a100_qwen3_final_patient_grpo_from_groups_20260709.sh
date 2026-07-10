#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
ENV_PATH="${AR_GRPO_ENV:-}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GROUP_DATA_JSONL FINAL_PATIENT_SFT_ADAPTER [RUN_TAG] [MAX_STEPS] [MAX_GROUPS]" >&2
  exit 2
fi

GROUP_DATA="$1"
SFT_ADAPTER_PATH="$2"
RUN_TAG="${3:-final_patient_grpo_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${4:-800}"
MAX_GROUPS="${5:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PHASE_DIR/outputs_qwen3_final_patient_grpo_${RUN_TAG}}"
MAX_CANDIDATES="${MAX_CANDIDATES:-8}"
MAX_LENGTH="${MAX_LENGTH:-768}"
EVAL_GROUPS="${EVAL_GROUPS:-256}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-6e-6}"
KL_COEF="${KL_COEF:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-0}"
SAVE_MILESTONES="${SAVE_MILESTONES:-200,400,800,1200,1600}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

if [[ ! -f "$GROUP_DATA" ]]; then
  echo "Missing group data: $GROUP_DATA" >&2
  exit 2
fi
if [[ ! -f "$SFT_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "Missing SFT adapter_config.json under: $SFT_ADAPTER_PATH" >&2
  exit 2
fi

export ACTIVE_REASONING_PROJECT="$PROJECT"
export HF_HOME="$PROJECT/cache/hf_home"
export HF_MODULES_CACHE="$PROJECT/cache/hf_modules"
export TRANSFORMERS_CACHE="$PROJECT/cache/transformers"
export MODELSCOPE_CACHE="$PROJECT/cache/modelscope"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT/.cache/pip}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TRANSFORMERS_CACHE" "$MODELSCOPE_CACHE" "$PHASE_DIR/logs"

if [[ -n "${ENV_PATH:-}" ]]; then source "$ENV_PATH/bin/activate"; fi
cd "$PHASE_DIR"

python scripts/train_qwen3_grpo_from_v6_groups.py \
  --model-path "$PROJECT/cache/qwen3-8b-hf-remote-code" \
  --sft-adapter-path "$SFT_ADAPTER_PATH" \
  --group-data "$GROUP_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --max-groups "$MAX_GROUPS" \
  --eval-groups "$EVAL_GROUPS" \
  --max-candidates "$MAX_CANDIDATES" \
  --max-length "$MAX_LENGTH" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "$GRAD_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --eval-steps 50 \
  --save-steps "$SAVE_STEPS" \
  --save-milestones "$SAVE_MILESTONES" \
  --advantage-mode zscore \
  --reward-clip 3.0 \
  --kl-coef "$KL_COEF" \
  --length-normalize \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --bf16 \
  --gradient-checkpointing
