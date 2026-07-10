#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
ENV_PATH="${AR_GRPO_ENV:-}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TRAIN_JSONL DEV_JSONL [RUN_TAG] [MAX_STEPS]" >&2
  exit 2
fi

TRAIN_PATH="$1"
DEV_PATH="$2"
RUN_TAG="${3:-final_patient_sft_r16_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${4:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PHASE_DIR/outputs_qwen3_final_patient_doctor_sft_lora_${RUN_TAG}}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

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

python scripts/train_qwen3_doctor_sft_lora.py \
  --model-path "$PROJECT/cache/qwen3-8b-hf-remote-code" \
  --train-path "$TRAIN_PATH" \
  --dev-path "$DEV_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "$MAX_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "$GRAD_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 200 \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --bf16 \
  --gradient-checkpointing \
  --report-to none
