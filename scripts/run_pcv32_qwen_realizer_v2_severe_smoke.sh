#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
ENV_PATH="${AR_GRPO_ENV:-}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
TAG="${1:-pcv32_qwen_realizer_v2_severe_smoke_20260707}"
MAX_REQUESTS="${2:-240}"

TRAJ="$PHASE_DIR/outputs_qwen3_li_ecr_rfv_v2_ckpt1600_online_replay_test108_turn24_pcv32_li_ecr_rfv_v2_ckpt1600_pcv32_test108_turn24_from_pcv31_20260707/mdd5k_llm_doctor_online_replay_records.jsonl"
OUT_DIR="$PHASE_DIR/outputs_${TAG}"
REQ="$OUT_DIR/mdd5k_llm_patient_realizer_requests.jsonl"
GEN="$OUT_DIR/qwen3_patient_realizer_outputs.jsonl"
LOG="$OUT_DIR/run.log"

mkdir -p "$OUT_DIR"
if [[ -n "${ENV_PATH:-}" ]]; then source "$ENV_PATH/bin/activate"; fi
cd "$PHASE_DIR"
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export ACTIVE_REASONING_PROJECT="$PROJECT"
export HF_HOME="$PROJECT/cache/hf_home"
export HF_MODULES_CACHE="$PROJECT/cache/hf_modules"
export TRANSFORMERS_CACHE="$PROJECT/cache/transformers"
export TOKENIZERS_PARALLELISM=false

{
  echo "=== v2 severe smoke start $(date) tag=$TAG max_requests=$MAX_REQUESTS ==="
  python scripts/prepare_llm_patient_realizer_requests_v2.py \
    --trajectory-path "$TRAJ" \
    --output-dir "$OUT_DIR" \
    --policies reward_centered_v6_patient_v2 \
    --severities severe_low_info \
    --max-requests "$MAX_REQUESTS" \
    --max-requests-per-cell 12 \
    --sample-seed 42

  echo "=== wait for other qwen generation jobs $(date) ==="
  while pgrep -af 'call_qwen3_hf_lora_for_pending_requests.py|call_qwen3_hf_for_patient_realizer.py' \
    | grep -v "$$" \
    | grep -v "grep" >/dev/null; do
    pgrep -af 'call_qwen3_hf_lora_for_pending_requests.py|call_qwen3_hf_for_patient_realizer.py' \
      | grep -v "$$" \
      | grep -v "grep" || true
    sleep 60
  done

  echo "=== generate qwen realizer $(date) ==="
  CUDA_VISIBLE_DEVICES=0 python scripts/call_qwen3_hf_for_patient_realizer.py \
    --input-path "$REQ" \
    --output-path "$GEN" \
    --model-path "$PROJECT/cache/qwen3-8b-hf-remote-code" \
    --provider-tag remote_qwen3_8b_patient_realizer_pcv32_json_contract_v2 \
    --model-tag Qwen3-8B-Patient-Realizer-PCV32-JsonContractV2 \
    --max-new-tokens 160 \
    --temperature 0.0 \
    --dtype bf16 \
    --batch-size 8 \
    --flush-every 10

  echo "=== verify qwen realizer $(date) ==="
  python scripts/verify_llm_patient_realizer_outputs_v1.py \
    --request-path "$REQ" \
    --output-path "$GEN" \
    --report-dir "$OUT_DIR"

  echo "=== build verified cache include_warned $(date) ==="
  python scripts/build_verified_patient_realizer_cache_v1.py \
    --verification-records "$OUT_DIR/mdd5k_patient_realizer_verification_records_llm_outputs.jsonl" \
    --request-path "$REQ" \
    --output-dir "$OUT_DIR" \
    --include-warned

  echo "=== v2 severe smoke done $(date) ==="
} 2>&1 | tee "$LOG"
