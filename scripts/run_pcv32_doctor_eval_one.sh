#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
ENV_PATH="${AR_GRPO_ENV:-}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"

MODEL_KEY="${1:?model key required}"
OUT="${2:?output dir required}"
MAX_TURNS="${3:-24}"

REPLAY_CLEAR_OUTPUT="${REPLAY_CLEAR_OUTPUT:-1}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"
CLOSED_MODEL="${CLOSED_MODEL:-gpt-4o-mini}"
CLOSED_ENV_FILE="${CLOSED_ENV_FILE:-}"
PATIENT_REALIZER_MODE="${PATIENT_REALIZER_MODE:-rule}"
PATIENT_REALIZER_CACHE_PATH="${PATIENT_REALIZER_CACHE_PATH:-}"
PATIENT_REALIZER_CACHE_POLICY="${PATIENT_REALIZER_CACHE_POLICY:-fallback}"

export ACTIVE_REASONING_PROJECT="$PROJECT"
export HF_HOME="$PROJECT/cache/hf_home"
export HF_MODULES_CACHE="$PROJECT/cache/hf_modules"
export TRANSFORMERS_CACHE="$PROJECT/cache/transformers"
export MODELSCOPE_CACHE="$PROJECT/cache/modelscope"
export TOKENIZERS_PARALLELISM=false

if [[ -n "${ENV_PATH:-}" ]]; then source "$ENV_PATH/bin/activate"; fi
cd "$PHASE_DIR"
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

POLICY="reward_centered_v6_patient_v2"
CALLER="qwen"
ADAPTER_PATH=""
NO_ADAPTER=0
MODEL_OUTPUT_FILENAME="${MODEL_KEY}_doctor_outputs.jsonl"
PROVIDER_TAG="pcv32_${MODEL_KEY}"
MODEL_TAG="$MODEL_KEY"

case "$MODEL_KEY" in
  closed_gpt4omini_evidence)
    CALLER="closed"
    POLICY="closed_llm_evidence_aware"
    MODEL_OUTPUT_FILENAME="closed_gpt4omini_evidence_doctor_outputs.jsonl"
    PROVIDER_TAG="closed_${CLOSED_MODEL}_pcv32_closed_evidence"
    MODEL_TAG="${CLOSED_MODEL}-PCV32-evidence-aware"
    if [[ -z "$CLOSED_ENV_FILE" || ! -s "$CLOSED_ENV_FILE" ]]; then
      echo "CLOSED_ENV_FILE must point to a readable env file for closed baseline" >&2
      exit 2
    fi
    ;;
  qwen_base)
    NO_ADAPTER=1
    MODEL_OUTPUT_FILENAME="qwen3_base_pcv32_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_base_pcv32"
    MODEL_TAG="Qwen3-8B-Base-PCV32"
    ;;
  qwen_sft_r16)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_doctor_sft_lora_f32_f41_2ksteps_bf16/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_sft_r16_pcv32_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_sft_r16_pcv32"
    MODEL_TAG="Qwen3-8B-SFT-r16-PCV32"
    ;;
  qwen_grpo_v6_300)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_reward_v6_patient_v2_grpo_a800_300step_fastacc1_v6_20260629_162408/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_grpo_v6_300_pcv32_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_grpo_v6_300_pcv32"
    MODEL_TAG="Qwen3-8B-GRPO-v6-300-PCV32"
    ;;
  qwen_grpo_v6_full1500)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_reward_v6_patient_v2_grpo_a800_full1500_groupsall_cand8_len768_20260630_165932/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_grpo_v6_full1500_pcv32_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_grpo_v6_full1500_pcv32"
    MODEL_TAG="Qwen3-8B-GRPO-v6-Full1500-PCV32"
    ;;
  qwen_valueaug_full1500)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_value_augmented_v6_patient_v2_grpo_a800_full1500_groupsall_cand8_len768_valueaug_patched_20260701_074131/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_valueaug_full1500_pcv32_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_valueaug_full1500_pcv32"
    MODEL_TAG="Qwen3-8B-ValueAug-GRPO-Full1500-PCV32"
    ;;
  qwen_grpo_rfv2_ckpt1600)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_li_ecr_rfv_v2_grpo_a100_li_ecr_rfv_v2_1600step_groupsall_cand4_len640_ga8_20260703_204129/checkpoint-1600"
    MODEL_OUTPUT_FILENAME="qwen3_grpo_rfv2_ckpt1600_pcv32_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_grpo_rfv2_ckpt1600_pcv32"
    MODEL_TAG="Qwen3-8B-GRPO-RFV-v2-ckpt1600-PCV32"
    ;;
  *)
    echo "Unknown MODEL_KEY=$MODEL_KEY" >&2
    exit 2
    ;;
esac

if [[ "$CALLER" == "qwen" && "$NO_ADAPTER" != "1" && ! -d "$ADAPTER_PATH" ]]; then
  echo "Missing adapter path: $ADAPTER_PATH" >&2
  exit 2
fi

if [[ "$REPLAY_CLEAR_OUTPUT" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

MODEL_OUTPUT_PATH="$OUT/$MODEL_OUTPUT_FILENAME"
MAIN_LOG="$OUT/pcv32_${MODEL_KEY}_main.log"

echo "=== PCV3.2 doctor eval start $(date) model=$MODEL_KEY out=$OUT ===" | tee "$MAIN_LOG"
echo "policy=$POLICY caller=$CALLER max_turns=$MAX_TURNS patient_realizer_mode=$PATIENT_REALIZER_MODE" | tee -a "$MAIN_LOG"
if [[ "$PATIENT_REALIZER_MODE" == "verified_cache" && -z "$PATIENT_REALIZER_CACHE_PATH" ]]; then
  echo "PATIENT_REALIZER_CACHE_PATH is required when PATIENT_REALIZER_MODE=verified_cache" >&2
  exit 2
fi

REALIZER_ARGS=(
  --patient-realizer-mode "$PATIENT_REALIZER_MODE"
  --patient-realizer-cache-policy "$PATIENT_REALIZER_CACHE_POLICY"
)
if [[ -n "$PATIENT_REALIZER_CACHE_PATH" ]]; then
  REALIZER_ARGS+=(--patient-realizer-cache-path "$PATIENT_REALIZER_CACHE_PATH")
fi

for ITER in $(seq 1 $((MAX_TURNS + 1))); do
  echo "--- ITER ${ITER} replay $(date) ---" | tee -a "$MAIN_LOG"
  python scripts/run_llm_doctor_online_replay_v1.py \
    --output-dir "$OUT" \
    --group-dir outputs_f32_f41_single_label_stratified_profile_split_v1 \
    --splits test \
    --max-groups 10000 \
    --max-per-slot 999 \
    --max-profiles 108 \
    --max-turns "$MAX_TURNS" \
    --patient-controller-version v3_2 \
    --provider cached \
    --model-output-path "$MODEL_OUTPUT_PATH" \
    --missing-output-policy stop \
    "${REALIZER_ARGS[@]}" \
    --severities mild_low_info moderate_low_info severe_low_info \
    --policies "$POLICY" \
    >"$OUT/replay_iter_${ITER}.log" 2>&1

  PENDING="$OUT/mdd5k_llm_doctor_online_replay_pending_requests.jsonl"
  if [[ -f "$PENDING" ]]; then
    N=$(python - "$PENDING" <<'PY'
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(sum(1 for _ in f))
PY
)
  else
    N=0
  fi
  echo "pending=${N}" | tee -a "$MAIN_LOG"
  if [[ "$N" -eq 0 ]]; then
    break
  fi

  echo "--- ITER ${ITER} generate $(date) ---" | tee -a "$MAIN_LOG"
  if [[ "$CALLER" == "closed" ]]; then
    python scripts/call_closed_llm_for_pending_requests.py \
      --env-file "$CLOSED_ENV_FILE" \
      --pending-path "$PENDING" \
      --output-path "$MODEL_OUTPUT_PATH" \
      --model "$CLOSED_MODEL" \
      --limit 0 \
      --max-output-tokens 96 \
      --temperature 0.0 \
      --sleep-seconds 0.1 \
      >"$OUT/generate_iter_${ITER}.log" 2>&1
  else
    QWEN_ARGS=(
      --input-path "$PENDING"
      --output-path "$MODEL_OUTPUT_PATH"
      --model-path "$PROJECT/cache/qwen3-8b-hf-remote-code"
      --provider-tag "$PROVIDER_TAG"
      --model-tag "$MODEL_TAG"
      --max-new-tokens 96
      --temperature 0.0
      --dtype bf16
      --batch-size "$REPLAY_BATCH_SIZE"
      --flush-every 10
    )
    if [[ "$NO_ADAPTER" == "1" ]]; then
      QWEN_ARGS+=(--no-adapter)
    else
      QWEN_ARGS+=(--adapter-path "$ADAPTER_PATH")
    fi
    CUDA_VISIBLE_DEVICES=0 python scripts/call_qwen3_hf_lora_for_pending_requests.py \
      "${QWEN_ARGS[@]}" \
      >"$OUT/generate_iter_${ITER}.log" 2>&1
  fi
  tail -10 "$OUT/generate_iter_${ITER}.log" | tee -a "$MAIN_LOG" || true
done

echo "=== canonical analysis start $(date) ===" | tee -a "$MAIN_LOG"
ANALYSIS_OUT="$OUT/tree_aligned_canonical_recovery"
python scripts/analyze_tree_aligned_canonical_evidence_recovery.py \
  --records "${MODEL_KEY}=$OUT/mdd5k_llm_doctor_online_replay_records.jsonl" \
  --output-dir "$ANALYSIS_OUT" \
  >"$OUT/canonical_analysis.log" 2>&1

python - "$MODEL_KEY" "$ANALYSIS_OUT" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

model_key = sys.argv[1]
analysis_out = Path(sys.argv[2])
out = Path(sys.argv[3])
summary_path = analysis_out / "tree_aligned_canonical_evidence_recovery_summary.json"
obj = json.load(open(summary_path, encoding="utf-8"))
rows = []
for row in obj.get("results", obj.get("summary", [])):
    if row.get("metric_name") == "keyword_supported_only":
        rows.append({"model": model_key, **row})
(out / "pcv32_keyword_supported_only.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

echo "=== PCV3.2 doctor eval done $(date) model=$MODEL_KEY ===" | tee -a "$MAIN_LOG"
