#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
ENV_PATH="${AR_GRPO_ENV:-}"
PY="${AR_GRPO_PYTHON:-python}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
MODEL_PATH="${MODEL_PATH:-$PROJECT/cache/qwen3-8b-hf-remote-code}"

MODEL_KEY="${1:?model key required}"
OUT="${2:?output dir required}"
MAX_TURNS="${3:-24}"

MAX_GROUPS="${MAX_GROUPS:-10000}"
MAX_PROFILES="${MAX_PROFILES:-108}"
MAX_PER_SLOT="${MAX_PER_SLOT:-999}"
EVAL_SPLITS="${EVAL_SPLITS:-test}"
SEVERITIES="${SEVERITIES:-mild_low_info moderate_low_info severe_low_info}"
RANDOM_LOW_DISCLOSURE_PROB="${RANDOM_LOW_DISCLOSURE_PROB:-0.5}"
RANDOM_DISCLOSURE_SEED="${RANDOM_DISCLOSURE_SEED:-0}"
GROUP_DIR="${GROUP_DIR:-}"
DATASET_PREFIX="${DATASET_PREFIX:-mdd5k}"
LANGUAGE="${LANGUAGE:-}"
PROFILE_PATH="${PROFILE_PATH:-}"
SCHEMA_PATH="${SCHEMA_PATH:-}"
CANONICAL_PREFIX="${CANONICAL_PREFIX:-$DATASET_PREFIX}"
CANONICAL_DIR="${CANONICAL_DIR:-}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"
REALIZER_BATCH_SIZE="${REALIZER_BATCH_SIZE:-4}"
CLOSED_MODEL="${CLOSED_MODEL:-gpt-4.1-mini}"
CLOSED_PROVIDER="${CLOSED_PROVIDER:-openai_compatible}"
CLOSED_ENV_FILE="${CLOSED_ENV_FILE:-}"

export ACTIVE_REASONING_PROJECT="$PROJECT"
export HF_HOME="$PROJECT/cache/hf_home"
export HF_MODULES_CACHE="$PROJECT/cache/hf_modules"
export TRANSFORMERS_CACHE="$PROJECT/cache/transformers"
export MODELSCOPE_CACHE="$PROJECT/cache/modelscope"
export TOKENIZERS_PARALLELISM=false

if [[ -n "${ENV_PATH:-}" ]]; then source "$ENV_PATH/bin/activate"; fi
cd "$PHASE_DIR"
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

if [[ -z "$GROUP_DIR" ]]; then
  if [[ -d "$PHASE_DIR/data/${DATASET_PREFIX}/profile_split" ]]; then
    GROUP_DIR="data/${DATASET_PREFIX}/profile_split"
  elif [[ -d "$PHASE_DIR/data/f32_f41_profile_split" ]]; then
    GROUP_DIR="data/f32_f41_profile_split"
  else
    GROUP_DIR="outputs_f32_f41_single_label_stratified_profile_split_v1"
  fi
fi
if [[ -z "$PROFILE_PATH" ]]; then
  if [[ "$DATASET_PREFIX" == "daic" && -f "$PHASE_DIR/data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl" ]]; then
    PROFILE_PATH="data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl"
  else
    PROFILE_PATH="data/patient_profiles/mdd5k_dialogue_derived_patient_profiles.jsonl"
  fi
fi
if [[ -z "$SCHEMA_PATH" ]]; then
  if [[ "$DATASET_PREFIX" == "daic" && -f "$PHASE_DIR/schemas/daic_symptom_slot_schema.json" ]]; then
    SCHEMA_PATH="schemas/daic_symptom_slot_schema.json"
  else
    SCHEMA_PATH="schemas/mdd5k_symptom_slot_schema.json"
  fi
fi
if [[ -z "$LANGUAGE" ]]; then
  if [[ "$DATASET_PREFIX" == "daic" ]]; then
    LANGUAGE="en"
  else
    LANGUAGE="zh"
  fi
fi
if [[ -z "$CANONICAL_DIR" ]]; then
  if [[ -d "$PHASE_DIR/data/${DATASET_PREFIX}/canonical_evidence" ]]; then
    CANONICAL_DIR="data/${DATASET_PREFIX}/canonical_evidence"
  elif [[ "$DATASET_PREFIX" == "mdd5k" && -d "$PHASE_DIR/data/tree_aligned_canonical_evidence" ]]; then
    CANONICAL_DIR="data/tree_aligned_canonical_evidence"
  fi
fi

CALLER="qwen"
POLICY="reward_centered_v6_patient_v2"
ADAPTER_PATH=""
NO_ADAPTER=0
MODEL_OUTPUT_FILENAME="${MODEL_KEY}_doctor_outputs.jsonl"
PROVIDER_TAG="pcv32_online_${MODEL_KEY}"
MODEL_TAG="$MODEL_KEY"

case "$MODEL_KEY" in
  closed_evidence|closed_gpt41mini_evidence)
    CALLER="closed"
    POLICY="closed_llm_evidence_aware"
    MODEL_OUTPUT_FILENAME="closed_${CLOSED_MODEL}_evidence_doctor_outputs.jsonl"
    PROVIDER_TAG="closed_${CLOSED_MODEL}_pcv32_online_evidence"
    MODEL_TAG="${CLOSED_MODEL}-PCV32-online-evidence-aware"
    if [[ -z "$CLOSED_ENV_FILE" || ! -s "$CLOSED_ENV_FILE" ]]; then
      echo "CLOSED_ENV_FILE must point to a readable env file for closed baseline" >&2
      exit 2
    fi
    ;;
  qwen_base)
    NO_ADAPTER=1
    MODEL_OUTPUT_FILENAME="qwen3_base_pcv32_online_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_base_pcv32_online"
    MODEL_TAG="Qwen3-8B-Base-PCV32-OnlineFinalPatient"
    ;;
  qwen_sft_r16)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_doctor_sft_lora_f32_f41_2ksteps_bf16/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_sft_r16_pcv32_online_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_sft_r16_pcv32_online"
    MODEL_TAG="Qwen3-8B-SFT-r16-PCV32-OnlineFinalPatient"
    ;;
  qwen_grpo_v6_300)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_reward_v6_patient_v2_grpo_a800_300step_fastacc1_v6_20260629_162408/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_grpo_v6_300_pcv32_online_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_grpo_v6_300_pcv32_online"
    MODEL_TAG="Qwen3-8B-GRPO-v6-300-PCV32-OnlineFinalPatient"
    ;;
  qwen_grpo_v6_full1500)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_reward_v6_patient_v2_grpo_a800_full1500_groupsall_cand8_len768_20260630_165932/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_grpo_v6_full1500_pcv32_online_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_grpo_v6_full1500_pcv32_online"
    MODEL_TAG="Qwen3-8B-GRPO-v6-Full1500-PCV32-OnlineFinalPatient"
    ;;
  qwen_valueaug_full1500)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_value_augmented_v6_patient_v2_grpo_a800_full1500_groupsall_cand8_len768_valueaug_patched_20260701_074131/final_lora_adapter"
    MODEL_OUTPUT_FILENAME="qwen3_valueaug_full1500_pcv32_online_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_valueaug_full1500_pcv32_online"
    MODEL_TAG="Qwen3-8B-ValueAug-GRPO-Full1500-PCV32-OnlineFinalPatient"
    ;;
  qwen_grpo_rfv2_ckpt1600)
    ADAPTER_PATH="$PHASE_DIR/outputs_qwen3_li_ecr_rfv_v2_grpo_a100_li_ecr_rfv_v2_1600step_groupsall_cand4_len640_ga8_20260703_204129/checkpoint-1600"
    MODEL_OUTPUT_FILENAME="qwen3_grpo_rfv2_ckpt1600_pcv32_online_doctor_outputs.jsonl"
    PROVIDER_TAG="remote_qwen3_8b_grpo_rfv2_ckpt1600_pcv32_online"
    MODEL_TAG="Qwen3-8B-GRPO-RFV-v2-ckpt1600-PCV32-OnlineFinalPatient"
    ;;
  qwen_lora_custom|custom_qwen_lora)
    ADAPTER_PATH="${CUSTOM_ADAPTER_PATH:-}"
    CUSTOM_MODEL_ID="${CUSTOM_MODEL_ID:-qwen_lora_custom}"
    MODEL_OUTPUT_FILENAME="${CUSTOM_MODEL_OUTPUT_FILENAME:-${CUSTOM_MODEL_ID}_pcv32_online_doctor_outputs.jsonl}"
    PROVIDER_TAG="${CUSTOM_PROVIDER_TAG:-remote_qwen3_8b_${CUSTOM_MODEL_ID}_pcv32_online}"
    MODEL_TAG="${CUSTOM_MODEL_TAG:-Qwen3-8B-${CUSTOM_MODEL_ID}-PCV32-OnlineFinalPatient}"
    if [[ -z "$ADAPTER_PATH" ]]; then
      echo "CUSTOM_ADAPTER_PATH is required for MODEL_KEY=$MODEL_KEY" >&2
      exit 2
    fi
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

rm -rf "$OUT"
mkdir -p "$OUT" "$OUT/online_patient_work" logs
MAIN_LOG="$OUT/pcv32_online_${MODEL_KEY}_main.log"
MODEL_OUTPUT_PATH="$OUT/$MODEL_OUTPUT_FILENAME"
GLOBAL_CACHE="$OUT/online_patient_work/current_verified_patient_cache.jsonl"
GLOBAL_CACHE_SUMMARY="$OUT/online_patient_work/current_verified_patient_cache_summary.json"
: > "$GLOBAL_CACHE"

echo "=== PCV3.2 online final-patient doctor eval start $(date) model=$MODEL_KEY out=$OUT ===" | tee "$MAIN_LOG"
echo "max_turns=$MAX_TURNS max_groups=$MAX_GROUPS max_profiles=$MAX_PROFILES max_per_slot=$MAX_PER_SLOT eval_splits=$EVAL_SPLITS group_dir=$GROUP_DIR" | tee -a "$MAIN_LOG"
echo "dataset_prefix=$DATASET_PREFIX language=$LANGUAGE profile_path=$PROFILE_PATH schema_path=$SCHEMA_PATH canonical_dir=$CANONICAL_DIR canonical_prefix=$CANONICAL_PREFIX" | tee -a "$MAIN_LOG"
echo "severities=$SEVERITIES random_low_disclosure_prob=$RANDOM_LOW_DISCLOSURE_PROB random_disclosure_seed=$RANDOM_DISCLOSURE_SEED" | tee -a "$MAIN_LOG"

run_replay() {
  local out_dir="$1"
  local mode="$2"
  local cache_policy="$3"
  local cache_path="$4"
  local missing_policy="$5"
  local provider="$6"
  local extra_args=()
  if [[ "$mode" == "verified_cache" ]]; then
    extra_args+=(--patient-realizer-cache-path "$cache_path" --patient-realizer-cache-policy "$cache_policy")
  fi
  "$PY" scripts/run_llm_doctor_online_replay.py \
    --output-dir "$out_dir" \
    --profiles "$PROFILE_PATH" \
    --schema "$SCHEMA_PATH" \
    --group-dir "$GROUP_DIR" \
    --dataset-prefix "$DATASET_PREFIX" \
    --language "$LANGUAGE" \
    --splits $EVAL_SPLITS \
    --max-groups "$MAX_GROUPS" \
    --max-per-slot "$MAX_PER_SLOT" \
    --max-profiles "$MAX_PROFILES" \
    --max-turns "$MAX_TURNS" \
    --random-low-disclosure-prob "$RANDOM_LOW_DISCLOSURE_PROB" \
    --random-disclosure-seed "$RANDOM_DISCLOSURE_SEED" \
    --patient-controller-version v3_2 \
    --provider "$provider" \
    --model-output-path "$MODEL_OUTPUT_PATH" \
    --missing-output-policy "$missing_policy" \
    --patient-realizer-mode "$mode" \
    "${extra_args[@]}" \
    --severities $SEVERITIES \
    --policies "$POLICY"
}

pending_count() {
  local pending="$1"
  if [[ ! -f "$pending" ]]; then
    echo 0
    return
  fi
  "$PY" - "$pending" <<'PY'
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(sum(1 for line in f if line.strip()))
PY
}

generate_doctor_outputs() {
  local pending="$1"
  local iter="$2"
  if [[ "$CALLER" == "closed" ]]; then
    "$PY" scripts/call_closed_llm_for_pending_requests.py \
      --env-file "$CLOSED_ENV_FILE" \
      --provider "$CLOSED_PROVIDER" \
      --pending-path "$pending" \
      --output-path "$MODEL_OUTPUT_PATH" \
      --model "$CLOSED_MODEL" \
      --limit 0 \
      --max-output-tokens 96 \
      --temperature 0.0 \
      --sleep-seconds 0.1 \
      >"$OUT/generate_doctor_iter_${iter}.log" 2>&1
  else
    local qwen_args=(
      --input-path "$pending"
      --output-path "$MODEL_OUTPUT_PATH"
      --model-path "$MODEL_PATH"
      --provider-tag "$PROVIDER_TAG"
      --model-tag "$MODEL_TAG"
      --max-new-tokens 96
      --temperature 0.0
      --dtype bf16
      --batch-size "$REPLAY_BATCH_SIZE"
      --flush-every 10
    )
    if [[ "$NO_ADAPTER" == "1" ]]; then
      qwen_args+=(--no-adapter)
    else
      qwen_args+=(--adapter-path "$ADAPTER_PATH")
    fi
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_lora_for_pending_requests.py \
      "${qwen_args[@]}" \
      >"$OUT/generate_doctor_iter_${iter}.log" 2>&1
  fi
  tail -10 "$OUT/generate_doctor_iter_${iter}.log" | tee -a "$MAIN_LOG" || true
}

generate_patient_cache_for_probe() {
  local probe_records="$1"
  local iter="$2"
  local work="$OUT/online_patient_work/iter_${iter}"
  mkdir -p "$work"
  local all_req_dir="$work/01_all_requests"
  local req_path="$all_req_dir/${DATASET_PREFIX}_llm_patient_realizer_requests.jsonl"
  local missing_req="$work/02_missing_requests/${DATASET_PREFIX}_llm_patient_realizer_requests.jsonl"
  local missing_summary="$work/02_missing_requests/filter_summary.json"

  "$PY" scripts/prepare_patient_realizer_requests.py \
    --trajectory-path "$probe_records" \
    --output-dir "$all_req_dir" \
    --dataset-prefix "$DATASET_PREFIX" \
    --language "$LANGUAGE" \
    --max-requests 0 \
    --max-requests-per-cell 0 \
    --sample-seed 909

  "$PY" scripts/filter_patient_realizer_requests_by_cache.py \
    --request-path "$req_path" \
    --cache-path "$GLOBAL_CACHE" \
    --output-path "$missing_req" \
    --summary-path "$missing_summary"

  local missing_n
  missing_n=$(pending_count "$missing_req")
  echo "patient_realizer_missing_requests=${missing_n}" | tee -a "$MAIN_LOG"
  if [[ "$missing_n" -eq 0 ]]; then
    return
  fi

  local primary_out_dir="$work/03_primary_qwen_outputs"
  local primary_out="$primary_out_dir/qwen3_patient_realizer_outputs.jsonl"
  mkdir -p "$primary_out_dir"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
    --input-path "$missing_req" \
    --output-path "$primary_out" \
    --model-path "$MODEL_PATH" \
    --no-adapter \
    --provider-tag remote_qwen3_8b_pcv32_online_patient_primary \
    --model-tag Qwen3-8B-Patient-Realizer-PCV32-Online \
    --limit 0 \
    --max-new-tokens 220 \
    --temperature 0.0 \
    --dtype bf16 \
    --batch-size "$REALIZER_BATCH_SIZE" \
    --flush-every 8 \
    >"$work/03_primary_qwen_outputs.log" 2>&1

  local primary_verify_dir="$work/04_primary_verify"
  "$PY" scripts/verify_patient_realizer_outputs.py \
    --request-path "$missing_req" \
    --output-path "$primary_out" \
    --report-dir "$primary_verify_dir" \
    --dataset-prefix "$DATASET_PREFIX" \
    --leak-threshold 0.72 \
    --allowed-threshold 0.45 \
    --reference-min-coverage 0.30 \
    --severe-max-coverage 0.45

  local primary_verify="$primary_verify_dir/${DATASET_PREFIX}_patient_realizer_verification_records_llm_outputs.jsonl"
  local repair_req_files=()
  local repair_verify_files=()
  local source_req="$missing_req"
  local source_verify="$primary_verify"

  for repair_round in 1 2; do
    local repair_req_dir="$work/05_repair${repair_round}_requests"
    "$PY" scripts/prepare_patient_realizer_repair_requests.py \
      --request-path "$source_req" \
      --verification-records "$source_verify" \
      --output-dir "$repair_req_dir" \
      --dataset-prefix "$DATASET_PREFIX"
    local repair_req="$repair_req_dir/${DATASET_PREFIX}_llm_patient_realizer_repair_requests.jsonl"
    local repair_n
    repair_n=$(pending_count "$repair_req")
    echo "repair_round_${repair_round}_requests=${repair_n}" | tee -a "$MAIN_LOG"
    if [[ "$repair_n" -eq 0 ]]; then
      break
    fi
    local repair_out_dir="$work/06_repair${repair_round}_qwen_outputs"
    local repair_out="$repair_out_dir/qwen3_patient_realizer_repair_outputs.jsonl"
    mkdir -p "$repair_out_dir"
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/call_qwen3_hf_for_patient_realizer.py \
      --input-path "$repair_req" \
      --output-path "$repair_out" \
      --model-path "$MODEL_PATH" \
      --no-adapter \
      --provider-tag "remote_qwen3_8b_pcv32_online_patient_repair${repair_round}" \
      --model-tag "Qwen3-8B-Patient-Realizer-PCV32-Online-Repair${repair_round}" \
      --limit 0 \
      --max-new-tokens 180 \
      --temperature 0.0 \
      --dtype bf16 \
      --batch-size "$REALIZER_BATCH_SIZE" \
      --flush-every 8 \
      >"$work/06_repair${repair_round}_qwen_outputs.log" 2>&1
    local repair_verify_dir="$work/07_repair${repair_round}_verify"
    "$PY" scripts/verify_patient_realizer_outputs.py \
      --request-path "$repair_req" \
      --output-path "$repair_out" \
      --report-dir "$repair_verify_dir" \
      --dataset-prefix "$DATASET_PREFIX" \
      --leak-threshold 0.72 \
      --allowed-threshold 0.45 \
      --reference-min-coverage 0.30 \
      --severe-max-coverage 0.45
    local repair_verify="$repair_verify_dir/${DATASET_PREFIX}_patient_realizer_verification_records_llm_outputs.jsonl"
    repair_req_files+=("$repair_req")
    repair_verify_files+=("$repair_verify")
    source_req="$repair_req"
    source_verify="$repair_verify"
  done

  local build_dir="$work/08_new_cache"
  local repair_args=()
  if [[ "${#repair_req_files[@]}" -gt 0 ]]; then
    local merged_repair_req="$work/merged_repair_requests.jsonl"
    local merged_repair_verify="$work/merged_repair_verify.jsonl"
    "$PY" - "$merged_repair_req" "${repair_req_files[@]}" <<'PY'
import sys
out = sys.argv[1]
with open(out, "w", encoding="utf-8", newline="\n") as w:
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    w.write(line)
PY
    "$PY" - "$merged_repair_verify" "${repair_verify_files[@]}" <<'PY'
import sys
out = sys.argv[1]
with open(out, "w", encoding="utf-8", newline="\n") as w:
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    w.write(line)
PY
    repair_args=(--repair-request-path "$merged_repair_req" --repair-verification-records "$merged_repair_verify")
  fi

  "$PY" scripts/build_verified_patient_realizer_cache.py \
    --primary-request-path "$missing_req" \
    --primary-verification-records "$primary_verify" \
    "${repair_args[@]}" \
    --output-dir "$build_dir" \
    --dataset-prefix "$DATASET_PREFIX" \
    --include-warned

  local new_cache="$build_dir/${DATASET_PREFIX}_verified_patient_response_cache_repair_include_warned.jsonl"
  local new_cache_n
  new_cache_n=$(pending_count "$new_cache")
  if [[ "$new_cache_n" -lt "$missing_n" ]]; then
    echo "ONLINE_PATIENT_FAIL insufficient new verified cache: new_cache=$new_cache_n missing=$missing_n" >&2
    exit 30
  fi

  "$PY" scripts/merge_patient_realizer_caches.py \
    --cache-path "$GLOBAL_CACHE" \
    --cache-path "$new_cache" \
    --output-path "$GLOBAL_CACHE.tmp" \
    --summary-path "$GLOBAL_CACHE_SUMMARY" \
    --prefer-later
  mv "$GLOBAL_CACHE.tmp" "$GLOBAL_CACHE"
}

for ITER in $(seq 1 $((MAX_TURNS + 1))); do
  echo "--- ITER ${ITER} actual replay before doctor generation $(date) ---" | tee -a "$MAIN_LOG"
  CACHE_N=$(pending_count "$GLOBAL_CACHE")
  if [[ "$CACHE_N" -gt 0 ]]; then
    run_replay "$OUT" verified_cache error "$GLOBAL_CACHE" stop cached >"$OUT/replay_actual_before_iter_${ITER}.log" 2>&1
  else
    run_replay "$OUT" rule fallback "$GLOBAL_CACHE" stop cached >"$OUT/replay_actual_before_iter_${ITER}.log" 2>&1
  fi

  PENDING="$OUT/${DATASET_PREFIX}_llm_doctor_online_replay_pending_requests.jsonl"
  N=$(pending_count "$PENDING")
  echo "doctor_pending=${N}" | tee -a "$MAIN_LOG"
  if [[ "$N" -eq 0 ]]; then
    break
  fi

  echo "--- ITER ${ITER} generate doctor outputs $(date) ---" | tee -a "$MAIN_LOG"
  generate_doctor_outputs "$PENDING" "$ITER"

  echo "--- ITER ${ITER} probe rule trajectory for patient realization $(date) ---" | tee -a "$MAIN_LOG"
  PROBE_DIR="$OUT/online_patient_work/probe_iter_${ITER}"
  rm -rf "$PROBE_DIR"
  run_replay "$PROBE_DIR" rule fallback "$GLOBAL_CACHE" stop cached >"$OUT/probe_iter_${ITER}.log" 2>&1
  PROBE_RECORDS="$PROBE_DIR/${DATASET_PREFIX}_llm_doctor_online_replay_records.jsonl"
  if [[ ! -s "$PROBE_RECORDS" ]]; then
    echo "Missing probe records at $PROBE_RECORDS" >&2
    exit 31
  fi
  generate_patient_cache_for_probe "$PROBE_RECORDS" "$ITER"

  echo "--- ITER ${ITER} actual replay with verified patient cache $(date) ---" | tee -a "$MAIN_LOG"
  run_replay "$OUT" verified_cache error "$GLOBAL_CACHE" stop cached >"$OUT/replay_actual_after_iter_${ITER}.log" 2>&1
done

echo "=== canonical analysis start $(date) ===" | tee -a "$MAIN_LOG"
ANALYSIS_OUT="$OUT/tree_aligned_canonical_recovery"
ANALYSIS_ARGS=(
  --records "${MODEL_KEY}=$OUT/${DATASET_PREFIX}_llm_doctor_online_replay_records.jsonl"
  --output-dir "$ANALYSIS_OUT"
  --canonical-prefix "$CANONICAL_PREFIX"
)
if [[ -n "$CANONICAL_DIR" ]]; then
  ANALYSIS_ARGS+=(--canonical-dir "$CANONICAL_DIR")
fi
"$PY" scripts/analyze_tree_aligned_canonical_evidence_recovery.py \
  "${ANALYSIS_ARGS[@]}" \
  >"$OUT/canonical_analysis.log" 2>&1

"$PY" - "$MODEL_KEY" "$ANALYSIS_OUT" "$OUT" <<'PY'
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

"$PY" - "$OUT/${DATASET_PREFIX}_llm_doctor_online_replay_records.jsonl" <<'PY'
import collections
import json
import sys

path = sys.argv[1]
counts = collections.Counter()
records = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        records += 1
        counts[str(row.get("patient_realizer_mode"))] += 1
print(json.dumps({"records": records, "patient_realizer_mode_counts": dict(counts)}, ensure_ascii=False, indent=2))
if records and any(key != "verified_llm_cache" for key in counts):
    raise SystemExit("ONLINE_PATIENT_FAIL non-verified patient response present")
PY

echo "=== PCV3.2 online final-patient doctor eval done $(date) model=$MODEL_KEY ===" | tee -a "$MAIN_LOG"
