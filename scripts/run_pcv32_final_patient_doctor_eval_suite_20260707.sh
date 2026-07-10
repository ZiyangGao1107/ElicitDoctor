#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
RUN_TAG="${1:-pcv32_final_patient_test108_turn24_doctor_suite_20260707}"
MAX_TURNS="${2:-24}"

REALIZER_CACHE_PATH="${REALIZER_CACHE_PATH:-$PHASE_DIR/outputs_llm_patient_realizer_v3_2_li_ecr_rfv_v2_ckpt1600_pcv32_test108_turn24_from_pcv31_20260707/qwen3_verified_cache_pcv32_qwenrealizer_full_20260707_0222/mdd5k_verified_patient_response_cache_include_warned.jsonl}"
REALIZER_CACHE_POLICY="${REALIZER_CACHE_POLICY:-fallback}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"

cd "$PHASE_DIR"

SUITE_DIR="$PHASE_DIR/outputs_${RUN_TAG}"
mkdir -p "$SUITE_DIR"
SUITE_LOG="$SUITE_DIR/suite.log"

echo "=== PCV3.2 final patient doctor suite start $(date) tag=$RUN_TAG turns=$MAX_TURNS ===" | tee "$SUITE_LOG"
echo "realizer_cache=$REALIZER_CACHE_PATH" | tee -a "$SUITE_LOG"
echo "realizer_cache_policy=$REALIZER_CACHE_POLICY" | tee -a "$SUITE_LOG"

if [[ ! -s "$REALIZER_CACHE_PATH" ]]; then
  echo "Missing realizer cache: $REALIZER_CACHE_PATH" >&2
  exit 2
fi

MODELS=(
  qwen_base
  qwen_sft_r16
  qwen_grpo_v6_300
  qwen_valueaug_full1500
  qwen_grpo_rfv2_ckpt1600
)

for MODEL_KEY in "${MODELS[@]}"; do
  OUT="$PHASE_DIR/outputs_${RUN_TAG}_${MODEL_KEY}"
  echo "=== run $MODEL_KEY $(date) out=$OUT ===" | tee -a "$SUITE_LOG"
  REPLAY_CLEAR_OUTPUT=1 \
  REPLAY_BATCH_SIZE="$REPLAY_BATCH_SIZE" \
  PATIENT_REALIZER_MODE=verified_cache \
  PATIENT_REALIZER_CACHE_PATH="$REALIZER_CACHE_PATH" \
  PATIENT_REALIZER_CACHE_POLICY="$REALIZER_CACHE_POLICY" \
    bash scripts/run_pcv32_doctor_eval_one.sh "$MODEL_KEY" "$OUT" "$MAX_TURNS" \
    >"$SUITE_DIR/${MODEL_KEY}.log" 2>&1
  echo "=== done $MODEL_KEY $(date) ===" | tee -a "$SUITE_LOG"
done

python3 - "$RUN_TAG" "$SUITE_DIR" <<'PY'
import json
import sys
from pathlib import Path

tag = sys.argv[1]
suite_dir = Path(sys.argv[2])
phase_dir = suite_dir.parent
models = [
    "qwen_base",
    "qwen_sft_r16",
    "qwen_grpo_v6_300",
    "qwen_valueaug_full1500",
    "qwen_grpo_rfv2_ckpt1600",
]
summary = []
for model in models:
    out = phase_dir / f"outputs_{tag}_{model}"
    p = out / "pcv32_keyword_supported_only.json"
    entry = {"model": model, "output_dir": str(out), "keyword_summary_path": str(p), "exists": p.exists()}
    if p.exists():
        rows = json.load(open(p, encoding="utf-8"))
        entry["rows"] = rows
        vals = [float(r.get("mean_tree_aligned_canonical_final_s", r.get("mean_score", r.get("score", 0.0))) or 0.0) for r in rows]
        entry["mean"] = round(sum(vals) / len(vals), 6) if vals else None
        entry["by_severity"] = {r.get("base_severity"): r.get("mean_tree_aligned_canonical_final_s") for r in rows}
    summary.append(entry)
(suite_dir / "suite_keyword_supported_only_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== PCV3.2 final patient doctor suite done $(date) ===" | tee -a "$SUITE_LOG"
