#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
RUN_TAG="${1:-pcv32_online_final_patient_doctor_suite_20260709}"
MAX_TURNS="${2:-24}"

MODELS_CSV="${MODELS_CSV:-qwen_base,qwen_sft_r16,qwen_grpo_v6_300,qwen_grpo_v6_full1500,qwen_valueaug_full1500,qwen_grpo_rfv2_ckpt1600}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"
REALIZER_BATCH_SIZE="${REALIZER_BATCH_SIZE:-4}"
MAX_GROUPS="${MAX_GROUPS:-10000}"
MAX_PROFILES="${MAX_PROFILES:-108}"
MAX_PER_SLOT="${MAX_PER_SLOT:-999}"

cd "$PHASE_DIR"
mkdir -p "outputs_${RUN_TAG}" logs
SUITE_DIR="$PHASE_DIR/outputs_${RUN_TAG}"
SUITE_LOG="$SUITE_DIR/suite.log"

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"

echo "=== PCV3.2 online final-patient doctor suite start $(date) tag=$RUN_TAG turns=$MAX_TURNS ===" | tee "$SUITE_LOG"
echo "models=${MODELS[*]}" | tee -a "$SUITE_LOG"
echo "max_groups=$MAX_GROUPS max_profiles=$MAX_PROFILES max_per_slot=$MAX_PER_SLOT" | tee -a "$SUITE_LOG"
echo "replay_batch_size=$REPLAY_BATCH_SIZE realizer_batch_size=$REALIZER_BATCH_SIZE" | tee -a "$SUITE_LOG"

for MODEL_KEY in "${MODELS[@]}"; do
  MODEL_KEY="$(echo "$MODEL_KEY" | xargs)"
  [[ -z "$MODEL_KEY" ]] && continue
  OUT="$PHASE_DIR/outputs_${RUN_TAG}_${MODEL_KEY}"
  echo "=== run $MODEL_KEY $(date) out=$OUT ===" | tee -a "$SUITE_LOG"
  MAX_GROUPS="$MAX_GROUPS" \
  MAX_PROFILES="$MAX_PROFILES" \
  MAX_PER_SLOT="$MAX_PER_SLOT" \
  REPLAY_BATCH_SIZE="$REPLAY_BATCH_SIZE" \
  REALIZER_BATCH_SIZE="$REALIZER_BATCH_SIZE" \
    bash scripts/run_pcv32_online_final_patient_doctor_eval_one_20260709.sh "$MODEL_KEY" "$OUT" "$MAX_TURNS" \
    >"$SUITE_DIR/${MODEL_KEY}.log" 2>&1
  echo "=== done $MODEL_KEY $(date) ===" | tee -a "$SUITE_LOG"
done

python3 - "$RUN_TAG" "$SUITE_DIR" "${MODELS[@]}" <<'PY'
import json
import sys
from pathlib import Path

tag = sys.argv[1]
suite_dir = Path(sys.argv[2])
models = [item.strip() for item in sys.argv[3:] if item.strip()]
phase_dir = suite_dir.parent

def score_from_row(row):
    for key in ("mean_tree_aligned_canonical_final_s", "mean_score", "score"):
        if row.get(key) is not None:
            return float(row.get(key) or 0.0)
    return 0.0

summary = []
for model in models:
    out = phase_dir / f"outputs_{tag}_{model}"
    p = out / "pcv32_keyword_supported_only.json"
    records_path = out / "mdd5k_llm_doctor_online_replay_records.jsonl"
    entry = {
        "model": model,
        "output_dir": str(out),
        "keyword_summary_path": str(p),
        "records_path": str(records_path),
        "exists": p.exists(),
    }
    if p.exists():
        rows = json.load(open(p, encoding="utf-8"))
        entry["rows"] = rows
        by_severity = {}
        for row in rows:
            severity = row.get("base_severity")
            if severity:
                by_severity[severity] = round(score_from_row(row), 6)
        entry["mild"] = by_severity.get("mild_low_info")
        entry["moderate"] = by_severity.get("moderate_low_info")
        entry["severe"] = by_severity.get("severe_low_info")
        values = [value for value in (entry["mild"], entry["moderate"], entry["severe"]) if value is not None]
        entry["mean"] = round(sum(values) / len(values), 6) if values else None
    if records_path.exists():
        counts = {}
        n = 0
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                row = json.loads(line)
                key = str(row.get("patient_realizer_mode"))
                counts[key] = counts.get(key, 0) + 1
        entry["records"] = n
        entry["patient_realizer_mode_counts"] = counts
        entry["verified_only"] = bool(counts) and set(counts) == {"verified_llm_cache"}
    summary.append(entry)

(suite_dir / "suite_keyword_supported_only_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = ["# PCV3.2 Online Final Patient Doctor Suite", "", "| model | mild | moderate | severe | mean | verified_only | records |", "|---|---:|---:|---:|---:|---|---:|"]
for entry in summary:
    lines.append(
        "| {model} | {mild} | {moderate} | {severe} | {mean} | {verified_only} | {records} |".format(
            model=entry["model"],
            mild="" if entry.get("mild") is None else f"{entry['mild']:.6f}",
            moderate="" if entry.get("moderate") is None else f"{entry['moderate']:.6f}",
            severe="" if entry.get("severe") is None else f"{entry['severe']:.6f}",
            mean="" if entry.get("mean") is None else f"{entry['mean']:.6f}",
            verified_only=entry.get("verified_only"),
            records=entry.get("records", ""),
        )
    )
(suite_dir / "SUITE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== PCV3.2 online final-patient doctor suite done $(date) ===" | tee -a "$SUITE_LOG"
