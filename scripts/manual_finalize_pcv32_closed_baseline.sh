#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
MODEL_KEY="${1:-closed_gpt4omini_evidence}"
OUT="${2:-outputs_pcv32_test108_turn24_doctor_suite_20260707_0330_closed_gpt4omini_evidence}"

cd "$PHASE_DIR"
export PYTHONPATH="$PHASE_DIR/scripts:${PYTHONPATH:-}"

ANALYSIS_OUT="$OUT/tree_aligned_canonical_recovery"
python3 scripts/analyze_tree_aligned_canonical_evidence_recovery.py \
  --records "$MODEL_KEY=$OUT/mdd5k_llm_doctor_online_replay_records.jsonl" \
  --output-dir "$ANALYSIS_OUT" \
  >"$OUT/canonical_analysis.manual.log" 2>&1

python3 - "$MODEL_KEY" "$ANALYSIS_OUT" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

model_key = sys.argv[1]
analysis_out = Path(sys.argv[2])
out = Path(sys.argv[3])
summary_path = analysis_out / "tree_aligned_canonical_evidence_recovery_summary.json"
data = json.load(open(summary_path, encoding="utf-8"))
rows = []
for row in data.get("results", []):
    if row.get("metric_name") != "keyword_supported_only":
        continue
    rows.append({"model": model_key, **row})
(out / "pcv32_keyword_supported_only.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
