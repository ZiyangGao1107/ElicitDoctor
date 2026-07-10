#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
ENV_PATH="${AR_GRPO_ENV:-}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
RUN_TAG="${1:-pcv32_test108_turn24_doctor_suite_20260707_0330}"
MAX_TURNS="${2:-24}"

cd "$PHASE_DIR"
if [[ -n "${ENV_PATH:-}" ]]; then source "$ENV_PATH/bin/activate"; fi
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

SUITE_DIR="$PHASE_DIR/outputs_${RUN_TAG}"
mkdir -p "$SUITE_DIR"
SUITE_LOG="$SUITE_DIR/suite.log"

echo "=== PCV3.2 doctor suite start $(date) tag=$RUN_TAG ===" | tee "$SUITE_LOG"

if [[ -z "${CLOSED_ENV_FILE:-}" || ! -s "${CLOSED_ENV_FILE:-}" ]]; then
  echo "CLOSED_ENV_FILE is not set/readable; closed baseline will not start" | tee -a "$SUITE_LOG"
  CLOSED_PID=""
else
  CLOSED_OUT="$PHASE_DIR/outputs_${RUN_TAG}_closed_gpt4omini_evidence"
  echo "=== launch closed baseline $(date) out=$CLOSED_OUT ===" | tee -a "$SUITE_LOG"
  REPLAY_CLEAR_OUTPUT=1 CLOSED_ENV_FILE="$CLOSED_ENV_FILE" CLOSED_MODEL="${CLOSED_MODEL:-gpt-4o-mini}" \
    bash scripts/run_pcv32_doctor_eval_one.sh closed_gpt4omini_evidence "$CLOSED_OUT" "$MAX_TURNS" \
    >"$SUITE_DIR/closed_gpt4omini_evidence.log" 2>&1 &
  CLOSED_PID=$!
  echo "closed_pid=$CLOSED_PID" | tee -a "$SUITE_LOG"
fi

for MODEL_KEY in qwen_base qwen_sft_r16 qwen_grpo_rfv2_ckpt1600; do
  OUT="$PHASE_DIR/outputs_${RUN_TAG}_${MODEL_KEY}"
  echo "=== run $MODEL_KEY $(date) out=$OUT ===" | tee -a "$SUITE_LOG"
  REPLAY_CLEAR_OUTPUT=1 bash scripts/run_pcv32_doctor_eval_one.sh "$MODEL_KEY" "$OUT" "$MAX_TURNS" \
    >"$SUITE_DIR/${MODEL_KEY}.log" 2>&1
  echo "=== done $MODEL_KEY $(date) ===" | tee -a "$SUITE_LOG"
done

if [[ -n "${CLOSED_PID:-}" ]]; then
  echo "=== wait closed baseline pid=$CLOSED_PID $(date) ===" | tee -a "$SUITE_LOG"
  wait "$CLOSED_PID"
  echo "=== closed baseline done $(date) ===" | tee -a "$SUITE_LOG"
fi

python - "$RUN_TAG" "$SUITE_DIR" <<'PY'
import json
import sys
from pathlib import Path

tag = sys.argv[1]
suite_dir = Path(sys.argv[2])
phase_dir = suite_dir.parent
models = [
    "closed_gpt4omini_evidence",
    "qwen_base",
    "qwen_sft_r16",
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
        scores = [float(r.get("mean_score", r.get("score", 0.0)) or 0.0) for r in rows]
        if scores:
            entry["mean_of_rows"] = round(sum(scores) / len(scores), 6)
    summary.append(entry)
(suite_dir / "suite_keyword_supported_only_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== PCV3.2 doctor suite done $(date) ===" | tee -a "$SUITE_LOG"
