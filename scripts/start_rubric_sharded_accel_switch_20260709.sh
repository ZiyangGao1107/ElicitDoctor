#!/usr/bin/env bash
set -uo pipefail

PROJECT="${ACTIVE_REASONING_PROJECT:-$(pwd)}"
PHASE_DIR="${ACTIVE_REASONING_PHASE_DIR:-$PROJECT}"
ENV_FILE="${1:-/tmp/pcv32_doctor_suite_env_20260707_0330}"
MODEL="${2:-gpt-4.1-mini}"
RUN_DIR="${3:-outputs_llm_patient_realizer_rubric_v3_2_final_verified_cache_20260708}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PHASE_DIR"

echo "=== start rubric sharded accel switch $(date) ==="
echo "model=$MODEL"
echo "run_dir=$RUN_DIR"
echo "num_shards=$NUM_SHARDS"

echo "[before]"
ps -eo pid,ppid,stat,etime,cmd \
  | grep -E 'run_patient_realizer_rubric_v3_2_final_verified_full_eval_20260708|call_closed_llm_for_patient_realizer.py --env-file .*mdd5k_patient_realizer_rubric_judge_requests' \
  | grep -v grep || true

pids="$(
  ps -eo pid,cmd \
    | grep -E 'run_patient_realizer_rubric_v3_2_final_verified_full_eval_20260708|call_closed_llm_for_patient_realizer.py --env-file .*mdd5k_patient_realizer_rubric_judge_requests' \
    | grep -v grep \
    | awk '{print $1}'
)"
if [[ -n "$pids" ]]; then
  echo "stopping rubric pids: $pids"
  kill $pids || true
  sleep 5
fi

echo "[after_stop]"
ps -eo pid,ppid,stat,etime,cmd \
  | grep -E 'run_patient_realizer_rubric_v3_2_final_verified_full_eval_20260708|call_closed_llm_for_patient_realizer.py --env-file .*mdd5k_patient_realizer_rubric_judge_requests' \
  | grep -v grep || true

TAG="patient_realizer_rubric_sharded_accel_gpt41mini_20260709_0209"
LOG_PATH="logs/${TAG}.log"
NUM_SHARDS="$NUM_SHARDS" nohup bash scripts/run_patient_realizer_rubric_sharded_accel_20260709.sh \
  "$ENV_FILE" \
  "$MODEL" \
  "$RUN_DIR" >"$LOG_PATH" 2>&1 < /dev/null &
echo "sharded_pid=$! log=$LOG_PATH"
echo "=== switch command end $(date) ==="
