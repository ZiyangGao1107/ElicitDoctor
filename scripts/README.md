# Script Index

This directory contains the implementation scripts for the final-patient active
inquiry pipeline. Time-stamped shell scripts are experiment runners; Python files
hold the reusable logic.

## Final Patient Simulator

| Script | Purpose |
|---|---|
| `build_dynamic_patient_controller_v3_2.py` | Final PCV3.2 controller with cross-turn disclosure state. |
| `build_dynamic_patient_controller_v3_1.py` | Previous severe/avoidance policy utilities reused by V3.2. |
| `build_dynamic_patient_controller_v3.py` | Trust/readiness state utilities. |
| `build_dynamic_patient_controller_v2.py` | Earlier controller with retention/weakening logic. |
| `build_dynamic_patient_controller_v1.py` | Base profile/evidence loading and controller helpers. |
| `map_mdd5k_question_slots.py` | Maps doctor questions to symptom/evidence slots. |
| `online_query_interpreter.py` | Online query interpretation helpers. |

## LLM Patient Realizer

| Script | Purpose |
|---|---|
| `prepare_llm_patient_realizer_requests_v2.py` | Builds structured realizer requests from controller outputs. |
| `prepare_llm_patient_realizer_requests_v1.py` | Earlier request builder and JSONL helpers. |
| `call_qwen3_hf_for_patient_realizer.py` | Runs the Qwen3-8B patient realizer. |
| `call_closed_llm_for_patient_realizer.py` | Optional closed-source patient realizer caller for audits. |

## Verification, Repair, and Cache

| Script | Purpose |
|---|---|
| `verify_llm_patient_realizer_outputs_v1.py` | Hard verifier for grounding, leakage, response validity, and severity policy. |
| `prepare_llm_patient_realizer_repair_requests_v1.py` | Builds repair requests from verifier failures. |
| `build_verified_patient_realizer_cache_v1.py` | Builds a verified cache from one verifier pass. |
| `build_verified_patient_realizer_cache_with_repair_v1.py` | Builds final cache from primary plus repaired outputs. |
| `merge_patient_realizer_caches_v1.py` | Merges cache shards or cache versions. |
| `filter_patient_realizer_requests_by_cache_v1.py` | Filters already cached requests. |
| `build_final_patient_freeze_report_v1.py` | Checks whether the final patient setting passes freeze criteria. |

## Rubric Judge

| Script | Purpose |
|---|---|
| `prepare_llm_patient_realizer_rubric_judge_requests_v2.py` | Builds final rubric judge requests. |
| `prepare_llm_patient_realizer_rubric_judge_requests_v1.py` | Earlier rubric request builder. |
| `summarize_patient_realizer_rubric_judge_outputs_v1.py` | Summarizes rubric judge outputs. |
| `prepare_and_merge_rubric_shards_v1.py` | Splits and merges large rubric runs. |
| `run_patient_realizer_rubric_sharded_accel_20260709.sh` | Sharded closed-source rubric runner. |

## Online Doctor Evaluation

| Script | Purpose |
|---|---|
| `run_llm_doctor_online_replay_v1.py` | Online replay engine for doctor-patient turns. |
| `run_pcv32_online_final_patient_doctor_eval_one_20260709.sh` | One-model final-patient doctor evaluation runner. |
| `run_pcv32_online_final_patient_doctor_eval_suite_20260709.sh` | Multi-model final-patient baseline suite runner. |
| `call_closed_llm_for_pending_requests.py` | Closed-source doctor API caller. |
| `call_qwen3_hf_lora_for_pending_requests.py` | Qwen base/LoRA doctor generator. |
| `analyze_tree_aligned_canonical_evidence_recovery.py` | Canonical evidence recovery analyzer. |
| `summarize_final_patient_baseline_suite_v1.py` | Summarizes baseline suite metrics. |

## Data Builders and Training

| Script | Purpose |
|---|---|
| `build_final_patient_sft_from_online_records_v1.py` | Builds SFT examples from verified online records. |
| `build_final_patient_state_bank_from_online_records_v1.py` | Extracts reusable states from online records. |
| `build_final_patient_candidate_rule_rollout_from_state_bank_v1.py` | Builds candidate rollouts from sampled states. |
| `apply_verified_patient_cache_to_candidate_rollout_v1.py` | Replaces branch patient responses with verified cache rows. |
| `build_final_patient_grpo_groups_from_candidate_rollout_v1.py` | Builds same-state GRPO candidate groups. |
| `build_final_patient_rfv_data_from_online_records_v1.py` | Builds residual future-value training data. |
| `train_final_patient_rfv_value_model_numpy_v1.py` | Lightweight RFV value model trainer. |
| `train_qwen3_doctor_sft_lora.py` | Qwen doctor SFT LoRA trainer. |
| `train_qwen3_grpo_from_v6_groups.py` | Qwen doctor GRPO trainer. |
| `run_a100_qwen3_final_patient_sft_lora_20260709.sh` | SFT shell runner. |
| `run_a100_qwen3_final_patient_grpo_from_groups_20260709.sh` | GRPO shell runner. |

## Legacy or Smoke Runners

| Script | Purpose |
|---|---|
| `run_pcv32_full_patient_realizer_hardened_repair_20260708.sh` | Full patient-realizer verification and repair pipeline. |
| `run_pcv32_final_patient_rollout_smoke_20260708.sh` | Small final-patient rollout smoke test. |
| `run_pcv32_qwen_realizer_v2_severe_smoke.sh` | Severe-policy smoke test. |
| `run_patient_realizer_rubric_v3_2_final_verified_full_eval_20260708.sh` | Full rubric runner. |
| `run_patient_realizer_rubric_v3_2_qwenrealizer_verified_sample_eval.sh` | Sample rubric runner. |
| `wait_closed_rubric_then_online_baseline_20260709.sh` | Watcher that starts baseline after freeze. |
| `manual_finalize_pcv32_closed_baseline.sh` | Manual baseline finalization helper. |

## Running Assumptions

Most shell scripts can be redirected to another machine by setting:

- `ACTIVE_REASONING_PROJECT`
- `ACTIVE_REASONING_PHASE_DIR`
- `AR_GRPO_ENV`
- `AR_GRPO_PYTHON`
- `MODEL_PATH`
- `CLOSED_ENV_FILE`

The repository does not include raw data, API keys, checkpoints, or generated
experiment outputs.
