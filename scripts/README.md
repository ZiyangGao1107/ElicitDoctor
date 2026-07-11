# Script Index

This directory exposes one maintained final-patient pipeline. Files with a
leading underscore are internal helpers used by the public entry points.

## Patient Simulator

| Script | Purpose |
|---|---|
| `patient_controller.py` | Final PCV3.2 controller with cross-turn disclosure state. |
| `prepare_patient_realizer_requests.py` | Builds structured LLM patient-realizer requests. |
| `call_qwen3_hf_for_patient_realizer.py` | Runs the Qwen3-8B patient realizer. |
| `verify_patient_realizer_outputs.py` | Hard verifier for grounding, leakage, response validity, and severity policy. |
| `prepare_patient_realizer_repair_requests.py` | Builds verifier-guided repair requests. |
| `build_verified_patient_realizer_cache.py` | Builds the final verified patient-response cache. |
| `run_final_patient_realizer_build_cache.sh` | End-to-end realizer, verifier, repair, and cache runner. |

## Rubric Judge

| Script | Purpose |
|---|---|
| `prepare_patient_realizer_rubric_requests.py` | Builds patient-simulation rubric judge requests. |
| `run_patient_realizer_rubric_sharded.sh` | Runs closed-source rubric judge calls in shards. |
| `prepare_and_merge_rubric_shards.py` | Splits and merges rubric request shards. |
| `summarize_patient_realizer_rubric_outputs.py` | Summarizes rubric judge scores. |

## Doctor Evaluation

| Script | Purpose |
|---|---|
| `run_llm_doctor_online_replay.py` | Online replay engine for doctor-patient turns. |
| `run_final_patient_doctor_eval_one.sh` | One-model final-patient doctor evaluation runner. |
| `run_final_patient_doctor_eval_suite.sh` | Multi-model final-patient baseline suite runner. |
| `call_closed_llm_for_pending_requests.py` | Closed-source doctor API caller; supports `openai_compatible`, `openai_responses`, `anthropic`, and `gemini`. |
| `call_qwen3_hf_lora_for_pending_requests.py` | Qwen base/LoRA doctor generator. |
| `analyze_tree_aligned_canonical_evidence_recovery.py` | Canonical evidence recovery analyzer. |
| `summarize_final_patient_baseline_suite.py` | Summarizes baseline suite metrics. |

## Data Builders and Training

| Script | Purpose |
|---|---|
| `build_final_patient_sft_from_online_records.py` | Builds SFT examples from verified online records. |
| `build_final_patient_state_bank_from_online_records.py` | Extracts reusable same-state doctor contexts. |
| `build_final_patient_candidate_rollout.py` | Builds candidate rollouts from sampled states. |
| `apply_verified_patient_cache_to_candidate_rollout.py` | Applies verified patient cache rows to candidate branches. |
| `build_final_patient_grpo_groups.py` | Builds same-state GRPO candidate groups. |
| `build_final_patient_action_value_data.py` | Builds same-state action-value records for Value Model V2. |
| `build_final_patient_rfv_data.py` | Builds residual future-value training data. |
| `train_final_patient_rfv_value_model.py` | Lightweight RFV/action-value model trainer with optional same-state pairwise ranking. |
| `score_final_patient_value_model.py` | Scores action-value records with a trained lightweight value model. |
| `train_qwen3_doctor_sft_lora.py` | Qwen doctor SFT LoRA trainer. |
| `train_qwen3_grpo_from_v6_groups.py` | Qwen doctor GRPO trainer. |
| `run_final_patient_sft_lora.sh` | SFT shell runner. |
| `run_final_patient_grpo_from_groups.sh` | GRPO shell runner. |
| `run_final_patient_value_model_v2.sh` | End-to-end Value Model V2 data, training, scoring, and value-augmented GRPO group runner. |

## Internal Helpers

| Script | Purpose |
|---|---|
| `_patient_controller_base.py` | Profile/evidence loading and base controller utilities. |
| `_patient_controller_disclosure.py` | Disclosure-budget helper logic. |
| `_patient_controller_state.py` | Trust/readiness state helper logic. |
| `_patient_controller_policy.py` | Severe/avoidance helper logic. |
| `_profile_grounded_controller.py` | Profile-grounded environment helper functions. |
| `_patient_realizer_io.py` | JSONL and evidence-unit helpers for realizer scripts. |
| `_doctor_request_prompts.py` | Doctor request prompt construction. |
| `_doctor_policy_baselines.py` | Doctor-policy baseline helper functions. |
| `map_mdd5k_question_slots.py` | Doctor-question to slot mapping. |
| `online_query_interpreter.py` | Online query interpretation helpers. |
| `filter_patient_realizer_requests_by_cache.py` | Cache-aware request filtering. |
| `merge_patient_realizer_caches.py` | Verified-cache merge helper. |

## Running Assumptions

Most shell scripts can be redirected to another machine by setting:

- `ACTIVE_REASONING_PROJECT`
- `ACTIVE_REASONING_PHASE_DIR`
- `AR_GRPO_ENV`
- `AR_GRPO_PYTHON`
- `MODEL_PATH`
- `CLOSED_ENV_FILE`

The repository does not include API keys, checkpoints, logs, or generated
experiment outputs.
