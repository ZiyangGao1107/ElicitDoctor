# Code Guide

This guide explains how the repository code is organized and how the major
scripts connect. It is written for readers who want to understand or reproduce
the final-patient active inquiry pipeline without reading every script first.

## Pipeline Overview

The final setting has four layers:

1. Patient controller
2. LLM patient realizer
3. Verifier-repair cache
4. Doctor evaluation and training

The intended flow is:

```text
public patient profile + canonical evidence
  -> PCV3.2 controller
  -> patient-realizer request JSONL
  -> Qwen3-8B patient response
  -> verifier
  -> repair loop if needed
  -> verified patient cache
  -> online doctor replay
  -> canonical evidence recovery summary
  -> SFT / RFV / GRPO data builders
```

Final records should use `verified_llm_cache` patient responses. Rule fallback is
only a debugging or probe mechanism.

## If You Only Read Five Files

Read these first:

- `scripts/build_dynamic_patient_controller_v3_2.py`
- `scripts/prepare_llm_patient_realizer_requests_v2.py`
- `scripts/verify_llm_patient_realizer_outputs_v1.py`
- `scripts/run_pcv32_online_final_patient_doctor_eval_one_20260709.sh`
- `scripts/call_closed_llm_for_pending_requests.py`

These files define the frozen patient simulator, the LLM-realizer interface, the
hard verifier, one-model online evaluation, and closed-source doctor calling.

## Patient Controller

Main files:

- `scripts/build_dynamic_patient_controller_v1.py`
- `scripts/build_dynamic_patient_controller_v2.py`
- `scripts/build_dynamic_patient_controller_v3.py`
- `scripts/build_dynamic_patient_controller_v3_1.py`
- `scripts/build_dynamic_patient_controller_v3_2.py`
- `scripts/map_mdd5k_question_slots.py`
- `scripts/online_query_interpreter.py`

The controller maps each doctor question to symptom or evidence slots and
decides which evidence can be disclosed at the current turn.

The final version is `DynamicPatientControllerV32`. It tracks cross-turn patient
state:

- disclosed slots
- withheld slots
- forbidden evidence
- trust/readiness
- engagement
- refusal or avoidance history
- doctor question quality and follow-up behavior

The controller outputs three evidence groups:

- `retained`: can be said directly.
- `weakened`: can be said vaguely or softly.
- `removed`: must not be revealed.

Severity controls how quickly the patient opens up:

- mild: more direct answers.
- moderate: partial answers and some avoidance.
- severe: more guarded, especially for sensitive evidence, but can open after
  relevant and supportive follow-up.

## Patient Realizer

Main files:

- `scripts/prepare_llm_patient_realizer_requests_v1.py`
- `scripts/prepare_llm_patient_realizer_requests_v2.py`
- `scripts/call_qwen3_hf_for_patient_realizer.py`
- `scripts/call_closed_llm_for_patient_realizer.py`

The request builder converts controller state into a JSONL prompt request. A
realizer model then writes the actual patient utterance.

The realizer should:

- answer as the patient, not as a doctor or evaluator
- respond to the current question
- stay inside retained/weakened evidence
- avoid removed evidence
- sound natural and conversational
- preserve severity-conditioned guardedness

`call_qwen3_hf_for_patient_realizer.py` is the open-source Qwen3-8B realizer
runner. `call_closed_llm_for_patient_realizer.py` exists for closed-source
patient-realizer comparison or auditing, but the frozen setting uses Qwen3-8B.

## Verifier and Repair

Main files:

- `scripts/verify_llm_patient_realizer_outputs_v1.py`
- `scripts/prepare_llm_patient_realizer_repair_requests_v1.py`
- `scripts/build_verified_patient_realizer_cache_v1.py`
- `scripts/build_verified_patient_realizer_cache_with_repair_v1.py`
- `scripts/merge_patient_realizer_caches_v1.py`
- `scripts/filter_patient_realizer_requests_by_cache_v1.py`

The verifier is a hard gate. It checks:

- grounding: no unsupported symptom or contradiction
- disclosure control: no forbidden evidence leakage
- response validity: not empty, not a doctor note, not unrelated
- severity policy: no severe over-disclosure

Failed outputs become repair requests. The repair prompt includes verifier
feedback and asks the realizer to regenerate within the same controller budget.

The final cache builder merges primary and repaired outputs into a verified
cache. Final evaluation/training should only consume records with:

- `patient_realizer_mode == "verified_llm_cache"`
- fallback count equal to zero
- verifier hard error count equal to zero

## Rubric Judge

Main files:

- `scripts/prepare_llm_patient_realizer_rubric_judge_requests_v1.py`
- `scripts/prepare_llm_patient_realizer_rubric_judge_requests_v2.py`
- `scripts/summarize_patient_realizer_rubric_judge_outputs_v1.py`
- `scripts/prepare_and_merge_rubric_shards_v1.py`
- `scripts/run_patient_realizer_rubric_sharded_accel_20260709.sh`

The verifier is not a quality score. The rubric judge evaluates whether the
patient simulator behaves like a plausible controlled patient simulation.

Rubric dimensions:

- grounding
- disclosure control
- avoidance quality
- query responsiveness
- dialogue naturalness

The sharded runner is used when many rubric requests need to be evaluated by a
closed-source judge model.

## Doctor Evaluation

Main files:

- `scripts/run_llm_doctor_online_replay_v1.py`
- `scripts/run_pcv32_online_final_patient_doctor_eval_one_20260709.sh`
- `scripts/run_pcv32_online_final_patient_doctor_eval_suite_20260709.sh`
- `scripts/call_closed_llm_for_pending_requests.py`
- `scripts/call_qwen3_hf_lora_for_pending_requests.py`
- `scripts/analyze_tree_aligned_canonical_evidence_recovery.py`
- `scripts/summarize_final_patient_baseline_suite_v1.py`

Online doctor replay alternates between doctor questions and verified patient
responses. The patient simulator must stay fixed across all doctor models.

The suite runner evaluates models such as:

- closed-source evidence doctor
- Qwen base
- Qwen SFT
- GRPO variants
- ValueAug variants
- RFV variants

The canonical analyzer computes evidence recovery by severity and the final
summary reports:

- mild
- moderate
- severe
- mean

## Closed-Source Doctor Calling

Main file:

- `scripts/call_closed_llm_for_pending_requests.py`

This script reads pending doctor requests and writes model outputs. It supports:

- OpenAI-compatible chat completions
- OpenAI Responses API
- Anthropic
- Gemini / Google GenAI

Keys should be provided by environment variables or an external `.env` file.
The repository does not include keys.

## Training Data Builders

Main files:

- `scripts/build_final_patient_sft_from_online_records_v1.py`
- `scripts/build_final_patient_state_bank_from_online_records_v1.py`
- `scripts/build_final_patient_candidate_rule_rollout_from_state_bank_v1.py`
- `scripts/apply_verified_patient_cache_to_candidate_rollout_v1.py`
- `scripts/build_final_patient_grpo_groups_from_candidate_rollout_v1.py`
- `scripts/build_final_patient_rfv_data_from_online_records_v1.py`
- `scripts/train_final_patient_rfv_value_model_numpy_v1.py`

These scripts rebuild training data under the final patient setting.

Use them after aligned baseline runs are complete. Do not mix records generated
under different patient simulators.

The intended training-data logic is:

```text
verified online records
  -> SFT examples
  -> state bank
  -> same-state candidate rollouts
  -> verified patient cache applied to candidate branches
  -> GRPO groups
  -> RFV value targets
```

## Training Runners

Main files:

- `scripts/run_a100_qwen3_final_patient_sft_lora_20260709.sh`
- `scripts/run_a100_qwen3_final_patient_grpo_from_groups_20260709.sh`
- `scripts/train_qwen3_doctor_sft_lora.py`
- `scripts/train_qwen3_grpo_from_v6_groups.py`

The shell runners are thin wrappers around the Python training scripts. They
expect model and data paths to be supplied through arguments or environment
variables.

Important variables:

- `ACTIVE_REASONING_PROJECT`: project root
- `ACTIVE_REASONING_PHASE_DIR`: data/evaluation working directory
- `AR_GRPO_ENV`: optional Python environment to activate
- `AR_GRPO_PYTHON`: optional Python executable
- `MODEL_PATH`: local Qwen model path
- `OUTPUT_DIR`: training output path

## Output Files

Common output files:

- `mdd5k_llm_doctor_online_replay_records.jsonl`
- `mdd5k_llm_doctor_online_replay_pending_requests.jsonl`
- `current_verified_patient_cache.jsonl`
- `pcv32_keyword_supported_only.json`
- `mdd5k_patient_realizer_verification_summary_llm_outputs.json`
- `mdd5k_patient_realizer_rubric_judge_summary.json`

See `examples/` for safe synthetic schemas.

## Included Data

The public data artifacts needed by the controller and canonical evaluator are
included under `data/`:

- `data/patient_profiles/`
- `data/tree_aligned_canonical_evidence/`
- `data/f32_f41_profile_split/`

Large JSONL files are tracked with Git LFS.

## What Is Not Included

This repository intentionally excludes:

- full generated dialogues
- paper PDFs or extracted paper text
- API credentials
- logs and generated output directories
- checkpoints and LoRA adapters

Generated records and model artifacts should be released separately only when
their size, license, and reproducibility purpose are clear.
