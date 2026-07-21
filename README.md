# Active Reasoning Final Patient

This repository contains the reproducible code package for the Active Reasoning
final patient setting used in the MDD active inquiry experiments.

The public package focuses on method code, patient simulator control logic,
closed-source doctor evaluation utilities, training-data builders, and the
public MDD-derived data artifacts needed to reproduce the environment.
Closed-source API keys, model checkpoints, logs, and generated experiment
outputs are not included.

## What Is Included

Start with `docs/code_guide.md` for a stage-by-stage explanation of the code.
The short script index is in `scripts/README.md`.

- `scripts/patient_controller.py`: PCV3.2 patient controller
  with cross-turn disclosure state.
- `scripts/prepare_patient_realizer_requests.py`: converts controller
  decisions into LLM patient-realizer requests.
- `scripts/verify_patient_realizer_outputs.py`: verifier for grounding,
  forbidden-evidence leakage, and response validity.
- `scripts/prepare_patient_realizer_repair_requests.py`: repair-loop
  request builder for failed realizer outputs.
- `scripts/build_verified_patient_realizer_cache.py`: merges
  primary and repaired outputs into a verified cache.
- `scripts/run_final_patient_doctor_eval_one.sh`: one-model
  online doctor evaluation driver under the final patient setting.
- `scripts/call_closed_llm_for_pending_requests.py`: closed-source LLM doctor API
  caller for OpenAI-compatible, OpenAI Responses, Anthropic, and Gemini-style
  providers.
- `scripts/build_final_patient_sft_from_online_records.py`: SFT data builder
  from verified final-patient online records.
- `scripts/build_belief_guided_query_reward_data.py`: builds visible-dialogue
  belief-guided query reward and long-horizon value labels without using
  canonical evidence as reward.
- `scripts/build_final_patient_rfv_data.py`: residual
  future-value data builder.
- `scripts/build_final_patient_grpo_groups.py`: GRPO
  group builder from same-state candidate rollouts.

## Final Patient Setting

The frozen simulator is:

`PCV3.2 controller + Qwen3-8B realizer + verifier-repair + verified cache`

Final evaluation records must satisfy:

- `patient_realizer_mode == "verified_llm_cache"`
- `fallback == 0`
- `hard_errors == 0`

See `docs/final_patient_setting.md` for the controller, realizer, verifier, and
repair-loop design.

The default disclosure settings remain `mild_low_info`, `moderate_low_info`,
and `severe_low_info`. Additional patient settings are available for controlled
stress tests: `random_disclosure` with configurable
`RANDOM_LOW_DISCLOSURE_PROB`, and `fully_cooperative`.

## Data Policy

The public MDD-derived patient profiles, canonical evidence files, and F32/F41
profile splits are stored under `data/`. Large JSONL files are tracked with Git
LFS. Generated model outputs, logs, checkpoints, and closed-source API traces are
not part of the release package.

DAIC-WoZ and Extended-DAIC are supported by the same profile-grounded simulator
contract, but the DAIC data itself should be transferred privately under the
original DAIC release terms. Place the prepared private copy at `data/daic/` and
run:

```bash
python scripts/build_daic_profile_environment.py
```

This builds DAIC as a PHQ-8 task with exactly eight symptom slots and exactly
two diagnosis labels: `Depressed` and `control`. DAIC-WoZ `train` is used for
training, DAIC-WoZ `valid` is used for validation/model selection, and all
Extended-DAIC rows are used only as `test`.

See `docs/dataset_card.md` for the expected JSONL schemas and release policy.

## Closed-Source Doctor Baselines

Closed-source doctor baselines are run by preparing pending doctor requests,
calling the configured API provider, then continuing the same online replay
against the verified final patient simulator.

For reproduction, use a fresh output directory for each baseline model/run. The
`current_verified_patient_cache.jsonl` produced during evaluation is valid only
inside that run; do not reuse patient caches or doctor-output JSONL files across
models, runs, splits, or turn budgets.

See `docs/closed_llm_doctor_eval.md`.

For DAIC closed-source doctor evaluation, use the same runner and set the
dataset environment variables:

```bash
export DATASET_PREFIX=daic
export LANGUAGE=en
export EVAL_SPLITS=test
export MAX_PROFILES=219
export MAX_GROUPS=1752
export GROUP_DIR=data/daic/profile_split
export CLOSED_PROVIDER=openai_compatible
export CLOSED_MODEL=gpt-4.1-mini
export CLOSED_ENV_FILE=.env
bash scripts/run_final_patient_doctor_eval_one.sh closed_evidence outputs_daic_closed_eval 24
```

## Minimal Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Closed-source doctor evaluation requires API credentials supplied through an
external `.env` file or environment variables. Do not commit keys.

## Value Model V2

The maintained value-model route is belief-guided: estimate whether a doctor
question targets unresolved belief regions, reduces visible diagnostic
uncertainty, and improves future patient openness. Canonical evidence recovery
is reserved for final evaluation and oracle-style RFV baselines, not the direct
query reward. See `docs/value_model_v2.md`.

## Repository Safety

The `.gitignore` excludes generated outputs, logs, checkpoints, local notes,
paper PDFs/text extracts, presentations, and credentials. Before publishing,
run a secret scan, confirm Git LFS is enabled, and review `git status --ignored`.
