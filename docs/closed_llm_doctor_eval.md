# Closed-Source LLM Doctor Evaluation

This package supports closed-source LLM doctors in the same online patient
environment used by open-source and trained Qwen doctors.

## Evaluation Flow

1. Build pending doctor requests for the current replay state.
2. Call the closed-source doctor model with `call_closed_llm_for_pending_requests.py`.
3. Build patient-realizer requests from the doctor outputs.
4. Realize patient responses using Qwen3-8B, or read an already verified
   response from this run's own patient cache.
5. Verify and repair patient responses when needed.
6. Run the actual replay update.
7. Run canonical evidence analysis and summarize mild/moderate/severe/mean.

The final patient setting must be identical across closed-source, base, SFT,
GRPO, ValueAug, and RFV doctors.

## Cache Boundary

The verified patient cache is an internal reproducibility mechanism for one
online run. It is safe to reuse only inside the same output directory, model,
split, turn budget, and patient setting. It prevents repeated Qwen3-8B patient
realizer calls for the exact same patient-realizer request.

Do not seed a new baseline with patient responses generated from another doctor
model or another run. In particular, do not evaluate a closed-source doctor by
reusing a cache produced from RFV, SFT, Qwen base, or another closed model's
doctor questions. That would test the wrong dialogue trajectory.

For a fresh baseline reproduction, the correct flow is:

```text
current doctor model asks a question
  -> controller builds the patient-realizer request
  -> Qwen3-8B realizer answers
  -> verifier/repair accepts it
  -> this run writes current_verified_patient_cache.jsonl
  -> replay advances
```

The final records should still show `patient_realizer_mode ==
"verified_llm_cache"` because replay consumes the verified response after it has
been generated and cached for the current run.

The online runner also enables a final deterministic rule fallback by default
for patient-realizer requests that still fail after two LLM repair rounds:

```bash
export PATIENT_REALIZER_FALLBACK_TO_RULE=1
```

Before a rule fallback response is added to the cache, it is checked with the
same verifier in `--use-source-rule-based` mode. This prevents
`reference_under_informative` repair failures from stopping the whole online
run, while still rejecting unsafe source responses. Set
`PATIENT_REALIZER_FALLBACK_TO_RULE=0` to require every patient response to come
from accepted LLM realization or repair only.

This fallback is dataset-agnostic. It is active for the default MDD-5K test
runner and for DAIC/Extended-DAIC runs that set `DATASET_PREFIX=daic`.

## API Configuration

Closed-source API keys must be supplied outside Git, either through an `.env`
file or through environment variables. The caller supports:

- OpenAI-compatible chat endpoints
- OpenAI Responses API
- Anthropic
- Gemini / Google GenAI

Relevant environment names include:

- `OPENAI_API_KEY`
- `OPENAI_API_KEYS`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- provider-specific base URL variables

Do not commit `.env` files or printed API responses containing secrets.

The final-patient one-model runner exposes closed-source doctor settings through
environment variables:

```bash
export CLOSED_PROVIDER=openai_compatible  # or openai_responses, anthropic, gemini
export CLOSED_MODEL=gpt-4.1-mini
export CLOSED_ENV_FILE=.env
bash scripts/run_final_patient_doctor_eval_one.sh closed_evidence outputs_closed_gpt41mini_turn24 24
```

This evaluates the closed model as the doctor. The patient is still the frozen
final patient simulator.

For a MDD-5K closed-source doctor run that uses only the cooperative
zero-avoidance patient setting, keep the default MDD-5K data paths and set:

```bash
export CLOSED_PROVIDER=openai_compatible
export CLOSED_MODEL=gpt-4.1-mini
export CLOSED_ENV_FILE=.env
export SEVERITIES="zero_avoidance"
bash scripts/run_final_patient_doctor_eval_one.sh \
  closed_evidence outputs_mdd5k_closed_zero_avoidance 24
```

To evaluate the added disclosure settings with the same runner, keep the model
configuration unchanged and override only the patient setting variables:

```bash
export CLOSED_PROVIDER=openai_compatible
export CLOSED_MODEL=gpt-4.1-mini
export CLOSED_ENV_FILE=.env
export SEVERITIES="random_disclosure fully_cooperative zero_avoidance"
export RANDOM_LOW_DISCLOSURE_PROB=0.5
export RANDOM_DISCLOSURE_SEED=0
bash scripts/run_final_patient_doctor_eval_one.sh \
  closed_evidence outputs_closed_gpt41mini_random_full_turn24 24
```

`fully_cooperative` and `zero_avoidance` are deterministic. `zero_avoidance` is
the cooperative-patient condition: it answers truthfully from the available
profile/evidence content with no intentional avoidance, while still not
inventing facts. `random_disclosure` is deterministic for the same profile,
turn, question, probability, and seed, so reruns are reproducible when these
inputs are unchanged.

Use a new output directory or run tag for every independent baseline run:

```bash
export CLOSED_PROVIDER=openai_compatible
export CLOSED_MODEL=gpt-4.1-mini
export CLOSED_ENV_FILE=.env
bash scripts/run_final_patient_doctor_eval_one.sh \
  closed_evidence outputs_closed_gpt41mini_turn24_run1 24
```

If a run is interrupted, resume only with a recovery script that preserves the
same output directory. Do not start a second independent baseline by pointing it
at the first run's directory.

## DAIC PHQ-8 Evaluation

DAIC uses the same replay, patient-controller, realizer, verifier, cache, and
canonical-recovery flow as MDD-5K. The only differences are the data paths,
schema, language, split, and canonical prefix.

Prepare the private DAIC copy under `data/daic/`, then normalize it to the
PHQ-8 profile-grounded contract:

```bash
python scripts/build_daic_profile_environment.py
```

Expected DAIC task contract:

- symptom slots: exactly the eight PHQ-8 items
- labels: exactly `Depressed` and `control`
- train: DAIC-WoZ train, used for training
- valid: DAIC-WoZ validation, used for validation/model selection
- test: all Extended-DAIC rows only; report final test results on this split
- canonical denominator: 8 PHQ-8 item units per profile

Run a closed-source doctor on DAIC test:

```bash
export DATASET_PREFIX=daic
export LANGUAGE=en
export EVAL_SPLITS=test
export MAX_PROFILES=219
export MAX_GROUPS=1752
export MAX_PER_SLOT=999
export MAX_TURNS=24
export GROUP_DIR=data/daic/profile_split
export PROFILE_PATH=data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl
export SCHEMA_PATH=schemas/daic_symptom_slot_schema.json
export CANONICAL_DIR=data/daic/canonical_evidence
export CANONICAL_PREFIX=daic

export CLOSED_PROVIDER=openai_compatible
export CLOSED_MODEL=gpt-4.1-mini
export CLOSED_ENV_FILE=.env

bash scripts/run_final_patient_doctor_eval_one.sh \
  closed_evidence outputs_daic_closed_gpt41mini_turn24 24
```

`--provider cached` in `run_llm_doctor_online_replay.py` does not choose the
closed-source model. It only tells replay to read already generated doctor
questions. The closed-source doctor is chosen by
`call_closed_llm_for_pending_requests.py` through `CLOSED_PROVIDER` and
`CLOSED_MODEL`, or by the one-model runner above.

For a local DAIC smoke test without API calls:

```bash
python scripts/run_llm_doctor_online_replay.py \
  --profiles data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl \
  --schema schemas/daic_symptom_slot_schema.json \
  --group-dir data/daic/profile_split \
  --dataset-prefix daic \
  --language en \
  --splits valid \
  --max-groups 1 \
  --max-per-slot 1 \
  --max-profiles 1 \
  --max-turns 2 \
  --patient-controller-version v3_2 \
  --provider scripted \
  --missing-output-policy scripted \
  --severities mild_low_info \
  --policies closed_llm_general \
  --output-dir outputs_daic_smoke
```

## Main Scripts

- `scripts/call_closed_llm_for_pending_requests.py`: generic closed-source
  doctor caller.
- `scripts/call_closed_llm_for_patient_realizer.py`: closed-source patient
  realizer helper, used mainly for comparison or auditing.
- `scripts/run_final_patient_doctor_eval_one.sh`: one
  model online final-patient evaluation driver.
- `scripts/run_final_patient_doctor_eval_suite.sh`: suite
  runner for multiple baselines.
- `scripts/summarize_final_patient_baseline_suite.py`: suite summary builder.

## Caution

`run_final_patient_doctor_eval_one.sh` recreates its output
directory at startup. Do not restart a running model output directory unless the
recovery plan explicitly preserves existing records.

Do not copy `doctor_outputs.jsonl` or `current_verified_patient_cache.jsonl`
from another model into a new baseline directory. A baseline is valid only if
its doctor outputs are produced by the model being evaluated and its patient
cache is generated from that same dialogue.
