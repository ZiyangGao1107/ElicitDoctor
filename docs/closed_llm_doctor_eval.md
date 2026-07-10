# Closed-Source LLM Doctor Evaluation

This package supports closed-source LLM doctors in the same online patient
environment used by open-source and trained Qwen doctors.

## Evaluation Flow

1. Build pending doctor requests for the current replay state.
2. Call the closed-source doctor model with `call_closed_llm_for_pending_requests.py`.
3. Build patient-realizer requests from the doctor outputs.
4. Realize patient responses using Qwen3-8B, or read them from the verified cache.
5. Verify and repair patient responses when needed.
6. Run the actual replay update.
7. Run canonical evidence analysis and summarize mild/moderate/severe/mean.

The final patient setting must be identical across closed-source, base, SFT,
GRPO, ValueAug, and RFV doctors.

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
