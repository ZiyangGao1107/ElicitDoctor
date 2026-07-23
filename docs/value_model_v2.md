# Belief-Guided Value Model V2

## Boundary

Evidence recovery is the final evaluation metric, not the direct query reward
for the main RL method.

The doctor policy only sees the visible dialogue state. The reward/value data
builder must not expose hidden patient profile fields, canonical evidence
labels, gold diagnosis labels, or simulator-only metadata as model-visible
signals.

RFV / RFV-v2 can still be kept as an outcome-supervision or oracle-style
baseline because it directly optimizes canonical evidence recovery. It should
not be presented as the same method as the belief-guided value model.

## Core Target

The maintained method separates immediate diagnostic uncertainty reduction from
long-horizon trajectory value:

```text
short_term_query_reward(s_t, a_t, r_t)
  = H_norm(b_before) - H_norm(b_after)

long_horizon_belief_value(s_t, a_t)
  = sum_{k=t+1}^{tau} gamma^{k-t-1}
      [H_norm(b_before,k) - H_norm(b_after,k)]
```

Here `s_t` is the visible dialogue state, `a_t` is the doctor question, and
`r_t` is the patient response under the frozen Final Patient Setting. `b` is a
probability distribution over a fixed diagnostic space induced only by visible
dialogue. `tau` is the T3 truncation point: normal dialogue end, patient active
termination, or a detected belief-plateau tail.

The current implementation builds before/after belief evaluator requests from
verified online replay records:

- `query_before`: visible dialogue before the doctor question.
- `query_after`: the same question plus the patient answer.

The belief evaluator estimates broad diagnostic hypotheses, top confidence,
unresolved regions, recommended next inquiry regions, query targets, query
relevance, redundancy, safety relevance, and belief update magnitude.

Query targeting, relevance, safety, and redundancy are diagnostic fields. They
can be used for analysis or auxiliary ablation, but they are not the main
short-term reward.

## Refusal / Low-Information Gate

Patient refusal or vague low-information answers must not create positive
belief gain. The scorer therefore gates visible responses such as "not willing
to say" or "cannot say clearly":

```text
if patient answer is low-information:
  positive entropy reduction := 0
  positive confidence gain := 0
  positive uncertainty-region reduction := 0
  belief_update_magnitude := 0
```

The query cannot receive positive short-term reward as if the patient disclosed
new diagnostic information.

## T3 Truncation

T3 is not GRPO and not a retry policy. It is a trajectory-label cleanup rule:
when a future tail is trapped in no-information turns, truncate the tail so it
does not corrupt earlier action credit.

The current detector marks a belief plateau when a window of turns has:

```text
abs(H_before - H_after) <= epsilon_H
JS(b_after,t, b_after,t-1) <= epsilon_JS
low-information response rate >= threshold
```

For a candidate at turn `t`, the value target only sums future uncertainty
reduction until the first T3 event after `t`:

```text
V_long(s_t, a_t)
  = discounted future entropy reduction before plateau / termination
```

## Current Builder

The implementation is:

```text
scripts/build_belief_guided_query_reward_data.py
```

Prepare belief evaluator requests:

```bash
python scripts/build_belief_guided_query_reward_data.py prepare \
  --records outputs_final_patient_ckpt_select_dev_turn24_20260712_rfvfirst_rfv2_ckpt800/mdd5k_llm_doctor_online_replay_records.jsonl \
  --output-dir outputs_belief_guided_query_reward_data \
  --max-turn-index 20
```

Run a local or closed-source belief evaluator over
`belief_guided_query_belief_requests.jsonl`. The evaluator must return strict
JSON in `raw_output`.

Score query reward and long-horizon value labels:

```bash
python scripts/build_belief_guided_query_reward_data.py score \
  --records outputs_final_patient_ckpt_select_dev_turn24_20260712_rfvfirst_rfv2_ckpt800/mdd5k_llm_doctor_online_replay_records.jsonl \
  --belief-outputs outputs_belief_guided_query_reward_data/qwen3_belief_outputs.jsonl \
  --output-dir outputs_belief_guided_query_reward_data \
  --max-turn-index 20
```

Important outputs:

- `belief_guided_query_belief_requests.jsonl`
- `belief_guided_query_reward_records.jsonl`
- `belief_guided_t3_value_model_records.jsonl`
- `belief_guided_query_reward_summary.json`

The summary explicitly records:

- parse failures
- severity distribution
- low-information response rate
- short-term reward distribution
- long-horizon belief value distribution
- T3 truncation reasons
- method boundary: no canonical evidence recovery, gold diagnosis, or hidden
  patient evidence is used as query reward

## Training Use

After the builder is validated on dev data, train the value model on same-state
candidate branches rather than single-policy trajectories:

```text
state s_t
  candidate a_1 -> final patient response / branch -> belief value y_1
  candidate a_2 -> final patient response / branch -> belief value y_2
  candidate a_3 -> final patient response / branch -> belief value y_3
  candidate a_4 -> final patient response / branch -> belief value y_4
```

The value model should be selected by held-out same-state ranking metrics:

- Spearman correlation with long-horizon belief value
- pairwise accuracy
- top-1 candidate accuracy
- oracle regret
- subgroup checks for severe scenarios

Then use the selected value model in policy training:

```text
reward = short_term_query_reward
       + lambda_value * predicted_long_horizon_belief_value
```

Final policy checkpoints are still selected by verified Final Patient online
recovery on dev, and final numbers are reported on test.

## Minimum Acceptance Bar

Before launching a full GRPO / ValueAug run with this target, require:

- verified final-patient records only
- `fallback = 0`
- `hard_errors = 0`
- no canonical evidence or gold diagnosis in reward prompts
- belief evaluator JSON parse failures near zero
- refusal / low-information gate active
- reward distribution not collapsed to all zeros
- severe subgroup inspected separately
- positive held-out same-state ranking metrics for the value model
