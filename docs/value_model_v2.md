# Value Model V2: Same-State Action Value

## Purpose

Value Model V2 estimates which doctor question is likely to recover more
canonical clinical evidence over the remaining dialogue budget under the frozen
Final Patient Setting.

The model target is not generic dialogue quality. It is:

```text
Q(s_t, a_t) = immediate canonical evidence gain
            + future branch canonical evidence gain
```

where `s_t` is the visible dialogue state and `a_t` is a candidate doctor
question.

## Why This Replaces the Older Value Target

The older residual-value builder can learn from complete trajectories, but each
state usually has only the action that was actually taken. That makes credit
assignment weak: the value model may learn trajectory or policy bias instead of
learning which question is better at the same state.

V2 uses same-state counterfactual candidates:

```text
state s_t
  candidate a_1 -> patient response / branch rollout -> recovery R_1
  candidate a_2 -> patient response / branch rollout -> recovery R_2
  candidate a_3 -> patient response / branch rollout -> recovery R_3
  candidate a_4 -> patient response / branch rollout -> recovery R_4
```

The value model is then trained to score higher-recovery candidates above
lower-recovery candidates.

## Data Pipeline

Build a state bank from verified final-patient online records:

```bash
python scripts/build_final_patient_state_bank_from_online_records.py \
  --source rfv2=outputs_pcv32_online_final_patient_baseline_turn24_..._qwen_grpo_rfv2_ckpt1600 \
  --output-dir outputs_final_patient_state_bank_rfv2 \
  --metric-name keyword_supported_only \
  --max-turn-index 20 \
  --candidates-per-state 4
```

Generate doctor candidates for
`final_patient_same_state_candidate_requests.jsonl` with a doctor model, then
run them through the final patient controller/realizer/cache path:

```bash
python scripts/build_final_patient_candidate_rollout.py \
  --state-bank outputs_final_patient_state_bank_rfv2/final_patient_state_bank.jsonl \
  --candidate-requests outputs_final_patient_state_bank_rfv2/final_patient_same_state_candidate_requests.jsonl \
  --candidate-outputs outputs_candidate_doctor_outputs.jsonl \
  --output-dir outputs_final_patient_candidate_rollout_rfv2

python scripts/apply_verified_patient_cache_to_candidate_rollout.py \
  --records outputs_final_patient_candidate_rollout_rfv2/final_patient_candidate_rule_rollout_records.jsonl \
  --cache outputs_verified_candidate_patient_cache/current_verified_patient_cache.jsonl \
  --output-dir outputs_final_patient_verified_candidate_rollout_rfv2 \
  --require-all \
  --drop-hard-errors
```

Build action-value records:

```bash
python scripts/build_final_patient_action_value_data.py \
  --records outputs_final_patient_verified_candidate_rollout_rfv2/final_patient_candidate_verified_rollout_records.jsonl \
  --output-dir outputs_final_patient_action_value_data_rfv2 \
  --metric-name keyword_supported_only
```

If candidate branches contain only the first patient response, the target is a
one-step action value. If candidate branches are continued for more turns, the
same builder automatically includes future branch gains.

## Training

Train the lightweight value model with action-value targets and same-state
ranking:

```bash
python scripts/train_final_patient_rfv_value_model.py \
  --record-path outputs_final_patient_action_value_data_rfv2/final_patient_action_value_records.jsonl \
  --output-dir outputs_final_patient_action_value_model_v2 \
  --target-mode action_value_total \
  --pairwise-weight 0.5 \
  --pair-min-margin 0.01 \
  --epochs 12
```

Important metrics in
`final_patient_rfv_value_model_train_summary.json`:

- `eval_metrics.spearman`: rank correlation with realized action value.
- `eval_pair_metrics.pair_accuracy`: same-state pairwise ordering accuracy.
- `eval_pair_metrics.top1_accuracy`: whether the top predicted candidate is
  the best realized candidate for that state.
- `eval_pair_metrics.mean_oracle_regret`: realized recovery lost by choosing
  the predicted best instead of the oracle best.

## Use in GRPO / RFV

After the V2 value model passes offline ranking checks, use it as a reward
component for same-state GRPO groups:

```text
reward = canonical evidence gain
       + lambda_value * predicted residual/action value
       - safety/avoidance penalties
```

Do not use hidden patient evidence as model-visible doctor input. Hidden
canonical labels may be used for reward and value targets during training, but
the deployed doctor policy should only see the dialogue history.

## Minimum Acceptance Bar

Before launching a full GRPO/RFV run, require:

- verified final-patient records only
- no fallback rows
- no hard-error rows
- at least two candidates per state
- non-zero reward margin for most groups
- positive held-out Spearman
- positive pairwise accuracy above random
- severe subgroup not worse than the non-value baseline
