# Checkpoint Selection

This repository uses one explicit checkpoint-selection policy for the frozen
Final Patient Setting. The goal is to avoid selecting checkpoints from training
loss alone when the paper metric is canonical evidence recovery.

## Stages

## Comparability Rule

Every proposed method must expose an explicit checkpoint set before final test
evaluation. Checkpoint choice is part of the method, so it must be reproducible.

The protocol is:

1. Train with predeclared save milestones.
2. Evaluate every candidate checkpoint on the same frozen Final Patient
   selection split, usually `EVAL_SPLITS=dev` or a predeclared validation
   profile subset.
3. Select exactly one checkpoint per method with the rules below.
4. Run the selected checkpoint once on the locked final test split for turn24
   and turn32 reporting.
5. Do not use final test metrics to choose checkpoints.

Closed-source doctor baselines and Qwen base have no trainable checkpoint. They
are evaluated once under the same final patient setting and are not eligible for
checkpoint tuning.

### SFT

SFT is a warm start for the doctor policy.

Selection rule:

1. If a checkpoint has frozen Final Patient online evaluation, rank it by the
   doctor-evaluation rule below.
2. Otherwise use the lowest held-out SFT `eval_loss`.
3. Before using it as the RL initialization, run frozen Final Patient selection
   evaluation and reject checkpoints with fallback rows, hard errors, or obvious
   severe degradation.

Default SFT candidates for historical SFT-only selection:

```text
checkpoint-200, checkpoint-400, checkpoint-600, checkpoint-800,
checkpoint-1000, final_lora_adapter
```

These SFT-only checkpoints are diagnostic for warm-start quality. They should
not be mixed directly with RL method checkpoints in the final method comparison
unless the same normalized milestone grid is available.

### Value Model V2

Value Model V2 predicts same-state long-horizon belief value: which doctor
question is likely to reduce visible uncertainty, preserve or improve patient
openness, and create better future belief updates. It must not use canonical
evidence recovery as the direct value target for the main method.

Selection rule:

1. Prefer higher held-out same-state Spearman.
2. Require pairwise accuracy above random.
3. Prefer higher top-1 accuracy.
4. Prefer lower mean oracle regret.

This selects the value model that best ranks candidate questions at the same
dialogue state by belief-guided long-horizon value, not the model with the
lowest generic regression loss.

### GRPO / ValueAug / RFV

Policy checkpoints are selected by frozen Final Patient online canonical
evidence recovery.

Primary score:

```text
selection_score = mean + 0.5 * severe
```

The severe term is explicit because a checkpoint that improves mild/moderate but
collapses severe is not acceptable for the final patient environment.

Hard filters:

- final records must be verified-only
- fallback rows must be zero
- hard-error rows must be zero
- optional KL/logp-shift ceilings may be applied when available
- optional baseline mean/severe margins may be applied when comparing against an
  existing best model

Official comparable policy grid:

```text
25%, 50%, 75%, 100%, final_lora_adapter
```

Use the same normalized candidate grid for standard GRPO, ValueAug-GRPO, and
RFV. For the current historical runs this maps to:

```text
RFV-v2 1600-step run:      checkpoint-400, checkpoint-800, checkpoint-1200, checkpoint-1600, final_lora_adapter
GRPO-v6 1500-step run:     checkpoint-400, checkpoint-800, checkpoint-1200, checkpoint-1500, final_lora_adapter
ValueAug 1500-step run:    checkpoint-400, checkpoint-800, checkpoint-1200, checkpoint-1500, final_lora_adapter
```

Earlier checkpoints such as `checkpoint-200` are allowed only as diagnostic
learning-curve points. They are not eligible for official method selection
unless every compared method has the same normalized early milestone declared
before evaluation.

Method-specific details:

- Standard GRPO: select from its GRPO checkpoints with the same online recovery
  rule.
- ValueAug-GRPO: first select the value model by the Value Model V2 rule, then
  train policy checkpoints and select the policy by the same online recovery
  rule.
- RFV: train on canonical evidence recovery / residual future-value reward and
  select policy checkpoints by the same online recovery rule. Treat this as an
  outcome-supervision or oracle-style baseline, separate from the belief-guided
  Value Model V2 method.

### Turn24 vs Turn32

Use turn24 on the selection split as the default checkpoint-selection pass
because it is cheaper and matches the main bounded-dialogue setting. Use turn32
as a secondary stress test only if it was predeclared before looking at final
test results:

- if turn24 is weak, do not rescue the checkpoint only because turn32 is longer
- if turn24 checkpoints are close on the selection split, choose the one with
  better selection-split turn32 severe and lower evidence-recovery regret
- final paper tables should report both turn24 and turn32 under the same frozen
  Final Patient Setting

## Command

Use `scripts/select_final_patient_checkpoint.py` after evaluations or value
model training finish.

Doctor / GRPO / RFV example:

```bash
python scripts/select_final_patient_checkpoint.py \
  --stage rfv \
  --suite-summary outputs_final_patient_baseline_suite_summary_final_turn32_20260711/summary.json \
  --output-dir outputs_checkpoint_selection_rfv_turn32 \
  --severe-weight 0.5
```

Value model example:

```bash
python scripts/select_final_patient_checkpoint.py \
  --stage value_model \
  --candidate v2=outputs_final_patient_action_value_model_v2 \
  --output-dir outputs_checkpoint_selection_value_model_v2
```

SFT example:

```bash
python scripts/select_final_patient_checkpoint.py \
  --stage sft \
  --candidate sft_run=outputs_qwen3_final_patient_doctor_sft_lora_run1 \
  --output-dir outputs_checkpoint_selection_sft
```

To evaluate arbitrary LoRA checkpoints under the same frozen patient setting,
create a JSONL manifest:

```json
{"method":"rfv","checkpoint_name":"ckpt400","adapter_path":"outputs_rfv/checkpoint-400"}
{"method":"rfv","checkpoint_name":"ckpt800","adapter_path":"outputs_rfv/checkpoint-800"}
{"method":"valueaug","checkpoint_name":"ckpt400","adapter_path":"outputs_valueaug/checkpoint-400"}
```

Then run the selection-split evaluation:

```bash
python scripts/run_final_patient_checkpoint_eval_manifest.py \
  --manifest checkpoint_manifest.jsonl \
  --run-tag final_patient_ckpt_select_dev_turn24 \
  --max-turns 24 \
  --eval-splits dev
```

Finally pass each output directory to the selector, grouped by method.

The selector writes:

- `checkpoint_selection_report.json`
- `CHECKPOINT_SELECTION.md`

These reports should be kept with the experiment output and referenced when
choosing the checkpoint for the next stage.
