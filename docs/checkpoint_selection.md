# Checkpoint Selection

This repository uses one explicit checkpoint-selection policy for the frozen
Final Patient Setting. The goal is to avoid selecting checkpoints from training
loss alone when the paper metric is canonical evidence recovery.

## Stages

### SFT

SFT is a warm start for the doctor policy.

Selection rule:

1. If a checkpoint has frozen Final Patient online evaluation, rank it by the
   doctor-evaluation rule below.
2. Otherwise use the lowest held-out SFT `eval_loss`.
3. Before using it as the RL initialization, run a small frozen Final Patient
   online evaluation and reject checkpoints with fallback rows, hard errors, or
   obvious severe degradation.

### Value Model V2

Value Model V2 predicts same-state action value: which doctor question is likely
to recover more canonical evidence under the remaining dialogue budget.

Selection rule:

1. Prefer higher held-out same-state Spearman.
2. Require pairwise accuracy above random.
3. Prefer higher top-1 accuracy.
4. Prefer lower mean oracle regret.

This selects the value model that best ranks candidate questions at the same
dialogue state, not the model with the lowest generic regression loss.

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

### Turn24 vs Turn32

Use turn24 as the first checkpoint-selection pass because it is cheaper and
matches the main bounded-dialogue setting. Use turn32 as a secondary stress test:

- if turn24 is weak, do not rescue the checkpoint only because turn32 is longer
- if turn24 checkpoints are close, choose the one with better turn32 severe and
  lower evidence-recovery regret
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

The selector writes:

- `checkpoint_selection_report.json`
- `CHECKPOINT_SELECTION.md`

These reports should be kept with the experiment output and referenced when
choosing the checkpoint for the next stage.
