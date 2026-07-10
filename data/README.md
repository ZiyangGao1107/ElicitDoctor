# Data Files

This directory contains the public data artifacts used by the final-patient
active inquiry environment.

## `patient_profiles/`

Dialogue-derived patient profiles used by the PCV3.2 controller.

Key files:

- `mdd5k_dialogue_derived_patient_profiles.jsonl`
- `mdd5k_dialogue_derived_patient_profile_summary.json`
- `MDD5K_DIALOGUE_DERIVED_PATIENT_PROFILES.md`

## `tree_aligned_canonical_evidence/`

Canonical evidence units and links used for evidence-recovery scoring.

Key files:

- `mdd5k_tree_aligned_canonical_evidence_units.jsonl`
- `mdd5k_surface_to_canonical_evidence_links.jsonl`
- `mdd5k_tree_aligned_canonical_evidence_summary.json`

## `f32_f41_profile_split/`

F32/F41 train/dev/test profile-grounded split.

Key files:

- `mdd5k_profile_grounded_environment_train_groups.jsonl`
- `mdd5k_profile_grounded_environment_dev_groups.jsonl`
- `mdd5k_profile_grounded_environment_test_groups.jsonl`
- `f32_f41_stratified_profile_split_summary.json`
- `F32_F41_STRATIFIED_PROFILE_SPLIT_V1.md`

## Storage

Large JSONL files under `data/` are tracked with Git LFS. Install Git LFS before
cloning or pushing:

```bash
git lfs install
```
