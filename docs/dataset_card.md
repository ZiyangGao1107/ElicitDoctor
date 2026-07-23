# Dataset Card

## Scope

This repository includes the public MDD-derived data artifacts needed to
reproduce the final-patient environment. It does not include full generated
experiment outputs, closed-source API traces, model checkpoints, or logs.

The project uses MDD-derived patient profiles and canonical evidence units for
active inquiry experiments. These artifacts are treated as public research data
for this release. Downstream users should still cite the original dataset source
and follow its license and usage terms.

## Included Data Files

### `data/patient_profiles/`

- `mdd5k_dialogue_derived_patient_profiles.jsonl`
- `mdd5k_dialogue_derived_patient_profile_summary.json`
- `MDD5K_DIALOGUE_DERIVED_PATIENT_PROFILES.md`

These files define the dialogue-derived patient profiles used by the controller.

### `data/tree_aligned_canonical_evidence/`

- `mdd5k_tree_aligned_canonical_evidence_units.jsonl`
- `mdd5k_surface_to_canonical_evidence_links.jsonl`
- `mdd5k_tree_aligned_canonical_evidence_summary.json`

These files define canonical evidence units and surface-to-canonical links used
for evidence recovery scoring.

### `data/f32_f41_profile_split/`

- `mdd5k_profile_grounded_environment_train_groups.jsonl`
- `mdd5k_profile_grounded_environment_dev_groups.jsonl`
- `mdd5k_profile_grounded_environment_test_groups.jsonl`
- `f32_f41_stratified_profile_split_summary.json`
- `F32_F41_STRATIFIED_PROFILE_SPLIT_V1.md`

These files define the F32/F41 profile-grounded train/dev/test split.

### Private DAIC Layout

DAIC-WoZ and Extended-DAIC are supported, but the raw/prepared DAIC artifacts
should not be published in this repository unless the distributor has the
necessary DAIC release rights. A private transfer should place the directory at
`data/daic/`, then run:

```bash
python scripts/build_daic_profile_environment.py
```

The generated DAIC environment follows the same profile-grounded contract as
MDD-5K:

- `data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl`
- `data/daic/profile_split/daic_profile_grounded_environment_train_groups.jsonl`
- `data/daic/profile_split/daic_profile_grounded_environment_valid_groups.jsonl`
- `data/daic/profile_split/daic_profile_grounded_environment_test_groups.jsonl`
- `data/daic/canonical_evidence/daic_tree_aligned_canonical_evidence_units.jsonl`
- `data/daic/canonical_evidence/daic_surface_to_canonical_evidence_links.jsonl`

DAIC is defined as a PHQ-8 depression-screening task: the evaluator slots are
exactly the eight PHQ-8 items, and labels are exactly `Depressed` and `control`.
DAIC-WoZ `train` is used for training, DAIC-WoZ `valid` is used for validation
and model selection, and all Extended-DAIC rows are used only as `test`.

## Expected Inputs

The full pipeline expects JSON or JSONL files for:

- patient profiles
- evidence schema and canonical evidence units
- train/test profile splits
- online replay records
- patient-realizer request/output/cache files
- rubric judge requests and outputs

Small synthetic format examples are also provided in `examples/`.

## Core JSONL Objects

### Patient Realizer Request

Required concepts:

- unique `request_id`
- profile or case id
- severity label
- current doctor question
- dialogue history
- retained evidence
- weakened evidence
- forbidden evidence
- controller state

### Patient Realizer Output

Required concepts:

- `request_id`
- generated patient response
- model name
- realization metadata

### Verified Cache Row

Required concepts:

- `request_id`
- final verified patient response
- verification status
- repair round if any
- error/warning counters

### Online Replay Record

Required concepts:

- profile id
- turn index
- doctor question
- patient response
- patient realizer mode
- verified-cache marker
- evidence recovery metrics
- severity label

## Release Policy

Published in this repository:

- method code
- public MDD-derived profile/evidence/split data under `data/`
- DAIC preprocessing and evaluation code, but not DAIC private data unless
  separately authorized
- schema documentation
- aggregate metrics
- scripts that read external data paths

Not published in this repository:

- full generated patient dialogues
- closed-source model outputs that may contain patient text
- API keys, endpoint credentials, or local environment files
- model checkpoints or adapters

## Reproducibility Note

The repo is designed so that a reviewer can inspect the method, run the patient
environment from the included public data artifacts, and reproduce the file
formats without requiring generated experiment outputs.
