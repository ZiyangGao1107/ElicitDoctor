# F32/F41 Stratified Profile Split V1

This split is built from F32/F41 single-label cases that have profile-grounded environment records.

## Why

The previous baseline pilot used 27 profiles and was too small for paper-facing results. This split expands held-out evaluation while preserving a train/dev/test boundary for later SFT/RL doctor training.

## Settings

- seed: 20260616
- train_ratio: 0.7
- dev_ratio: 0.1
- test_ratio: 0.2
- source profile-grounded cases: 541

## Split Counts

| Split | Cases | F32 cases | F41 cases | Groups |
|---|---:|---:|---:|---:|
| train | 379 | 206 | 173 | 5631 |
| dev | 54 | 29 | 25 | 791 |
| test | 108 | 59 | 49 | 1669 |

## Use

Use this directory as `--group-dir` for formal baseline runs when a larger held-out evaluation set is needed.
