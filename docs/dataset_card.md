# Dataset Card

## Scope

This repository does not include raw patient profiles, raw interviews, or full
generated dialogue records. It includes code and schema examples needed to
reproduce the pipeline when the controlled data is available.

The project uses MDD-derived patient profiles and canonical evidence units for
active inquiry experiments. These records may contain sensitive mental-health
content and should be stored in controlled infrastructure.

## Expected Inputs

The full pipeline expects JSON or JSONL files for:

- patient profiles
- evidence schema and canonical evidence units
- train/test profile splits
- online replay records
- patient-realizer request/output/cache files
- rubric judge requests and outputs

Synthetic examples are provided in `examples/`.

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

Safe to publish:

- method code
- synthetic examples
- schema documentation
- aggregate metrics
- scripts that read external data paths

Do not publish without an explicit data-release review:

- raw patient text
- raw profile evidence
- full generated patient dialogues
- closed-source model outputs that may contain patient text
- API keys, endpoint credentials, or local environment files
- model checkpoints or adapters

## Reproducibility Note

The repo is designed so that a reviewer with authorized access to the controlled
data can run the same pipeline, while public readers can inspect the method and
file formats without seeing sensitive records.
