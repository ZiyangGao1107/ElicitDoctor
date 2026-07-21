from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DAIC_DIR = BASE_DIR / "data" / "daic"
DEFAULT_SCHEMA_PATH = BASE_DIR / "schemas" / "daic_symptom_slot_schema.json"

PHQ8_SLOTS = [
    "anhedonia",
    "hopelessness_or_crying",
    "sleep",
    "fatigue",
    "appetite_loss",
    "self_worth",
    "attention_decline",
    "psychomotor_change",
]

PHQ8_SLOT_SET = set(PHQ8_SLOTS)

PHQ8_ITEM_TEXT = {
    "anhedonia": "little interest or pleasure in doing things",
    "hopelessness_or_crying": "feeling down, depressed, or hopeless",
    "sleep": "trouble falling or staying asleep, or sleeping too much",
    "fatigue": "feeling tired or having little energy",
    "appetite_loss": "poor appetite or overeating",
    "self_worth": "feeling bad about myself or that I am a failure",
    "attention_decline": "trouble concentrating on things",
    "psychomotor_change": "moving or speaking slowly, or being fidgety/restless",
}

PHQ8_FREQUENCY = {
    0: "not at all",
    1: "several days",
    2: "more than half the days",
    3: "nearly every day",
}


def normalize_daic_split(split: Any) -> str:
    split_text = str(split or "").strip().lower()
    if split_text in {"dev", "validation", "val"}:
        return "valid"
    return split_text


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from exc


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def phq8_binary(record: dict[str, Any]) -> int:
    value = record.get("phq8_binary")
    if value is None:
        value = record.get("PHQ8_Binary")
    if value is None:
        score = int(record.get("phq8_score") or record.get("PHQ8_Score") or 0)
        return 1 if score >= 10 else 0
    return 1 if int(value) == 1 else 0


def diagnosis_label(record: dict[str, Any]) -> str:
    return "Depressed" if phq8_binary(record) == 1 else "control"


def phq8_item_score(record: dict[str, Any], slot: str) -> int:
    items = record.get("phq8_items") or {}
    try:
        return max(0, min(3, int(items.get(slot) or 0)))
    except (TypeError, ValueError):
        return 0


def phq8_anchor_unit(profile: dict[str, Any], slot: str, score: int) -> dict[str, Any]:
    item_text = PHQ8_ITEM_TEXT[slot]
    frequency = PHQ8_FREQUENCY[score]
    if score == 0:
        unit_text = f"I have been bothered by {item_text} {frequency}."
    else:
        unit_text = f"I have been bothered by {item_text} on {frequency}."
    return {
        "unit_id": f"{slot}_phq8_score_anchor",
        "unit_text": unit_text,
        "target_relevance": "core",
        "source_count": 1,
        "source_refs": [
            {
                "source_dataset": profile.get("source_dataset"),
                "participant_id": profile.get("participant_id"),
                "speaker": "PHQ8_Label",
                "phq8_item": slot,
                "phq8_item_text": item_text,
                "phq8_item_score": score,
                "phq8_frequency": frequency,
                "match_type": "phq8_label_anchor",
                "matched_keywords": [],
            }
        ],
        "match_types": {"phq8_label_anchor": 1},
        "matched_keywords": [],
        "phq8_item_score": score,
        "phq8_frequency": frequency,
    }


def normalize_slot_profile(profile: dict[str, Any], slot: str, schema_slots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_slot_profiles = profile.get("slot_profiles") or {}
    source = dict(raw_slot_profiles.get(slot) or {})
    score = phq8_item_score(profile, slot)
    transcript_units = [
        unit
        for unit in list(source.get("evidence_units") or [])
        if not str(unit.get("unit_id") or "").endswith("_phq8_score_anchor")
    ]

    # For score-0 items, broad keyword hits are often false positives. The PHQ-8
    # score anchor is still kept so every profile has all eight PHQ-8 slots.
    if score <= 0:
        transcript_units = []
    evidence_units = [phq8_anchor_unit(profile, slot, score), *transcript_units]

    schema_slot = schema_slots.get(slot) or {}
    source["slot"] = slot
    source["evidence_units"] = evidence_units
    source["num_evidence_units"] = len(evidence_units)
    source["num_unique_units_before_cap"] = len(evidence_units)
    source["support_turn_count"] = sum(int(unit.get("source_count") or 0) for unit in evidence_units)
    source["criticality"] = schema_slot.get("criticality", source.get("criticality", "medium"))
    source["evidence_unit_hints"] = list(schema_slot.get("evidence_unit_hints") or source.get("evidence_unit_hints") or [])
    source["profile_status"] = "observed"
    source["phq8_item_score"] = score
    source["daic_phq8_normalized"] = True
    return source


def normalize_profile(profile: dict[str, Any], schema_slots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label = diagnosis_label(profile)
    original_split = profile.get("split")
    normalized_split = normalize_daic_split(original_split)
    slot_profiles = {
        slot: normalize_slot_profile(profile, slot, schema_slots)
        for slot in PHQ8_SLOTS
    }
    observed_slots = list(PHQ8_SLOTS)
    normalized = dict(profile)
    normalized.update(
        {
            "diagnoses": [label],
            "binary_task_label": label,
            "binary_label": label,
            "diagnosis_label": label,
            "source_split": original_split,
            "split": normalized_split,
            "phq8_binary": phq8_binary(profile),
            "primary_tree_type": "daic_phq8",
            "tree_type_counts": {"daic_phq8": len(PHQ8_SLOTS)},
            "active_tree_slots": list(PHQ8_SLOTS),
            "num_observed_slots": len(PHQ8_SLOTS),
            "observed_slots": observed_slots,
            "slot_profiles": slot_profiles,
        }
    )
    notes = list(normalized.get("construction_notes") or [])
    notes.append("normalized_to_daic_phq8_8_slots_depressed_control_labels")
    normalized["construction_notes"] = notes
    return normalized


def build_group(profile: dict[str, Any], slot: str) -> dict[str, Any]:
    slot_profile = (profile.get("slot_profiles") or {}).get(slot) or {}
    label = diagnosis_label(profile)
    return {
        "counterfactual_group_id": f"{profile['profile_id']}_{slot}",
        "profile_id": profile["profile_id"],
        "case_id": profile.get("case_id"),
        "target_tree_node": slot,
        "target_node_role": "simulator_internal_target_node",
        "target_node_visibility": "simulator_internal_not_doctor_visible",
        "active_tree_type": "daic_phq8",
        "target_slot_profile_status": slot_profile.get("profile_status"),
        "target_slot_evidence_unit_count": int(slot_profile.get("num_evidence_units") or 0),
        "binary_task_label": label,
        "diagnosis_label": label,
        "phq8_binary": phq8_binary(profile),
        "phq8_score": profile.get("phq8_score"),
        "phq8_item_score": phq8_item_score(profile, slot),
        "phq8_severity": profile.get("phq8_severity"),
        "source_dataset": profile.get("source_dataset"),
        "participant_id": profile.get("participant_id"),
        "resplit_source": "daic_woz_train_valid_extended_daic_test_v1",
        "resplit_name": profile.get("split"),
        "source_split": profile.get("source_split"),
    }


def unit_support_refs(unit: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for ref in unit.get("source_refs") or []:
        enriched = dict(ref)
        enriched["surface_unit_id"] = unit.get("unit_id")
        enriched["surface_unit_text"] = unit.get("unit_text")
        enriched.setdefault("match_type", "transcript_evidence")
        enriched.setdefault("matched_keywords", unit.get("matched_keywords") or [])
        refs.append(enriched)
    if refs:
        return refs
    return [
        {
            "surface_unit_id": unit.get("unit_id"),
            "surface_unit_text": unit.get("unit_text"),
            "match_type": "profile_unit",
            "matched_keywords": unit.get("matched_keywords") or [],
        }
    ]


def build_canonical_and_links(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_units: list[dict[str, Any]] = []
    surface_links: list[dict[str, Any]] = []
    for profile in profiles:
        for slot in PHQ8_SLOTS:
            slot_profile = (profile.get("slot_profiles") or {}).get(slot) or {}
            evidence_units = list(slot_profile.get("evidence_units") or [])
            canonical_unit_id = f"{slot}::phq8_item_score"
            support_refs = []
            match_types: Counter[str] = Counter()
            surface_unit_ids = []
            for unit in evidence_units:
                unit_id = str(unit.get("unit_id") or "")
                if not unit_id:
                    continue
                surface_unit_ids.append(unit_id)
                refs = unit_support_refs(unit)
                support_refs.extend(refs)
                for ref in refs:
                    match_types.update([str(ref.get("match_type") or "profile_unit")])
                surface_links.append(
                    {
                        "profile_id": profile["profile_id"],
                        "case_id": profile.get("case_id"),
                        "surface_unit_id": unit_id,
                        "canonical_unit_id": canonical_unit_id,
                        "tree_node": slot,
                        "dimension": "phq8_item_score",
                        "match_type": next(iter((unit.get("match_types") or {"profile_unit": 1}).keys())),
                        "matched_keywords": unit.get("matched_keywords") or [],
                        "phq8_item_score": phq8_item_score(profile, slot),
                        "diagnosis_label": diagnosis_label(profile),
                    }
                )
            canonical_units.append(
                {
                    "profile_id": profile["profile_id"],
                    "case_id": profile.get("case_id"),
                    "tree_type": "daic_phq8",
                    "tree_node": slot,
                    "dimension": "phq8_item_score",
                    "canonical_unit_id": canonical_unit_id,
                    "criticality": slot_profile.get("criticality", "medium"),
                    "is_clinical_key": True,
                    "support_count": len(support_refs),
                    "supporting_variant_ids": sorted(
                        {
                            int(ref["variant_id"])
                            for ref in support_refs
                            if str(ref.get("variant_id", "")).isdigit()
                        }
                    ),
                    "support_refs": support_refs,
                    "profile_surface_unit_ids": sorted(surface_unit_ids),
                    "match_types": dict(sorted(match_types.items())),
                    "phq8_item_score": phq8_item_score(profile, slot),
                    "phq8_item_text": PHQ8_ITEM_TEXT[slot],
                    "diagnosis_label": diagnosis_label(profile),
                }
            )
    return canonical_units, surface_links


def build_summary(
    *,
    daic_dir: Path,
    schema_path: Path,
    profiles: list[dict[str, Any]],
    groups_by_split: dict[str, list[dict[str, Any]]],
    canonical_units: list[dict[str, Any]],
    surface_links: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_slot_counts = Counter()
    label_counts = Counter()
    profiles_by_split = Counter()
    for profile in profiles:
        observed_slot_counts.update(profile.get("observed_slots") or [])
        label_counts.update([profile.get("binary_task_label")])
        profiles_by_split.update([profile.get("split")])
    return {
        "dataset_prefix": "daic",
        "schema": str(schema_path),
        "schema_name": "daic_phq8_symptom_slot_schema",
        "phq8_slots": list(PHQ8_SLOTS),
        "profiles": len(profiles),
        "canonical_units": len(canonical_units),
        "surface_links": len(surface_links),
        "groups_by_split": {split: len(groups) for split, groups in sorted(groups_by_split.items())},
        "profiles_by_split": dict(sorted(profiles_by_split.items())),
        "binary_label_counts": dict(sorted(label_counts.items())),
        "observed_slot_counts": dict(sorted(observed_slot_counts.items())),
        "paths": {
            "profiles": str(daic_dir / "patient_profiles" / "daic_dialogue_derived_patient_profiles.jsonl"),
            "profile_split_dir": str(daic_dir / "profile_split"),
            "canonical_dir": str(daic_dir / "canonical_evidence"),
        },
        "split_policy": {
            "train": "DAIC-WoZ train split",
            "valid": "DAIC-WoZ validation split",
            "test": "all Extended-DAIC rows only",
        },
        "compatibility_aliases": {
            "dev": "alias file for DAIC-WoZ valid split; use valid in new commands",
        },
        "label_policy": "Use PHQ8_Binary when provided; otherwise derive Depressed when PHQ8 total >= 10. Labels are exactly Depressed/control.",
        "normalization_policy": "Keep exactly the eight PHQ-8 symptom slots. Every profile has eight profile-slot groups and eight canonical PHQ-8 item units; score-0 items use label-derived absence anchors and do not keep broad transcript keyword false positives.",
    }


def normalize_environment(daic_dir: Path, schema_path: Path) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_slots = {slot["slot"]: slot for slot in schema.get("slots", [])}
    if list(schema_slots) != PHQ8_SLOTS:
        raise ValueError(f"Schema must define exactly PHQ-8 slots in order: {PHQ8_SLOTS}")

    profile_path = daic_dir / "patient_profiles" / "daic_dialogue_derived_patient_profiles.jsonl"
    group_dir = daic_dir / "profile_split"
    canonical_dir = daic_dir / "canonical_evidence"
    canonical_units_path = canonical_dir / "daic_tree_aligned_canonical_evidence_units.jsonl"
    surface_links_path = canonical_dir / "daic_surface_to_canonical_evidence_links.jsonl"

    profiles = [normalize_profile(record, schema_slots) for record in iter_jsonl(profile_path)]
    groups_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for profile in profiles:
        split = str(profile.get("split") or "")
        if split not in groups_by_split:
            continue
        groups_by_split[split].extend(build_group(profile, slot) for slot in PHQ8_SLOTS)

    canonical_units, surface_links = build_canonical_and_links(profiles)

    write_jsonl(profile_path, profiles)
    for split, groups in groups_by_split.items():
        write_jsonl(group_dir / f"daic_profile_grounded_environment_{split}_groups.jsonl", groups)
    write_jsonl(group_dir / "daic_profile_grounded_environment_dev_groups.jsonl", groups_by_split["valid"])
    write_jsonl(canonical_units_path, canonical_units)
    write_jsonl(surface_links_path, surface_links)

    summary = build_summary(
        daic_dir=daic_dir,
        schema_path=schema_path,
        profiles=profiles,
        groups_by_split=groups_by_split,
        canonical_units=canonical_units,
        surface_links=surface_links,
    )
    write_json(daic_dir / "patient_profiles" / "daic_dialogue_derived_patient_profile_summary.json", summary)
    write_json(canonical_dir / "daic_tree_aligned_canonical_evidence_summary.json", summary)
    return summary


def write_readme(daic_dir: Path) -> None:
    text = """# DAIC PHQ-8 Profile-Grounded Environment

This directory is generated by `scripts/build_daic_profile_environment.py`.
It follows the same profile-grounded artifact contract as the MDD-5K environment, while using the DAIC PHQ-8 task schema.

Split policy:

- `train`: DAIC-WoZ train split.
- `valid`: DAIC-WoZ validation split.
- `test`: all Extended-DAIC rows only.
- `dev`: compatibility alias for `valid`; new commands should use `valid`.

Task definition:

- Symptom slots are exactly the eight PHQ-8 items.
- Diagnosis labels are exactly `Depressed` and `control`.
- `PHQ8_Binary` is used when available; otherwise `PHQ8_Score >= 10` maps to `Depressed`.
- Non-PHQ-8 interview context such as self-harm, anxiety, trauma, substance use, work, social support, treatment history, family history, and current stressors is not used as an evaluator slot.
- The controller still retrieves evidence from `slot_profiles[simulator_internal_target_node]`, and the split files still enumerate profile-slot groups just like MDD-5K.
- Each profile has eight PHQ-8 profile-slot groups and eight canonical PHQ-8 item units; score-0 items are represented by label-derived absence anchors.

Generated files:

- `patient_profiles/daic_dialogue_derived_patient_profiles.jsonl`
- `profile_split/daic_profile_grounded_environment_train_groups.jsonl`
- `profile_split/daic_profile_grounded_environment_valid_groups.jsonl`
- `profile_split/daic_profile_grounded_environment_test_groups.jsonl`
- `canonical_evidence/daic_tree_aligned_canonical_evidence_units.jsonl`
- `canonical_evidence/daic_surface_to_canonical_evidence_links.jsonl`

Replay smoke test:

```bash
python scripts/run_llm_doctor_online_replay.py \\
  --profiles data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl \\
  --schema schemas/daic_symptom_slot_schema.json \\
  --group-dir data/daic/profile_split \\
  --dataset-prefix daic \\
  --language en \\
  --splits valid \\
  --max-groups 1 \\
  --max-per-slot 1 \\
  --max-profiles 1 \\
  --max-turns 2 \\
  --patient-controller-version v3_2 \\
  --provider scripted \\
  --missing-output-policy scripted \\
  --severities mild_low_info \\
  --policies closed_llm_general \\
  --output-dir outputs_daic_smoke
```

These artifacts contain transcript-derived participant text. Keep the original DAIC license and release restrictions in mind before publishing data.
"""
    daic_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize DAIC artifacts to the PHQ-8 eight-slot Depressed/control task.")
    parser.add_argument("--daic-dir", type=Path, default=DEFAULT_DAIC_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = normalize_environment(args.daic_dir, args.schema)
    write_readme(args.daic_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
