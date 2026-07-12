from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from online_query_interpreter import load_json


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(r"D:\Active Reasoning\Code\Dataset")
DEFAULT_SCHEMA_PATH = BASE_DIR / "schemas" / "daic_symptom_slot_schema.json"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "data" / "daic"
DEFAULT_DATASET_PREFIX = "daic"

DAIC_WOZ_DIRNAME = "DAIC-WOZ_Pre"
EXTENDED_DAIC_DIRNAME = "Extend-DAIC-WOZ_Pre"

PHQ_SLOT_KEYS = {
    "anhedonia": ("PHQ8_NoInterest", "PHQ_8NoInterest"),
    "hopelessness_or_crying": ("PHQ8_Depressed", "PHQ_8Depressed"),
    "sleep": ("PHQ8_Sleep", "PHQ_8Sleep"),
    "fatigue": ("PHQ8_Tired", "PHQ_8Tired"),
    "appetite_loss": ("PHQ8_Appetite", "PHQ_8Appetite"),
    "self_worth": ("PHQ8_Failure", "PHQ_8Failure"),
    "attention_decline": ("PHQ8_Concentrating", "PHQ_8Concentrating"),
    "psychomotor_change": ("PHQ8_Moving", "PHQ_8Moving"),
}

SPEAKER_RE = re.compile(r"^\[(?P<speaker>[^\]]+)\]:\s*(?P<text>.*)$")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SPACE_RE = re.compile(r"\s+")

FILLER_TEXT = {
    "",
    "okay",
    "ok",
    "yeah",
    "yep",
    "uh huh",
    "mhm",
    "mm hmm",
    "thanks",
    "thank you",
}


def read_text_with_fallback(path: Path, encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "cp1252", "latin-1")) -> str:
    last_error: UnicodeDecodeError | None = None
    raw = path.read_bytes()
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return raw.decode("utf-8", errors="replace")


def iter_csv(path: Path) -> list[dict[str, str]]:
    text = read_text_with_fallback(path)
    return list(csv.DictReader(text.splitlines()))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").replace("\u3000", " ").strip())


def normalize_text(value: Any) -> str:
    return clean_text(value).lower()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and str(row.get(key, "")).strip() != "":
            return row.get(key)
    return default


def load_label_rows(path: Path, *, split: str, source_dataset: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_csv(path):
        participant_id = clean_text(row.get("Participant_ID"))
        if not participant_id:
            continue
        phq8_score = safe_int(first_present(row, ("PHQ8_Score", "PHQ_8Total")), default=0) or 0
        phq8_binary_raw = first_present(row, ("PHQ8_Binary",), default=None)
        phq8_binary = safe_int(phq8_binary_raw, default=None)
        if phq8_binary is None:
            phq8_binary = 1 if phq8_score >= 10 else 0
        phq8_items = {
            slot: safe_int(first_present(row, keys), default=0) or 0
            for slot, keys in PHQ_SLOT_KEYS.items()
        }
        result[participant_id] = {
            "participant_id": participant_id,
            "split": split,
            "source_dataset": source_dataset,
            "phq8_score": phq8_score,
            "phq8_binary": phq8_binary,
            "phq8_items": phq8_items,
            "raw_label_row": row,
            "gender": safe_int(first_present(row, ("Gender",), default=None), default=None),
            "label": safe_int(first_present(row, ("label",), default=None), default=None),
            "ds_score": safe_int(first_present(row, ("DS score",), default=None), default=None),
        }
    return result


def read_daic_woz_turns(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current_doctor = ""
    current_doctor_turn = -1
    doctor_turn_count = 0
    participant_turn_count = 0
    for raw_line in read_text_with_fallback(path).splitlines():
        match = SPEAKER_RE.match(raw_line.strip())
        if not match:
            continue
        speaker = clean_text(match.group("speaker")).lower()
        text = clean_text(match.group("text"))
        if not text:
            continue
        if speaker == "ellie":
            current_doctor = text
            current_doctor_turn = doctor_turn_count
            doctor_turn_count += 1
        elif speaker == "participant":
            turns.append(
                {
                    "turn_id": participant_turn_count,
                    "speaker": "Participant",
                    "doctor_utterance": current_doctor,
                    "doctor_turn_id": current_doctor_turn,
                    "patient_utterance": text,
                    "transcript_row": None,
                    "confidence": None,
                }
            )
            participant_turn_count += 1
    return turns


def read_extended_turns(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_csv(path)):
        text = clean_text(row.get("Text"))
        if not text:
            continue
        turns.append(
            {
                "turn_id": idx,
                "speaker": "Participant",
                "doctor_utterance": "",
                "doctor_turn_id": None,
                "patient_utterance": text,
                "transcript_row": idx,
                "start_time": row.get("Start_Time"),
                "end_time": row.get("End_Time"),
                "confidence": row.get("Confidence"),
            }
        )
    return turns


def schema_slots(schema: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    slots = [slot["slot"] for slot in schema.get("slots", [])]
    return slots, {slot["slot"]: slot for slot in schema.get("slots", [])}


def slot_matches(text: str, slot_defs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    lowered = normalize_text(text)
    matches: dict[str, list[str]] = {}
    for slot, slot_def in slot_defs.items():
        matched = []
        for keyword in slot_def.get("question_keywords") or []:
            keyword_text = normalize_text(keyword)
            if keyword_text and keyword_text in lowered:
                matched.append(str(keyword))
        if matched:
            matches[slot] = matched
    return matches


def unit_text_for_turn(turn: dict[str, Any], slots_from_question: set[str]) -> str:
    text = clean_text(turn.get("patient_utterance"))
    if word_count(text) >= 3 or not slots_from_question:
        return text
    doctor = clean_text(turn.get("doctor_utterance"))
    if doctor:
        return text
    return text


def should_keep_unit(text: str, has_slot_context: bool) -> bool:
    lowered = normalize_text(text).strip(" .,!?:;")
    if lowered in FILLER_TEXT and not has_slot_context:
        return False
    return bool(text) and (word_count(text) >= 3 or has_slot_context)


def add_unit(
    slot_units: dict[str, dict[str, dict[str, Any]]],
    *,
    slot: str,
    unit_text: str,
    source_ref: dict[str, Any],
    match_type: str,
    matched_keywords: list[str],
    target_relevance: str,
) -> None:
    key = normalize_text(unit_text)
    bucket = slot_units.setdefault(slot, {})
    unit = bucket.setdefault(
        key,
        {
            "unit_text": unit_text,
            "target_relevance": target_relevance,
            "source_refs": [],
            "match_types": Counter(),
            "matched_keywords": set(),
        },
    )
    unit["source_refs"].append(source_ref)
    unit["match_types"][match_type] += 1
    unit["matched_keywords"].update(matched_keywords)
    if target_relevance == "core":
        unit["target_relevance"] = "core"


def phq8_severity(score: int) -> str:
    if score >= 20:
        return "severe"
    if score >= 15:
        return "moderately_severe"
    if score >= 10:
        return "moderate"
    if score >= 5:
        return "mild"
    return "minimal"


def build_profile(
    *,
    label: dict[str, Any],
    transcript_path: Path,
    turns: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_prefix: str,
    max_units_per_slot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    active_slots, slot_defs = schema_slots(schema)
    source_dataset = label["source_dataset"]
    participant_id = label["participant_id"]
    source_id = "daic_woz" if source_dataset == "DAIC-WoZ" else "extended_daic"
    case_id = f"{source_id}_{participant_id}"
    profile_id = f"{case_id}_dialogue_derived_profile"
    slot_units: dict[str, dict[str, dict[str, Any]]] = {}

    for turn in turns:
        patient_text = clean_text(turn.get("patient_utterance"))
        doctor_text = clean_text(turn.get("doctor_utterance"))
        question_matches = slot_matches(doctor_text, slot_defs) if doctor_text else {}
        patient_matches = slot_matches(patient_text, slot_defs)
        slots = set(question_matches) | set(patient_matches)
        if not slots:
            continue
        unit_text = unit_text_for_turn(turn, set(question_matches))
        if not should_keep_unit(unit_text, has_slot_context=bool(question_matches)):
            continue
        for slot in sorted(slots, key=lambda name: active_slots.index(name) if name in active_slots else 999):
            matched_keywords = patient_matches.get(slot) or question_matches.get(slot) or []
            match_type = "patient_keyword" if slot in patient_matches else "question_context"
            item_score = int((label.get("phq8_items") or {}).get(slot) or 0)
            relevance = "core" if item_score > 0 or match_type == "patient_keyword" else "supporting"
            source_ref = {
                "source_dataset": source_dataset,
                "participant_id": participant_id,
                "transcript_file": str(transcript_path),
                "turn_id": turn.get("turn_id"),
                "speaker": turn.get("speaker"),
                "doctor_utterance": doctor_text,
                "patient_utterance": patient_text,
                "source_span": {"start_char": 0, "end_char": len(patient_text)},
                "confidence": turn.get("confidence"),
                "match_type": match_type,
                "matched_keywords": matched_keywords,
            }
            add_unit(
                slot_units,
                slot=slot,
                unit_text=unit_text,
                source_ref=source_ref,
                match_type=match_type,
                matched_keywords=matched_keywords,
                target_relevance=relevance,
            )

    slot_profiles: dict[str, Any] = {}
    canonical_units: list[dict[str, Any]] = []
    surface_links: list[dict[str, Any]] = []
    observed_slots: list[str] = []
    relevance_rank = {"core": 0, "supporting": 1, "peripheral": 2}
    criticality_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    for slot in active_slots:
        raw_units = list(slot_units.get(slot, {}).values())
        raw_units.sort(
            key=lambda unit: (
                relevance_rank.get(unit.get("target_relevance"), 3),
                -len(unit.get("source_refs") or []),
                str(unit.get("unit_text") or ""),
            )
        )
        selected = raw_units[:max_units_per_slot]
        evidence_units: list[dict[str, Any]] = []
        slot_def = slot_defs[slot]
        criticality = str(slot_def.get("criticality") or "medium")
        for idx, unit in enumerate(selected, start=1):
            unit_id = f"{slot}_u{idx}"
            match_types = dict(unit["match_types"])
            source_refs = unit.get("source_refs") or []
            evidence_unit = {
                "unit_id": unit_id,
                "unit_text": unit["unit_text"],
                "target_relevance": unit.get("target_relevance", "supporting"),
                "source_count": len(source_refs),
                "source_refs": source_refs,
                "match_types": match_types,
                "matched_keywords": sorted(unit.get("matched_keywords") or []),
            }
            evidence_units.append(evidence_unit)
            canonical_unit_id = f"{slot}::{unit_id}"
            is_clinical_key = criticality_rank.get(criticality, 2) >= 2
            canonical_units.append(
                {
                    "profile_id": profile_id,
                    "case_id": case_id,
                    "tree_type": "daic_adult",
                    "tree_node": slot,
                    "dimension": "transcript_evidence",
                    "canonical_unit_id": canonical_unit_id,
                    "criticality": criticality,
                    "is_clinical_key": is_clinical_key,
                    "support_count": len(source_refs),
                    "supporting_variant_ids": [0],
                    "support_refs": source_refs,
                    "profile_surface_unit_ids": [unit_id],
                    "match_types": match_types or {"keyword": len(source_refs)},
                }
            )
            surface_links.append(
                {
                    "profile_id": profile_id,
                    "case_id": case_id,
                    "surface_unit_id": unit_id,
                    "canonical_unit_id": canonical_unit_id,
                    "tree_node": slot,
                    "dimension": "transcript_evidence",
                    "match_type": "keyword",
                    "matched_keywords": evidence_unit["matched_keywords"],
                }
            )
        slot_profiles[slot] = {
            "slot": slot,
            "evidence_units": evidence_units,
            "num_evidence_units": len(evidence_units),
            "num_unique_units_before_cap": len(raw_units),
            "support_turn_count": sum(len(unit.get("source_refs") or []) for unit in raw_units),
            "criticality": criticality,
            "evidence_unit_hints": slot_def.get("evidence_unit_hints") or [],
            "profile_status": "observed" if evidence_units else "empty",
            "phq8_item_score": int((label.get("phq8_items") or {}).get(slot) or 0),
        }
        if evidence_units:
            observed_slots.append(slot)

    phq8_binary = int(label.get("phq8_binary") or 0)
    profile = {
        "profile_id": profile_id,
        "case_id": case_id,
        "source_dataset": source_dataset,
        "source_dataset_family": "DAIC",
        "dataset_prefix": dataset_prefix,
        "language": "en",
        "profile_source": "daic_transcript_and_phq8_labels",
        "profile_type": "dialogue_derived_not_original_clinical_case_profile",
        "participant_id": participant_id,
        "split": label.get("split"),
        "diagnoses": ["PHQ8-positive depression screening"] if phq8_binary else ["PHQ8-negative depression screening"],
        "icd_codes": [],
        "binary_task_label": "depression" if phq8_binary else "non_depression",
        "phq8_binary": phq8_binary,
        "phq8_score": int(label.get("phq8_score") or 0),
        "phq8_severity": phq8_severity(int(label.get("phq8_score") or 0)),
        "phq8_items": label.get("phq8_items") or {},
        "label": label.get("label"),
        "ds_score": label.get("ds_score"),
        "gender": label.get("gender"),
        "primary_tree_type": "daic_adult",
        "tree_type_counts": {"daic_adult": len(turns)},
        "active_tree_slots": active_slots,
        "num_dialogue_variants_observed": 1,
        "dialogue_ids": [case_id],
        "variant_ids": [0],
        "num_mapped_turns_used": len(turns),
        "num_observed_slots": len(observed_slots),
        "observed_slots": observed_slots,
        "slot_profiles": slot_profiles,
        "construction_notes": [
            "DAIC-WoZ train/dev and Extended-DAIC test are converted into the same profile-grounded environment schema as MDD-5K.",
            "Evidence units are extracted from participant transcript text and mapped to English depression-screening slots with deterministic keyword rules.",
            "PHQ-8 labels are metadata for supervised evaluation/training; patient simulator evidence is transcript-grounded.",
            "This profile is a simulator hidden state, not a clinically validated diagnostic case record.",
        ],
    }
    return profile, canonical_units, surface_links


def group_rows_for_profile(profile: dict[str, Any], *, split_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in profile.get("observed_slots") or []:
        slot_profile = (profile.get("slot_profiles") or {}).get(slot) or {}
        count = int(slot_profile.get("num_evidence_units") or 0)
        if count <= 0:
            continue
        rows.append(
            {
                "counterfactual_group_id": f"{profile['profile_id']}_{slot}",
                "profile_id": profile["profile_id"],
                "case_id": profile["case_id"],
                "target_tree_node": slot,
                "target_node_role": "simulator_internal_target_node",
                "target_node_visibility": "simulator_internal_not_doctor_visible",
                "active_tree_type": profile.get("primary_tree_type"),
                "target_slot_evidence_unit_count": count,
                "binary_task_label": profile.get("binary_task_label"),
                "phq8_binary": profile.get("phq8_binary"),
                "phq8_score": profile.get("phq8_score"),
                "phq8_severity": profile.get("phq8_severity"),
                "source_dataset": profile.get("source_dataset"),
                "participant_id": profile.get("participant_id"),
                "resplit_source": split_source,
                "resplit_name": profile.get("split"),
            }
        )
    return rows


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    schema = load_json(args.schema)
    dataset_root = args.dataset_root
    train_labels = load_label_rows(
        args.train_labels or dataset_root / "train_split_Depression_AVEC2017.csv",
        split="train",
        source_dataset="DAIC-WoZ",
    )
    dev_labels = load_label_rows(
        args.dev_labels or dataset_root / "dev_split_Depression_AVEC2017.csv",
        split="dev",
        source_dataset="DAIC-WoZ",
    )
    extended_labels = load_label_rows(
        args.extended_labels or dataset_root / "Extend_PHQ8_Labels.csv",
        split="test",
        source_dataset="Extended-DAIC",
    )
    sources = [
        ("DAIC-WoZ", "train", train_labels, args.daic_woz_dir or dataset_root / DAIC_WOZ_DIRNAME, read_daic_woz_turns),
        ("DAIC-WoZ", "dev", dev_labels, args.daic_woz_dir or dataset_root / DAIC_WOZ_DIRNAME, read_daic_woz_turns),
        ("Extended-DAIC", "test", extended_labels, args.extended_daic_dir or dataset_root / EXTENDED_DAIC_DIRNAME, read_extended_turns),
    ]

    profiles: list[dict[str, Any]] = []
    canonical_units: list[dict[str, Any]] = []
    surface_links: list[dict[str, Any]] = []
    groups_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    missing: list[dict[str, Any]] = []

    for source_dataset, split, labels, transcript_dir, reader in sources:
        for participant_id, label in sorted(labels.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
            transcript_path = transcript_dir / f"{participant_id}_TRANSCRIPT.csv"
            if not transcript_path.exists():
                counters[f"{split}_missing_transcript"] += 1
                missing.append(
                    {
                        "source_dataset": source_dataset,
                        "split": split,
                        "participant_id": participant_id,
                        "expected_transcript": str(transcript_path),
                    }
                )
                continue
            turns = reader(transcript_path)
            if not turns:
                counters[f"{split}_empty_transcript"] += 1
                continue
            profile, units, links = build_profile(
                label=label,
                transcript_path=transcript_path,
                turns=turns,
                schema=schema,
                dataset_prefix=args.dataset_prefix,
                max_units_per_slot=args.max_units_per_slot,
            )
            profiles.append(profile)
            canonical_units.extend(units)
            surface_links.extend(links)
            profile_groups = group_rows_for_profile(profile, split_source="daic_woz_train_dev_extended_test_v1")
            groups_by_split[split].extend(profile_groups)
            counters[f"{split}_profiles"] += 1
            counters[f"{split}_groups"] += len(profile_groups)

    output_root = args.output_root
    profile_dir = output_root / "patient_profiles"
    split_dir = output_root / "profile_split"
    canonical_dir = output_root / "canonical_evidence"
    profile_path = profile_dir / f"{args.dataset_prefix}_dialogue_derived_patient_profiles.jsonl"
    write_jsonl(profile_path, profiles)
    for split in ("train", "dev", "test"):
        write_jsonl(
            split_dir / f"{args.dataset_prefix}_profile_grounded_environment_{split}_groups.jsonl",
            groups_by_split.get(split, []),
        )
    write_jsonl(canonical_dir / f"{args.dataset_prefix}_tree_aligned_canonical_evidence_units.jsonl", canonical_units)
    write_jsonl(canonical_dir / f"{args.dataset_prefix}_surface_to_canonical_evidence_links.jsonl", surface_links)

    profile_counts_by_split = Counter(str(profile.get("split")) for profile in profiles)
    label_counts = Counter(str(profile.get("binary_task_label")) for profile in profiles)
    observed_slot_counts = Counter()
    for profile in profiles:
        observed_slot_counts.update(profile.get("observed_slots") or [])
    summary = {
        "dataset_prefix": args.dataset_prefix,
        "dataset_root": str(dataset_root),
        "schema": str(args.schema),
        "profiles": len(profiles),
        "canonical_units": len(canonical_units),
        "surface_links": len(surface_links),
        "groups_by_split": {split: len(rows) for split, rows in sorted(groups_by_split.items())},
        "profiles_by_split": dict(sorted(profile_counts_by_split.items())),
        "binary_label_counts": dict(sorted(label_counts.items())),
        "observed_slot_counts": dict(observed_slot_counts.most_common()),
        "missing_inputs": missing,
        "counters": dict(counters),
        "paths": {
            "profiles": str(profile_path),
            "profile_split_dir": str(split_dir),
            "canonical_dir": str(canonical_dir),
        },
        "split_policy": {
            "train": "DAIC-WoZ train_split_Depression_AVEC2017.csv",
            "dev": "DAIC-WoZ dev_split_Depression_AVEC2017.csv",
            "test": "all Extended-DAIC rows from Extend_PHQ8_Labels.csv",
        },
        "label_policy": "Use PHQ8_Binary when provided; otherwise derive binary label as PHQ8 total >= 10. Store PHQ8 score/items as metadata.",
    }
    write_json(profile_dir / f"{args.dataset_prefix}_dialogue_derived_patient_profile_summary.json", summary)
    write_json(canonical_dir / f"{args.dataset_prefix}_tree_aligned_canonical_evidence_summary.json", summary)
    write_readme(output_root / "README.md", summary)
    return summary


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DAIC Profile-Grounded Environment",
        "",
        "This directory is generated by `scripts/build_daic_profile_environment.py`.",
        "",
        "Split policy:",
        "",
        "- `train`: DAIC-WoZ train split.",
        "- `dev`: DAIC-WoZ dev split.",
        "- `test`: all Extended-DAIC rows.",
        "",
        "Generated files:",
        "",
        f"- `patient_profiles/{summary['dataset_prefix']}_dialogue_derived_patient_profiles.jsonl`",
        f"- `profile_split/{summary['dataset_prefix']}_profile_grounded_environment_train_groups.jsonl`",
        f"- `profile_split/{summary['dataset_prefix']}_profile_grounded_environment_dev_groups.jsonl`",
        f"- `profile_split/{summary['dataset_prefix']}_profile_grounded_environment_test_groups.jsonl`",
        f"- `canonical_evidence/{summary['dataset_prefix']}_tree_aligned_canonical_evidence_units.jsonl`",
        f"- `canonical_evidence/{summary['dataset_prefix']}_surface_to_canonical_evidence_links.jsonl`",
        "",
        "These artifacts contain transcript-derived participant text. Keep the original DAIC license and release restrictions in mind before publishing.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DAIC-WoZ train/dev and Extended-DAIC test artifacts for the profile-grounded active-reasoning pipeline."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--daic-woz-dir", type=Path, default=None)
    parser.add_argument("--extended-daic-dir", type=Path, default=None)
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--dev-labels", type=Path, default=None)
    parser.add_argument("--extended-labels", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--max-units-per-slot", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_all(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
