from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DIR = BASE_DIR / "data" / "tree_aligned_canonical_evidence"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_action_value_data"
DEFAULT_DATASET_PREFIX = "mdd5k"


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("patient_verify_hard_error")
        or row.get("patient_hard_error")
        or row.get("hard_error")
        or row.get("has_forbidden_leak")
        or row.get("forbidden_evidence_leak")
    )


def load_canonical_units(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    all_supported: dict[str, set[str]] = defaultdict(set)
    keyword_supported: dict[str, set[str]] = defaultdict(set)
    for unit in iter_jsonl(path):
        if not unit.get("is_clinical_key"):
            continue
        profile_id = str(unit["profile_id"])
        canonical_unit_id = str(unit["canonical_unit_id"])
        all_supported[profile_id].add(canonical_unit_id)
        match_types = unit.get("match_types") or {}
        if int(match_types.get("keyword") or 0) > 0:
            keyword_supported[profile_id].add(canonical_unit_id)
    return all_supported, keyword_supported


def load_surface_links(path: Path) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
    all_links: dict[tuple[str, str], set[str]] = defaultdict(set)
    keyword_links: dict[tuple[str, str], set[str]] = defaultdict(set)
    for link in iter_jsonl(path):
        profile_id = str(link["profile_id"])
        surface_unit_id = str(link["surface_unit_id"])
        canonical_unit_id = str(link["canonical_unit_id"])
        key = (profile_id, surface_unit_id)
        all_links[key].add(canonical_unit_id)
        if link.get("match_type") == "keyword":
            keyword_links[key].add(canonical_unit_id)
    return all_links, keyword_links


def select_metric_maps(
    canonical_dir: Path,
    metric_name: str,
    dataset_prefix: str = DEFAULT_DATASET_PREFIX,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    units_path = canonical_dir / f"{dataset_prefix}_tree_aligned_canonical_evidence_units.jsonl"
    links_path = canonical_dir / f"{dataset_prefix}_surface_to_canonical_evidence_links.jsonl"
    all_denominator, keyword_denominator = load_canonical_units(units_path)
    all_links, keyword_links = load_surface_links(links_path)
    if metric_name == "all_supported":
        return all_denominator, all_links
    if metric_name == "keyword_supported_only":
        return keyword_denominator, keyword_links
    raise ValueError(f"Unsupported metric_name={metric_name!r}")


def canonical_from_surface_units(
    *,
    profile_id: str,
    unit_ids: list[Any],
    denominator: set[str],
    surface_to_canonical: dict[tuple[str, str], set[str]],
    weight: float,
) -> dict[str, float]:
    gained: dict[str, float] = {}
    for unit_id in unit_ids:
        for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
            if canonical_unit_id in denominator:
                gained[canonical_unit_id] = max(gained.get(canonical_unit_id, 0.0), weight)
    return gained


def merge_gain(target: dict[str, float], gain: dict[str, float]) -> None:
    for unit_id, value in gain.items():
        target[unit_id] = max(target.get(unit_id, 0.0), value)


def record_gain(
    record: dict[str, Any],
    *,
    profile_id: str,
    denominator: set[str],
    surface_to_canonical: dict[tuple[str, str], set[str]],
) -> dict[str, float]:
    gained: dict[str, float] = {}
    merge_gain(
        gained,
        canonical_from_surface_units(
            profile_id=profile_id,
            unit_ids=list(record.get("retained_profile_unit_ids") or []),
            denominator=denominator,
            surface_to_canonical=surface_to_canonical,
            weight=1.0,
        ),
    )
    merge_gain(
        gained,
        canonical_from_surface_units(
            profile_id=profile_id,
            unit_ids=list(record.get("weakened_profile_unit_ids") or []),
            denominator=denominator,
            surface_to_canonical=surface_to_canonical,
            weight=0.5,
        ),
    )
    return gained


def recovered_before_state(
    record: dict[str, Any],
    *,
    profile_id: str,
    denominator: set[str],
    surface_to_canonical: dict[tuple[str, str], set[str]],
) -> dict[str, float]:
    state = record.get("controller_state_before_replay") or record.get("controller_state_before") or {}
    by_slot = state.get("disclosed_profile_unit_ids_by_slot") or {}
    recovered: dict[str, float] = {}
    if isinstance(by_slot, dict):
        for unit_ids in by_slot.values():
            if isinstance(unit_ids, list):
                merge_gain(
                    recovered,
                    canonical_from_surface_units(
                        profile_id=profile_id,
                        unit_ids=unit_ids,
                        denominator=denominator,
                        surface_to_canonical=surface_to_canonical,
                        weight=1.0,
                    ),
                )
    return recovered


def weighted_sum(values: dict[str, float], denominator: set[str]) -> float:
    return sum(values.get(unit_id, 0.0) for unit_id in denominator)


def visible_history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no visible dialogue history)"
    lines: list[str] = []
    for idx, turn in enumerate(history[-12:], start=max(1, len(history) - 11)):
        doctor = clean_text(turn.get("doctor") or turn.get("doctor_utterance"))
        patient = clean_text(turn.get("patient") or turn.get("patient_utterance"))
        lines.append(f"Turn {idx} Doctor: {doctor}")
        lines.append(f"Turn {idx} Patient: {patient}")
    return "\n".join(lines)


def build_value_model_input(first: dict[str, Any]) -> str:
    history = first.get("dialogue_history") or (first.get("source_state_identity") or {}).get("visible_history") or []
    return (
        "Visible dialogue history before candidate action:\n"
        f"{visible_history_text(history)}\n\n"
        f"Candidate doctor question:\n{clean_text(first.get('doctor_question'))}\n\n"
        f"Observed patient response:\n{clean_text(first.get('patient_response'))}"
    )


def branch_key(record: dict[str, Any]) -> tuple[str, str]:
    state_id = str(record.get("source_state_id") or record.get("scenario_id") or "")
    candidate_index = str(record.get("candidate_index"))
    if candidate_index in {"", "None"}:
        candidate_index = str(record.get("request_id") or record.get("record_id") or "")
    return state_id, candidate_index


def sort_branch_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            int(row.get("turn_index") or 0),
            str(row.get("record_id") or ""),
        ),
    )


def build_rows(
    *,
    records_path: Path,
    denominator_by_profile: dict[str, set[str]],
    surface_to_canonical: dict[tuple[str, str], set[str]],
    metric_name: str,
    require_verified: bool,
    max_branch_records: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for record in iter_jsonl(records_path):
        counters["records_seen"] += 1
        mode = str(record.get("patient_realizer_mode") or "")
        if require_verified and mode != "verified_llm_cache":
            counters["skip_non_verified_patient"] += 1
            continue
        if hard_error(record):
            counters["skip_hard_error"] += 1
            continue
        question = clean_text(record.get("doctor_question"))
        patient = clean_text(record.get("patient_response"))
        if not question or not patient:
            counters["skip_empty_question_or_response"] += 1
            continue
        state_id, candidate_index = branch_key(record)
        if not state_id or not candidate_index:
            counters["skip_missing_branch_key"] += 1
            continue
        grouped[(state_id, candidate_index)].append(record)

    rows: list[dict[str, Any]] = []
    target_values: list[float] = []
    states: set[str] = set()
    candidates_by_state: Counter[str] = Counter()
    for (state_id, candidate_index), branch_records in sorted(grouped.items()):
        branch_records = sort_branch_records(branch_records)
        if max_branch_records is not None and max_branch_records > 0:
            branch_records = branch_records[:max_branch_records]
        if not branch_records:
            continue
        first = branch_records[0]
        profile_id = str(first.get("profile_id") or "")
        denominator = denominator_by_profile.get(profile_id) or set()
        if not denominator:
            counters["skip_no_denominator"] += 1
            continue

        before = recovered_before_state(
            first,
            profile_id=profile_id,
            denominator=denominator,
            surface_to_canonical=surface_to_canonical,
        )
        running = dict(before)
        before_weight = weighted_sum(before, denominator)

        first_gain = record_gain(
            first,
            profile_id=profile_id,
            denominator=denominator,
            surface_to_canonical=surface_to_canonical,
        )
        immediate_running = dict(running)
        merge_gain(immediate_running, first_gain)
        immediate_gain = max(0.0, weighted_sum(immediate_running, denominator) - before_weight) / len(denominator)

        future_records: list[dict[str, Any]] = []
        for record in branch_records:
            merge_gain(
                running,
                record_gain(
                    record,
                    profile_id=profile_id,
                    denominator=denominator,
                    surface_to_canonical=surface_to_canonical,
                ),
            )
            future_records.append(
                {
                    "turn_index": record.get("turn_index"),
                    "response_type": record.get("response_type"),
                    "target_tree_node": record.get("target_tree_node"),
                    "delta_cumulative_slot_sufficiency": record.get("delta_cumulative_slot_sufficiency"),
                    "patient_realizer_mode": record.get("patient_realizer_mode"),
                }
            )

        total_gain = max(0.0, weighted_sum(running, denominator) - before_weight) / len(denominator)
        residual_after_first = max(0.0, total_gain - immediate_gain)
        row = {
            "record_id": f"{state_id}::candidate_{candidate_index}",
            "state_id": state_id,
            "profile_id": profile_id,
            "case_id": first.get("case_id"),
            "base_severity": first.get("base_severity"),
            "prefix_name": f"turn_{first.get('turn_index')}",
            "candidate_action": first.get("candidate_method") or first.get("question_type") or "same_state_candidate_question",
            "previous_low_info": bool((first.get("controller_state_before_replay") or {}).get("prior_boundary_refusal_by_slot")),
            "prior_boundary_for_target": bool(first.get("prior_boundary_refusal")),
            "first_response": {
                "response_type": first.get("response_type"),
                "doctor_recovery_quality": first.get("doctor_recovery_quality"),
                "patient_realizer_mode": first.get("patient_realizer_mode"),
            },
            "immediate_target_gain": round(immediate_gain, 6),
            "immediate_any_gain": round(immediate_gain, 6),
            "future_target_gain": round(residual_after_first, 6),
            "future_any_gain": 0.0,
            "action_value_total_gain": round(total_gain, 6),
            "target_sufficiency_before_action": round(before_weight / len(denominator), 6),
            "target_sufficiency_after_branch": round(weighted_sum(running, denominator) / len(denominator), 6),
            "delta_readiness_final": round(
                safe_float(branch_records[-1].get("disclosure_readiness_after"))
                - safe_float(first.get("disclosure_readiness_before")),
                6,
            ),
            "future_records": future_records,
            "value_model_input": build_value_model_input(first),
            "metadata": {
                "source_records": str(records_path),
                "source_state_id": state_id,
                "candidate_index": candidate_index,
                "source_record_id": first.get("record_id"),
                "metric_name": metric_name,
                "canonical_denominator_count": len(denominator),
                "branch_record_count": len(branch_records),
                "turn_index": first.get("turn_index"),
                "target_tree_node_for_audit_only": first.get("target_tree_node"),
                "request_id": first.get("request_id"),
                "target_definition": "action_value_total_gain = immediate canonical gain plus future branch canonical gain",
            },
        }
        rows.append(row)
        states.add(state_id)
        candidates_by_state[state_id] += 1
        target_values.append(total_gain)

    counters["branches_seen"] = len(grouped)
    counters["examples_built"] = len(rows)
    summary: dict[str, Any] = {
        "counters": dict(counters),
        "states": len(states),
        "candidate_count_distribution": {
            str(k): v for k, v in sorted(Counter(candidates_by_state.values()).items())
        },
    }
    if target_values:
        summary.update(
            {
                "action_value_mean": round(sum(target_values) / len(target_values), 6),
                "action_value_min": round(min(target_values), 6),
                "action_value_max": round(max(target_values), 6),
            }
        )
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build same-state action-value data for the final-patient value model. "
            "The output schema is compatible with train_final_patient_rfv_value_model.py."
        )
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--metric-name", choices=["keyword_supported_only", "all_supported"], default="keyword_supported_only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--max-branch-records", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    denominator_by_profile, surface_to_canonical = select_metric_maps(
        args.canonical_dir,
        args.metric_name,
        dataset_prefix=args.dataset_prefix,
    )
    rows, build_summary = build_rows(
        records_path=args.records,
        denominator_by_profile=denominator_by_profile,
        surface_to_canonical=surface_to_canonical,
        metric_name=args.metric_name,
        require_verified=not args.allow_non_verified,
        max_branch_records=args.max_branch_records if args.max_branch_records > 0 else None,
    )
    record_path = args.output_dir / "final_patient_action_value_records.jsonl"
    write_jsonl(record_path, rows)
    summary = {
        "settings": {
            "records": str(args.records),
            "canonical_dir": str(args.canonical_dir),
            "dataset_prefix": args.dataset_prefix,
            "metric_name": args.metric_name,
            "require_verified": not args.allow_non_verified,
            "max_branch_records": args.max_branch_records,
        },
        **build_summary,
        "records_built": len(rows),
        "record_path": str(record_path),
        "target_definition": (
            "same-state candidate action value: immediate canonical evidence gain "
            "plus residual future branch canonical evidence gain"
        ),
    }
    write_json(args.output_dir / "final_patient_action_value_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
