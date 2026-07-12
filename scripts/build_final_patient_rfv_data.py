from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DIR = BASE_DIR / "outputs_tree_aligned_canonical_evidence_20260629"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_rfv_data"
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


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Use LABEL=OUTPUT_DIR for --source.")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Empty source label.")
    return label, Path(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("patient_verify_hard_error")
        or row.get("patient_hard_error")
        or row.get("hard_error")
    )


def record_gain(
    record: dict[str, Any],
    *,
    profile_id: str,
    denominator: set[str],
    surface_to_canonical: dict[tuple[str, str], set[str]],
) -> dict[str, float]:
    gained: dict[str, float] = {}
    for unit_id in record.get("retained_profile_unit_ids") or []:
        for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
            if canonical_unit_id in denominator:
                gained[canonical_unit_id] = max(gained.get(canonical_unit_id, 0.0), 1.0)
    for unit_id in record.get("weakened_profile_unit_ids") or []:
        for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
            if canonical_unit_id in denominator:
                gained[canonical_unit_id] = max(gained.get(canonical_unit_id, 0.0), 0.5)
    return gained


def weighted_sum(values: dict[str, float], denominator: set[str]) -> float:
    return sum(values.get(unit_id, 0.0) for unit_id in denominator)


def visible_history_text(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no visible dialogue history)"
    lines: list[str] = []
    for idx, turn in enumerate(history[-12:], start=1):
        lines.append(f"Turn {idx} Doctor: {turn.get('doctor_utterance', '')}")
        lines.append(f"Turn {idx} Patient: {turn.get('patient_utterance', '')}")
    return "\n".join(lines)


def build_value_model_input(history: list[dict[str, str]], doctor_question: str, patient_response: str) -> str:
    return (
        "Visible dialogue history before candidate action:\n"
        f"{visible_history_text(history)}\n\n"
        f"Candidate doctor question:\n{doctor_question}\n\n"
        f"Observed patient response:\n{patient_response}"
    )


def source_records_path(output_dir: Path, dataset_prefix: str = DEFAULT_DATASET_PREFIX) -> Path:
    return output_dir / f"{dataset_prefix}_llm_doctor_online_replay_records.jsonl"


def build_rows_for_source(
    *,
    label: str,
    output_dir: Path,
    denominator_by_profile: dict[str, set[str]],
    surface_to_canonical: dict[tuple[str, str], set[str]],
    metric_name: str,
    dataset_prefix: str,
    require_verified: bool,
    max_turn_index: int | None,
    min_final_score: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_path = source_records_path(output_dir, dataset_prefix=dataset_prefix)
    if not records_path.exists():
        raise FileNotFoundError(records_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for record in iter_jsonl(records_path):
        counters["records_seen"] += 1
        if require_verified and record.get("patient_realizer_mode") != "verified_llm_cache":
            counters["skip_non_verified_patient"] += 1
            continue
        if hard_error(record):
            counters["skip_hard_error"] += 1
            continue
        scenario_id = str(record.get("scenario_id") or "")
        if not scenario_id:
            counters["skip_missing_scenario"] += 1
            continue
        grouped[scenario_id].append(record)

    rows: list[dict[str, Any]] = []
    target_values: list[float] = []
    immediate_values: list[float] = []
    for scenario_id, records in sorted(grouped.items()):
        records = sorted(records, key=lambda item: int(item.get("turn_index") or 0))
        if not records:
            continue
        profile_id = str(records[0].get("profile_id") or "")
        denominator = denominator_by_profile.get(profile_id) or set()
        if not denominator:
            counters["skip_no_denominator"] += len(records)
            continue

        cumulative_before: list[dict[str, float]] = []
        cumulative_after: list[dict[str, float]] = []
        running: dict[str, float] = {}
        immediate_gains: list[float] = []
        for record in records:
            before = dict(running)
            gained = record_gain(
                record,
                profile_id=profile_id,
                denominator=denominator,
                surface_to_canonical=surface_to_canonical,
            )
            for unit_id, weight in gained.items():
                running[unit_id] = max(running.get(unit_id, 0.0), weight)
            after = dict(running)
            immediate = (weighted_sum(after, denominator) - weighted_sum(before, denominator)) / len(denominator)
            immediate_gains.append(round(immediate, 6))
            cumulative_before.append(before)
            cumulative_after.append(after)

        final_weight = weighted_sum(running, denominator)
        final_score = final_weight / len(denominator)
        if min_final_score is not None and final_score < min_final_score:
            counters["skip_low_final_score"] += len(records)
            continue

        history: list[dict[str, str]] = []
        for idx, record in enumerate(records):
            turn_index = record.get("turn_index")
            if max_turn_index is not None and isinstance(turn_index, int) and turn_index > max_turn_index:
                counters["skip_after_max_turn"] += 1
                question = str(record.get("doctor_question") or "").strip()
                patient = str(record.get("patient_response") or "").strip()
                if question or patient:
                    history.append({"doctor_utterance": question, "patient_utterance": patient})
                continue

            question = str(record.get("doctor_question") or "").strip()
            patient = str(record.get("patient_response") or "").strip()
            if not question or not patient:
                counters["skip_empty_question_or_response"] += 1
                continue
            after_weight = weighted_sum(cumulative_after[idx], denominator)
            residual_future = max(0.0, (final_weight - after_weight) / len(denominator))
            immediate = immediate_gains[idx]
            future_records = [
                {
                    "turn_index": future.get("turn_index"),
                    "response_type": future.get("response_type"),
                    "target_tree_node": future.get("target_tree_node"),
                    "delta_cumulative_slot_sufficiency": future.get("delta_cumulative_slot_sufficiency"),
                }
                for future in records[idx + 1 :]
            ]
            record_id = str(record.get("record_id") or f"{scenario_id}::turn_{turn_index}")
            row = {
                "record_id": f"{label}::{record_id}",
                "state_id": f"{label}::{scenario_id}::turn_{turn_index}",
                "profile_id": profile_id,
                "case_id": record.get("case_id"),
                "base_severity": record.get("base_severity"),
                "prefix_name": f"turn_{turn_index}",
                "candidate_action": record.get("question_type") or "llm_generated_question",
                "previous_low_info": bool(idx > 0 and str(records[idx - 1].get("response_type") or "") in {
                    "vague_uncertain",
                    "topic_deflection",
                    "boundary_refusal",
                    "no_profile_evidence",
                    "unmapped_question",
                }),
                "prior_boundary_for_target": bool(record.get("prior_boundary_refusal")),
                "first_response": {
                    "response_type": record.get("response_type"),
                    "doctor_recovery_quality": record.get("doctor_recovery_quality"),
                    "patient_realizer_mode": record.get("patient_realizer_mode"),
                },
                "immediate_target_gain": immediate,
                "immediate_any_gain": immediate,
                "future_target_gain": round(residual_future, 6),
                "future_any_gain": 0.0,
                "target_sufficiency_after_first": round(after_weight / len(denominator), 6),
                "delta_readiness_final": round(
                    safe_float(records[-1].get("disclosure_readiness_after"))
                    - safe_float(record.get("disclosure_readiness_after")),
                    6,
                ),
                "future_records": future_records,
                "value_model_input": build_value_model_input(history, question, patient),
                "metadata": {
                    "source_label": label,
                    "source_output_dir": str(output_dir),
                    "scenario_id": scenario_id,
                    "metric_name": metric_name,
                    "final_recovery_score": round(final_score, 6),
                    "canonical_denominator_count": len(denominator),
                    "turn_index": turn_index,
                    "target_tree_node_for_audit_only": record.get("target_tree_node"),
                    "request_id": record.get("request_id"),
                },
            }
            rows.append(row)
            target_values.append(round(residual_future, 6))
            immediate_values.append(immediate)
            history.append({"doctor_utterance": question, "patient_utterance": patient})

        counters["scenarios_used"] += 1

    counters["examples_built"] = len(rows)
    summary = dict(counters)
    if target_values:
        summary.update(
            {
                "target_mean": round(sum(target_values) / len(target_values), 6),
                "target_min": round(min(target_values), 6),
                "target_max": round(max(target_values), 6),
                "immediate_gain_mean": round(sum(immediate_values) / len(immediate_values), 6),
            }
        )
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final-patient residual future value records from online verified trajectories."
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--metric-name", choices=["keyword_supported_only", "all_supported"], default="keyword_supported_only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--max-turn-index", type=int, default=None)
    parser.add_argument("--min-final-score", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    denominator_by_profile, surface_to_canonical = select_metric_maps(
        args.canonical_dir,
        args.metric_name,
        dataset_prefix=args.dataset_prefix,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    seen_ids: set[str] = set()
    for label, output_dir in args.source:
        rows, summary = build_rows_for_source(
            label=label,
            output_dir=output_dir,
            denominator_by_profile=denominator_by_profile,
            surface_to_canonical=surface_to_canonical,
            metric_name=args.metric_name,
            dataset_prefix=args.dataset_prefix,
            require_verified=not args.allow_non_verified,
            max_turn_index=args.max_turn_index,
            min_final_score=args.min_final_score,
        )
        source_summaries[label] = summary
        for row in rows:
            if row["record_id"] in seen_ids:
                continue
            seen_ids.add(row["record_id"])
            all_rows.append(row)

    record_path = args.output_dir / "final_patient_rfv_value_records.jsonl"
    write_jsonl(record_path, all_rows)
    summary = {
        "sources": {label: str(path) for label, path in args.source},
        "source_summaries": source_summaries,
        "settings": {
            "canonical_dir": str(args.canonical_dir),
            "dataset_prefix": args.dataset_prefix,
            "metric_name": args.metric_name,
            "require_verified": not args.allow_non_verified,
            "max_turn_index": args.max_turn_index,
            "min_final_score": args.min_final_score,
        },
        "records": len(all_rows),
        "record_path": str(record_path),
        "target_definition": "future residual tree-aligned canonical evidence recovery after the current turn",
    }
    write_json(args.output_dir / "final_patient_rfv_value_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
