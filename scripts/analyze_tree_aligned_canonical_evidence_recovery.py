from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DIR = BASE_DIR / "outputs_tree_aligned_canonical_evidence_20260629"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_tree_aligned_canonical_recovery_20260629"


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


def load_canonical_units(
    path: Path,
    *,
    strict_supported_match_types: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    all_supported: dict[str, set[str]] = defaultdict(set)
    keyword_supported: dict[str, set[str]] = defaultdict(set)
    for unit in iter_jsonl(path):
        if not unit.get("is_clinical_key"):
            continue
        profile_id = str(unit["profile_id"])
        canonical_unit_id = str(unit["canonical_unit_id"])
        all_supported[profile_id].add(canonical_unit_id)
        match_types = unit.get("match_types") or {}
        if any(int(match_types.get(match_type) or 0) > 0 for match_type in strict_supported_match_types):
            keyword_supported[profile_id].add(canonical_unit_id)
    return all_supported, keyword_supported


def load_surface_links(
    path: Path,
    *,
    strict_supported_match_types: set[str],
) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
    all_links: dict[tuple[str, str], set[str]] = defaultdict(set)
    keyword_links: dict[tuple[str, str], set[str]] = defaultdict(set)
    for link in iter_jsonl(path):
        profile_id = str(link["profile_id"])
        surface_unit_id = str(link["surface_unit_id"])
        canonical_unit_id = str(link["canonical_unit_id"])
        key = (profile_id, surface_unit_id)
        all_links[key].add(canonical_unit_id)
        if link.get("match_type") in strict_supported_match_types:
            keyword_links[key].add(canonical_unit_id)
    return all_links, keyword_links


def get_slot(record: dict[str, Any]) -> str:
    slot = record.get("target_tree_node")
    if slot:
        return str(slot)
    query_interpreter = record.get("query_interpreter")
    if isinstance(query_interpreter, dict):
        slot = query_interpreter.get("target_tree_node") or query_interpreter.get("simulator_internal_target_node")
        if slot:
            return str(slot)
    return "UNMAPPED"


def is_unmapped(record: dict[str, Any]) -> bool:
    query_interpreter = record.get("query_interpreter")
    if isinstance(query_interpreter, dict) and query_interpreter.get("query_interpreter_status") == "mapped":
        return False
    return get_slot(record) == "UNMAPPED"


def analyze_records(
    *,
    label: str,
    records_path: Path,
    denominator_by_profile: dict[str, set[str]],
    surface_to_canonical: dict[tuple[str, str], set[str]],
    metric_name: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(records_path):
        scenario_id = str(record.get("scenario_id") or "")
        if not scenario_id:
            scenario_id = "::".join(
                [
                    str(record.get("profile_id") or ""),
                    str(record.get("base_severity") or ""),
                    label,
                ]
            )
        groups[scenario_id].append(record)

    rows: list[dict[str, Any]] = []
    for scenario_id, records in groups.items():
        records = sorted(records, key=lambda item: int(item.get("turn_index") or 0))
        first = records[0]
        profile_id = str(first.get("profile_id") or "")
        denominator = denominator_by_profile.get(profile_id) or set()
        if not denominator:
            continue

        recovered: dict[str, float] = {}
        for record in records:
            for unit_id in record.get("retained_profile_unit_ids") or []:
                for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
                    if canonical_unit_id in denominator:
                        recovered[canonical_unit_id] = max(recovered.get(canonical_unit_id, 0.0), 1.0)
            for unit_id in record.get("weakened_profile_unit_ids") or []:
                for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
                    if canonical_unit_id in denominator:
                        recovered[canonical_unit_id] = max(recovered.get(canonical_unit_id, 0.0), 0.5)

        final_s = sum(recovered.get(unit_id, 0.0) for unit_id in denominator) / len(denominator)
        rows.append(
            {
                "source_label": label,
                "metric_name": metric_name,
                "scenario_id": scenario_id,
                "profile_id": profile_id,
                "base_severity": first.get("base_severity"),
                "turns": len(records),
                "canonical_denominator_count": len(denominator),
                "recovered_canonical_weight": round(sum(recovered.get(unit_id, 0.0) for unit_id in denominator), 6),
                "tree_aligned_canonical_final_s": round(final_s, 6),
                "unmapped_rate": round(sum(1 for record in records if is_unmapped(record)) / len(records), 6),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["source_label"]),
                str(row["metric_name"]),
                str(row["base_severity"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (label, metric_name, severity), severity_rows in sorted(grouped.items()):
        summary.append(
            {
                "source_label": label,
                "metric_name": metric_name,
                "base_severity": severity,
                "n_scenarios": len(severity_rows),
                "mean_turns": round(mean(float(row["turns"]) for row in severity_rows), 4),
                "mean_canonical_denominator_count": round(
                    mean(float(row["canonical_denominator_count"]) for row in severity_rows), 4
                ),
                "mean_tree_aligned_canonical_final_s": round(
                    mean(float(row["tree_aligned_canonical_final_s"]) for row in severity_rows), 6
                ),
                "mean_recovered_canonical_weight": round(
                    mean(float(row["recovered_canonical_weight"]) for row in severity_rows), 6
                ),
                "mean_unmapped_rate": round(mean(float(row["unmapped_rate"]) for row in severity_rows), 6),
            }
        )
    return summary


def parse_record_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH for --records.")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze recovery of tree-aligned canonical evidence units.")
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--canonical-prefix", default="mdd5k")
    parser.add_argument(
        "--strict-supported-match-types",
        nargs="+",
        default=["keyword", "phq8_label_anchor"],
        help=(
            "Match types included in the strict denominator/link variant. "
            "The default preserves MDD keyword support and also supports DAIC PHQ-8 label anchors."
        ),
    )
    parser.add_argument("--records", action="append", type=parse_record_arg, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    strict_supported_match_types = set(args.strict_supported_match_types)
    canonical_units_path = args.canonical_dir / f"{args.canonical_prefix}_tree_aligned_canonical_evidence_units.jsonl"
    surface_links_path = args.canonical_dir / f"{args.canonical_prefix}_surface_to_canonical_evidence_links.jsonl"
    all_denominator, keyword_denominator = load_canonical_units(
        canonical_units_path,
        strict_supported_match_types=strict_supported_match_types,
    )
    all_links, keyword_links = load_surface_links(
        surface_links_path,
        strict_supported_match_types=strict_supported_match_types,
    )

    rows: list[dict[str, Any]] = []
    for label, path in args.records:
        rows.extend(
            analyze_records(
                label=label,
                records_path=path,
                denominator_by_profile=all_denominator,
                surface_to_canonical=all_links,
                metric_name="all_supported",
            )
        )
        rows.extend(
            analyze_records(
                label=label,
                records_path=path,
                denominator_by_profile=keyword_denominator,
                surface_to_canonical=keyword_links,
                metric_name="keyword_supported_only",
            )
        )

    result = {
        "reference_definition": "tree-aligned canonical evidence units with observed support spans",
        "canonical_prefix": args.canonical_prefix,
        "canonical_dir": str(args.canonical_dir),
        "strict_supported_match_types": sorted(strict_supported_match_types),
        "metric_variants": {
            "all_supported": "includes keyword-aligned and fallback-aligned canonical units",
            "keyword_supported_only": "stricter; includes canonical units with configured strict support match types, including PHQ-8 label anchors for DAIC",
        },
        "results": summarize(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "tree_aligned_canonical_evidence_recovery_rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    write_json(args.output_dir / "tree_aligned_canonical_evidence_recovery_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
