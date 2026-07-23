from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BASE_DIR / "outputs_remote_baseline_20260712"
DEFAULT_CANONICAL_DIR = BASE_DIR / "data" / "tree_aligned_canonical_evidence"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "baseline_evidence_analysis"

MODELS = [
    "closed_evidence",
    "qwen_base",
    "qwen_sft_r16",
    "qwen_grpo_v6_300",
    "qwen_grpo_v6_full1500",
    "qwen_valueaug_full1500",
    "qwen_grpo_rfv2_ckpt1600",
]

MODEL_ORDER = {model: idx for idx, model in enumerate(MODELS)}
SEVERITY_ORDER = {
    "mild_low_info": 0,
    "moderate_low_info": 1,
    "severe_low_info": 2,
}


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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


def update_recovered(
    recovered: dict[str, float],
    *,
    record: dict[str, Any],
    profile_id: str,
    denominator: set[str],
    surface_to_canonical: dict[tuple[str, str], set[str]],
) -> None:
    for unit_id in record.get("retained_profile_unit_ids") or []:
        for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
            if canonical_unit_id in denominator:
                recovered[canonical_unit_id] = max(recovered.get(canonical_unit_id, 0.0), 1.0)
    for unit_id in record.get("weakened_profile_unit_ids") or []:
        for canonical_unit_id in surface_to_canonical.get((profile_id, str(unit_id)), set()):
            if canonical_unit_id in denominator:
                recovered[canonical_unit_id] = max(recovered.get(canonical_unit_id, 0.0), 0.5)


def recovery_score(recovered: dict[str, float], denominator: set[str]) -> float:
    if not denominator:
        return 0.0
    return sum(recovered.get(unit_id, 0.0) for unit_id in denominator) / len(denominator)


def audit_records(records_path: Path) -> dict[str, Any]:
    modes: Counter[str] = Counter()
    hard_errors = 0
    max_turn = None
    scenario_ids: set[str] = set()
    severity_counts: Counter[str] = Counter()
    if not records_path.exists():
        return {
            "records": 0,
            "n_scenarios": 0,
            "severity_counts": {},
            "patient_realizer_mode_counts": {},
            "verified_only": False,
            "hard_error_rows": 0,
            "max_turn_index": None,
        }
    for row in iter_jsonl(records_path):
        scenario_ids.add(str(row.get("scenario_id") or ""))
        severity_counts[str(row.get("base_severity"))] += 1
        modes[str(row.get("patient_realizer_mode"))] += 1
        if row.get("patient_verify_hard_error") or row.get("patient_hard_error") or row.get("hard_error"):
            hard_errors += 1
        turn_index = row.get("turn_index")
        if isinstance(turn_index, int):
            max_turn = turn_index if max_turn is None else max(max_turn, turn_index)
    return {
        "records": line_count(records_path) or 0,
        "n_scenarios": len(scenario_ids),
        "severity_counts": dict(severity_counts),
        "patient_realizer_mode_counts": dict(modes),
        "verified_only": bool(modes) and set(modes) == {"verified_llm_cache"},
        "hard_error_rows": hard_errors,
        "max_turn_index": max_turn,
    }


def group_records(records_path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(records_path):
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            scenario_id = "::".join(
                [
                    str(row.get("profile_id") or ""),
                    str(row.get("base_severity") or ""),
                    "missing_scenario_id",
                ]
            )
        groups[scenario_id].append(row)
    return groups


def analyze_records(
    *,
    records_path: Path,
    turn_budget: int,
    denominator_by_profile: dict[str, set[str]],
    surface_to_canonical: dict[tuple[str, str], set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    groups = group_records(records_path)
    for scenario_id, records in groups.items():
        records = sorted(records, key=lambda item: int(item.get("turn_index") or 0))
        if not records:
            continue
        first = records[0]
        profile_id = str(first.get("profile_id") or "")
        severity = str(first.get("base_severity") or "")
        denominator = denominator_by_profile.get(profile_id) or set()
        if not denominator:
            continue

        recovered: dict[str, float] = {}
        score_by_turn: dict[int, float] = {}
        observed_turns: set[int] = set()
        for record in records:
            turn_index = int(record.get("turn_index") or 0)
            observed_turns.add(turn_index)
            update_recovered(
                recovered,
                record=record,
                profile_id=profile_id,
                denominator=denominator,
                surface_to_canonical=surface_to_canonical,
            )
            score_by_turn[turn_index] = recovery_score(recovered, denominator)

        final_score = score_by_turn[max(score_by_turn)] if score_by_turn else 0.0
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "profile_id": profile_id,
                "base_severity": severity,
                "observed_turns": len(records),
                "max_observed_turn_index": max(observed_turns) if observed_turns else None,
                "denominator_count": len(denominator),
                "final_s": round(final_score, 6),
            }
        )

        current = 0.0
        for turn_index in range(turn_budget):
            if turn_index in score_by_turn:
                current = score_by_turn[turn_index]
            turn_rows.append(
                {
                    "scenario_id": scenario_id,
                    "profile_id": profile_id,
                    "base_severity": severity,
                    "turn_index": turn_index,
                    "observed_at_turn": turn_index in observed_turns,
                    "cumulative_s": round(current, 6),
                }
            )
    return scenario_rows, turn_rows


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def aggregate_final(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["turn_budget"],
                row["model"],
                row["metric_variant"],
                row["base_severity"],
            )
        ].append(row)
    out = []
    for key, values in grouped.items():
        turn_budget, model, metric_variant, severity = key
        out.append(
            {
                "turn_budget": turn_budget,
                "model": model,
                "metric_variant": metric_variant,
                "base_severity": severity,
                "n_scenarios": len(values),
                "mean_final_s": mean_or_none([float(row["final_s"]) for row in values]),
                "mean_observed_turns": mean_or_none([float(row["observed_turns"]) for row in values]),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            int(row["turn_budget"]),
            MODEL_ORDER.get(str(row["model"]), 999),
            str(row["metric_variant"]),
            SEVERITY_ORDER.get(str(row["base_severity"]), 999),
        ),
    )


def aggregate_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["turn_budget"],
                row["model"],
                row["metric_variant"],
                row["base_severity"],
                row["turn_index"],
            )
        ].append(row)
    out = []
    previous: dict[tuple[Any, ...], float] = {}
    for key, values in sorted(grouped.items()):
        turn_budget, model, metric_variant, severity, turn_index = key
        current = mean_or_none([float(row["cumulative_s"]) for row in values])
        prev_key = (turn_budget, model, metric_variant, severity)
        prev = previous.get(prev_key)
        delta = None if current is None or prev is None else round(current - prev, 6)
        if current is not None:
            previous[prev_key] = current
        out.append(
            {
                "turn_budget": turn_budget,
                "model": model,
                "metric_variant": metric_variant,
                "base_severity": severity,
                "turn_index": turn_index,
                "n_scenarios": len(values),
                "observed_at_turn": sum(1 for row in values if row["observed_at_turn"]),
                "observed_rate": round(sum(1 for row in values if row["observed_at_turn"]) / len(values), 6),
                "mean_cumulative_s": current,
                "mean_delta_s": delta,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            int(row["turn_budget"]),
            MODEL_ORDER.get(str(row["model"]), 999),
            str(row["metric_variant"]),
            SEVERITY_ORDER.get(str(row["base_severity"]), 999),
            int(row["turn_index"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    final_rows = [
        row
        for row in summary["final_by_severity"]
        if row["metric_variant"] == "keyword_supported_only"
    ]
    audit_by_key = {
        (row["turn_budget"], row["model"]): row
        for row in summary["run_audit"]
    }
    lines = [
        "# Baseline Evidence Response Analysis",
        "",
        "Metric: tree-aligned canonical evidence recovery. Retained evidence counts as 1.0; weakened evidence counts as 0.5.",
        "Primary table uses `keyword_supported_only`, the stricter metric used by the baseline suite summary.",
        "",
        "## Final Score By Severity",
        "",
        "| turn | model | status | mild | moderate | severe | mean | records | pending | verified_only | hard_errors |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    by_run: dict[tuple[int, str], dict[str, Any]] = defaultdict(dict)
    for row in final_rows:
        by_run[(int(row["turn_budget"]), str(row["model"]))][str(row["base_severity"])] = row
    for turn_budget, model in sorted(by_run, key=lambda item: (item[0], MODEL_ORDER.get(item[1], 999))):
        values = by_run[(turn_budget, model)]
        nums = []
        for severity in ("mild_low_info", "moderate_low_info", "severe_low_info"):
            value = values.get(severity, {}).get("mean_final_s")
            if isinstance(value, (int, float)):
                nums.append(float(value))
        audit = audit_by_key.get((turn_budget, model), {})
        status = "complete" if audit.get("summary_complete") else "partial"
        lines.append(
            "| {turn} | {model} | {status} | {mild} | {moderate} | {severe} | {mean} | {records} | {pending} | {verified} | {hard} |".format(
                turn=turn_budget,
                model=model,
                status=status,
                mild=format_num(values.get("mild_low_info", {}).get("mean_final_s")),
                moderate=format_num(values.get("moderate_low_info", {}).get("mean_final_s")),
                severe=format_num(values.get("severe_low_info", {}).get("mean_final_s")),
                mean=format_num(mean(nums) if nums else None),
                records=audit.get("records", ""),
                pending="" if audit.get("pending") is None else audit.get("pending"),
                verified=audit.get("verified_only", ""),
                hard=audit.get("hard_error_rows", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Turn Curve Snapshots",
            "",
            "Rows below show cumulative keyword-supported recovery at selected turns. Early-stopped scenarios are carried forward.",
            "",
            "| turn budget | eval turn | model | mild | moderate | severe | mean |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    curve = [
        row
        for row in summary["turn_curve"]
        if row["metric_variant"] == "keyword_supported_only"
    ]
    curve_by_key: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(dict)
    for row in curve:
        curve_by_key[(int(row["turn_budget"]), int(row["turn_index"]), str(row["model"]))][
            str(row["base_severity"])
        ] = row
    for turn_budget in (24, 32):
        selected_turns = [0, 3, 7, 11, 15, 19, 23]
        if turn_budget == 32:
            selected_turns.extend([27, 31])
        for eval_turn in selected_turns:
            for model in MODELS:
                values = curve_by_key.get((turn_budget, eval_turn, model))
                if not values:
                    continue
                nums = []
                for severity in ("mild_low_info", "moderate_low_info", "severe_low_info"):
                    value = values.get(severity, {}).get("mean_cumulative_s")
                    if isinstance(value, (int, float)):
                        nums.append(float(value))
                lines.append(
                    "| {budget} | {turn} | {model} | {mild} | {moderate} | {severe} | {mean} |".format(
                        budget=turn_budget,
                        turn=eval_turn + 1,
                        model=model,
                        mild=format_num(values.get("mild_low_info", {}).get("mean_cumulative_s")),
                        moderate=format_num(values.get("moderate_low_info", {}).get("mean_cumulative_s")),
                        severe=format_num(values.get("severe_low_info", {}).get("mean_cumulative_s")),
                        mean=format_num(mean(nums) if nums else None),
                    )
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(input_dir: Path, canonical_dir: Path, output_dir: Path) -> dict[str, Any]:
    canonical_units_path = canonical_dir / "mdd5k_tree_aligned_canonical_evidence_units.jsonl"
    surface_links_path = canonical_dir / "mdd5k_surface_to_canonical_evidence_links.jsonl"
    all_denominator, keyword_denominator = load_canonical_units(canonical_units_path)
    all_links, keyword_links = load_surface_links(surface_links_path)
    metric_defs = {
        "all_supported": (all_denominator, all_links),
        "keyword_supported_only": (keyword_denominator, keyword_links),
    }

    run_audit = []
    scenario_rows_all = []
    turn_rows_all = []
    manifest_path = input_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"items": []}
    for item in manifest.get("items", []):
        turn_budget = int(item["turn"])
        model = str(item["model"])
        out_dir = input_dir / str(item["output_dir"])
        records_path = out_dir / "mdd5k_llm_doctor_online_replay_records.jsonl"
        summary_path = out_dir / "tree_aligned_canonical_recovery" / "tree_aligned_canonical_evidence_recovery_summary.json"
        pending_path = out_dir / "mdd5k_llm_doctor_online_replay_pending_requests.jsonl"
        audit = audit_records(records_path)
        audit.update(
            {
                "turn_budget": turn_budget,
                "model": model,
                "output_dir": str(out_dir),
                "summary_complete": summary_path.exists(),
                "pending": line_count(pending_path),
            }
        )
        run_audit.append(audit)
        if not records_path.exists():
            continue
        for metric_variant, (denominator_by_profile, surface_to_canonical) in metric_defs.items():
            scenario_rows, turn_rows = analyze_records(
                records_path=records_path,
                turn_budget=turn_budget,
                denominator_by_profile=denominator_by_profile,
                surface_to_canonical=surface_to_canonical,
            )
            for row in scenario_rows:
                row.update({"turn_budget": turn_budget, "model": model, "metric_variant": metric_variant})
            for row in turn_rows:
                row.update({"turn_budget": turn_budget, "model": model, "metric_variant": metric_variant})
            scenario_rows_all.extend(scenario_rows)
            turn_rows_all.extend(turn_rows)

    final_by_severity = aggregate_final(scenario_rows_all)
    turn_curve = aggregate_turns(turn_rows_all)
    summary = {
        "input_dir": str(input_dir),
        "canonical_dir": str(canonical_dir),
        "metric": {
            "name": "tree_aligned_canonical_evidence_recovery",
            "retained_weight": 1.0,
            "weakened_weight": 0.5,
            "primary_variant": "keyword_supported_only",
            "turn_curve": "cumulative; early-stopped scenarios carry their last score forward",
        },
        "run_audit": sorted(run_audit, key=lambda row: (int(row["turn_budget"]), MODEL_ORDER.get(str(row["model"]), 999))),
        "final_by_severity": final_by_severity,
        "turn_curve": turn_curve,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "baseline_evidence_analysis_summary.json", summary)
    write_csv(output_dir / "run_audit.csv", summary["run_audit"])
    write_csv(output_dir / "final_by_severity.csv", final_by_severity)
    write_csv(output_dir / "turn_curve.csv", turn_curve)
    write_markdown(output_dir / "BASELINE_EVIDENCE_ANALYSIS.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pulled final-patient baseline evidence response.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = analyze(args.input_dir, args.canonical_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(summary["run_audit"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
