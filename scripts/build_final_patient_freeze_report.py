from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_score(summary: dict[str, Any], key: str) -> float | None:
    value = (summary.get("mean_scores") or {}).get(key)
    if value is None:
        return None
    return float(value)


def final_cache_quality(path: Path) -> dict[str, Any]:
    hard_errors: dict[str, int] = {}
    warnings: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    records = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records += 1
        source = str(record.get("realizer_source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        for key in record.get("hard_errors") or []:
            hard_errors[str(key)] = hard_errors.get(str(key), 0) + 1
        for key in record.get("warnings") or []:
            warnings[str(key)] = warnings.get(str(key), 0) + 1
    return {
        "records": records,
        "source_counts": source_counts,
        "hard_errors": hard_errors,
        "warnings": warnings,
    }


def bool_gate(gates: list[dict[str, Any]], name: str, passed: bool, evidence: Any, threshold: Any = None) -> None:
    gates.append(
        {
            "name": name,
            "passed": bool(passed),
            "threshold": threshold,
            "evidence": evidence,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build freeze report for the final controlled LLM patient simulator.")
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--closed-rubric-output-path", type=Path, required=True)
    parser.add_argument("--closed-rubric-summary-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--cache-summary-path", type=Path, required=True)
    parser.add_argument("--primary-verify-summary-path", type=Path, required=True)
    parser.add_argument("--repair-verify-summary-path", type=Path, default=None)
    parser.add_argument("--repair2-verify-summary-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-grounding", type=float, default=4.8)
    parser.add_argument("--min-disclosure-control", type=float, default=4.8)
    parser.add_argument("--min-overall", type=float, default=4.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request_count = count_jsonl(args.request_path)
    rubric_output_count = count_jsonl(args.closed_rubric_output_path)
    rubric_summary = read_json(args.closed_rubric_summary_path)
    cache_summary = read_json(args.cache_summary_path)
    cache_quality = final_cache_quality(args.cache_path)
    primary_verify_summary = read_json(args.primary_verify_summary_path)
    repair_verify_summary = read_json(args.repair_verify_summary_path) if args.repair_verify_summary_path else None
    repair2_verify_summary = read_json(args.repair2_verify_summary_path) if args.repair2_verify_summary_path else None

    generation_hard_errors = dict(primary_verify_summary.get("hard_errors") or {})
    generation_warnings = dict(primary_verify_summary.get("warnings") or {})
    for summary in (repair_verify_summary, repair2_verify_summary):
        if not summary:
            continue
        for key, value in (summary.get("hard_errors") or {}).items():
            generation_hard_errors[key] = generation_hard_errors.get(key, 0) + value
        for key, value in (summary.get("warnings") or {}).items():
            generation_warnings[key] = generation_warnings.get(key, 0) + value

    grounding = get_score(rubric_summary, "grounding")
    disclosure = get_score(rubric_summary, "disclosure_control")
    overall = get_score(rubric_summary, "overall_score")

    gates: list[dict[str, Any]] = []
    bool_gate(gates, "closed_rubric_complete", rubric_output_count >= request_count > 0, {"outputs": rubric_output_count, "requests": request_count}, "outputs >= requests")
    bool_gate(gates, "cache_full_coverage", cache_summary.get("num_cached") == cache_summary.get("num_primary_requests") and cache_summary.get("coverage") == 1.0, cache_summary, "coverage == 1.0")
    bool_gate(gates, "no_rule_fallback", not cache_summary.get("fallback_to_rule") and "rule_fallback_after_failed_llm_or_repair" not in (cache_summary.get("source_counts") or {}), cache_summary.get("source_counts"), "no rule fallback")
    bool_gate(gates, "final_cache_record_count", cache_quality["records"] == request_count, {"cache_records": cache_quality["records"], "requests": request_count}, "cache_records == requests")
    bool_gate(gates, "no_verifier_hard_errors_remaining", not cache_quality["hard_errors"], cache_quality["hard_errors"], "{}")
    bool_gate(gates, "rubric_parse_clean", not rubric_summary.get("parse_errors"), rubric_summary.get("parse_errors"), "{}")
    bool_gate(gates, "rubric_grounding", grounding is not None and grounding >= args.min_grounding, grounding, args.min_grounding)
    bool_gate(gates, "rubric_disclosure_control", disclosure is not None and disclosure >= args.min_disclosure_control, disclosure, args.min_disclosure_control)
    bool_gate(gates, "rubric_overall", overall is not None and overall >= args.min_overall, overall, args.min_overall)

    freeze_pass = all(gate["passed"] for gate in gates)
    report = {
        "freeze_pass": freeze_pass,
        "gates": gates,
        "request_path": str(args.request_path),
        "closed_rubric_output_path": str(args.closed_rubric_output_path),
        "closed_rubric_summary_path": str(args.closed_rubric_summary_path),
        "cache_summary_path": str(args.cache_summary_path),
        "cache_path": str(args.cache_path),
        "primary_verify_summary_path": str(args.primary_verify_summary_path),
        "repair_verify_summary_path": str(args.repair_verify_summary_path) if args.repair_verify_summary_path else None,
        "repair2_verify_summary_path": str(args.repair2_verify_summary_path) if args.repair2_verify_summary_path else None,
        "request_count": request_count,
        "rubric_output_count": rubric_output_count,
        "rubric_mean_scores": rubric_summary.get("mean_scores"),
        "rubric_pass_fail": rubric_summary.get("pass_fail"),
        "rubric_error_tags": rubric_summary.get("error_tags"),
        "rubric_by_severity": rubric_summary.get("by_severity"),
        "cache_summary": cache_summary,
        "final_cache_quality": cache_quality,
        "generation_verifier_hard_errors_repaired_or_removed": generation_hard_errors,
        "generation_verifier_warnings_repaired_or_allowed": generation_warnings,
    }

    report_path = args.output_dir / "final_patient_freeze_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Final Patient Freeze Report",
        "",
        f"- freeze_pass: `{freeze_pass}`",
        f"- rubric outputs: `{rubric_output_count}/{request_count}`",
        f"- grounding: `{grounding}`",
        f"- disclosure_control: `{disclosure}`",
        f"- overall_score: `{overall}`",
        "",
        "## Gates",
        "",
        "| gate | passed | threshold | evidence |",
        "|---|---:|---|---|",
    ]
    for gate in gates:
        evidence = gate.get("evidence")
        if isinstance(evidence, (dict, list)):
            evidence_text = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        else:
            evidence_text = str(evidence)
        if len(evidence_text) > 180:
            evidence_text = evidence_text[:177] + "..."
        lines.append(f"| {gate['name']} | {gate['passed']} | {gate.get('threshold')} | `{evidence_text}` |")
    lines.extend(
        [
            "",
            "## Rubric Error Tags",
            "",
            "```json",
            json.dumps(rubric_summary.get("error_tags") or {}, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Cache Source Counts",
            "",
            "```json",
            json.dumps(cache_summary.get("source_counts") or {}, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    md_path = args.output_dir / "FINAL_PATIENT_FREEZE_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_pass": freeze_pass, "report_path": str(report_path), "markdown_path": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
