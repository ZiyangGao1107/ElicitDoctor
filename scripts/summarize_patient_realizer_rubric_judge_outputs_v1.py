from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from prepare_llm_patient_realizer_requests_v1 import iter_jsonl, write_json, write_jsonl


DIMENSIONS = [
    "grounding",
    "disclosure_control",
    "avoidance_quality",
    "query_responsiveness",
    "dialogue_naturalness",
]
PASS_FAIL_VALUES = {"pass", "soft_fail", "hard_fail"}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def strip_json_fence(text: str) -> str:
    text = clean_text(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_json_output(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    text = clean_text(raw)
    if not text:
        return None, "empty_raw_output"
    candidate = strip_json_fence(text)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    return obj, None


def request_index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["request_id"]): record for record in iter_jsonl(path)}


def output_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    outputs = {}
    for record in iter_jsonl(path):
        request_id = record.get("request_id")
        if request_id:
            outputs[str(request_id)] = record
    return outputs


def score_value(obj: dict[str, Any], key: str) -> int | None:
    scores = obj.get("scores") or {}
    value = scores.get(key)
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if 1 <= value_int <= 5 else None


def normalize_pass_fail(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in PASS_FAIL_VALUES else "invalid"


def normalize_error_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def build_record(request: dict[str, Any], output: dict[str, Any] | None) -> dict[str, Any]:
    output = output or {}
    parsed, parse_error = parse_json_output(output.get("raw_output"))
    parsed = parsed or {}
    scores = {dimension: score_value(parsed, dimension) for dimension in DIMENSIONS}
    valid_scores = [score for score in scores.values() if score is not None]
    pass_fail = normalize_pass_fail(parsed.get("pass_fail"))
    error_tags = normalize_error_tags(parsed.get("error_tags"))
    try:
        overall_score = int(parsed.get("overall_score"))
    except (TypeError, ValueError):
        overall_score = round(mean(valid_scores), 4) if valid_scores else None
    return {
        "request_id": request.get("request_id"),
        "source_record_id": request.get("source_record_id"),
        "scenario_id": request.get("scenario_id"),
        "profile_id": request.get("profile_id"),
        "policy_name": request.get("policy_name"),
        "base_severity": request.get("base_severity"),
        "target_tree_node": request.get("target_tree_node"),
        "patient_realizer_mode": request.get("patient_realizer_mode"),
        "provider": output.get("provider"),
        "model": output.get("model"),
        "parse_error": parse_error,
        "scores": scores,
        "overall_score": overall_score,
        "pass_fail": pass_fail,
        "error_tags": error_tags,
        "brief_rationale": clean_text(parsed.get("brief_rationale")),
        "raw_output": output.get("raw_output"),
    }


def summarize(records: list[dict[str, Any]], request_path: Path, output_path: Path) -> dict[str, Any]:
    score_lists: dict[str, list[float]] = defaultdict(list)
    pass_fail_counter = Counter()
    error_tag_counter = Counter()
    parse_error_counter = Counter()
    by_severity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if record.get("parse_error"):
            parse_error_counter.update([record["parse_error"]])
        pass_fail_counter.update([record.get("pass_fail") or "invalid"])
        error_tag_counter.update(record.get("error_tags") or [])
        by_severity[str(record.get("base_severity"))].append(record)
        by_target[str(record.get("target_tree_node"))].append(record)
        for dimension, value in (record.get("scores") or {}).items():
            if value is not None:
                score_lists[dimension].append(float(value))
        if record.get("overall_score") is not None:
            score_lists["overall_score"].append(float(record["overall_score"]))

    def bucket_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        local_scores: dict[str, list[float]] = defaultdict(list)
        local_pf = Counter()
        local_tags = Counter()
        for item in items:
            local_pf.update([item.get("pass_fail") or "invalid"])
            local_tags.update(item.get("error_tags") or [])
            for dimension, value in (item.get("scores") or {}).items():
                if value is not None:
                    local_scores[dimension].append(float(value))
            if item.get("overall_score") is not None:
                local_scores["overall_score"].append(float(item["overall_score"]))
        return {
            "n": len(items),
            "mean_scores": {
                key: round(mean(values), 4) for key, values in sorted(local_scores.items()) if values
            },
            "pass_fail": dict(local_pf),
            "error_tags": dict(local_tags),
        }

    return {
        "request_path": str(request_path),
        "output_path": str(output_path),
        "n": len(records),
        "parse_errors": dict(parse_error_counter),
        "mean_scores": {
            key: round(mean(values), 4) for key, values in sorted(score_lists.items()) if values
        },
        "pass_fail": dict(pass_fail_counter),
        "error_tags": dict(error_tag_counter),
        "by_severity": {key: bucket_summary(value) for key, value in sorted(by_severity.items())},
        "top_targets": {
            key: bucket_summary(value)
            for key, value in sorted(by_target.items(), key=lambda item: len(item[1]), reverse=True)[:12]
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Patient Realizer Rubric Judge Summary",
        "",
        f"- n: {summary['n']}",
        f"- mean_scores: `{json.dumps(summary['mean_scores'], ensure_ascii=False)}`",
        f"- pass_fail: `{json.dumps(summary['pass_fail'], ensure_ascii=False)}`",
        f"- error_tags: `{json.dumps(summary['error_tags'], ensure_ascii=False)}`",
        f"- parse_errors: `{json.dumps(summary['parse_errors'], ensure_ascii=False)}`",
        "",
        "## By Severity",
        "",
        "| severity | n | overall | grounding | disclosure | avoidance | responsiveness | naturalness | pass/fail | tags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for severity, row in summary.get("by_severity", {}).items():
        scores = row.get("mean_scores") or {}
        lines.append(
            "| {sev} | {n} | {overall} | {grounding} | {disclosure} | {avoid} | {resp} | {nat} | `{pf}` | `{tags}` |".format(
                sev=severity,
                n=row.get("n"),
                overall=scores.get("overall_score", ""),
                grounding=scores.get("grounding", ""),
                disclosure=scores.get("disclosure_control", ""),
                avoid=scores.get("avoidance_quality", ""),
                resp=scores.get("query_responsiveness", ""),
                nat=scores.get("dialogue_naturalness", ""),
                pf=json.dumps(row.get("pass_fail"), ensure_ascii=False),
                tags=json.dumps(row.get("error_tags"), ensure_ascii=False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize closed-LLM rubric judge outputs for patient simulator responses.")
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    requests = request_index(args.request_path)
    outputs = output_index(args.output_path)
    records = [build_record(request, outputs.get(request_id)) for request_id, request in requests.items() if request_id in outputs]
    summary = summarize(records, args.request_path, args.output_path)
    record_path = args.report_dir / "mdd5k_patient_realizer_rubric_judge_records.jsonl"
    summary_path = args.report_dir / "mdd5k_patient_realizer_rubric_judge_summary.json"
    report_path = args.report_dir / "PATIENT_REALIZER_RUBRIC_JUDGE_SUMMARY.md"
    write_jsonl(record_path, records)
    write_json(summary_path, summary)
    write_report(report_path, summary)
    print(json.dumps({**summary, "record_path": str(record_path), "summary_path": str(summary_path), "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
