from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from _patient_realizer_io import iter_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_PATH = (
    BASE_DIR
    / "outputs_llm_patient_realizer_v3_1"
    / "mdd5k_llm_patient_realizer_requests.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    BASE_DIR
    / "outputs_llm_patient_realizer_v3_1"
    / "llm_patient_realizer_outputs.jsonl"
)
DEFAULT_REPORT_DIR = BASE_DIR / "outputs_llm_patient_realizer_v3_1"

REFUSAL_OR_VAGUE_CUES = (
    "不想细说",
    "不太想说",
    "不太想谈",
    "不想谈",
    "不太想详细说",
    "不想详细说",
    "不太想详细聊",
    "不想详细聊",
    "暂时不想展开",
    "不太想展开",
    "不方便说",
    "可以先跳过",
    "先跳过",
    "说不清",
    "不太清楚",
    "不知道怎么说",
    "记不清",
    "不确定",
    "没法说",
    "不愿意",
    "不舒服",
    "有点模糊",
)

PUNCT_RE = re.compile(r"[\s\.,;:!?，。；：！？、（）()【】\[\]《》<>\"'“”‘’`~_\-—]+")

TARGET_TOPIC_KEYWORDS = {
    "appetite_loss": ("没胃口", "食欲", "吃不下", "胃口", "体重", "吃饭"),
    "attention_decline": ("注意力", "集中", "分心", "走神"),
    "binge_eating": ("暴食", "暴饮暴食", "吃得多", "吃很多", "控制不住", "饮食", "食物", "吃东西"),
    "chest_tightness": ("胸闷", "胸口", "喘不过气"),
    "cognitive_slowing": ("脑子", "反应慢", "变笨", "迟钝", "思考"),
    "dizziness_or_headache": ("头晕", "头痛", "头疼"),
    "hallucination": ("幻觉", "异常知觉", "听到", "看到", "声音", "影子", "奇怪的事情"),
    "mania_screen": ("兴奋", "精力", "冲动", "话多", "睡很少"),
    "memory_problem": ("记忆", "健忘", "记不住"),
    "palpitation": ("心慌", "心悸", "心跳"),
    "parent_awareness": ("父母", "爸爸", "妈妈", "家里", "沟通", "支持"),
    "school_or_study_status": ("学习", "上学", "学校", "成绩", "作业"),
    "sleep": ("睡眠", "睡不着", "失眠", "入睡", "早醒", "做梦"),
    "suicide_or_self_harm": ("自杀", "自残", "伤害自己", "轻生", "活着没意思", "结束生命"),
}


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def normalize_text(text: Any) -> str:
    return PUNCT_RE.sub("", clean_text(text).lower())


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[idx : idx + n] for idx in range(len(text) - n + 1)}


def overlap_score(unit_text: str, response: str) -> float:
    unit_norm = normalize_text(unit_text)
    response_norm = normalize_text(response)
    if not unit_norm or not response_norm:
        return 0.0
    if len(unit_norm) >= 4 and unit_norm in response_norm:
        return 1.0
    unit_grams = char_ngrams(unit_norm, n=3)
    if not unit_grams:
        return 0.0
    response_grams = char_ngrams(response_norm, n=3)
    return len(unit_grams & response_grams) / len(unit_grams)


def strip_json_fence(text: str) -> str:
    text = clean_text(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_patient_response(raw_output: Any) -> tuple[str, str | None]:
    raw = clean_text(raw_output)
    if not raw:
        return "", "empty_raw_output"
    candidate = strip_json_fence(raw)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return raw, "invalid_json"
    if not isinstance(parsed, dict):
        return raw, "json_not_object"
    response = clean_text(parsed.get("patient_response"))
    if not response:
        return "", "missing_patient_response"
    return response, None


def request_index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["request_id"]): record for record in iter_jsonl(path)}


def output_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    outputs: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        request_id = str(record.get("request_id"))
        if request_id:
            outputs[request_id] = record
    return outputs


def unit_scores(response: str, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = []
    for unit in units:
        scores.append(
            {
                "unit_id": unit.get("unit_id"),
                "profile_unit_id": unit.get("profile_unit_id"),
                "unit_text": unit.get("unit_text"),
                "target_relevance": unit.get("target_relevance"),
                "overlap": round(overlap_score(str(unit.get("unit_text") or ""), response), 4),
            }
        )
    return scores


def has_refusal_or_vague_cue(response: str) -> bool:
    return any(cue in response for cue in REFUSAL_OR_VAGUE_CUES)


def topic_specific_terms(response: str, target_tree_node: str | None) -> list[str]:
    if not target_tree_node:
        return []
    keywords = TARGET_TOPIC_KEYWORDS.get(str(target_tree_node), ())
    return [keyword for keyword in keywords if keyword in response]


def duplicate_phrase_warnings(response: str) -> list[str]:
    warnings = []
    clauses = [
        normalize_text(part)
        for part in re.split(r"[，。；！？、,.!?;]", clean_text(response))
        if len(normalize_text(part)) >= 5
    ]
    repeated_clauses = [clause for clause, count in Counter(clauses).items() if count >= 2]
    if repeated_clauses:
        warnings.append("duplicate_clause")

    norm = normalize_text(response)
    if len(norm) >= 16:
        grams = [norm[idx : idx + 6] for idx in range(len(norm) - 5)]
        if any(count >= 3 for count in Counter(grams).values()):
            warnings.append("repeated_6gram")
    return warnings


def verify_one(
    *,
    request: dict[str, Any],
    output: dict[str, Any] | None,
    use_source_rule_based: bool,
    leak_threshold: float,
    allowed_threshold: float,
    reference_min_coverage: float,
    severe_max_coverage: float,
    strict_readability: bool,
) -> dict[str, Any]:
    metadata = request.get("hidden_verifier_metadata_not_for_realizer") or {}
    retained_units = metadata.get("retained_units") or []
    weakened_units = metadata.get("weakened_units") or []
    removed_units = metadata.get("removed_units") or []
    withheld_units = metadata.get("withheld_units") or removed_units
    forbidden_units = metadata.get("forbidden_units") or []
    observed_units = metadata.get("observed_evidence_units") or []
    visible_units = retained_units + weakened_units
    visible_unit_ids = {str(unit.get("unit_id")) for unit in visible_units}
    hidden_observed_units = [
        unit
        for unit in observed_units
        if str(unit.get("unit_id")) not in visible_unit_ids
    ]

    parse_error = None
    if use_source_rule_based:
        response = clean_text(request.get("source_rule_based_patient_response"))
        output_model = "deterministic_rule_based_source"
        provider = "source_record"
    else:
        output = output or {}
        response, parse_error = parse_patient_response(output.get("raw_output"))
        output_model = output.get("model")
        provider = output.get("provider")

    hard_errors: list[str] = []
    warnings: list[str] = []
    if parse_error:
        hard_errors.append(parse_error)
    if not response:
        hard_errors.append("empty_patient_response")

    retained_scores = unit_scores(response, retained_units)
    weakened_scores = unit_scores(response, weakened_units)
    removed_scores = unit_scores(response, removed_units)
    withheld_scores = unit_scores(response, withheld_units)
    forbidden_scores = unit_scores(response, forbidden_units)
    hidden_observed_scores = unit_scores(response, hidden_observed_units)
    visible_scores = retained_scores + weakened_scores

    leaked_removed = [
        score
        for score in removed_scores
        if score["overlap"] >= leak_threshold and len(normalize_text(score.get("unit_text"))) >= 4
    ]
    leaked_withheld = [
        score
        for score in withheld_scores
        if score["overlap"] >= leak_threshold and len(normalize_text(score.get("unit_text"))) >= 4
    ]
    leaked_forbidden = [
        score
        for score in forbidden_scores
        if score["overlap"] >= leak_threshold and len(normalize_text(score.get("unit_text"))) >= 4
    ]
    leaked_hidden_observed = [
        score
        for score in hidden_observed_scores
        if score["overlap"] >= leak_threshold and len(normalize_text(score.get("unit_text"))) >= 4
    ]
    if leaked_removed:
        hard_errors.append("removed_evidence_leakage")
    if leaked_withheld:
        hard_errors.append("withheld_evidence_leakage")
    if leaked_forbidden:
        hard_errors.append("forbidden_evidence_leakage")
    if leaked_hidden_observed:
        hard_errors.append("hidden_observed_evidence_leakage")
    empty_visible_topic_terms = topic_specific_terms(response, request.get("target_tree_node")) if not visible_units else []
    if empty_visible_topic_terms:
        hard_errors.append("empty_visible_topic_specific_claim")

    allowed_coverages = [score["overlap"] for score in visible_scores]
    mean_allowed_coverage = mean(allowed_coverages) if allowed_coverages else 0.0
    covered_visible = sum(score["overlap"] >= allowed_threshold for score in visible_scores)

    severity = str(request.get("base_severity") or "")
    low_info_category = str(request.get("low_info_category") or "")
    vague_or_refusal = has_refusal_or_vague_cue(response)

    if severity == "reference_informative" and visible_units and mean_allowed_coverage < reference_min_coverage:
        hard_errors.append("reference_under_informative")
    if severity == "severe_low_info":
        if not visible_units and not vague_or_refusal:
            hard_errors.append("severe_missing_boundary_or_vagueness")
        if visible_units and mean_allowed_coverage > severe_max_coverage:
            hard_errors.append("severe_over_disclosure")
    if severity == "moderate_low_info" and visible_units and mean_allowed_coverage > 0.85:
        warnings.append("moderate_may_be_too_informative")

    warnings.extend(duplicate_phrase_warnings(response))
    if len(response) > 180:
        warnings.append("long_response")
    if len(response) < 4:
        warnings.append("very_short_response")

    exact_visible_hits = [
        score
        for score in visible_scores
        if normalize_text(score.get("unit_text")) and normalize_text(score.get("unit_text")) in normalize_text(response)
    ]
    if len(exact_visible_hits) >= 4:
        warnings.append("copy_like_concatenation")

    accepted = not hard_errors and (not strict_readability or not warnings)
    return {
        "request_id": request.get("request_id"),
        "source_record_id": request.get("source_record_id"),
        "scenario_id": request.get("scenario_id"),
        "policy_name": request.get("policy_name"),
        "base_severity": severity,
        "target_tree_node": request.get("target_tree_node"),
        "low_info_category": low_info_category,
        "doctor_question": request.get("doctor_question"),
        "patient_response": response,
        "provider": provider,
        "model": output_model,
        "parse_error": parse_error,
        "hard_errors": hard_errors,
        "warnings": sorted(set(warnings)),
        "accepted": accepted,
        "retained_unit_count": len(retained_units),
        "weakened_unit_count": len(weakened_units),
        "removed_unit_count": len(removed_units),
        "withheld_unit_count": len(withheld_units),
        "forbidden_unit_count": len(forbidden_units),
        "covered_visible_count": covered_visible,
        "mean_allowed_coverage": round(mean_allowed_coverage, 4),
        "vague_or_refusal_cue": vague_or_refusal,
        "empty_visible_topic_terms": empty_visible_topic_terms,
        "leaked_removed_units": leaked_removed,
        "leaked_withheld_units": leaked_withheld,
        "leaked_forbidden_units": leaked_forbidden,
        "leaked_hidden_observed_units": leaked_hidden_observed,
        "retained_scores": retained_scores,
        "weakened_scores": weakened_scores,
        "removed_scores": removed_scores,
        "withheld_scores": withheld_scores,
        "forbidden_scores": forbidden_scores,
        "hidden_observed_scores": hidden_observed_scores,
        "g_target": metadata.get("g_target"),
        "controller_information_retention": metadata.get("information_retention"),
        "controller_clarity": metadata.get("clarity"),
        "controller_topic_responsiveness": metadata.get("topic_responsiveness"),
    }


def summarize(records: list[dict[str, Any]], *, source_mode: str, request_path: Path, output_path: Path | None) -> dict[str, Any]:
    hard_error_counter = Counter()
    warning_counter = Counter()
    by_severity: dict[str, dict[str, Any]] = {}
    severity_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        hard_error_counter.update(record.get("hard_errors") or [])
        warning_counter.update(record.get("warnings") or [])
        severity_buckets[str(record.get("base_severity"))].append(record)

    for severity, items in sorted(severity_buckets.items()):
        sev_hard = Counter()
        sev_warn = Counter()
        for item in items:
            sev_hard.update(item.get("hard_errors") or [])
            sev_warn.update(item.get("warnings") or [])
        by_severity[severity] = {
            "n": len(items),
            "accepted_rate": round(sum(1 for item in items if item.get("accepted")) / len(items), 4),
            "mean_allowed_coverage": round(mean(float(item.get("mean_allowed_coverage") or 0.0) for item in items), 4),
            "hard_errors": dict(sev_hard),
            "warnings": dict(sev_warn),
        }

    total = len(records)
    return {
        "source_mode": source_mode,
        "request_path": str(request_path),
        "output_path": str(output_path) if output_path else None,
        "n": total,
        "accepted_rate": round(sum(1 for record in records if record.get("accepted")) / total, 4) if total else 0.0,
        "hard_error_rate": round(sum(1 for record in records if record.get("hard_errors")) / total, 4) if total else 0.0,
        "warning_rate": round(sum(1 for record in records if record.get("warnings")) / total, 4) if total else 0.0,
        "mean_allowed_coverage": round(mean(float(record.get("mean_allowed_coverage") or 0.0) for record in records), 4)
        if records
        else 0.0,
        "hard_errors": dict(hard_error_counter),
        "warnings": dict(warning_counter),
        "by_severity": by_severity,
    }


def write_report(path: Path, summary: dict[str, Any], sample_records: list[dict[str, Any]]) -> None:
    lines = [
        "# LLM Patient Realizer Verification Report V3.1",
        "",
        "Date: 2026-07-07",
        "",
        "## Purpose",
        "",
        "This verifier is a conservative gate for the controlled patient simulator. It checks whether a realized patient response respects controller-level disclosure constraints before it can replace the deterministic fallback.",
        "",
        "## What It Can and Cannot Guarantee",
        "",
        "- It can catch exact or near-lexical leakage of removed/withheld evidence units.",
        "- It can catch invalid JSON, empty responses, severe-setting over-disclosure, and obvious copy-like/repetitive responses.",
        "- It cannot fully prove semantic absence of new facts; small sampled stronger-model or human audits are still needed for final paper credibility.",
        "",
        "## Summary",
        "",
        f"- source mode: `{summary['source_mode']}`",
        f"- n: {summary['n']}",
        f"- accepted rate: {summary['accepted_rate']}",
        f"- hard error rate: {summary['hard_error_rate']}",
        f"- warning rate: {summary['warning_rate']}",
        f"- mean allowed coverage: {summary['mean_allowed_coverage']}",
        f"- hard errors: `{json.dumps(summary['hard_errors'], ensure_ascii=False)}`",
        f"- warnings: `{json.dumps(summary['warnings'], ensure_ascii=False)}`",
        "",
        "## By Severity",
        "",
        "| setting | n | accepted | mean allowed coverage | hard errors | warnings |",
        "|---|---:|---:|---:|---|---|",
    ]
    for severity, row in summary.get("by_severity", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    severity,
                    str(row["n"]),
                    str(row["accepted_rate"]),
                    str(row["mean_allowed_coverage"]),
                    "`" + json.dumps(row["hard_errors"], ensure_ascii=False) + "`",
                    "`" + json.dumps(row["warnings"], ensure_ascii=False) + "`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Sample Failed or Warned Records", ""])
    highlighted = [
        record
        for record in sample_records
        if (record.get("hard_errors") or record.get("warnings"))
    ][:8]
    if not highlighted:
        lines.append("No failed or warned records in the displayed sample.")
    for record in highlighted:
        lines.extend(
            [
                f"### {record.get('request_id')}",
                "",
                f"- setting: `{record.get('base_severity')}`",
                f"- target slot: `{record.get('target_tree_node')}`",
                f"- hard errors: `{json.dumps(record.get('hard_errors'), ensure_ascii=False)}`",
                f"- warnings: `{json.dumps(record.get('warnings'), ensure_ascii=False)}`",
                f"- doctor: {record.get('doctor_question')}",
                f"- patient: {record.get('patient_response')}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LLM patient realizer outputs against controller metadata.")
    parser.add_argument("--request-path", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--use-source-rule-based", action="store_true", help="Verify deterministic source responses instead of LLM outputs.")
    parser.add_argument("--strict-readability", action="store_true", help="Treat readability warnings as rejection.")
    parser.add_argument("--leak-threshold", type=float, default=0.72)
    parser.add_argument("--allowed-threshold", type=float, default=0.45)
    parser.add_argument("--reference-min-coverage", type=float, default=0.30)
    parser.add_argument("--severe-max-coverage", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    requests = request_index(args.request_path)
    outputs = {} if args.use_source_rule_based else output_index(args.output_path)

    records: list[dict[str, Any]] = []
    for request_id, request in requests.items():
        if not args.use_source_rule_based and request_id not in outputs:
            continue
        records.append(
            verify_one(
                request=request,
                output=outputs.get(request_id),
                use_source_rule_based=args.use_source_rule_based,
                leak_threshold=args.leak_threshold,
                allowed_threshold=args.allowed_threshold,
                reference_min_coverage=args.reference_min_coverage,
                severe_max_coverage=args.severe_max_coverage,
                strict_readability=args.strict_readability,
            )
        )

    source_mode = "source_rule_based" if args.use_source_rule_based else "llm_outputs"
    summary = summarize(
        records,
        source_mode=source_mode,
        request_path=args.request_path,
        output_path=None if args.use_source_rule_based else args.output_path,
    )
    record_path = args.report_dir / f"mdd5k_patient_realizer_verification_records_{source_mode}.jsonl"
    summary_path = args.report_dir / f"mdd5k_patient_realizer_verification_summary_{source_mode}.json"
    report_path = args.report_dir / f"LLM_PATIENT_REALIZER_VERIFICATION_REPORT_{source_mode}.md"
    write_jsonl(record_path, records)
    write_json(summary_path, summary)
    write_report(report_path, summary, records)
    print(
        json.dumps(
            {
                **summary,
                "record_path": str(record_path),
                "summary_path": str(summary_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
