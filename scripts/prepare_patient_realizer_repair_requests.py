from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _patient_realizer_io import iter_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]

SEVERE_SAFE_CUE_PHRASES = [
    "不太想说",
    "说不清",
    "不太清楚",
    "不知道怎么说",
    "可以先跳过",
]


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(record.get(key)): record for record in iter_jsonl(path) if record.get(key) is not None}


def parse_user_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "user":
            try:
                return json.loads(message.get("content") or "{}")
            except json.JSONDecodeError:
                return {}
    return {}


def replace_user_payload(messages: list[dict[str, str]], payload: dict[str, Any]) -> list[dict[str, str]]:
    updated: list[dict[str, str]] = []
    replaced = False
    for message in messages:
        if message.get("role") == "user" and not replaced:
            updated.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)})
            replaced = True
        else:
            updated.append(dict(message))
    if not replaced:
        updated.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)})
    return updated


def dedupe_texts(items: list[Any]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        text = clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def collect_verifier_forbidden_terms(verification: dict[str, Any]) -> list[str]:
    terms: list[Any] = []
    terms.extend(verification.get("empty_visible_topic_terms") or [])
    for key in (
        "leaked_removed_units",
        "leaked_withheld_units",
        "leaked_forbidden_units",
        "leaked_hidden_observed_units",
    ):
        for item in verification.get(key) or []:
            if isinstance(item, dict):
                terms.append(item.get("unit_text"))
                terms.append(item.get("profile_unit_id"))
            else:
                terms.append(item)
    # Keep the full rejected answer visible as context, but only blacklist short
    # explicit leaked phrases here. Long phrases make the repair prompt unstable.
    return [term[:48] for term in dedupe_texts(terms) if len(term) <= 80]


def repair_instructions_for_errors(hard_errors: list[str]) -> list[str]:
    instructions = []
    error_set = set(hard_errors)
    if error_set & {
        "removed_evidence_leakage",
        "withheld_evidence_leakage",
        "forbidden_evidence_leakage",
        "hidden_observed_evidence_leakage",
    }:
        instructions.append(
            "Verifier detected leakage of hidden evidence. Do not reuse the leaked phrase or any close synonym."
        )
    if "empty_visible_topic_specific_claim" in error_set:
        instructions.append(
            "No clinical evidence was allowed, but the response named a topic-specific symptom. Replace it with a generic boundary reply."
        )
    if "severe_over_disclosure" in error_set:
        instructions.append(
            "The severe patient disclosed too much. Downgrade to boundary_only: zero clinical facts and no symptom naming."
        )
    if "severe_missing_boundary_or_vagueness" in error_set:
        instructions.append(
            "The severe patient needs a natural boundary or uncertainty cue, without adding clinical details."
        )
    if not instructions:
        instructions.append("Repair the response by being shorter, less specific, and strictly grounded.")
    return instructions


def requires_ultra_safe_boundary(hard_errors: list[str]) -> bool:
    return bool(
        set(hard_errors)
        & {
            "removed_evidence_leakage",
            "withheld_evidence_leakage",
            "forbidden_evidence_leakage",
            "hidden_observed_evidence_leakage",
            "empty_visible_topic_specific_claim",
            "severe_over_disclosure",
        }
    )


def tighten_payload(payload: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    hard_errors = verification.get("hard_errors") or []
    warnings = verification.get("warnings") or []
    patient_response = clean_text(verification.get("patient_response"))
    severity = str(verification.get("base_severity") or payload.get("patient_setting") or "")
    verifier_forbidden_terms = collect_verifier_forbidden_terms(verification)
    repair_instructions = repair_instructions_for_errors(hard_errors)
    ultra_safe_boundary = requires_ultra_safe_boundary(hard_errors)

    repair = {
        "repair_attempt": 1,
        "previous_response_rejected": patient_response,
        "verifier_hard_errors": hard_errors,
        "verifier_warnings": warnings,
        "verifier_forbidden_terms_from_failed_response": verifier_forbidden_terms,
        "verifier_repair_instructions": repair_instructions,
        "ultra_safe_boundary_required": ultra_safe_boundary,
        "repair_goal": (
            "Generate a safer replacement patient_response. Do not preserve wording from the rejected response "
            "if it caused leakage, over-disclosure, hallucination, or missing boundary/vagueness."
        ),
    }
    payload["repair_feedback"] = repair

    constraints = list(payload.get("hard_constraints") or [])
    constraints.extend(
        [
            "This is a repair generation. Be stricter than the previous response.",
            "Do not reuse any phrase that may have caused verifier leakage.",
            "Follow verifier_repair_instructions exactly; they are higher priority than style preferences.",
            "If any hard error mentions leakage or over-disclosure, reduce clinical specificity.",
            "If any hard error mentions missing boundary or vagueness, add a brief natural boundary while still avoiding clinical details.",
            "Return only valid JSON with patient_response and brief_self_check.",
        ]
    )
    if verifier_forbidden_terms:
        constraints.append(
            "The repaired patient_response must not contain these verifier-forbidden terms or close synonyms: "
            + ", ".join(verifier_forbidden_terms)
        )
    if ultra_safe_boundary:
        constraints.append(
            "Because verifier found leakage or topic-specific claims, ignore the mild/moderate disclosure style for this repair and use a generic boundary-only answer with zero clinical facts."
        )
    if severity == "severe_low_info":
        constraints.append(
            "The repaired severe_low_info patient_response must include at least one exact boundary/vague cue phrase: "
            + ", ".join(SEVERE_SAFE_CUE_PHRASES)
        )
    payload["hard_constraints"] = constraints

    budget = dict(payload.get("response_budget") or {})
    if severity == "severe_low_info" or ultra_safe_boundary:
        budget["max_sentences"] = 1
        budget["max_chinese_chars"] = min(int(budget.get("max_chinese_chars") or 32), 24)
        budget["clinical_fact_budget"] = 0
        budget["required_style"] = "minimal bounded vague reply; no specific clinical fact"
        payload["target_topic_for_patient_realization"] = "the doctor's current concern; exact internal symptom slot is hidden"
        progressive_state = dict(payload.get("progressive_disclosure_state") or {})
        progressive_state["disclosure_stage"] = "boundary_only"
        progressive_state["repair_override"] = "Verifier rejected the previous response; use boundary_only now."
        payload["progressive_disclosure_state"] = progressive_state
        visibility_contract = dict(payload.get("visibility_contract") or {})
        existing_terms = list(visibility_contract.get("forbidden_surface_terms") or [])
        visibility_contract["progressive_disclosure_stage"] = "boundary_only"
        visibility_contract["can_say_exact"] = []
        visibility_contract["can_paraphrase_weakly"] = []
        visibility_contract["allowed_hint_count"] = 0
        visibility_contract["forbidden_surface_terms"] = dedupe_texts(existing_terms + verifier_forbidden_terms)
        visibility_contract["can_hint_about_topic"] = (
            "generic words only, such as 这个/这方面/这件事; do not name the exact symptom slot"
        )
        visibility_contract["allowed_response_templates"] = [
            "这个我现在不太想说。",
            "这方面我还说不清。",
            "可以先跳过这个吗？",
            "我现在不知道怎么说。",
        ]
        payload["visibility_contract"] = visibility_contract
        payload["style_requirement"] = (
            "Repair unsafe response: give a natural but very short boundary/vague reply. "
            "Do not name symptoms, duration, frequency, risk, diagnosis, event, or behavior."
        )
    else:
        budget["max_sentences"] = min(int(budget.get("max_sentences") or 2), 2)
        budget["max_chinese_chars"] = min(int(budget.get("max_chinese_chars") or 60), 48)
        budget["clinical_fact_budget"] = min(int(budget.get("clinical_fact_budget") or 1), 1)
    payload["response_budget"] = budget
    return payload


def build_repair_request(original: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    messages = list(original.get("messages") or [])
    payload = parse_user_payload(messages)
    payload = tighten_payload(payload, verification)
    request = dict(original)
    request["request_id"] = f"{original.get('request_id')}::repair1"
    request["repair_of_request_id"] = original.get("repair_of_request_id") or original.get("request_id")
    request["repair_parent_request_id"] = original.get("request_id")
    request["prompt_protocol_version"] = f"{original.get('prompt_protocol_version', 'unknown')}+repair_v1"
    request["messages"] = replace_user_payload(messages, payload)
    request["expected_output"] = {
        "patient_response": "safer repaired natural Chinese response constrained by allowed evidence",
        "brief_self_check": "short no-new-fact self check",
    }
    return request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build repair requests for verifier-failed patient realizer outputs.")
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--verification-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-warned", action="store_true", help="Repair warned accepted records as well as hard failures.")
    parser.add_argument("--max-repairs", type=int, default=0, help="0 means all failed records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = load_jsonl_by_id(args.request_path, "request_id")
    repair_requests: list[dict[str, Any]] = []
    skipped = Counter()
    hard_error_counter = Counter()
    warning_counter = Counter()

    for verification in iter_jsonl(args.verification_records):
        hard_errors = verification.get("hard_errors") or []
        warnings = verification.get("warnings") or []
        accepted = bool(verification.get("accepted"))
        needs_repair = bool(hard_errors) or (args.include_warned and bool(warnings))
        if accepted and not needs_repair:
            skipped["accepted_clean"] += 1
            continue
        if not needs_repair:
            skipped["no_repair_needed"] += 1
            continue
        original = requests.get(str(verification.get("request_id")))
        if not original:
            skipped["missing_original_request"] += 1
            continue
        repair_requests.append(build_repair_request(original, verification))
        hard_error_counter.update(hard_errors)
        warning_counter.update(warnings)
        if args.max_repairs and len(repair_requests) >= args.max_repairs:
            break

    request_path = args.output_dir / "mdd5k_llm_patient_realizer_repair_requests.jsonl"
    summary_path = args.output_dir / "mdd5k_llm_patient_realizer_repair_request_summary.json"
    summary = {
        "source_request_path": str(args.request_path),
        "source_verification_records": str(args.verification_records),
        "repair_request_path": str(request_path),
        "num_repair_requests": len(repair_requests),
        "skipped": dict(skipped),
        "hard_errors_targeted": dict(hard_error_counter),
        "warnings_targeted": dict(warning_counter),
        "include_warned": args.include_warned,
    }
    write_jsonl(request_path, repair_requests)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
