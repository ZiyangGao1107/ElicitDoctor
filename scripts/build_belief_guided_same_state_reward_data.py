from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_belief_guided_same_state_reward_data"

LOW_INFORMATION_RESPONSE_PATTERNS = (
    "\u4e0d\u60f3\u8bf4",
    "\u4e0d\u592a\u60f3\u8bf4",
    "\u8fd8\u4e0d\u60f3\u8bf4",
    "\u6682\u65f6\u4e0d\u60f3\u8bf4",
    "\u4e0d\u613f\u610f\u8bf4",
    "\u4e0d\u592a\u613f\u610f\u8bf4",
    "\u4e0d\u65b9\u4fbf\u8bf4",
    "\u4e0d\u60f3\u8c08",
    "\u4e0d\u592a\u60f3\u8c08",
    "\u8bf4\u4e0d\u6e05\u695a",
    "\u8bf4\u4e0d\u6e05",
    "\u6682\u65f6\u8bf4\u4e0d\u592a\u6e05\u695a",
    "\u4e0d\u77e5\u9053\u600e\u4e48\u8bf4",
    "\u6ca1\u6cd5\u8bf4",
    "\u4e0d\u60f3\u8bb2",
    "\u4e0d\u613f\u8bb2",
    "i don't know",
    "i do not know",
    "not sure",
    "i'm not sure",
    "i am not sure",
    "hard to say",
    "i don't want to talk",
    "i do not want to talk",
    "i'd rather not",
    "i would rather not",
    "rather not say",
    "can we skip",
    "let's skip",
    "nothing much",
    "not really",
)


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


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def record_is_verified(row: dict[str, Any]) -> bool:
    return str(row.get("patient_realizer_mode") or "") == "verified_llm_cache"


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("hard_error")
        or row.get("patient_hard_error")
        or row.get("patient_verify_hard_error")
        or row.get("has_forbidden_leak")
        or row.get("forbidden_evidence_leak")
    )


def is_low_information_response(text: Any) -> bool:
    cleaned = clean_text(text).lower()
    if not cleaned:
        return True
    return any(pattern in cleaned for pattern in LOW_INFORMATION_RESPONSE_PATTERNS)


def state_id_for(row: dict[str, Any]) -> str:
    return str(row.get("source_state_id") or row.get("scenario_id") or "")


def candidate_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row.get("candidate_index") or 0), str(row.get("record_id") or ""))


def render_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(No visible dialogue yet.)"
    lines: list[str] = []
    start_idx = max(1, len(history) - 11)
    for idx, turn in enumerate(history[-12:], start=start_idx):
        doctor = clean_text(turn.get("doctor") or turn.get("doctor_utterance"))
        patient = clean_text(turn.get("patient") or turn.get("patient_utterance"))
        if doctor:
            lines.append(f"{idx}. Doctor: {doctor}")
        if patient:
            lines.append(f"   Patient: {patient}")
    return "\n".join(lines) if lines else "(No visible dialogue yet.)"


def render_history_plus_candidate(row: dict[str, Any]) -> str:
    history = list(row.get("dialogue_history") or [])
    lines = [render_history(history)]
    question = clean_text(row.get("doctor_question"))
    response = clean_text(row.get("patient_response"))
    turn_no = len(history) + 1
    if question:
        lines.append(f"{turn_no}. Doctor: {question}")
    if response:
        lines.append(f"   Patient: {response}")
    return "\n".join(line for line in lines if line)


def visible_doctor_prompt(row: dict[str, Any], *, language: str = "zh") -> str:
    history = row.get("dialogue_history") or []
    if language == "en":
        if not history:
            return (
                "You are a doctor in a research psychiatric interview. "
                "Output exactly one natural, specific, safe English doctor question.\n\n"
                "Visible interview history: no prior dialogue.\n\n"
                "Next doctor question:"
            )
        lines: list[str] = []
        start_idx = max(1, len(history) - 11)
        for idx, turn in enumerate(history[-12:], start=start_idx):
            doctor = clean_text(turn.get("doctor") or turn.get("doctor_utterance"))
            patient = clean_text(turn.get("patient") or turn.get("patient_utterance"))
            lines.append(f"{idx}. Doctor: {doctor}")
            lines.append(f"   Patient: {patient}")
        return (
            "You are a doctor in a research psychiatric interview. "
            "Output exactly one natural, specific, safe English doctor question.\n\n"
            f"Visible interview history:\n{chr(10).join(lines)}\n\n"
            "Next doctor question:"
        )
    if not history:
        return (
            "\u4f60\u662f\u4e00\u4e2a\u7814\u7a76\u573a\u666f\u4e2d\u7684\u7cbe\u795e\u5fc3\u7406\u4e3b\u52a8\u95ee\u8bca\u533b\u751f\u3002"
            "\u8bf7\u53ea\u8f93\u51fa\u4e0b\u4e00\u53e5\u81ea\u7136\u3001\u5177\u4f53\u3001\u5b89\u5168\u7684\u533b\u751f\u95ee\u9898\u3002\n\n"
            "\u53ef\u89c1\u95ee\u8bca\u5386\u53f2\uff1a\u6682\u65e0\u5bf9\u8bdd\u5386\u53f2\u3002\n\n\u4e0b\u4e00\u53e5\u533b\u751f\u95ee\u9898\uff1a"
        )
    lines: list[str] = []
    start_idx = max(1, len(history) - 11)
    for idx, turn in enumerate(history[-12:], start=start_idx):
        doctor = clean_text(turn.get("doctor") or turn.get("doctor_utterance"))
        patient = clean_text(turn.get("patient") or turn.get("patient_utterance"))
        lines.append(f"{idx}. \u533b\u751f\uff1a{doctor}")
        lines.append(f"   \u60a3\u8005\uff1a{patient}")
    return (
        "\u4f60\u662f\u4e00\u4e2a\u7814\u7a76\u573a\u666f\u4e2d\u7684\u7cbe\u795e\u5fc3\u7406\u4e3b\u52a8\u95ee\u8bca\u533b\u751f\u3002"
        "\u8bf7\u53ea\u8f93\u51fa\u4e0b\u4e00\u53e5\u81ea\u7136\u3001\u5177\u4f53\u3001\u5b89\u5168\u7684\u533b\u751f\u95ee\u9898\u3002\n\n"
        f"\u53ef\u89c1\u95ee\u8bca\u5386\u53f2\uff1a\n{chr(10).join(lines)}\n\n\u4e0b\u4e00\u53e5\u533b\u751f\u95ee\u9898\uff1a"
    )


def build_belief_messages(
    *,
    stage: str,
    dialogue_text: str,
    doctor_question: str,
    patient_response: str,
) -> list[dict[str, str]]:
    system = (
        "You are a calibrated psychiatric belief-state evaluator. Judge only from the visible "
        "doctor-patient dialogue. Do not use hidden patient profiles, simulator metadata, "
        "canonical evidence, gold diagnosis labels, or external facts. Return strict JSON only."
    )
    if stage == "before":
        stage_text = (
            "Stage: query_before. You see the dialogue before the current candidate doctor "
            "question, plus that candidate question. Estimate whether the question targets an "
            "unresolved belief region and is likely to reduce diagnostic uncertainty."
        )
        response_block = "Current patient answer: not observed yet."
    else:
        stage_text = (
            "Stage: query_after. You see the same candidate doctor question and the realized "
            "patient answer. Estimate how diagnostic belief and unresolved regions changed. "
            "Concrete disclosed information can reduce uncertainty; vague/refusal answers should "
            "keep belief nearly unchanged."
        )
        response_block = f"Current patient answer: {patient_response or '(empty)'}"
    user = (
        f"{stage_text}\n\n"
        f"Visible dialogue history:\n{dialogue_text}\n\n"
        f"Current doctor question: {doctor_question or '(empty)'}\n"
        f"{response_block}\n\n"
        "Return exactly one JSON object, no markdown, no extra text. Required fields:\n"
        "- diagnostic_hypotheses: array of objects with label:string and prob:number in [0,1]. "
        "Use broad hypotheses such as depressive_episode, anxiety_disorder, adjustment_stress, "
        "bipolar_related, substance_related, insufficient_information.\n"
        "- top_confidence: number in [0,1], equal to the highest current hypothesis probability.\n"
        "- uncertainty_regions: array of short strings for unresolved regions, e.g. mood, "
        "anhedonia, sleep, appetite, duration, impairment, suicide_risk, mania, substance, medical_causes.\n"
        "- recommended_next_inquiry_regions: array of short strings for high-value next inquiry regions.\n"
        "- candidate_query_targets: array of short strings naming what the current doctor question targets.\n"
        "- query_targets_unresolved_region: integer 0-5.\n"
        "- query_relevance: integer 0-5.\n"
        "- query_redundancy: integer 0-5.\n"
        "- safety_relevance: integer 0-5.\n"
        "- belief_update_magnitude: integer 0-5; for query_after, high only when the patient answer changes belief/uncertainty.\n"
        "- brief_reason: one short reason grounded only in visible dialogue.\n\n"
        "Calibration rules: do not use hidden evidence; do not infer certainty from diagnosis names alone; "
        "confidence gain without visible support is bad calibration; vague/refusal answers should not reduce uncertainty much."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def group_same_state_records(
    records_path: Path,
    *,
    require_verified: bool,
    max_turn_index: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for row in iter_jsonl(records_path):
        counters["records_seen"] += 1
        if require_verified and not record_is_verified(row):
            counters["skip_non_verified"] += 1
            continue
        if hard_error(row):
            counters["skip_hard_error"] += 1
            continue
        turn = int(row.get("turn_index") or 0)
        if max_turn_index is not None and turn > max_turn_index:
            counters["skip_after_max_turn"] += 1
            continue
        state_id = state_id_for(row)
        if not state_id:
            counters["skip_missing_state_id"] += 1
            continue
        if not clean_text(row.get("doctor_question")):
            counters["skip_empty_question"] += 1
            continue
        grouped[state_id].append(row)
    for rows in grouped.values():
        rows.sort(key=candidate_sort_key)
    return grouped, counters


def prepare(args: argparse.Namespace) -> None:
    grouped, counters = group_same_state_records(
        args.records,
        require_verified=not args.allow_non_verified,
        max_turn_index=args.max_turn_index if args.max_turn_index >= 0 else None,
    )
    rows: list[dict[str, Any]] = []
    selected_states = 0
    for state_id, candidates in sorted(grouped.items()):
        if args.min_candidates > 0 and len(candidates) < args.min_candidates:
            counters["skip_state_too_few_candidates"] += 1
            continue
        if args.max_states > 0 and selected_states >= args.max_states:
            break
        selected_states += 1
        for row in candidates:
            record_id = str(row.get("record_id"))
            question = clean_text(row.get("doctor_question"))
            response = clean_text(row.get("patient_response"))
            common = {
                "task_name": "belief_guided_same_state_query_reward_eval",
                "source_record_id": record_id,
                "same_state_group_id": state_id,
                "scenario_id": row.get("scenario_id"),
                "source_state_id": row.get("source_state_id"),
                "source_state_hash": row.get("source_state_hash"),
                "profile_id": row.get("profile_id"),
                "case_id": row.get("case_id"),
                "policy_name": row.get("policy_name"),
                "base_severity": row.get("base_severity"),
                "turn_index": row.get("turn_index"),
                "candidate_index": row.get("candidate_index"),
                "doctor_question": question,
                "patient_response": response,
                "prompt_protocol_version": "belief_guided_same_state_query_reward_v1",
            }
            before_dialogue = render_history(row.get("dialogue_history") or [])
            after_dialogue = render_history_plus_candidate(row)
            rows.append(
                {
                    **common,
                    "request_id": f"{record_id}::belief_before",
                    "belief_stage": "before",
                    "messages": build_belief_messages(
                        stage="before",
                        dialogue_text=before_dialogue,
                        doctor_question=question,
                        patient_response="",
                    ),
                }
            )
            rows.append(
                {
                    **common,
                    "request_id": f"{record_id}::belief_after",
                    "belief_stage": "after",
                    "messages": build_belief_messages(
                        stage="after",
                        dialogue_text=after_dialogue,
                        doctor_question=question,
                        patient_response=response,
                    ),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "belief_guided_same_state_belief_requests.jsonl"
    write_jsonl(request_path, rows)
    summary = {
        "mode": "prepare",
        "records": str(args.records),
        "require_verified": not args.allow_non_verified,
        "max_turn_index": args.max_turn_index,
        "max_states": args.max_states,
        "min_candidates": args.min_candidates,
        "states_available": len(grouped),
        "states_selected": selected_states,
        "belief_requests": len(rows),
        "counters": dict(counters),
        "request_path": str(request_path),
        "method_boundary": "same-state candidate reward; visible dialogue only; no canonical evidence or gold diagnosis in belief prompts",
    }
    write_json(args.output_dir / "belief_guided_same_state_belief_request_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = clean_text(raw)
    if not text:
        return None
    if "```" in text:
        parts = [part.strip() for part in text.split("```") if part.strip()]
        for part in parts:
            candidate = part[4:].strip() if part.lower().startswith("json") else part
            if "{" in candidate and "}" in candidate:
                text = candidate
                break
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def load_outputs(path: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        outputs[request_id] = {**row, "parsed_belief_json": parse_json_object(str(row.get("raw_output") or ""))}
    return outputs


def probs_from_belief(parsed: dict[str, Any] | None) -> list[float]:
    if not parsed:
        return []
    probs: list[float] = []
    for item in parsed.get("diagnostic_hypotheses") or []:
        if isinstance(item, dict):
            p = clamp(safe_float(item.get("prob"), 0.0), 0.0, 1.0)
            if p > 0:
                probs.append(p)
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else []


def entropy(probs: list[float]) -> float:
    if not probs:
        return 1.0
    value = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    denom = math.log(max(2, len(probs)))
    return value / denom if denom > 0 else 0.0


def belief_distribution(parsed: dict[str, Any] | None) -> dict[str, float]:
    if not parsed:
        return {}
    values: dict[str, float] = {}
    for item in parsed.get("diagnostic_hypotheses") or []:
        if not isinstance(item, dict):
            continue
        label = clean_text(item.get("label")).lower()
        prob = clamp(safe_float(item.get("prob"), 0.0), 0.0, 1.0)
        if label and prob > 0:
            values[label] = values.get(label, 0.0) + prob
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total > 0 else {}


def list_field(parsed: dict[str, Any] | None, key: str) -> list[str]:
    if not parsed:
        return []
    value = parsed.get(key)
    raw = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = clean_text(item).lower()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def score_field(parsed: dict[str, Any] | None, key: str) -> float:
    if not parsed:
        return 0.0
    return clamp(safe_float(parsed.get(key), 0.0) / 5.0, 0.0, 1.0)


def target_hit_rate(query_targets: list[str], unresolved_or_recommended: list[str]) -> float:
    if not query_targets or not unresolved_or_recommended:
        return 0.0
    unresolved = set(unresolved_or_recommended)
    return sum(1 for target in query_targets if target in unresolved) / max(1, len(query_targets))


def patient_state_value(row: dict[str, Any], stage: str, key: str) -> float:
    state = row.get(f"patient_state_{stage}") or {}
    return safe_float(state.get(key), 0.0) if isinstance(state, dict) else 0.0


def row_is_terminal_failure(row: dict[str, Any]) -> bool:
    if bool(row.get("patient_active_termination") or row.get("patient_terminated")):
        return True
    for key in ("response_type", "patient_response_type", "patient_status", "termination_reason"):
        value = str(row.get(key) or "").lower()
        if value in {"patient_active_termination", "active_termination", "terminated"}:
            return True
    return False


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "min": round(ordered[0], 6),
        "q1": round(ordered[n // 4], 6),
        "mean": round(sum(ordered) / n, 6),
        "q3": round(ordered[(3 * n) // 4], 6),
        "max": round(ordered[-1], 6),
        "nonzero_rate": round(sum(1 for value in values if abs(value) > 1e-9) / n, 6),
    }


def score(args: argparse.Namespace) -> None:
    outputs = load_outputs(args.belief_outputs)
    grouped, counters = group_same_state_records(
        args.records,
        require_verified=not args.allow_non_verified,
        max_turn_index=args.max_turn_index if args.max_turn_index >= 0 else None,
    )
    reward_rows: list[dict[str, Any]] = []
    parse_counter: Counter[str] = Counter()
    selected_states = 0
    for state_id, candidates in sorted(grouped.items()):
        if args.min_candidates > 0 and len(candidates) < args.min_candidates:
            counters["skip_state_too_few_candidates"] += 1
            continue
        if args.max_states > 0 and selected_states >= args.max_states:
            break
        selected_states += 1
        for row in candidates:
            record_id = str(row.get("record_id") or "")
            before = outputs.get(f"{record_id}::belief_before", {}).get("parsed_belief_json")
            after = outputs.get(f"{record_id}::belief_after", {}).get("parsed_belief_json")
            if before is None:
                parse_counter["missing_or_unparsed_before"] += 1
            if after is None:
                parse_counter["missing_or_unparsed_after"] += 1

            before_dist = belief_distribution(before)
            after_dist = belief_distribution(after)
            before_entropy = entropy(probs_from_belief(before))
            after_entropy = entropy(probs_from_belief(after))
            entropy_reduction = before_entropy - after_entropy
            before_conf = clamp(safe_float((before or {}).get("top_confidence"), 0.0), 0.0, 1.0)
            after_conf = clamp(safe_float((after or {}).get("top_confidence"), 0.0), 0.0, 1.0)
            confidence_delta = after_conf - before_conf
            before_uncertain = list_field(before, "uncertainty_regions")
            after_uncertain = list_field(after, "uncertainty_regions")
            uncertainty_reduction = (len(before_uncertain) - len(after_uncertain)) / max(1, len(before_uncertain))
            raw_entropy_reduction = entropy_reduction
            raw_confidence_delta = confidence_delta
            raw_uncertainty_reduction = uncertainty_reduction
            low_info = is_low_information_response(row.get("patient_response"))
            if low_info:
                entropy_reduction = min(0.0, entropy_reduction)
                confidence_delta = min(0.0, confidence_delta)
                uncertainty_reduction = min(0.0, uncertainty_reduction)
            query_targets = list_field(before, "candidate_query_targets") or list_field(after, "candidate_query_targets")
            recommended = list_field(before, "recommended_next_inquiry_regions") + before_uncertain
            target_alignment = target_hit_rate(query_targets, recommended)
            query_quality = (
                0.40 * score_field(before, "query_targets_unresolved_region")
                + 0.30 * score_field(before, "query_relevance")
                + 0.20 * score_field(before, "safety_relevance")
                - 0.20 * score_field(before, "query_redundancy")
            )
            belief_update = score_field(after, "belief_update_magnitude")
            if low_info and belief_update > 0:
                belief_update = 0.0
            short_reward = entropy_reduction
            reward_rows.append(
                {
                    "record_id": record_id,
                    "same_state_group_id": state_id,
                    "scenario_id": row.get("scenario_id"),
                    "source_state_id": row.get("source_state_id"),
                    "source_state_hash": row.get("source_state_hash"),
                    "profile_id": row.get("profile_id"),
                    "case_id": row.get("case_id"),
                    "policy_name": row.get("policy_name"),
                    "base_severity": row.get("base_severity"),
                    "turn_index": row.get("turn_index"),
                    "candidate_index": row.get("candidate_index"),
                    "doctor_question": row.get("doctor_question"),
                    "patient_response": row.get("patient_response"),
                    "patient_realizer_mode": row.get("patient_realizer_mode"),
                    "belief_entropy_before": round(before_entropy, 6),
                    "belief_entropy_after": round(after_entropy, 6),
                    "belief_entropy_reduction": round(entropy_reduction, 6),
                    "raw_belief_entropy_reduction": round(raw_entropy_reduction, 6),
                    "belief_top_confidence_before": round(before_conf, 6),
                    "belief_top_confidence_after": round(after_conf, 6),
                    "belief_confidence_delta": round(confidence_delta, 6),
                    "raw_belief_confidence_delta": round(raw_confidence_delta, 6),
                    "uncertainty_regions_before": before_uncertain,
                    "uncertainty_regions_after": after_uncertain,
                    "uncertainty_region_reduction": round(uncertainty_reduction, 6),
                    "raw_uncertainty_region_reduction": round(raw_uncertainty_reduction, 6),
                    "candidate_query_targets": query_targets,
                    "target_alignment_to_unresolved_belief": round(target_alignment, 6),
                    "query_quality_score": round(query_quality, 6),
                    "belief_update_magnitude_score": round(belief_update, 6),
                    "belief_distribution_before": before_dist,
                    "belief_distribution_after": after_dist,
                    "low_information_response": low_info,
                    "terminal_failure": row_is_terminal_failure(row),
                    "trust_delta": round(patient_state_value(row, "after", "trust") - patient_state_value(row, "before", "trust"), 6),
                    "engagement_delta": round(patient_state_value(row, "after", "engagement") - patient_state_value(row, "before", "engagement"), 6),
                    "disclosure_readiness_delta": round(safe_float(row.get("delta_disclosure_readiness")), 6),
                    "short_term_query_reward": round(short_reward, 6),
                    "parsed_before": before is not None,
                    "parsed_after": after is not None,
                    "reward_definition": "H(b_before)-H(b_after), gated so low-information/refusal answers cannot create positive gain; no canonical evidence reward",
                }
            )

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        by_state[str(row.get("same_state_group_id"))].append(row)
    grpo_groups: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for state_id, rows in sorted(by_state.items()):
        rows.sort(key=lambda item: (int(item.get("candidate_index") or 0), str(item.get("record_id") or "")))
        rewards = [safe_float(row.get("short_term_query_reward")) for row in rows]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        var = sum((value - mean) ** 2 for value in rewards) / len(rewards) if rewards else 0.0
        std = math.sqrt(var)
        margin = max(rewards) - min(rewards) if rewards else 0.0
        for row in rows:
            reward = safe_float(row.get("short_term_query_reward"))
            row["same_state_group_size"] = len(rows)
            row["same_state_reward_mean"] = round(mean, 6)
            row["same_state_reward_std"] = round(std, 6)
            row["same_state_reward_margin"] = round(margin, 6)
            row["same_state_reward_advantage"] = round((reward - mean) / std, 6) if std > 1e-9 else 0.0
        if len(rows) >= args.min_candidates and (args.allow_zero_reward_margin or margin > 1e-9):
            first = rows[0]
            grpo_groups.append(
                {
                    "id": f"belief_same_state_group::{state_id}",
                    "prompt": visible_doctor_prompt(first, language=args.language),
                    "responses": [
                        {
                            "text": row.get("doctor_question"),
                            "reward": safe_float(row.get("short_term_query_reward")),
                            "metadata": {
                                "record_id": row.get("record_id"),
                                "same_state_group_id": state_id,
                                "candidate_index": row.get("candidate_index"),
                                "base_severity": row.get("base_severity"),
                                "turn_index": row.get("turn_index"),
                                "patient_response": row.get("patient_response"),
                                "reward_source": "belief_short_term_entropy_reduction",
                                "same_state_reward_advantage": row.get("same_state_reward_advantage"),
                            },
                        }
                        for row in rows
                    ],
                    "metadata": {
                        "source_state_id": state_id,
                        "candidate_count": len(rows),
                        "base_severity": first.get("base_severity"),
                        "turn_index": first.get("turn_index"),
                        "profile_id": first.get("profile_id"),
                        "reward_source": "belief_short_term_entropy_reduction",
                        "same_state_boundary": "same source_state_id; candidates are alternatives, not sequential turns",
                        "uses_hidden_evidence_reward": False,
                    },
                }
            )
        else:
            counters["skip_group_zero_or_too_few_reward_margin"] += 1
        for row in rows:
            value_rows.append(
                {
                    "value_model_task": "belief_guided_same_state_immediate",
                    "record_id": row.get("record_id"),
                    "same_state_group_id": state_id,
                    "base_severity": row.get("base_severity"),
                    "turn_index": row.get("turn_index"),
                    "candidate_action": row.get("doctor_question"),
                    "patient_response": row.get("patient_response"),
                    "immediate_target_gain": safe_float(row.get("short_term_query_reward")),
                    "future_target_gain": None,
                    "action_value_total_gain": safe_float(row.get("short_term_query_reward")),
                    "metadata": {
                        "same_state_candidates": True,
                        "trajectory_level_validation": False,
                        "long_horizon_available": False,
                        "note": "One-step same-state branch; do not treat as long-horizon value label.",
                    },
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reward_path = args.output_dir / "belief_guided_same_state_reward_records.jsonl"
    group_path = args.output_dir / "belief_guided_same_state_grpo_groups.jsonl"
    value_path = args.output_dir / "belief_guided_same_state_value_model_records.jsonl"
    write_jsonl(reward_path, reward_rows)
    write_jsonl(group_path, grpo_groups)
    write_jsonl(value_path, value_rows)
    rewards = [safe_float(row.get("short_term_query_reward")) for row in reward_rows]
    margins = [safe_float((group.get("metadata") or {}).get("candidate_count")) for group in grpo_groups]
    summary = {
        "mode": "score",
        "records": str(args.records),
        "belief_outputs": str(args.belief_outputs),
        "require_verified": not args.allow_non_verified,
        "states_available": len(grouped),
        "states_selected": selected_states,
        "reward_records": len(reward_rows),
        "grpo_groups": len(grpo_groups),
        "parse_counters": dict(parse_counter),
        "counters": dict(counters),
        "severity_distribution": dict(Counter(str(row.get("base_severity")) for row in reward_rows)),
        "turn_index_distribution": dict(
            sorted(Counter(str(row.get("turn_index")) for row in reward_rows).items(), key=lambda item: int(item[0]))
        ) if reward_rows else {},
        "low_information_response_count": sum(1 for row in reward_rows if row.get("low_information_response")),
        "short_term_reward": summarize(rewards),
        "group_candidate_count": summarize(margins),
        "reward_path": str(reward_path),
        "group_path": str(group_path),
        "value_model_record_path": str(value_path),
        "method_boundary": "same-state short-term belief reward only; no canonical evidence recovery, gold diagnosis, or hidden patient evidence is used as reward",
    }
    write_json(args.output_dir / "belief_guided_same_state_reward_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-state belief-guided query reward data.")
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--records", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--allow-non-verified", action="store_true")
    prepare_parser.add_argument("--max-turn-index", type=int, default=-1)
    prepare_parser.add_argument("--max-states", type=int, default=0)
    prepare_parser.add_argument("--min-candidates", type=int, default=2)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--records", type=Path, required=True)
    score_parser.add_argument("--belief-outputs", type=Path, required=True)
    score_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    score_parser.add_argument("--allow-non-verified", action="store_true")
    score_parser.add_argument("--max-turn-index", type=int, default=-1)
    score_parser.add_argument("--max-states", type=int, default=0)
    score_parser.add_argument("--min-candidates", type=int, default=2)
    score_parser.add_argument("--allow-zero-reward-margin", action="store_true")
    score_parser.add_argument("--language", choices=["zh", "en"], default="zh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "score":
        score(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
