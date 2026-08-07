from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_belief_guided_query_reward_data"


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


LOW_INFORMATION_RESPONSE_PATTERNS = (
    "不想说",
    "不太想说",
    "还不想说",
    "暂时不想说",
    "不愿意说",
    "不太愿意说",
    "不方便说",
    "不想谈",
    "不太想谈",
    "说不清楚",
    "说不清",
    "暂时说不太清楚",
    "不知道怎么说",
    "没法说",
    "不想讲",
    "不愿讲",
)


def is_low_information_response(text: Any) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return True
    return any(pattern in cleaned for pattern in LOW_INFORMATION_RESPONSE_PATTERNS)


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
    mode = str(row.get("patient_realizer_mode") or "")
    return mode == "verified_llm_cache"


def group_records(records_path: Path, require_verified: bool, max_turn_index: int | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(records_path):
        if require_verified and not record_is_verified(row):
            continue
        turn = int(row.get("turn_index") or 0)
        if max_turn_index is not None and turn > max_turn_index:
            continue
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            continue
        grouped[scenario_id].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("turn_index") or 0))
    return grouped


def render_visible_dialogue(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        question = clean_text(row.get("doctor_question"))
        response = clean_text(row.get("patient_response"))
        if question:
            lines.append(f"{idx}. 医生：{question}")
        if response:
            lines.append(f"   患者：{response}")
    return "\n".join(lines) if lines else "（目前还没有可见对话。）"


def build_belief_messages(
    *,
    stage: str,
    dialogue_text: str,
    doctor_question: str,
    patient_response: str,
) -> list[dict[str, str]]:
    system = (
        "You are a calibrated psychiatric belief-state evaluator. Judge only from the visible doctor-patient dialogue. "
        "Do not use hidden patient profiles, simulator metadata, canonical evidence, gold diagnosis labels, or external facts. "
        "Your output will be used to build query reward/value data, so it must be conservative, calibrated, and strictly JSON. "
        "Do not copy placeholder values from the instruction; estimate every score from the visible dialogue."
    )
    if stage == "before":
        stage_text = (
            "Stage: query_before. You see the dialogue before the current doctor question and the candidate doctor question. "
            "Judge whether this question targets an unresolved belief region and whether it is likely to reduce uncertainty."
        )
        response_block = "Current patient answer: not observed yet."
    else:
        stage_text = (
            "Stage: query_after. You see the same doctor question plus the patient answer. "
            "Estimate how diagnostic belief, unresolved regions, and information state changed after this turn. "
            "If the patient answer concretely reveals symptoms, duration, impairment, safety risk, or negation, update confidence and uncertainty. "
            "If the patient refuses or gives a vague answer, keep belief nearly unchanged and set belief_update_magnitude low."
        )
        response_block = f"Current patient answer: {patient_response or '(empty)'}"

    user = (
        f"{stage_text}\n\n"
        f"Visible dialogue history:\n{dialogue_text}\n\n"
        f"Current doctor question: {doctor_question or '(empty)'}\n"
        f"{response_block}\n\n"
        "Return exactly one JSON object, no markdown, no extra text. Required fields:\n"
        "- diagnostic_hypotheses: array of objects with label:string and prob:number in [0,1]. Use broad hypotheses such as depressive_episode, anxiety_disorder, adjustment_stress, bipolar_related, substance_related, insufficient_information.\n"
        "- top_confidence: number in [0,1], equal to the highest current hypothesis probability.\n"
        "- uncertainty_regions: array of short strings for still-unresolved regions, e.g. mood, anhedonia, sleep, appetite, duration, impairment, suicide_risk, mania, substance, medical_causes.\n"
        "- recommended_next_inquiry_regions: array of short strings for high-value next inquiry regions.\n"
        "- candidate_query_targets: array of short strings naming what the current doctor question targets.\n"
        "- query_targets_unresolved_region: integer 0-5; high if the question targets a still-unresolved belief region.\n"
        "- query_relevance: integer 0-5; high if the question is clinically relevant to current uncertainty.\n"
        "- query_redundancy: integer 0-5; high if the question repeats already answered information.\n"
        "- safety_relevance: integer 0-5; high if the question appropriately covers safety risk.\n"
        "- belief_update_magnitude: integer 0-5; for query_after, high if the patient answer changes belief/uncertainty; for query_before, estimate expected update potential.\n"
        "- brief_reason: one short reason grounded only in visible dialogue.\n\n"
        "Calibration rules: do not use hidden evidence; do not infer certainty from diagnosis names alone; concrete disclosed information should reduce matching uncertainty regions; vague/refusal answers should not reduce uncertainty much."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prepare_requests(args: argparse.Namespace) -> None:
    grouped = group_records(
        args.records,
        require_verified=not args.allow_non_verified,
        max_turn_index=args.max_turn_index if args.max_turn_index >= 0 else None,
    )
    rows: list[dict[str, Any]] = []
    scenario_count = 0
    for scenario_id, records in grouped.items():
        if args.max_scenarios > 0 and scenario_count >= args.max_scenarios:
            break
        scenario_count += 1
        for idx, row in enumerate(records):
            record_id = str(row.get("record_id") or f"{scenario_id}::turn_{row.get('turn_index')}")
            before_rows = records[:idx]
            after_rows = records[: idx + 1]
            doctor_question = clean_text(row.get("doctor_question"))
            patient_response = clean_text(row.get("patient_response"))
            common = {
                "task_name": "belief_guided_query_reward_eval",
                "source_record_id": record_id,
                "scenario_id": scenario_id,
                "profile_id": row.get("profile_id"),
                "case_id": row.get("case_id"),
                "policy_name": row.get("policy_name"),
                "base_severity": row.get("base_severity"),
                "turn_index": row.get("turn_index"),
                "doctor_question": doctor_question,
                "patient_response": patient_response,
                "prompt_protocol_version": "belief_guided_query_reward_v1",
            }
            rows.append(
                {
                    **common,
                    "request_id": f"{record_id}::belief_before",
                    "belief_stage": "before",
                    "messages": build_belief_messages(
                        stage="before",
                        dialogue_text=render_visible_dialogue(before_rows),
                        doctor_question=doctor_question,
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
                        dialogue_text=render_visible_dialogue(after_rows),
                        doctor_question=doctor_question,
                        patient_response=patient_response,
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "belief_guided_query_belief_requests.jsonl"
    write_jsonl(request_path, rows)
    summary = {
        "mode": "prepare",
        "records": str(args.records),
        "require_verified": not args.allow_non_verified,
        "max_turn_index": args.max_turn_index,
        "max_scenarios": args.max_scenarios,
        "scenarios_used": scenario_count,
        "belief_requests": len(rows),
        "request_path": str(request_path),
        "note": "Requests use visible dialogue only; no canonical evidence or gold diagnosis is included in the prompt.",
    }
    write_json(args.output_dir / "belief_guided_query_belief_request_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = clean_text(raw)
    if not text:
        return None
    if "```" in text:
        parts = [part.strip() for part in text.split("```") if part.strip()]
        for part in parts:
            candidate = part
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
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


def load_outputs(output_path: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(output_path):
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        parsed = parse_json_object(str(row.get("raw_output") or ""))
        outputs[request_id] = {**row, "parsed_belief_json": parsed}
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
    if total > 0:
        probs = [p / total for p in probs]
    return probs


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
    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}


def kl_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    if not left:
        return 0.0
    total = 0.0
    for key, p in left.items():
        if p <= 0:
            continue
        q = max(right.get(key, 0.0), 1e-12)
        total += p * math.log(p / q)
    return total


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    if not left and not right:
        return 0.0
    keys = set(left) | set(right)
    midpoint = {key: 0.5 * left.get(key, 0.0) + 0.5 * right.get(key, 0.0) for key in keys}
    value = 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)
    return value / math.log(2.0)


def list_field(parsed: dict[str, Any] | None, key: str) -> list[str]:
    if not parsed:
        return []
    value = parsed.get(key)
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = clean_text(item).lower()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


def target_hit_rate(query_targets: list[str], unresolved_or_recommended: list[str]) -> float:
    if not query_targets or not unresolved_or_recommended:
        return 0.0
    unresolved = set(unresolved_or_recommended)
    hits = sum(1 for target in query_targets if target in unresolved)
    return hits / max(1, len(query_targets))


def score_field(parsed: dict[str, Any] | None, key: str) -> float:
    if not parsed:
        return 0.0
    return clamp(safe_float(parsed.get(key), 0.0) / 5.0, 0.0, 1.0)


def patient_state_value(row: dict[str, Any], stage: str, key: str) -> float:
    state = row.get(f"patient_state_{stage}") or {}
    return safe_float(state.get(key), 0.0)


def row_is_terminal_failure(row: dict[str, Any]) -> bool:
    if bool(row.get("patient_active_termination")):
        return True
    for key in ("response_type", "patient_response_type", "patient_status", "termination_reason"):
        value = str(row.get(key) or "").lower()
        if value in {"patient_active_termination", "active_termination", "terminated"}:
            return True
    for key in ("patient_state_after", "patient_state"):
        state = row.get(key) or {}
        if isinstance(state, dict) and bool(state.get("terminated")):
            return True
    return False


def plateau_window_triggered(
    rows: list[dict[str, Any]],
    end_idx: int,
    *,
    window: int,
    entropy_epsilon: float,
    js_epsilon: float,
    low_info_rate: float,
) -> bool:
    if window <= 0 or end_idx - window + 1 < 0:
        return False
    values = rows[end_idx - window + 1 : end_idx + 1]
    entropy_ok = all(abs(safe_float(row.get("belief_entropy_reduction"))) <= entropy_epsilon for row in values)
    js_ok = all(safe_float(row.get("belief_js_from_prev_after")) <= js_epsilon for row in values)
    low_info = sum(1 for row in values if row.get("low_information_response")) / max(1, len(values))
    return entropy_ok and js_ok and low_info >= low_info_rate


def build_value_model_input(prefix_dialogue: str, row: dict[str, Any]) -> str:
    return (
        "Task: estimate long-horizon belief value for the current doctor query.\n"
        "Only visible dialogue and the current candidate query are available.\n"
        f"Base severity: {row.get('base_severity') or ''}\n"
        f"Turn index: {row.get('turn_index')}\n"
        f"Current belief entropy before query: {safe_float(row.get('belief_entropy_before')):.6f}\n"
        f"Uncertainty regions before query: {', '.join(row.get('uncertainty_regions_before') or [])}\n"
        f"Candidate query targets: {', '.join(row.get('candidate_query_targets') or [])}\n"
        f"Target alignment to unresolved belief: {safe_float(row.get('target_alignment_to_unresolved_belief')):.6f}\n"
        f"Visible dialogue before query:\n{prefix_dialogue}\n\n"
        f"Candidate doctor query:\n{row.get('doctor_question') or ''}"
    ).strip()


def score_requests(args: argparse.Namespace) -> None:
    outputs = load_outputs(args.belief_outputs)
    records_by_id: dict[str, dict[str, Any]] = {}
    prefix_by_id: dict[str, str] = {}
    grouped = group_records(
        args.records,
        require_verified=not args.allow_non_verified,
        max_turn_index=args.max_turn_index if args.max_turn_index >= 0 else None,
    )
    ordered_records: list[dict[str, Any]] = []
    scenario_count = 0
    for scenario_records in grouped.values():
        if args.max_scenarios > 0 and scenario_count >= args.max_scenarios:
            break
        scenario_count += 1
        for idx, row in enumerate(scenario_records):
            record_id = str(row.get("record_id") or f"{row.get('scenario_id')}::turn_{row.get('turn_index')}")
            prefix_by_id[record_id] = render_visible_dialogue(scenario_records[:idx])
            ordered_records.append(row)
    for row in ordered_records:
        record_id = str(row.get("record_id") or f"{row.get('scenario_id')}::turn_{row.get('turn_index')}")
        records_by_id[record_id] = row

    reward_rows: list[dict[str, Any]] = []
    parse_counter: Counter[str] = Counter()
    for record_id, row in records_by_id.items():
        before = outputs.get(f"{record_id}::belief_before", {}).get("parsed_belief_json")
        after = outputs.get(f"{record_id}::belief_after", {}).get("parsed_belief_json")
        if not before:
            parse_counter["missing_or_unparsed_before"] += 1
        if not after:
            parse_counter["missing_or_unparsed_after"] += 1
        low_information_response = is_low_information_response(row.get("patient_response"))
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
        if low_information_response:
            if entropy_reduction > 0:
                after_entropy = before_entropy
                entropy_reduction = 0.0
                after_dist = before_dist
            if confidence_delta > 0:
                after_conf = before_conf
                confidence_delta = 0.0
            if uncertainty_reduction > 0:
                after_uncertain = before_uncertain
                uncertainty_reduction = 0.0
        query_targets = list_field(before, "candidate_query_targets") or list_field(after, "candidate_query_targets")
        recommended = list_field(before, "recommended_next_inquiry_regions") + before_uncertain
        target_alignment = target_hit_rate(query_targets, recommended)
        query_quality = (
            0.40 * score_field(before, "query_targets_unresolved_region")
            + 0.30 * score_field(before, "query_relevance")
            + 0.20 * score_field(before, "safety_relevance")
            - 0.20 * score_field(before, "query_redundancy")
        )
        belief_update_magnitude = score_field(after, "belief_update_magnitude")
        if low_information_response:
            belief_update_magnitude = 0.0
        trust_delta = patient_state_value(row, "after", "trust") - patient_state_value(row, "before", "trust")
        engagement_delta = patient_state_value(row, "after", "engagement") - patient_state_value(row, "before", "engagement")
        readiness_delta = safe_float(row.get("delta_disclosure_readiness"), 0.0)
        short_term_reward = entropy_reduction
        reward_rows.append(
            {
                "record_id": record_id,
                "scenario_id": row.get("scenario_id"),
                "profile_id": row.get("profile_id"),
                "case_id": row.get("case_id"),
                "policy_name": row.get("policy_name"),
                "base_severity": row.get("base_severity"),
                "turn_index": row.get("turn_index"),
                "doctor_question": row.get("doctor_question"),
                "patient_response": row.get("patient_response"),
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
                "belief_update_magnitude_score": round(belief_update_magnitude, 6),
                "belief_distribution_before": before_dist,
                "belief_distribution_after": after_dist,
                "low_information_response": low_information_response,
                "terminal_failure": row_is_terminal_failure(row),
                "trust_delta": round(trust_delta, 6),
                "engagement_delta": round(engagement_delta, 6),
                "disclosure_readiness_delta": round(readiness_delta, 6),
                "short_term_query_reward": round(short_term_reward, 6),
                "reward_definition": (
                    "short-term reward is normalized diagnostic belief entropy reduction H(before)-H(after); "
                    "low-information/refusal answers cannot create positive belief gain; "
                    "no canonical evidence recovery or gold diagnosis is used"
                ),
                "parsed_before": before is not None,
                "parsed_after": after is not None,
            }
        )

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        by_scenario[str(row.get("scenario_id"))].append(row)
    for rows in by_scenario.values():
        rows.sort(key=lambda item: int(item.get("turn_index") or 0))
        if not rows:
            continue
        for idx, item in enumerate(rows):
            if idx == 0:
                item["belief_js_from_prev_after"] = 1.0
            else:
                prev = rows[idx - 1].get("belief_distribution_after") or {}
                curr = item.get("belief_distribution_after") or {}
                item["belief_js_from_prev_after"] = round(js_divergence(prev, curr), 6)

        plateau_events: list[dict[str, Any]] = []
        for idx in range(len(rows)):
            if plateau_window_triggered(
                rows,
                idx,
                window=args.t3_plateau_window,
                entropy_epsilon=args.t3_entropy_epsilon,
                js_epsilon=args.t3_js_epsilon,
                low_info_rate=args.t3_low_info_rate,
            ):
                plateau_events.append(
                    {
                        "start": max(0, idx - args.t3_plateau_window + 1),
                        "end": idx,
                        "reason": "belief_plateau_low_info_tail",
                    }
                )

        terminal_indices = [idx for idx, item in enumerate(rows) if item.get("terminal_failure")]
        for idx, item in enumerate(rows):
            cutoff = len(rows)
            truncation_reason = "max_turn_or_dialogue_end"
            truncation_event_start = None
            truncation_event_end = None
            for terminal_idx in terminal_indices:
                if terminal_idx >= idx + 1 and terminal_idx < cutoff:
                    cutoff = terminal_idx
                    truncation_reason = "patient_active_termination"
                    truncation_event_start = terminal_idx
                    truncation_event_end = terminal_idx
            for event in plateau_events:
                if int(event["end"]) >= idx + 1:
                    event_cutoff = max(idx + 1, int(event["start"]))
                    if event_cutoff < cutoff:
                        cutoff = event_cutoff
                        truncation_reason = str(event["reason"])
                        truncation_event_start = int(event["start"])
                        truncation_event_end = int(event["end"])

            long_value = 0.0
            future_steps_used = 0
            for future_idx in range(idx + 1, cutoff):
                discount = args.discount_gamma ** (future_idx - idx - 1)
                long_value += discount * safe_float(rows[future_idx].get("belief_entropy_reduction"))
                future_steps_used += 1
            action_value_total = safe_float(item.get("short_term_query_reward")) + args.value_model_lambda * long_value
            item["t3_future_cutoff_index"] = cutoff
            item["t3_truncation_reason"] = truncation_reason
            item["t3_truncation_event_start_index"] = truncation_event_start
            item["t3_truncation_event_end_index"] = truncation_event_end
            item["t3_future_steps_used"] = future_steps_used
            item["future_discounted_uncertainty_reduction"] = round(long_value, 6)
            item["long_horizon_belief_value_label"] = round(long_value, 6)
            item["action_value_total_belief"] = round(action_value_total, 6)
            item["long_horizon_label_definition"] = (
                "discounted future diagnostic-belief entropy reduction until T3 truncation; "
                "truncates belief-plateau/low-information tails and terminal patient failures; "
                "does not use canonical evidence recovery, gold diagnosis, or hidden patient evidence"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "belief_guided_query_reward_records.jsonl"
    write_jsonl(record_path, reward_rows)
    value_model_rows: list[dict[str, Any]] = []
    for item in reward_rows:
        record_id = str(item.get("record_id") or "")
        value_model_rows.append(
            {
                "value_model_task": "belief_guided_t3_long_horizon",
                "record_id": record_id,
                "state_id": str(item.get("scenario_id") or ""),
                "scenario_id": item.get("scenario_id"),
                "profile_id": item.get("profile_id"),
                "case_id": item.get("case_id"),
                "base_severity": item.get("base_severity"),
                "turn_index": item.get("turn_index"),
                "candidate_action": item.get("doctor_question"),
                "patient_response": item.get("patient_response"),
                "immediate_target_gain": safe_float(item.get("short_term_query_reward")),
                "future_target_gain": safe_float(item.get("long_horizon_belief_value_label")),
                "future_any_gain": 1.0 if safe_float(item.get("long_horizon_belief_value_label")) > 0 else 0.0,
                "action_value_total_gain": safe_float(item.get("action_value_total_belief")),
                "value_model_input": build_value_model_input(prefix_by_id.get(record_id, ""), item),
                "first_response": {
                    "response_type": "low_information" if item.get("low_information_response") else "informative_or_partial",
                    "doctor_recovery_quality": "belief_guided",
                },
                "metadata": {
                    "target_family": "belief_guided_t3",
                    "short_term_definition": "H(b_before)-H(b_after)",
                    "long_term_definition": "discounted future entropy reduction until T3 truncation",
                    "same_state_candidates": False,
                    "trajectory_level_validation": True,
                    "t3_truncation_reason": item.get("t3_truncation_reason"),
                    "t3_future_steps_used": item.get("t3_future_steps_used"),
                    "low_information_response": item.get("low_information_response"),
                    "terminal_failure": item.get("terminal_failure"),
                },
            }
        )
    value_record_path = args.output_dir / "belief_guided_t3_value_model_records.jsonl"
    write_jsonl(value_record_path, value_model_rows)
    values = [safe_float(row.get("short_term_query_reward")) for row in reward_rows]
    long_values = [safe_float(row.get("long_horizon_belief_value_label")) for row in reward_rows]
    t3_reasons = Counter(str(row.get("t3_truncation_reason")) for row in reward_rows)
    summary = {
        "mode": "score",
        "records": str(args.records),
        "belief_outputs": str(args.belief_outputs),
        "require_verified": not args.allow_non_verified,
        "max_turn_index": args.max_turn_index,
        "max_scenarios": args.max_scenarios,
        "scenarios_used": scenario_count,
        "reward_records": len(reward_rows),
        "parse_counters": dict(parse_counter),
        "severity_distribution": dict(Counter(str(row.get("base_severity")) for row in reward_rows)),
        "low_information_response_count": sum(1 for row in reward_rows if row.get("low_information_response")),
        "low_information_response_rate": round(
            sum(1 for row in reward_rows if row.get("low_information_response")) / max(1, len(reward_rows)),
            6,
        ),
        "short_term_reward": summarize_values(values),
        "long_horizon_belief_value_label": summarize_values(long_values),
        "t3": {
            "discount_gamma": args.discount_gamma,
            "plateau_window": args.t3_plateau_window,
            "entropy_epsilon": args.t3_entropy_epsilon,
            "js_epsilon": args.t3_js_epsilon,
            "low_info_rate": args.t3_low_info_rate,
            "truncation_reasons": dict(t3_reasons),
        },
        "record_path": str(record_path),
        "value_model_record_path": str(value_record_path),
        "method_boundary": "No canonical evidence recovery, gold diagnosis, or hidden patient evidence is used as query reward.",
    }
    write_json(args.output_dir / "belief_guided_query_reward_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_values(values: list[float]) -> dict[str, float]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build belief-guided query reward/value data without evidence-recovery reward leakage.")
    sub = parser.add_subparsers(dest="mode", required=True)

    prepare = sub.add_parser("prepare", help="Prepare before/after belief evaluator requests from online replay records.")
    prepare.add_argument("--records", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--allow-non-verified", action="store_true")
    prepare.add_argument("--max-turn-index", type=int, default=-1)
    prepare.add_argument("--max-scenarios", type=int, default=0)

    score = sub.add_parser("score", help="Score belief before/after outputs into query reward and long-horizon belief value labels.")
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--belief-outputs", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    score.add_argument("--allow-non-verified", action="store_true")
    score.add_argument("--max-turn-index", type=int, default=-1)
    score.add_argument("--max-scenarios", type=int, default=0)
    score.add_argument("--discount-gamma", type=float, default=0.95)
    score.add_argument("--value-model-lambda", type=float, default=1.0)
    score.add_argument("--t3-plateau-window", type=int, default=3)
    score.add_argument("--t3-entropy-epsilon", type=float, default=0.01)
    score.add_argument("--t3-js-epsilon", type=float, default=0.02)
    score.add_argument("--t3-low-info-rate", type=float, default=0.67)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare_requests(args)
    elif args.mode == "score":
        score_requests(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
