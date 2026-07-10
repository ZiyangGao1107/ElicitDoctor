from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_PATH = (
    BASE_DIR
    / "outputs_doctor_policy_baselines_reference_lowinfo_turn8_v1"
    / "mdd5k_doctor_policy_baseline_records.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_llm_patient_realizer_v3_2_json_contract_v2"


SLOT_FORBIDDEN_TERMS = {
    "sleep": ["睡眠", "失眠", "睡不着", "入睡", "早醒", "半夜醒", "醒来"],
    "appetite": ["食欲", "胃口", "吃饭", "吃不下", "不想吃", "饭量"],
    "suicide": ["自杀", "自伤", "轻生", "伤害自己", "结束生命", "危险念头"],
    "self_harm": ["自杀", "自伤", "轻生", "伤害自己", "结束生命", "危险念头"],
    "hopeless": ["绝望", "没希望", "撑不下去", "哭", "崩溃"],
    "cry": ["哭", "流泪", "崩溃", "情绪低落"],
    "hallucination": ["幻听", "幻觉", "声音", "听到", "看见"],
    "work": ["工作", "上班", "绩效", "任务", "学习", "效率"],
    "attention": ["注意力", "走神", "集中不了", "分心"],
    "chest": ["胸闷", "胸口", "呼吸", "喘不过气"],
    "palpitation": ["心慌", "心跳", "心悸"],
    "dizziness": ["头晕", "头痛", "眩晕"],
    "headache": ["头晕", "头痛", "眩晕"],
    "mania": ["兴奋", "精力", "冲动", "狂躁", "停不下来"],
    "cognitive": ["反应慢", "脑子", "思考慢", "记忆"],
}

SEVERE_SAFE_CUE_PHRASES = [
    "不太想说",
    "说不清",
    "不太清楚",
    "不知道怎么说",
    "可以先跳过",
]


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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def units_by_ids(record: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    id_set = set(ids or [])
    result = []
    for unit in record.get("observed_evidence_units") or []:
        if unit.get("unit_id") in id_set:
            result.append(
                {
                    "unit_id": unit.get("unit_id"),
                    "profile_unit_id": unit.get("profile_unit_id"),
                    "unit_text": clean_text(unit.get("unit_text")),
                    "target_relevance": unit.get("target_relevance"),
                }
            )
    return result


def scenario_histories(records: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("scenario_id"))].append(record)
    histories: dict[str, list[dict[str, str]]] = {}
    for scenario_id, items in grouped.items():
        history: list[dict[str, str]] = []
        for item in sorted(items, key=lambda row: int(row.get("turn_index") or 0)):
            histories[f"{scenario_id}::turn_{item.get('turn_index')}"] = list(history)
            history.append(
                {
                    "doctor": clean_text(item.get("doctor_question")),
                    "patient": clean_text(item.get("patient_response")),
                }
            )
    return histories


def render_history(history: list[dict[str, str]], max_turns: int = 4) -> str:
    if not history:
        return "No previous dialogue."
    lines = []
    recent = history[-max_turns:]
    for idx, turn in enumerate(recent, start=1):
        lines.append(f"{idx}. Doctor: {turn.get('doctor', '')}")
        lines.append("   Patient: [hidden to prevent factual leakage; do not infer clinical facts from history]")
    return "\n".join(lines)


def severity_instruction(severity: str, low_info_category: str) -> str:
    if low_info_category == "rapport_open_narrative_no_evidence":
        return (
            "Rapport/opening turn with no factual evidence allowed: acknowledge the doctor gently, "
            "sound willing but hesitant, and do not name any specific symptom, behavior, duration, diagnosis, or risk."
        )
    if severity == "reference_informative":
        return (
            "Reference/informative: answer naturally and clearly. Include the allowed evidence as much as possible, "
            "but avoid robotic listing or exact repetition."
        )
    if severity == "mild_low_info":
        return (
            "Mild low-information: answer relevantly, but omit some details. The answer should feel natural and slightly incomplete."
        )
    if severity == "moderate_low_info":
        return (
            "Moderate low-information: disclose only a small amount of the allowed evidence. Use vague or uncertain language, "
            "such as not being able to explain all details clearly."
        )
    if low_info_category in {"direct_refusal_or_boundary", "topic_deflection"}:
        return (
            "Severe low-information: it is acceptable to avoid, set a boundary, or give a minimal answer. "
            "Do not disclose hidden clinical details."
        )
    return (
        "Severe low-information with partial recovery: give only a small, hesitant amount of allowed evidence. "
        "Do not reveal details that were not allowed."
    )


def response_budget(severity: str, retained_count: int, weakened_count: int) -> dict[str, Any]:
    if severity == "severe_low_info":
        return {
            "max_sentences": 1,
            "max_chinese_chars": 32,
            "clinical_fact_budget": 0 if retained_count == 0 else 1,
            "required_style": "boundary/vague/minimal; leave room for gentle follow-up",
        }
    if severity == "moderate_low_info":
        return {
            "max_sentences": 2,
            "max_chinese_chars": 60,
            "clinical_fact_budget": min(1, retained_count + weakened_count),
            "required_style": "partial and uncertain",
        }
    return {
        "max_sentences": 3,
        "max_chinese_chars": 90,
        "clinical_fact_budget": retained_count + weakened_count,
        "required_style": "natural and incomplete",
    }


def coarse_topic(slot: Any) -> str:
    slot_text = str(slot or "current topic")
    return slot_text.replace("_", " ")


def forbidden_terms_for_slot(slot: Any) -> list[str]:
    slot_text = str(slot or "").lower()
    terms: list[str] = []
    for key, values in SLOT_FORBIDDEN_TERMS.items():
        if key in slot_text:
            terms.extend(values)
    # Severe responses should not echo internal slot labels either.
    for part in slot_text.replace("-", "_").split("_"):
        if part and len(part) > 2:
            terms.append(part)
    seen = set()
    deduped = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bucket(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value >= high:
        return "high"
    return "medium"


def patient_state_before(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("patient_state_before") or record.get("cross_turn_patient_state") or {}
    return state if isinstance(state, dict) else {}


def slot_readiness(record: dict[str, Any], state: dict[str, Any]) -> float:
    slot = str(record.get("target_tree_node") or "")
    readiness_map = state.get("slot_disclosure_readiness") or record.get("slot_disclosure_readiness") or {}
    if isinstance(readiness_map, dict) and slot in readiness_map:
        return _as_float(readiness_map.get(slot), 0.0)
    return _as_float(record.get("slot_disclosure_readiness_before"), 0.0)


def severe_disclosure_stage(record: dict[str, Any], retained_count: int, weakened_count: int) -> str:
    state = patient_state_before(record)
    trust = _as_float(state.get("trust"), 0.25)
    defensiveness = _as_float(state.get("defensiveness"), 0.65)
    readiness = slot_readiness(record, state)
    low_info_category = str(record.get("low_info_category") or "")
    if low_info_category in {"direct_refusal_or_boundary", "topic_deflection"}:
        return "boundary_only"
    if retained_count + weakened_count <= 0:
        return "boundary_only"
    if trust < 0.40 or readiness < 0.30 or defensiveness > 0.72:
        return "boundary_only"
    if readiness < 0.55:
        return "vague_hint_only"
    return "one_weak_hint"


def progressive_disclosure_state(record: dict[str, Any], stage: str) -> dict[str, Any]:
    state = patient_state_before(record)
    trust = _as_float(state.get("trust"), 0.25)
    engagement = _as_float(state.get("engagement"), 0.45)
    defensiveness = _as_float(state.get("defensiveness"), 0.65)
    readiness = slot_readiness(record, state)
    return {
        "disclosure_stage": stage,
        "trust_level": _bucket(trust, low=0.35, high=0.60),
        "engagement_level": _bucket(engagement, low=0.35, high=0.65),
        "defensiveness_level": _bucket(defensiveness, low=0.35, high=0.65),
        "slot_readiness_level": _bucket(readiness, low=0.30, high=0.55),
        "instruction": (
            "This state controls how much the patient is ready to disclose this turn. "
            "Do not mention these state variables to the doctor."
        ),
    }


def build_visibility_contract(
    *,
    record: dict[str, Any],
    retained_units: list[dict[str, Any]],
    weakened_units: list[dict[str, Any]],
) -> dict[str, Any]:
    severity = str(record.get("base_severity") or "")
    topic = coarse_topic(record.get("target_tree_node"))
    if severity == "severe_low_info":
        stage = severe_disclosure_stage(record, len(retained_units), len(weakened_units))
        if stage in {"boundary_only", "vague_hint_only"}:
            allowed_hint_count = 0
        else:
            allowed_hint_count = min(1, len(weakened_units) + len(retained_units))
        # For severe turns, exact unit text is deliberately hidden from the
        # realizer. The verifier still sees the true retained/weakened units.
        return {
            "contract_version": "pcv3_2_json_allowed_only_v3_prompt_hardened",
            "progressive_disclosure_stage": stage,
            "can_say_exact": [],
            "can_hint_about_topic": (
                "the doctor's current concern only in generic words such as 这个/这方面/这件事; "
                "do not name the exact symptom slot"
            ),
            "internal_target_topic_hidden_from_realizer": True,
            "allowed_hint_count": allowed_hint_count,
            "forbidden_surface_terms": forbidden_terms_for_slot(record.get("target_tree_node")),
            "must_not_invent": [
                "new symptoms",
                "duration or frequency",
                "diagnosis labels",
                "risk or self-harm details",
                "causes or life events not stated by the doctor",
            ],
            "must_not_do": [
                "do not restate a symptom phrase as a full fact",
                "do not echo the doctor's symptom words as a patient admission",
                "do not answer with a complete clinical detail",
                "do not add a second clinical fact",
                "do not summarize the hidden profile",
            ],
            "safe_response_moves": [
                "acknowledge the topic without confirming details",
                "say it is hard to explain right now",
                "set a soft boundary",
                "ask to slow down or come back later",
                "give a very small vague hint only when progressive_disclosure_stage is one_weak_hint",
            ],
            "allowed_response_templates": [
                "这个我现在不太想说。",
                "这方面我还说不清。",
                "可以先跳过这个吗？",
                "我现在不知道怎么说。",
            ],
        }
    return {
        "contract_version": "pcv3_2_json_allowed_only_v2",
        "can_say_exact": [unit["unit_text"] for unit in retained_units],
        "can_paraphrase_weakly": [unit["unit_text"] for unit in weakened_units],
        "can_hint_about_topic": topic,
        "must_not_invent": [
            "new symptoms",
            "duration or frequency not listed",
            "diagnosis labels",
            "risk or self-harm details unless explicitly listed in can_say_exact",
            "causes or life events not listed",
        ],
    }


def build_messages(record: dict[str, Any], history: list[dict[str, str]]) -> list[dict[str, str]]:
    retained_units = units_by_ids(record, record.get("retained_unit_ids") or [])
    weakened_units = units_by_ids(record, record.get("weakened_unit_ids") or [])
    severity = str(record.get("base_severity") or "")
    low_info_category = str(record.get("low_info_category") or "")
    visibility_contract = build_visibility_contract(
        record=record,
        retained_units=retained_units,
        weakened_units=weakened_units,
    )
    disclosure_stage = (
        visibility_contract.get("progressive_disclosure_stage")
        if severity == "severe_low_info"
        else "not_applicable"
    )
    progressive_state = progressive_disclosure_state(record, str(disclosure_stage))
    target_topic_for_realizer = (
        "the doctor's current concern; exact internal symptom slot is hidden"
        if severity == "severe_low_info"
        else record.get("target_tree_node")
    )
    hard_constraints = [
        "Use first-person patient language.",
        "Return only JSON.",
        "The patient_response must be the only patient-facing text.",
        "Do not add symptoms, durations, plans, behaviors, diagnoses, or risks not present in the visibility_contract.",
        "Do not reveal details that are not authorized by the visibility_contract.",
        "Do not infer facts from the doctor question or hidden history placeholders.",
        "If the style requires avoidance, make it natural and bounded rather than a dead-end refusal.",
        "If can_say_exact is empty and the setting is severe_low_info, do not name a specific symptom or event; give a minimal bounded reply.",
        "For severe_low_info, do not paraphrase a full clinical fact from the topic. Keep it vague and short.",
        "For severe_low_info, do not echo a symptom word from the doctor's question as a patient admission.",
        "Do not repeat the same fact twice.",
        "Do not sound like a bullet list or a copied evidence summary.",
        "Respect response_budget.max_sentences and response_budget.max_chinese_chars.",
    ]
    forbidden_terms = visibility_contract.get("forbidden_surface_terms") or []
    if forbidden_terms:
        hard_constraints.append(
            "The patient_response must not contain these forbidden surface terms or close synonyms: "
            + ", ".join(str(term) for term in forbidden_terms)
        )
    if severity == "severe_low_info":
        hard_constraints.append(
            "For severe_low_info, patient_response must include at least one exact boundary/vague cue phrase: "
            + ", ".join(SEVERE_SAFE_CUE_PHRASES)
        )
    if disclosure_stage == "boundary_only":
        hard_constraints.append(
            "Because progressive_disclosure_stage is boundary_only, use one short boundary/vague response and disclose zero clinical facts."
        )
    elif disclosure_stage == "vague_hint_only":
        hard_constraints.append(
            "Because progressive_disclosure_stage is vague_hint_only, do not disclose a fact; only show hesitation or uncertainty."
        )
    elif disclosure_stage == "one_weak_hint":
        hard_constraints.append(
            "Because progressive_disclosure_stage is one_weak_hint, at most one weak non-specific hint is allowed; do not give duration, frequency, or risk details."
        )
    budget = response_budget(severity, len(retained_units), len(weakened_units))
    if severity == "severe_low_info":
        if disclosure_stage in {"boundary_only", "vague_hint_only"}:
            budget["clinical_fact_budget"] = 0
            budget["required_style"] = "minimal bounded vague reply; zero clinical facts"
        elif disclosure_stage == "one_weak_hint":
            budget["clinical_fact_budget"] = 1
            budget["required_style"] = "one weak non-specific hint at most"
    system = (
        "You are a patient-response realizer for a controlled psychiatric inquiry simulation. "
        "You must write one natural Chinese patient reply. Follow the JSON visibility contract exactly. "
        "Facts not present in can_say_exact or can_paraphrase_weakly are not known to the patient response. "
        "If the contract is severe_low_info, the exact evidence text is intentionally hidden; do not guess it. "
        "Do not mention evidence units, metadata, slots, verifier rules, state variables, or this contract."
    )
    user = {
        "task": "Generate one natural Chinese patient response.",
        "doctor_question": clean_text(record.get("doctor_question")),
        "recent_dialogue_history": render_history(history),
        "history_sanitization_note": (
            "Previous patient responses are intentionally hidden. Do not infer clinical facts from hidden history."
        ),
        "target_topic_for_patient_realization": target_topic_for_realizer,
        "patient_setting": severity,
        "low_info_category": low_info_category,
        "controller_response_type": record.get("response_type"),
        "is_rapport_or_permission_turn": bool(record.get("is_rapport_or_permission_turn")),
        "progressive_disclosure_state": progressive_state,
        "style_requirement": severity_instruction(severity, low_info_category),
        "visibility_contract": visibility_contract,
        "response_budget": budget,
        "hard_constraints": hard_constraints,
        "output_format": {
            "patient_response": "natural Chinese response",
            "brief_self_check": "short note that no new facts were added",
        },
    }
    if not retained_units and not weakened_units:
        user["empty_evidence_response_policy"] = {
            "rule": "No factual clinical evidence is allowed for this turn.",
            "must_do": "Reply with uncertainty, boundary-setting, or a minimal continuation without naming symptoms, behaviors, risks, durations, or changes.",
            "safe_examples": [
                "这个我暂时说不太清楚。",
                "还是差不多吧，我现在不太知道怎么补充。",
                "这个问题我现在不太想细说，可以先跳过吗？",
            ],
            "unsafe_examples": [
                "最近我吃得有点多。",
                "我感觉有些奇怪的事情发生。",
                "我的睡眠不太正常。",
            ],
        }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def build_requests(
    *,
    trajectory_path: Path,
    policies: set[str] | None,
    severities: set[str] | None,
    target_slots: set[str] | None,
    max_requests: int | None,
    max_requests_per_cell: int | None,
    sample_seed: int | None,
) -> list[dict[str, Any]]:
    records = list(iter_jsonl(trajectory_path))
    histories = scenario_histories(records)
    candidates = [
        record
        for record in records
        if record.get("doctor_question")
        and record.get("patient_response")
        and (policies is None or record.get("policy_name") in policies)
        and (severities is None or record.get("base_severity") in severities)
        and (target_slots is None or record.get("target_tree_node") in target_slots)
    ]
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(candidates)

    selected: list[dict[str, Any]] = []
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    for record in candidates:
        cell = (
            str(record.get("base_severity")),
            str(record.get("target_tree_node")),
            str(record.get("low_info_category")),
        )
        if max_requests_per_cell is not None and cell_counts[cell] >= max_requests_per_cell:
            continue
        selected.append(record)
        cell_counts[cell] += 1
        if max_requests is not None and len(selected) >= max_requests:
            break

    requests = []
    for record in selected:
        turn_key = f"{record.get('scenario_id')}::turn_{record.get('turn_index')}"
        retained_units = units_by_ids(record, record.get("retained_unit_ids") or [])
        weakened_units = units_by_ids(record, record.get("weakened_unit_ids") or [])
        withheld_units = units_by_ids(record, record.get("withheld_unit_ids") or record.get("removed_unit_ids") or [])
        removed_units = units_by_ids(record, record.get("removed_unit_ids") or [])
        forbidden_units = units_by_ids(record, record.get("forbidden_unit_ids") or [])
        request_id = f"{record.get('record_id')}::llm_patient_realizer"
        requests.append(
            {
                "request_id": request_id,
                "task_name": "mdd5k_llm_patient_realizer",
                "prompt_protocol_version": "llm_patient_realizer_pcv3_2_json_contract_v3_prompt_hardened",
                "history_mode": "doctor_history_only_patient_text_hidden",
                "source_trajectory_file": str(trajectory_path),
                "source_record_id": record.get("record_id"),
                "scenario_id": record.get("scenario_id"),
                "profile_id": record.get("profile_id"),
                "case_id": record.get("case_id"),
                "policy_name": record.get("policy_name"),
                "base_severity": record.get("base_severity"),
                "turn_index": record.get("turn_index"),
                "target_tree_node": record.get("target_tree_node"),
                "low_info_category": record.get("low_info_category"),
                "doctor_question": clean_text(record.get("doctor_question")),
                "source_rule_based_patient_response": clean_text(record.get("patient_response")),
                "controller_version": record.get("controller_version"),
                "controller_response_type": record.get("response_type"),
                "is_rapport_or_permission_turn": bool(record.get("is_rapport_or_permission_turn")),
                "pcv3_1_routing_source": record.get("pcv3_1_routing_source"),
                "messages": build_messages(record, histories.get(turn_key, [])),
                "model_visible_fields": ["messages"],
                "hidden_verifier_metadata_not_for_realizer": {
                    "retained_units": retained_units,
                    "weakened_units": weakened_units,
                    "removed_units": removed_units,
                    "withheld_units": withheld_units,
                    "forbidden_units": forbidden_units,
                    "observed_evidence_units": record.get("observed_evidence_units") or [],
                    "retained_unit_ids": record.get("retained_unit_ids") or [],
                    "weakened_unit_ids": record.get("weakened_unit_ids") or [],
                    "removed_unit_ids": record.get("removed_unit_ids") or [],
                    "withheld_unit_ids": record.get("withheld_unit_ids") or record.get("removed_unit_ids") or [],
                    "forbidden_unit_ids": record.get("forbidden_unit_ids") or [],
                    "topic_responsiveness": record.get("topic_responsiveness"),
                    "information_retention": record.get("information_retention"),
                    "clarity": record.get("clarity"),
                    "g_target": record.get("g_target"),
                    "target_slot_evidence_unit_count": record.get("target_slot_evidence_unit_count"),
                    "patient_state_before": record.get("patient_state_before"),
                    "patient_state_after": record.get("patient_state_after"),
                    "cross_turn_patient_state": record.get("cross_turn_patient_state"),
                },
                "expected_output": {
                    "patient_response": "natural Chinese response constrained by allowed evidence",
                    "brief_self_check": "short no-new-fact self check",
                },
            }
        )
    return requests


def write_protocol(path: Path, request_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LLM Patient Realizer Request Protocol V3.1",
        "",
        "Date: 2026-07-07",
        "",
        "## Goal",
        "",
        "Improve patient-response readability while preserving controller-level factual and disclosure constraints.",
        "",
        "## Division of Labor",
        "",
        "- Controller decides target slot, retained evidence, weakened evidence, removed evidence, and low-information category.",
        "- LLM realizer only verbalizes allowed/weakened evidence in natural Chinese.",
        "- Rule-based verifier checks hard constraints before a response can replace the deterministic fallback.",
        "- Previous patient responses are hidden from the LLM prompt to prevent history-induced evidence leakage.",
        "- If no evidence unit is allowed, the LLM must produce a non-factual low-information reply.",
        "",
        "## Request File",
        "",
        f"- `{request_path.name}`",
        f"- requests: {summary['num_requests']}",
        "",
        "## Realizer Cannot See",
        "",
        "- withheld/removed evidence units",
        "- gold diagnosis",
        "- verifier scores",
        "- full patient profile",
        "",
        "## Required Output",
        "",
        "```json",
        "{",
        '  "patient_response": "自然中文患者回答",',
        '  "brief_self_check": "未新增事实"',
        "}",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LLM patient realizer requests from controller trajectory records.")
    parser.add_argument("--trajectory-path", type=Path, default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--severities", nargs="*", default=None)
    parser.add_argument("--target-slots", nargs="*", default=None)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--max-requests-per-cell", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_requests = args.max_requests if args.max_requests and args.max_requests > 0 else None
    max_requests_per_cell = args.max_requests_per_cell if args.max_requests_per_cell and args.max_requests_per_cell > 0 else None
    requests = build_requests(
        trajectory_path=args.trajectory_path,
        policies=set(args.policies) if args.policies else None,
        severities=set(args.severities) if args.severities else None,
        target_slots=set(args.target_slots) if args.target_slots else None,
        max_requests=max_requests,
        max_requests_per_cell=max_requests_per_cell,
        sample_seed=args.sample_seed,
    )
    request_path = args.output_dir / "mdd5k_llm_patient_realizer_requests.jsonl"
    summary_path = args.output_dir / "mdd5k_llm_patient_realizer_request_summary.json"
    protocol_path = args.output_dir / "LLM_PATIENT_REALIZER_REQUEST_PROTOCOL_V3_1.md"
    write_jsonl(request_path, requests)
    summary = {
        "num_requests": len(requests),
        "trajectory_path": str(args.trajectory_path),
        "request_path": str(request_path),
        "protocol_path": str(protocol_path),
        "policies": sorted({str(record.get("policy_name")) for record in requests}),
        "severities": sorted({str(record.get("base_severity")) for record in requests}),
        "target_slots": sorted({str(record.get("target_tree_node")) for record in requests}),
        "low_info_categories": sorted({str(record.get("low_info_category")) for record in requests}),
        "prompt_protocol_version": "llm_patient_realizer_pcv3_2_json_contract_v3_prompt_hardened",
        "history_mode": "doctor_history_only_patient_text_hidden",
        "max_requests": max_requests,
        "max_requests_per_cell": max_requests_per_cell,
        "sample_seed": args.sample_seed,
    }
    write_json(summary_path, summary)
    write_protocol(protocol_path, request_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
