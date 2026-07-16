from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _profile_grounded_controller import (
    BROAD_CONTEXT_SLOTS,
    SENSITIVE_TARGETS,
    SLOT_DISPLAY,
    STRESS_TARGETS,
    alternative_deflection_unit,
    clean_text,
    compute_information_retention,
    ensure_sentence,
    join_units,
    make_doctor_question,
    ordered_units_by_ids,
    priority_unit_ids,
    selected_profile_units,
)
from online_query_interpreter import OnlineQueryInterpreter, load_json


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = (
    BASE_DIR
    / "outputs_patient_profiles"
    / "mdd5k_dialogue_derived_patient_profiles.jsonl"
)
DEFAULT_SCHEMA_PATH = BASE_DIR / "schemas" / "mdd5k_symptom_slot_schema.json"
DEFAULT_GROUP_DIR = BASE_DIR / "outputs_profile_grounded_environment"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_patient_controller_base"

SEVERITIES = [
    "reference_informative",
    "fully_cooperative",
    "random_disclosure",
    "mild_low_info",
    "moderate_low_info",
    "severe_low_info",
]

SPECIFICITY_CUES = [
    "具体",
    "频率",
    "多久",
    "持续",
    "程度",
    "严重",
    "什么时候",
    "什么情况下",
    "最近一次",
    "例子",
    "影响",
    "强度",
    "计划",
    "准备",
    "控制",
    "几次",
    "多大",
    "有没有",
]

ANAPHORA_CUES = ["刚才", "这个", "这方面", "这件事", "再说", "再讲", "继续说", "多说"]


INITIAL_QUESTION_OVERRIDES = {
    "binge_eating": "我想了解一下最近有没有暴饮暴食、吃很多或控制不住进食的情况？",
    "school_or_study_status": "我想了解一下你最近的学习、学校上课、作业考试或完成任务有没有受到影响？",
    "work_status": "我想了解一下你最近的工作、上班、绩效或完成任务有没有受到影响？",
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_severity(level: str) -> str:
    aliases = {
        "reference": "reference_informative",
        "full": "fully_cooperative",
        "cooperative": "fully_cooperative",
        "full_cooperation": "fully_cooperative",
        "random": "random_disclosure",
        "random_low_info": "random_disclosure",
        "random_low_disclosure": "random_disclosure",
        "probabilistic_low_info": "random_disclosure",
        "mild": "mild_low_info",
        "moderate": "moderate_low_info",
        "severe": "severe_low_info",
    }
    level = aliases.get(level, level)
    if level not in SEVERITIES:
        raise ValueError(f"Unknown base severity: {level}")
    return level


def contains_any(text: str, cues: list[str]) -> bool:
    return any(cue in (text or "") for cue in cues)


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stable_unit_float(*parts: Any) -> float:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def unit_profile_id(unit: dict[str, Any]) -> str:
    return str(unit.get("profile_unit_id") or unit.get("unit_id"))


def unit_order(unit_id: str) -> int:
    digits = "".join(ch for ch in unit_id if ch.isdigit())
    return int(digits) if digits else 10**9


def order_profile_ids(units: list[dict[str, Any]]) -> list[str]:
    return [unit_profile_id(unit) for unit in units]


def select_units_for_response(
    units: list[dict[str, Any]],
    disclosed_profile_unit_ids: set[str],
    retain_count: int,
    weaken_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    priority_ids = priority_unit_ids(units)
    local_by_id = {unit["unit_id"]: unit for unit in units}
    ordered_units = [
        local_by_id[local_id]
        for local_id in priority_ids
        if local_id in local_by_id and unit_profile_id(local_by_id[local_id]) not in disclosed_profile_unit_ids
    ]

    retained = ordered_units[: max(0, retain_count)]
    weakened = ordered_units[max(0, retain_count) : max(0, retain_count) + max(0, weaken_count)]
    used = {unit_profile_id(unit) for unit in retained + weakened}
    removed = [unit for unit in units if unit_profile_id(unit) not in disclosed_profile_unit_ids | used]
    return retained, weakened, removed


def make_initial_question(slot: str) -> str:
    return INITIAL_QUESTION_OVERRIDES.get(slot, make_doctor_question(slot))


def make_targeted_followup_question(slot: str) -> str:
    display = SLOT_DISPLAY.get(slot, slot)
    if slot == "suicide_or_self_harm":
        return "关于自伤或自杀相关想法，我想更具体地确认一下：这些念头最近出现的频率、强度、是否有计划或准备行为，以及你能不能控制住？"
    if slot == "sleep":
        return "关于睡眠情况，能具体说说入睡困难、早醒或半夜醒来的频率，持续多久，以及白天受到什么影响吗？"
    if slot == "school_or_study_status":
        return "关于学习或学校这部分，能具体说说上课、作业、考试、成绩或完成任务最近受到了什么影响吗？"
    if slot == "work_status":
        return "关于工作或上班这部分，能具体说说绩效、请假、同事老板或完成任务最近受到了什么影响吗？"
    if slot == "binge_eating":
        return "关于暴饮暴食这部分，能具体说说最近吃很多或控制不住进食的频率、程度和触发情况吗？"
    if slot == "appetite_loss":
        return f"关于{display}，能具体说说最近频率、程度、持续多久，以及体重或日常生活有没有受到影响吗？"
    return f"关于你的{display}，能具体说说出现频率、持续多久、严重程度，以及对生活的影响吗？"


def make_second_targeted_followup_question(slot: str) -> str:
    display = SLOT_DISPLAY.get(slot, slot)
    if slot == "school_or_study_status":
        return "如果可以的话，能不能再补充一个最近一次学习、上课、作业或考试受到影响的具体例子？比如什么时候发生、影响多严重、后来怎么处理的。"
    if slot == "work_status":
        return "如果可以的话，能不能再补充一个最近一次工作、上班、绩效或完成任务受到影响的具体例子？比如什么时候发生、影响多严重、后来怎么处理的。"
    if slot == "binge_eating":
        return "如果可以的话，能不能再补充一个最近一次暴饮暴食或控制不住吃很多东西的具体例子？比如什么时候发生、当时吃了多少、后来怎么缓解的。"
    return f"如果可以的话，能不能再补充一个最近一次和{display}有关的具体例子？比如什么时候发生、当时有多严重、后来怎么缓解的。"


def make_generic_clarification_question() -> str:
    return "你能再说说刚才这个情况吗？"


def make_response_text(
    *,
    category: str,
    retained_units: list[dict[str, Any]],
    weakened_units: list[dict[str, Any]],
    profile: dict[str, Any],
    target_slot: str,
    max_chars: int,
) -> str:
    retained_text = join_units(retained_units, max_chars=max_chars)
    weakened_text = join_units(weakened_units, max_chars=90)

    if category == "informative_reference":
        return retained_text or "这个我暂时说不太清楚。"
    if category == "partial_omission":
        return retained_text or "这个我暂时说不太清楚。"
    if category == "vague_or_uncertain":
        if retained_text:
            return ensure_sentence(f"大概就是{retained_text.rstrip('。')}这些吧，其他细节我一时也说不太清楚")
        return "这个我有点说不清楚，暂时也不太想展开。"
    if category == "targeted_recovery_partial":
        if retained_text and weakened_text:
            return ensure_sentence(
                f"如果只是说一点的话，{retained_text.rstrip('。')}。还有些像{weakened_text.rstrip('。')}，但细节我还是不太想展开"
            )
        if retained_text:
            return ensure_sentence(f"如果只是说一点的话，{retained_text.rstrip('。')}，但更多细节我还是不太想展开")
        return "我能说的还是不多，具体细节现在有点说不出口。"
    if category == "generic_clarification_no_recovery":
        return "还是差不多吧，我有点不知道怎么说清楚。"
    if category == "direct_refusal_or_boundary":
        return "这个问题我现在不太想细说，可以先跳过吗？"
    if category == "topic_deflection":
        alt_slot, alt_unit = alternative_deflection_unit(profile, target_slot)
        if alt_unit:
            alt_text = clean_text(alt_unit.get("unit_text") or "")
            return ensure_sentence(f"{alt_text}。至于刚才问的那个，我暂时不太想展开")
        return "我现在有点乱，可能还是先说别的吧。"
    if category == "minimal_generic":
        return "这个我有点说不清楚，暂时也不太想展开。"
    return retained_text or "这个我暂时说不太清楚。"


class DynamicPatientControllerV1:
    """Stateful profile-grounded controller for dynamic low-informativeness."""

    def __init__(
        self,
        schema: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
        max_units_per_slot: int = 8,
        random_low_disclosure_prob: float = 0.5,
        random_disclosure_seed: int = 0,
    ):
        self.schema = schema
        self.profiles = profiles
        self.max_units_per_slot = max_units_per_slot
        self.random_low_disclosure_prob = clamp_probability(random_low_disclosure_prob)
        self.random_disclosure_seed = int(random_disclosure_seed)
        self.interpreter = OnlineQueryInterpreter(schema)

    @staticmethod
    def initial_state() -> dict[str, Any]:
        return {
            "turn_index": 0,
            "last_target_slot": None,
            "asked_count_by_slot": {},
            "disclosed_profile_unit_ids_by_slot": {},
            "last_g_target_by_slot": {},
            "last_cumulative_coverage_by_slot": {},
        }

    def _route_target(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
        doctor_question: str,
        dialogue_history: list[dict[str, str]] | None,
    ) -> tuple[str | None, dict[str, Any], str]:
        pred = self.interpreter.interpret(
            doctor_question,
            dialogue_history=dialogue_history or [],
            hidden_profile_tree_type=profile.get("primary_tree_type"),
        )
        target = pred.get("simulator_internal_target_node")
        routing_source = "query_interpreter"
        if not target and state.get("last_target_slot") and contains_any(doctor_question, ANAPHORA_CUES):
            target = state["last_target_slot"]
            pred["simulator_internal_target_node"] = target
            pred["target_tree_node"] = target
            pred["query_interpreter_status"] = "anaphora_fallback"
            pred["query_interpreter_confidence"] = "medium"
            routing_source = "anaphora_to_previous_target"
        return target, pred, routing_source

    def _random_low_disclosure_triggered(self, *parts: Any) -> bool:
        return (
            stable_unit_float("random_disclosure", self.random_disclosure_seed, *parts)
            < self.random_low_disclosure_prob
        )

    @staticmethod
    def _fully_cooperative_budget(total_units: int) -> dict[str, Any]:
        return {
            "retain_count": total_units,
            "weaken_count": 0,
            "topic": 1.0,
            "clarity": 1.0,
            "category": "informative_reference",
            "response_type": "informative_response",
        }

    def _budget(
        self,
        *,
        severity: str,
        target_slot: str,
        total_units: int,
        asked_before: int,
        is_targeted_followup: bool,
        has_new_units: bool,
        profile_id: str = "",
        doctor_question: str = "",
    ) -> dict[str, Any]:
        if total_units <= 0:
            return {
                "retain_count": 0,
                "weaken_count": 0,
                "topic": 0.0,
                "clarity": 0.0,
                "category": "no_profile_evidence",
                "response_type": "no_profile_evidence",
                "disclosure_mode": "reference" if severity == "reference_informative" else severity,
                "random_low_disclosure_prob": self.random_low_disclosure_prob
                if severity == "random_disclosure"
                else None,
                "random_low_disclosure_triggered": False if severity == "random_disclosure" else None,
            }

        if severity in {"reference_informative", "fully_cooperative"}:
            budget = self._fully_cooperative_budget(total_units)
            budget["disclosure_mode"] = "fully_cooperative" if severity == "fully_cooperative" else "reference"
            return budget

        if severity == "random_disclosure":
            low_triggered = self._random_low_disclosure_triggered(
                profile_id,
                target_slot,
                asked_before,
                is_targeted_followup,
                doctor_question,
            )
            if not low_triggered:
                budget = self._fully_cooperative_budget(total_units)
                budget.update(
                    {
                        "disclosure_mode": "random_disclosure",
                        "random_low_disclosure_prob": self.random_low_disclosure_prob,
                        "random_low_disclosure_triggered": False,
                    }
                )
                return budget
            if asked_before == 0:
                budget = {
                    "retain_count": max(1, min(3, math.floor(total_units * 0.25))),
                    "weaken_count": max(1, min(2, math.ceil(total_units * 0.2))),
                    "topic": 1.0,
                    "clarity": 0.5,
                    "category": "vague_or_uncertain",
                    "response_type": "vague_uncertain",
                }
            elif is_targeted_followup and has_new_units:
                budget = {
                    "retain_count": max(1, min(3, math.ceil(total_units * 0.25))),
                    "weaken_count": 1,
                    "topic": 1.0,
                    "clarity": 0.75,
                    "category": "targeted_recovery_partial",
                    "response_type": "partial_disclosure",
                }
            else:
                budget = {
                    "retain_count": 0,
                    "weaken_count": 0,
                    "topic": 0.7,
                    "clarity": 0.4,
                    "category": "generic_clarification_no_recovery",
                    "response_type": "vague_uncertain",
                }
            budget.update(
                {
                    "disclosure_mode": "random_disclosure",
                    "random_low_disclosure_prob": self.random_low_disclosure_prob,
                    "random_low_disclosure_triggered": True,
                }
            )
            return budget

        if severity == "mild_low_info":
            if asked_before == 0:
                return {
                    "retain_count": max(1, math.ceil(total_units * 0.7)),
                    "weaken_count": 0,
                    "topic": 1.0,
                    "clarity": 1.0,
                    "category": "partial_omission",
                }
            if is_targeted_followup and has_new_units:
                return {
                    "retain_count": max(1, min(3, math.ceil(total_units * 0.25))),
                    "weaken_count": 0,
                    "topic": 1.0,
                    "clarity": 0.95,
                    "category": "targeted_recovery_partial",
                }
            return {
                "retain_count": 0,
                "weaken_count": 0,
                "topic": 0.8,
                "clarity": 0.55,
                "category": "generic_clarification_no_recovery",
            }

        if severity == "moderate_low_info":
            if asked_before == 0:
                return {
                    "retain_count": max(1, min(3, math.floor(total_units * 0.25))),
                    "weaken_count": max(1, min(2, math.ceil(total_units * 0.2))),
                    "topic": 1.0,
                    "clarity": 0.5,
                    "category": "vague_or_uncertain",
                }
            if is_targeted_followup and has_new_units:
                return {
                    "retain_count": max(1, min(3, math.ceil(total_units * 0.25))),
                    "weaken_count": 1,
                    "topic": 1.0,
                    "clarity": 0.75,
                    "category": "targeted_recovery_partial",
                }
            return {
                "retain_count": 0,
                "weaken_count": 0,
                "topic": 0.7,
                "clarity": 0.4,
                "category": "generic_clarification_no_recovery",
            }

        # Severe low-info: initial response can be refusal/deflection/minimal, but targeted
        # follow-up can recover a small amount of evidence.
        if asked_before == 0:
            if target_slot in SENSITIVE_TARGETS:
                return {
                    "retain_count": 0,
                    "weaken_count": 0,
                    "topic": 0.0,
                    "clarity": 0.0,
                    "category": "direct_refusal_or_boundary",
                }
            if target_slot not in STRESS_TARGETS:
                return {
                    "retain_count": 0,
                    "weaken_count": 0,
                    "topic": 0.0,
                    "clarity": 0.0,
                    "category": "topic_deflection",
                }
            return {
                "retain_count": 0,
                "weaken_count": 0,
                "topic": 0.5,
                "clarity": 0.0,
                "category": "minimal_generic",
            }
        if is_targeted_followup and has_new_units:
            sensitive = target_slot in SENSITIVE_TARGETS
            return {
                "retain_count": 1 if sensitive else min(2, max(1, total_units // 4)),
                "weaken_count": 1 if total_units > 1 else 0,
                "topic": 0.65 if sensitive else 0.85,
                "clarity": 0.45 if sensitive else 0.6,
                "category": "targeted_recovery_partial",
            }
        return {
            "retain_count": 0,
            "weaken_count": 0,
            "topic": 0.35,
            "clarity": 0.0,
            "category": "generic_clarification_no_recovery",
        }

    def step(
        self,
        *,
        profile_id: str,
        doctor_question: str,
        base_severity: str,
        state: dict[str, Any] | None = None,
        dialogue_history: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if profile_id not in self.profiles:
            raise KeyError(f"Unknown profile_id: {profile_id}")
        profile = self.profiles[profile_id]
        state = dict(state or self.initial_state())
        state.setdefault("asked_count_by_slot", {})
        state.setdefault("disclosed_profile_unit_ids_by_slot", {})
        state.setdefault("last_g_target_by_slot", {})
        state.setdefault("last_cumulative_coverage_by_slot", {})

        severity = normalize_severity(base_severity)
        target_slot, interpreter_output, routing_source = self._route_target(
            profile,
            state,
            doctor_question,
            dialogue_history,
        )

        if not target_slot:
            response = {
                "patient_response": "这个我有点不知道怎么回答。",
                "base_severity": severity,
                "dynamic_stage": "unmapped_question",
                "profile_id": profile_id,
                "case_id": profile.get("case_id"),
                "diagnoses": profile.get("diagnoses"),
                "icd_codes": profile.get("icd_codes"),
                "active_tree_type": profile.get("primary_tree_type"),
                "doctor_question": doctor_question,
                "target_tree_node": None,
                "target_node_visibility": "simulator_internal_not_doctor_visible",
                "query_interpreter": interpreter_output,
                "topic_responsiveness": 0.0,
                "information_retention": 0.0,
                "clarity": 0.0,
                "g_target": 0.0,
                "previous_g_target_for_slot": 0.0,
                "delta_g_target_for_slot": 0.0,
                "cumulative_slot_sufficiency": 0.0,
                "previous_cumulative_slot_sufficiency": 0.0,
                "delta_cumulative_slot_sufficiency": 0.0,
                "asked_count_for_slot_before": 0,
                "asked_count_for_slot_after": 0,
                "is_targeted_followup": False,
                "is_generic_clarification": False,
                "routing_source": "unmapped",
                "retained_unit_ids": [],
                "weakened_unit_ids": [],
                "removed_unit_ids": [],
            }
            state["turn_index"] = int(state.get("turn_index") or 0) + 1
            return response, state

        slot_profile = profile.get("slot_profiles", {}).get(target_slot) or {}
        units = selected_profile_units(slot_profile, max_units=self.max_units_per_slot)
        total_units = len(units)
        asked_before = int(state["asked_count_by_slot"].get(target_slot, 0))
        disclosed_before = set(state["disclosed_profile_unit_ids_by_slot"].get(target_slot, []))
        has_new_units = any(unit_profile_id(unit) not in disclosed_before for unit in units)
        is_targeted_followup = (
            asked_before > 0
            and target_slot == state.get("last_target_slot")
            and contains_any(doctor_question, SPECIFICITY_CUES)
        )
        is_generic_clarification = (
            asked_before > 0
            and target_slot == state.get("last_target_slot")
            and contains_any(doctor_question, ANAPHORA_CUES)
            and not is_targeted_followup
        )

        budget = self._budget(
            severity=severity,
            profile_id=profile_id,
            target_slot=target_slot,
            total_units=total_units,
            asked_before=asked_before,
            is_targeted_followup=is_targeted_followup,
            has_new_units=has_new_units,
            doctor_question=doctor_question,
        )
        retained_units, weakened_units, removed_units = select_units_for_response(
            units,
            disclosed_before,
            budget["retain_count"],
            budget["weaken_count"],
        )
        retained_ids = [unit["unit_id"] for unit in retained_units]
        weakened_ids = [unit["unit_id"] for unit in weakened_units]
        removed_ids = [unit["unit_id"] for unit in removed_units]
        new_disclosed = {unit_profile_id(unit) for unit in retained_units + weakened_units}
        disclosed_after = disclosed_before | new_disclosed

        information_retention = compute_information_retention(
            retained_ids,
            weakened_ids,
            total_units,
        )
        topic = float(budget["topic"])
        clarity = float(budget["clarity"])
        g_target = round(topic * information_retention * clarity, 4)
        cumulative_coverage = round(len(disclosed_after) / total_units, 4) if total_units else 0.0
        prev_g = float(state["last_g_target_by_slot"].get(target_slot, 0.0))
        prev_coverage = float(state["last_cumulative_coverage_by_slot"].get(target_slot, 0.0))

        response_text = make_response_text(
            category=budget["category"],
            retained_units=retained_units,
            weakened_units=weakened_units,
            profile=profile,
            target_slot=target_slot,
            max_chars=220,
        )

        state["asked_count_by_slot"][target_slot] = asked_before + 1
        state["disclosed_profile_unit_ids_by_slot"][target_slot] = sorted(disclosed_after)
        state["last_target_slot"] = target_slot
        state["last_g_target_by_slot"][target_slot] = g_target
        state["last_cumulative_coverage_by_slot"][target_slot] = cumulative_coverage
        state["turn_index"] = int(state.get("turn_index") or 0) + 1

        dynamic_stage = "initial_low_info"
        if severity == "reference_informative":
            dynamic_stage = "reference"
        elif is_targeted_followup:
            dynamic_stage = "targeted_followup_recovery"
        elif is_generic_clarification:
            dynamic_stage = "generic_clarification"
        elif asked_before > 0:
            dynamic_stage = "repeated_question"

        response = {
            "patient_response": response_text,
            "base_severity": severity,
            "dynamic_stage": dynamic_stage,
            "low_info_category": budget["category"],
            "low_info_cause": budget.get("response_type", budget["category"]),
            "response_type": budget.get("response_type", budget["category"]),
            "disclosure_mode": budget.get("disclosure_mode", severity),
            "random_low_disclosure_prob": budget.get("random_low_disclosure_prob"),
            "random_low_disclosure_triggered": budget.get("random_low_disclosure_triggered"),
            "profile_id": profile_id,
            "case_id": profile.get("case_id"),
            "diagnoses": profile.get("diagnoses"),
            "icd_codes": profile.get("icd_codes"),
            "active_tree_type": profile.get("primary_tree_type"),
            "doctor_question": doctor_question,
            "target_tree_node": target_slot,
            "target_node_role": "simulator_internal_target_node",
            "target_node_visibility": "simulator_internal_not_doctor_visible",
            "routing_source": routing_source,
            "query_interpreter": interpreter_output,
            "asked_count_for_slot_before": asked_before,
            "asked_count_for_slot_after": asked_before + 1,
            "is_targeted_followup": is_targeted_followup,
            "is_generic_clarification": is_generic_clarification,
            "topic_responsiveness": topic,
            "information_retention": information_retention,
            "clarity": clarity,
            "g_target": g_target,
            "previous_g_target_for_slot": prev_g,
            "delta_g_target_for_slot": round(g_target - prev_g, 4),
            "cumulative_slot_sufficiency": cumulative_coverage,
            "previous_cumulative_slot_sufficiency": prev_coverage,
            "delta_cumulative_slot_sufficiency": round(cumulative_coverage - prev_coverage, 4),
            "target_slot_evidence_unit_count": total_units,
            "retained_unit_ids": retained_ids,
            "weakened_unit_ids": weakened_ids,
            "removed_unit_ids": removed_ids,
            "retained_profile_unit_ids": [unit_profile_id(unit) for unit in retained_units],
            "weakened_profile_unit_ids": [unit_profile_id(unit) for unit in weakened_units],
            "observed_evidence_units": units,
            "disclosed_profile_unit_ids_before": sorted(disclosed_before),
            "disclosed_profile_unit_ids_after": sorted(disclosed_after),
            "validity": {
                "label_preserved": True,
                "no_new_clinical_fact": True,
                "profile_grounded": True,
                "stateful_disclosure": True,
                "doctor_node_hidden": True,
            },
            "realizer": {
                "type": "deterministic_rule_based",
                "model": "none",
                "version": "dynamic_profile_grounded_controller_v1",
            },
        }
        return response, state


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    return {record["profile_id"]: record for record in iter_jsonl(path)}


def load_group_records(group_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        path = group_dir / f"mdd5k_profile_grounded_environment_{split}_groups.jsonl"
        for record in iter_jsonl(path):
            record = dict(record)
            record["split"] = split
            records.append(record)
    return records


def select_pilot_groups(
    groups: list[dict[str, Any]],
    max_groups: int,
    max_per_slot: int,
) -> list[dict[str, Any]]:
    selected = []
    slot_counts: Counter[str] = Counter()
    seen_profiles: Counter[str] = Counter()
    for group in sorted(groups, key=lambda item: (item["target_tree_node"], item["case_id"])):
        slot = group["target_tree_node"]
        if slot in BROAD_CONTEXT_SLOTS:
            continue
        if slot_counts[slot] >= max_per_slot:
            continue
        if seen_profiles[group["profile_id"]] >= 4:
            continue
        selected.append(group)
        slot_counts[slot] += 1
        seen_profiles[group["profile_id"]] += 1
        if len(selected) >= max_groups:
            break
    return selected


def build_pilot_records(
    controller: DynamicPatientControllerV1,
    groups: list[dict[str, Any]],
    severities: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in groups:
        profile_id = group["profile_id"]
        target = group["target_tree_node"]
        questions = [
            ("initial_question", make_initial_question(target)),
            ("generic_clarification", make_generic_clarification_question()),
            ("targeted_followup", make_targeted_followup_question(target)),
            ("second_targeted_followup", make_second_targeted_followup_question(target)),
        ]
        for severity in severities:
            state = controller.initial_state()
            history: list[dict[str, str]] = []
            scenario_id = f"{group['counterfactual_group_id']}::{severity}"
            for turn_idx, (question_type, question) in enumerate(questions):
                response, state = controller.step(
                    profile_id=profile_id,
                    doctor_question=question,
                    base_severity=severity,
                    state=state,
                    dialogue_history=history,
                )
                history.append({"doctor_utterance": question, "patient_utterance": response["patient_response"]})
                records.append(
                    {
                        "record_id": f"{scenario_id}::turn_{turn_idx}_{question_type}",
                        "scenario_id": scenario_id,
                        "split": group.get("split"),
                        "turn_index": turn_idx,
                        "question_type": question_type,
                        "counterfactual_group_id": group["counterfactual_group_id"],
                        **response,
                    }
                )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_severity = Counter(record["base_severity"] for record in records)
    counts_by_stage = Counter(record["dynamic_stage"] for record in records)
    counts_by_question_type = Counter(record["question_type"] for record in records)
    g_by_stage: dict[str, list[float]] = defaultdict(list)
    coverage_delta_by_stage: dict[str, list[float]] = defaultdict(list)
    g_by_severity_turn: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    recovery_positive = Counter()
    recovery_total = Counter()

    for record in records:
        stage = record["dynamic_stage"]
        g_by_stage[stage].append(float(record["g_target"]))
        coverage_delta_by_stage[stage].append(float(record["delta_cumulative_slot_sufficiency"]))
        g_by_severity_turn[record["base_severity"]][str(record["turn_index"])].append(float(record["g_target"]))
        if stage in {"targeted_followup_recovery", "generic_clarification"}:
            recovery_total[stage] += 1
            if float(record["delta_cumulative_slot_sufficiency"]) > 0:
                recovery_positive[stage] += 1

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    scenario_ids = {record["scenario_id"] for record in records}
    profile_ids = {record["profile_id"] for record in records}
    target_counts = Counter(record["target_tree_node"] for record in records if record.get("turn_index") == 0)

    return {
        "num_records": len(records),
        "num_scenarios": len(scenario_ids),
        "num_profiles": len(profile_ids),
        "counts_by_severity": dict(counts_by_severity),
        "counts_by_stage": dict(counts_by_stage),
        "counts_by_question_type": dict(counts_by_question_type),
        "target_slot_scenario_counts": dict(target_counts),
        "mean_g_by_stage": {stage: avg(values) for stage, values in g_by_stage.items()},
        "mean_delta_coverage_by_stage": {
            stage: avg(values) for stage, values in coverage_delta_by_stage.items()
        },
        "mean_g_by_severity_turn": {
            severity: {turn: avg(values) for turn, values in turns.items()}
            for severity, turns in g_by_severity_turn.items()
        },
        "positive_recovery_rate": {
            stage: round(recovery_positive[stage] / recovery_total[stage], 6)
            if recovery_total[stage]
            else 0.0
            for stage in sorted(recovery_total)
        },
        "routing_source_counts": dict(Counter(record["routing_source"] for record in records)),
        "query_status_counts": dict(Counter(record["query_interpreter"]["query_interpreter_status"] for record in records)),
    }


def write_report(path: Path, summary: dict[str, Any], pilot_path: Path) -> None:
    lines = [
        "# MDD-5K Dynamic Patient Controller V1",
        "",
        "Date: 2026-06-11",
        "",
        "## Purpose",
        "",
        "This is a recovery-enabled, profile-grounded patient controller for training and evaluating robust doctor agents under low-informative responses.",
        "",
        "It is not a clinical realism benchmark. It is a reproducible stress-test environment derived from MDD-5K dialogue-derived profiles.",
        "",
        "## Online Interface",
        "",
        "```text",
        "doctor-visible:",
        "  dialogue_history",
        "  doctor_question",
        "",
        "hidden simulator:",
        "  profile_id",
        "  hidden_profile_tree_type",
        "  simulator_internal_target_node",
        "  disclosed_evidence_state",
        "  base_severity",
        "```",
        "",
        "The doctor never observes diagnosis-tree nodes or controller metadata.",
        "",
        "## Dynamic Rule",
        "",
        "```text",
        "patient_response_t = f(",
        "  doctor_question_t,",
        "  simulator_internal_target_node_t,",
        "  base_severity,",
        "  disclosed_units_so_far,",
        "  asked_count_for_slot,",
        "  whether_question_is_targeted_followup",
        ")",
        "```",
        "",
        "`reference/mild/moderate/severe` are base severity conditions, not fixed refusal scripts.",
        "",
        "## Pilot Output",
        "",
        f"- Pilot JSONL: `{pilot_path.name}`",
        f"- Records: {summary['num_records']}",
        f"- Scenarios: {summary['num_scenarios']}",
        f"- Profiles: {summary['num_profiles']}",
        "",
        "## Mean Gate Target By Stage",
        "",
        "| Stage | Mean g_target | Mean delta coverage |",
        "|---|---:|---:|",
    ]
    for stage, value in summary["mean_g_by_stage"].items():
        lines.append(
            f"| `{stage}` | {value:.6f} | {summary['mean_delta_coverage_by_stage'].get(stage, 0.0):.6f} |"
        )
    lines.extend(["", "## Recovery Rate", "", "| Stage | Positive recovery rate |", "|---|---:|"])
    for stage, value in summary["positive_recovery_rate"].items():
        lines.append(f"| `{stage}` | {value:.6f} |")

    lines.extend(["", "## Mean g_target By Severity And Turn", "", "| Severity | Turn 0 | Turn 1 | Turn 2 | Turn 3 |", "|---|---:|---:|---:|---:|"])
    for severity, turns in summary["mean_g_by_severity_turn"].items():
        lines.append(
            f"| `{severity}` | {turns.get('0', 0.0):.6f} | {turns.get('1', 0.0):.6f} | {turns.get('2', 0.0):.6f} | {turns.get('3', 0.0):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Generic clarification should not reliably recover evidence.",
            "- Targeted follow-up should increase cumulative evidence coverage when undisclosed profile evidence remains.",
            "- Severe low-info can still recover partially, but less than mild/moderate, especially for sensitive slots.",
            "- This controller enables baseline comparisons such as Closed LLM-General, Closed LLM-Evidence-Aware, Simple Clarification, and evidence-gated RL.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic patient controller V1 pilot cache.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--max-groups", type=int, default=90)
    parser.add_argument("--max-per-slot", type=int, default=5)
    parser.add_argument("--max-units-per-slot", type=int, default=8)
    parser.add_argument("--random-low-disclosure-prob", type=float, default=0.5)
    parser.add_argument("--random-disclosure-seed", type=int, default=0)
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["mild_low_info", "moderate_low_info", "severe_low_info"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_json(args.schema)
    profiles = load_profiles(args.profiles)
    groups = load_group_records(args.group_dir, args.splits)
    selected_groups = select_pilot_groups(groups, args.max_groups, args.max_per_slot)

    controller = DynamicPatientControllerV1(
        schema=schema,
        profiles=profiles,
        max_units_per_slot=args.max_units_per_slot,
        random_low_disclosure_prob=args.random_low_disclosure_prob,
        random_disclosure_seed=args.random_disclosure_seed,
    )
    severities = [normalize_severity(level) for level in args.severities]
    records = build_pilot_records(controller, selected_groups, severities)
    summary = {
        "settings": {
            "splits": args.splits,
            "max_groups": args.max_groups,
            "max_per_slot": args.max_per_slot,
            "max_units_per_slot": args.max_units_per_slot,
            "random_low_disclosure_prob": args.random_low_disclosure_prob,
            "random_disclosure_seed": args.random_disclosure_seed,
            "severities": severities,
        },
        **summarize(records),
    }
    pilot_path = args.output_dir / "mdd5k_dynamic_patient_controller_v1_pilot.jsonl"
    summary_path = args.output_dir / "mdd5k_dynamic_patient_controller_v1_summary.json"
    report_path = args.output_dir / "MDD5K_DYNAMIC_PATIENT_CONTROLLER_V1.md"

    write_jsonl(pilot_path, records)
    write_json(summary_path, summary)
    write_report(report_path, summary, pilot_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
