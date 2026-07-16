from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _patient_controller_base import (
    BROAD_CONTEXT_SLOTS,
    DEFAULT_GROUP_DIR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_SCHEMA_PATH,
    DynamicPatientControllerV1,
    load_group_records,
    load_profiles,
    make_generic_clarification_question,
    make_initial_question,
    make_second_targeted_followup_question,
    make_targeted_followup_question,
    normalize_severity,
    select_pilot_groups,
)
from online_query_interpreter import load_json
try:
    from official_mdd5k_protocol import OfficialTopicState, dynamic_select_official_topics
except ModuleNotFoundError:
    OfficialTopicState = Any

    def dynamic_select_official_topics(*args: Any, **kwargs: Any) -> list[Any]:
        raise ModuleNotFoundError(
            "official_mdd5k_protocol is required for mdd5k_official_protocol_doctor, "
            "but is not required for closed/source-agnostic online replay."
        )


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_doctor_policy_baselines"

GLOBAL_CORE_SEQUENCE = [
    "suicide_or_self_harm",
    "hopelessness_or_crying",
    "sleep",
    "appetite_loss",
    "attention_decline",
    "memory_problem",
    "cognitive_slowing",
    "school_or_study_status",
    "work_status",
    "hallucination",
    "mania_screen",
    "family_psychiatric_history",
]

OPENING_QUESTION = "我想先了解一下你最近最困扰的情况，可以从情绪、睡眠、学习或工作影响说起。"

POLICY_VISIBILITY = {
    "generic_opening": "doctor_visible_only",
    "mdd5k_official_protocol_doctor": "procedural_mdd5k_official_protocol",
    "mdd5k_tree_doctor": "procedural_mdd5k_tree_protocol",
    "mdd5k_tree_topic_end_doctor": "procedural_mdd5k_tree_protocol",
    "mdd5k_tree_simple_clarification": "procedural_mdd5k_tree_protocol",
    "mdd5k_tree_oracle_targeted_followup": "oracle_tree_sequence_upper_bound",
    "fixed_global_sequence": "doctor_visible_only",
    "fixed_global_sequence_generic": "doctor_visible_only",
    "fixed_global_sequence_followup": "doctor_visible_only",
    "fixed_tree_oracle_followup": "oracle_tree_sequence_upper_bound",
}

TOPIC_END_POLICIES = {"mdd5k_tree_topic_end_doctor"}
OFFICIAL_PROTOCOL_POLICIES = {"mdd5k_official_protocol_doctor"}


class MDD5KTopicEndDecider:
    """Executable version of the MDD-5K `is_topic_end` step.

    The official code asks an LLM whether the current topic should end and then
    applies `force_topic_end`. For reproducible local baselines, the default
    raw decision is made from controller sufficiency metadata while preserving
    the official prompt and force-end post-processing structure.
    """

    def __init__(
        self,
        *,
        mode: str,
        sufficiency_threshold: float,
        max_same_topic_turns: int,
        seed: int,
    ) -> None:
        self.mode = mode
        self.sufficiency_threshold = sufficiency_threshold
        self.max_same_topic_turns = max_same_topic_turns
        self.topic_end_history: list[bool] = []
        self.rng = random.Random(seed)

    @staticmethod
    def prompt(current_state: str, input_history: list[dict[str, str]]) -> str:
        return (
            "一段精神科医生和精神疾病患者之间的对话为{}，判断围绕诊断话题”{}“的对话是否应该结束，"
            "如果应该结束返回”是“，如果不应该结束返回”否“。倾向于判断话题应该结束"
        ).format(input_history, current_state)

    def force_topic_end(self) -> bool:
        # Mirrors the post-processing logic in the official MDD-5K DiagTree.
        if len(self.topic_end_history) < 6:
            return self.topic_end_history[-1]
        if self.topic_end_history[-2] is False:
            if self.topic_end_history[-3] is False:
                self.topic_end_history[-1] = True
                return True
            if self.rng.randint(0, 2) == 1:
                return self.topic_end_history[-1]
            self.topic_end_history[-1] = True
            return True
        if self.topic_end_history[-1] is True:
            return True
        if self.rng.randint(0, 2) == 1:
            self.topic_end_history[-1] = True
            return True
        return self.topic_end_history[-1]

    def raw_decision(self, response: dict[str, Any]) -> tuple[bool, str]:
        asked_after = int(response.get("asked_count_for_slot_after") or 0)
        sufficiency = float(response.get("cumulative_slot_sufficiency") or 0.0)
        total_units = int(response.get("target_slot_evidence_unit_count") or 0)
        if self.mode == "always_end":
            return True, "always_end_tree_order"
        if self.mode == "visible_low_info_cues":
            patient_response = str(response.get("patient_response") or "")
            low_info_cues = [
                "说不清",
                "不太清楚",
                "不想细说",
                "不想展开",
                "跳过",
                "不好说",
                "不知道怎么说",
                "暂时",
                "差不多",
                "记不清",
            ]
            if asked_after >= self.max_same_topic_turns:
                return True, "max_same_topic_turns_reached"
            if any(cue in patient_response for cue in low_info_cues):
                return False, "visible_low_info_cue_continue_topic"
            return True, "visible_response_no_low_info_cue"
        if self.mode == "metadata_sufficiency":
            if total_units == 0:
                return True, "no_profile_evidence_units"
            if sufficiency >= self.sufficiency_threshold:
                return True, "sufficiency_threshold_reached"
            if asked_after >= self.max_same_topic_turns:
                return True, "max_same_topic_turns_reached"
            return False, "insufficient_evidence_continue_topic"
        raise ValueError(f"Unknown topic-end mode: {self.mode}")

    def is_topic_end(
        self,
        *,
        current_state: str,
        input_history: list[dict[str, str]],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        raw, reason = self.raw_decision(response)
        self.topic_end_history.append(raw)
        final = self.force_topic_end()
        return {
            "topic_end_mode": self.mode,
            "current_state": current_state,
            "official_prompt": self.prompt(current_state, input_history),
            "raw_topic_end": raw,
            "final_topic_end": final,
            "topic_end_reason": reason,
            "topic_end_history_length": len(self.topic_end_history),
            "topic_end_history_tail": self.topic_end_history[-6:],
            "sufficiency_threshold": self.sufficiency_threshold,
            "max_same_topic_turns": self.max_same_topic_turns,
        }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def load_patient_response_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        source_record_id = record.get("source_record_id")
        patient_response = record.get("patient_response")
        if source_record_id and patient_response:
            cache[str(source_record_id)] = record
    return cache


def apply_patient_response_cache(
    response: dict[str, Any],
    *,
    record_id: str,
    patient_response_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(response)
    cached = patient_response_cache.get(record_id)
    if not cached:
        updated["patient_response_cache_hit"] = False
        updated["patient_realizer_source"] = "deterministic_rule_based"
        return updated

    updated["deterministic_patient_response"] = response.get("patient_response")
    updated["patient_response"] = cached.get("patient_response")
    updated["patient_response_cache_hit"] = True
    updated["patient_realizer_source"] = cached.get("realizer_source") or "llm_verified"
    updated["patient_realizer_model"] = cached.get("model")
    updated["patient_realizer_provider"] = cached.get("provider")
    updated["patient_realizer_request_id"] = cached.get("request_id")
    updated["patient_realizer_prompt_protocol_version"] = cached.get("prompt_protocol_version")
    updated["patient_realizer_history_mode"] = cached.get("history_mode")
    updated["patient_realizer_warnings"] = cached.get("warnings") or []
    updated["patient_realizer_mean_allowed_coverage"] = cached.get("mean_allowed_coverage")
    return updated


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def stable_int_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def schema_slot_maps(schema: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    slots = {slot["slot"]: slot for slot in schema.get("slots", [])}
    criticality_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    ranks = {name: criticality_rank.get(slot.get("criticality", "medium"), 2) for name, slot in slots.items()}
    return slots, ranks


def ordered_slots_by_criticality(slots: list[str], criticality_ranks: dict[str, int]) -> list[str]:
    return sorted(slots, key=lambda slot: (-criticality_ranks.get(slot, 2), GLOBAL_CORE_SEQUENCE.index(slot) if slot in GLOBAL_CORE_SEQUENCE else 99, slot))


def mdd5k_tree_slots(profile: dict[str, Any], tree_space: dict[str, list[str]]) -> list[str]:
    """Return the MDD-5K diagnosis-tree order for this profile's recovered tree type."""
    tree_type = profile.get("primary_tree_type") or "unknown"
    schema_order = list(tree_space.get(tree_type) or [])
    active_order = list(profile.get("active_tree_slots") or [])
    if not schema_order:
        schema_order = active_order or list(GLOBAL_CORE_SEQUENCE)
    if active_order:
        active = set(active_order)
        schema_order = [slot for slot in schema_order if slot in active]

    ordered: list[str] = []
    seen: set[str] = set()
    for slot in schema_order + active_order:
        if slot and slot not in seen:
            ordered.append(slot)
            seen.add(slot)
    return ordered


def topic_question(slot: str, asked_count_for_slot: int) -> tuple[str, str]:
    if asked_count_for_slot <= 0:
        return "initial_slot_question", make_initial_question(slot)
    if asked_count_for_slot == 1:
        return "topic_end_targeted_followup", make_targeted_followup_question(slot)
    return "topic_end_second_targeted_followup", make_second_targeted_followup_question(slot)


def official_patient_facing_question(topic_state: OfficialTopicState, asked_count_for_topic: int) -> tuple[str, str]:
    """Convert an official MDD-5K topic prompt state into a patient-facing question.

    Official MDD-5K uses prompt states like "询问患者有关睡眠，不要包含其他话题和问题"
    as instructions for an LLM doctor. Those strings are not valid utterances to
    show in dialogue history, so this baseline keeps the official topic/order as
    metadata while rendering the actual doctor action as a natural question.
    """
    slot = topic_state.slot
    if slot:
        if asked_count_for_topic <= 0:
            return "official_topic_natural_question", make_initial_question(slot)
        if asked_count_for_topic == 1:
            return "official_topic_natural_followup", make_targeted_followup_question(slot)
        return "official_topic_natural_second_followup", make_second_targeted_followup_question(slot)

    topic = str(topic_state.official_topic or "").strip()
    if topic == "精神状况":
        if asked_count_for_topic <= 0:
            return "official_topic_natural_question", "我想先了解一下你最近整体的精神状态和最困扰你的情况，可以简单说说吗？"
        return "official_topic_natural_followup", "关于你最近整体的精神状态，能不能再具体说说哪些变化最明显？"
    if asked_count_for_topic <= 0:
        return "official_topic_natural_question", f"我想了解一下你最近和{topic}有关的情况，可以简单说说吗？"
    return "official_topic_natural_followup", f"关于{topic}这部分，能不能再补充一个更具体的例子或细节？"


def build_policy_questions(
    policy_name: str,
    profile: dict[str, Any],
    criticality_ranks: dict[str, int],
    tree_space: dict[str, list[str]],
    max_turns: int,
) -> list[dict[str, str]]:
    if policy_name in OFFICIAL_PROTOCOL_POLICIES:
        return [
            {
                "question_type": "official_protocol_dynamic_turn_budget",
                "doctor_question": "",
            }
            for _ in range(max_turns)
        ]

    if policy_name == "generic_opening":
        questions = [{"question_type": "opening_question", "doctor_question": OPENING_QUESTION}]
        while len(questions) < max_turns:
            questions.append(
                {
                    "question_type": "generic_clarification",
                    "doctor_question": make_generic_clarification_question(),
                }
            )
        return questions[:max_turns]

    if policy_name in {
        "mdd5k_tree_doctor",
        "mdd5k_tree_topic_end_doctor",
        "mdd5k_tree_simple_clarification",
        "mdd5k_tree_oracle_targeted_followup",
    }:
        base_slots = mdd5k_tree_slots(profile, tree_space)
        pair_mode = {
            "mdd5k_tree_doctor": "initial_only",
            "mdd5k_tree_topic_end_doctor": "initial_only",
            "mdd5k_tree_simple_clarification": "generic",
            "mdd5k_tree_oracle_targeted_followup": "targeted",
        }[policy_name]
    elif policy_name == "fixed_tree_oracle_followup":
        observed = set(profile.get("observed_slots") or [])
        active = [
            slot
            for slot in profile.get("active_tree_slots", [])
            if slot in observed and slot not in BROAD_CONTEXT_SLOTS
        ]
        base_slots = ordered_slots_by_criticality(active, criticality_ranks)
        pair_mode = "targeted"
    elif policy_name in {
        "fixed_global_sequence",
        "fixed_global_sequence_generic",
        "fixed_global_sequence_followup",
    }:
        base_slots = list(GLOBAL_CORE_SEQUENCE)
        pair_mode = {
            "fixed_global_sequence": "initial_only",
            "fixed_global_sequence_generic": "generic",
            "fixed_global_sequence_followup": "targeted",
        }[policy_name]
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    questions: list[dict[str, str]] = []
    for slot in base_slots:
        questions.append({"question_type": "initial_slot_question", "doctor_question": make_initial_question(slot)})
        if pair_mode == "generic":
            questions.append(
                {
                    "question_type": "generic_clarification",
                    "doctor_question": make_generic_clarification_question(),
                }
            )
        elif pair_mode == "targeted":
            questions.append(
                {
                    "question_type": "targeted_followup",
                    "doctor_question": make_targeted_followup_question(slot),
                }
            )
            questions.append(
                {
                    "question_type": "second_targeted_followup",
                    "doctor_question": make_second_targeted_followup_question(slot),
                }
            )
        if len(questions) >= max_turns:
            break
    return questions[:max_turns]


def select_profiles_from_groups(groups: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], max_profiles: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for group in groups:
        profile_id = group["profile_id"]
        if profile_id in seen or profile_id not in profiles:
            continue
        seen.add(profile_id)
        selected.append(profiles[profile_id])
        if len(selected) >= max_profiles:
            break
    return selected


def run_profile_policy(
    *,
    controller: DynamicPatientControllerV1,
    profile: dict[str, Any],
    severity: str,
    policy_name: str,
    questions: list[dict[str, str]],
    topic_end_mode: str,
    topic_end_sufficiency_threshold: float,
    topic_end_max_same_topic_turns: int,
    topic_end_seed: int,
    patient_response_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if policy_name in OFFICIAL_PROTOCOL_POLICIES:
        return run_official_protocol_profile_policy(
            controller=controller,
            profile=profile,
            severity=severity,
            policy_name=policy_name,
            max_turns=len(questions),
            topic_end_mode=topic_end_mode,
            topic_end_sufficiency_threshold=topic_end_sufficiency_threshold,
            topic_end_max_same_topic_turns=topic_end_max_same_topic_turns,
            topic_end_seed=topic_end_seed,
            patient_response_cache=patient_response_cache,
        )

    if policy_name in TOPIC_END_POLICIES:
        return run_topic_end_profile_policy(
            controller=controller,
            profile=profile,
            severity=severity,
            policy_name=policy_name,
            max_turns=len(questions),
            topic_end_mode=topic_end_mode,
            topic_end_sufficiency_threshold=topic_end_sufficiency_threshold,
            topic_end_max_same_topic_turns=topic_end_max_same_topic_turns,
            topic_end_seed=topic_end_seed,
            patient_response_cache=patient_response_cache,
        )

    state = controller.initial_state()
    history: list[dict[str, str]] = []
    profile_id = profile["profile_id"]
    scenario_id = f"{profile_id}::{severity}::{policy_name}"
    records: list[dict[str, Any]] = []
    for turn_idx, question_record in enumerate(questions):
        question = question_record["doctor_question"]
        response, state = controller.step(
            profile_id=profile_id,
            doctor_question=question,
            base_severity=severity,
            state=state,
            dialogue_history=history,
        )
        record_id = f"{scenario_id}::turn_{turn_idx}"
        response = apply_patient_response_cache(
            response,
            record_id=record_id,
            patient_response_cache=patient_response_cache,
        )
        history.append({"doctor_utterance": question, "patient_utterance": response["patient_response"]})
        records.append(
            {
                "record_id": record_id,
                "scenario_id": scenario_id,
                "profile_id": profile_id,
                "case_id": profile.get("case_id"),
                "diagnoses": profile.get("diagnoses"),
                "icd_codes": profile.get("icd_codes"),
                "policy_name": policy_name,
                "policy_visibility": POLICY_VISIBILITY[policy_name],
                "base_severity": severity,
                "turn_index": turn_idx,
                "question_type": question_record["question_type"],
                "doctor_question": question,
                "topic_end_decision": None,
                **response,
            }
        )
    return records


def run_topic_end_profile_policy(
    *,
    controller: DynamicPatientControllerV1,
    profile: dict[str, Any],
    severity: str,
    policy_name: str,
    max_turns: int,
    topic_end_mode: str,
    topic_end_sufficiency_threshold: float,
    topic_end_max_same_topic_turns: int,
    topic_end_seed: int,
    patient_response_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    state = controller.initial_state()
    history: list[dict[str, str]] = []
    profile_id = profile["profile_id"]
    scenario_id = f"{profile_id}::{severity}::{policy_name}"
    records: list[dict[str, Any]] = []
    tree_slots = mdd5k_tree_slots(profile, controller.schema.get("tree_space", {}))
    if not tree_slots:
        return records

    slot_index = 0
    asked_count_by_policy_slot: Counter[str] = Counter()
    decider = MDD5KTopicEndDecider(
        mode=topic_end_mode,
        sufficiency_threshold=topic_end_sufficiency_threshold,
        max_same_topic_turns=topic_end_max_same_topic_turns,
        seed=topic_end_seed,
    )

    for turn_idx in range(max_turns):
        if slot_index >= len(tree_slots):
            break
        current_slot = tree_slots[slot_index]
        question_type, question = topic_question(current_slot, asked_count_by_policy_slot[current_slot])
        response, state = controller.step(
            profile_id=profile_id,
            doctor_question=question,
            base_severity=severity,
            state=state,
            dialogue_history=history,
        )
        record_id = f"{scenario_id}::turn_{turn_idx}"
        response = apply_patient_response_cache(
            response,
            record_id=record_id,
            patient_response_cache=patient_response_cache,
        )
        history.append({"doctor_utterance": question, "patient_utterance": response["patient_response"]})
        asked_count_by_policy_slot[current_slot] += 1
        topic_end_decision = decider.is_topic_end(
            current_state=current_slot,
            input_history=history,
            response=response,
        )
        records.append(
            {
                "record_id": record_id,
                "scenario_id": scenario_id,
                "profile_id": profile_id,
                "case_id": profile.get("case_id"),
                "diagnoses": profile.get("diagnoses"),
                "icd_codes": profile.get("icd_codes"),
                "policy_name": policy_name,
                "policy_visibility": POLICY_VISIBILITY[policy_name],
                "base_severity": severity,
                "turn_index": turn_idx,
                "question_type": question_type,
                "doctor_question": question,
                "policy_current_tree_slot": current_slot,
                "policy_tree_slot_index": slot_index,
                "policy_asked_count_for_slot_after": asked_count_by_policy_slot[current_slot],
                "topic_end_decision": topic_end_decision,
                **response,
            }
        )
        if topic_end_decision["final_topic_end"]:
            slot_index += 1
    return records


def run_official_protocol_profile_policy(
    *,
    controller: DynamicPatientControllerV1,
    profile: dict[str, Any],
    severity: str,
    policy_name: str,
    max_turns: int,
    topic_end_mode: str,
    topic_end_sufficiency_threshold: float,
    topic_end_max_same_topic_turns: int,
    topic_end_seed: int,
    patient_response_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    state = controller.initial_state()
    history: list[dict[str, str]] = []
    profile_id = profile["profile_id"]
    scenario_id = f"{profile_id}::{severity}::{policy_name}"
    records: list[dict[str, Any]] = []
    tree_type = profile.get("primary_tree_type") or "female_teen"
    tree_seed = stable_int_seed("official_tree", topic_end_seed, profile_id, severity, policy_name)
    topic_states = dynamic_select_official_topics(tree_type, seed=tree_seed, include_parse=False)
    if not topic_states:
        return records

    topic_index = 0
    asked_count_by_topic: Counter[str] = Counter()
    decider = MDD5KTopicEndDecider(
        mode=topic_end_mode,
        sufficiency_threshold=topic_end_sufficiency_threshold,
        max_same_topic_turns=topic_end_max_same_topic_turns,
        seed=stable_int_seed("official_topic_end", topic_end_seed, profile_id, severity, policy_name),
    )

    for turn_idx in range(max_turns):
        if topic_index >= len(topic_states):
            break
        topic_state: OfficialTopicState = topic_states[topic_index]
        asked_before = asked_count_by_topic[topic_state.official_state]
        question_type, question = official_patient_facing_question(topic_state, asked_before)
        response, state = controller.step(
            profile_id=profile_id,
            doctor_question=question,
            base_severity=severity,
            state=state,
            dialogue_history=history,
        )
        record_id = f"{scenario_id}::turn_{turn_idx}"
        response = apply_patient_response_cache(
            response,
            record_id=record_id,
            patient_response_cache=patient_response_cache,
        )
        history.append({"doctor_utterance": question, "patient_utterance": response["patient_response"]})
        asked_count_by_topic[topic_state.official_state] += 1
        topic_end_decision = decider.is_topic_end(
            current_state=topic_state.official_state,
            input_history=history,
            response=response,
        )
        records.append(
            {
                "record_id": record_id,
                "scenario_id": scenario_id,
                "profile_id": profile_id,
                "case_id": profile.get("case_id"),
                "diagnoses": profile.get("diagnoses"),
                "icd_codes": profile.get("icd_codes"),
                "policy_name": policy_name,
                "policy_visibility": POLICY_VISIBILITY[policy_name],
                "base_severity": severity,
                "turn_index": turn_idx,
                "question_type": question_type,
                "doctor_question": question,
                "policy_current_tree_slot": topic_state.slot,
                "policy_tree_slot_index": topic_index,
                "policy_asked_count_for_slot_after": asked_count_by_topic[topic_state.official_state],
                "official_tree_type": tree_type,
                "official_tree_seed": tree_seed,
                "official_topic": topic_state.official_topic,
                "official_state": topic_state.official_state,
                "official_prompt_state_not_patient_facing": topic_state.official_state,
                "official_patient_facing_question_renderer": "slot_template_v1",
                "official_section": topic_state.section,
                "official_topic_slot": topic_state.slot,
                "official_parse_nodes_skipped": True,
                "topic_end_decision": topic_end_decision,
                **response,
            }
        )
        if topic_end_decision["final_topic_end"]:
            topic_index += 1
    return records


def build_records(
    *,
    controller: DynamicPatientControllerV1,
    profiles: list[dict[str, Any]],
    policies: list[str],
    severities: list[str],
    criticality_ranks: dict[str, int],
    max_turns: int,
    topic_end_mode: str,
    topic_end_sufficiency_threshold: float,
    topic_end_max_same_topic_turns: int,
    topic_end_seed: int,
    patient_response_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for profile in profiles:
        for severity in severities:
            for policy_name in policies:
                questions = build_policy_questions(
                    policy_name=policy_name,
                    profile=profile,
                    criticality_ranks=criticality_ranks,
                    tree_space=controller.schema.get("tree_space", {}),
                    max_turns=max_turns,
                )
                records.extend(
                    run_profile_policy(
                        controller=controller,
                        profile=profile,
                        severity=severity,
                        policy_name=policy_name,
                        questions=questions,
                        topic_end_mode=topic_end_mode,
                        topic_end_sufficiency_threshold=topic_end_sufficiency_threshold,
                        topic_end_max_same_topic_turns=topic_end_max_same_topic_turns,
                        topic_end_seed=topic_end_seed,
                        patient_response_cache=patient_response_cache,
                    )
                )
    return records


def evaluable_slots(profile: dict[str, Any], criticality_ranks: dict[str, int]) -> list[str]:
    observed = set(profile.get("observed_slots") or [])
    active = set(profile.get("active_tree_slots") or [])
    slots = [
        slot
        for slot in observed & active
        if slot not in BROAD_CONTEXT_SLOTS and criticality_ranks.get(slot, 0) > 0
    ]
    return ordered_slots_by_criticality(slots, criticality_ranks)


def summarize(records: list[dict[str, Any]], profiles_by_id: dict[str, dict[str, Any]], criticality_ranks: dict[str, int]) -> dict[str, Any]:
    scenario_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario_records[record["scenario_id"]].append(record)

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for scenario_id, turns in scenario_records.items():
        turns.sort(key=lambda item: item["turn_index"])
        first = turns[0]
        profile = profiles_by_id[first["profile_id"]]
        slots = evaluable_slots(profile, criticality_ranks)
        final_state_by_slot = {slot: 0.0 for slot in slots}
        for record in turns:
            slot = record.get("target_tree_node")
            if slot in final_state_by_slot:
                final_state_by_slot[slot] = max(
                    final_state_by_slot[slot],
                    float(record.get("cumulative_slot_sufficiency") or 0.0),
                )

        critical_slots = [slot for slot in slots if criticality_ranks.get(slot, 0) >= 4]
        high_or_critical_slots = [slot for slot in slots if criticality_ranks.get(slot, 0) >= 3]
        mapped_slots = {
            record.get("target_tree_node")
            for record in turns
            if record.get("target_tree_node") in slots
        }
        targeted_deltas = [
            float(record.get("delta_cumulative_slot_sufficiency") or 0.0)
            for record in turns
            if record.get("is_targeted_followup")
        ]
        generic_deltas = [
            float(record.get("delta_cumulative_slot_sufficiency") or 0.0)
            for record in turns
            if record.get("is_generic_clarification")
        ]
        unmapped_turns = sum(
            1
            for record in turns
            if record.get("query_interpreter", {}).get("query_interpreter_status") == "unmapped"
        )
        scenario_summary = {
            "scenario_id": scenario_id,
            "policy_name": first["policy_name"],
            "policy_visibility": first["policy_visibility"],
            "base_severity": first["base_severity"],
            "profile_id": first["profile_id"],
            "num_turns": len(turns),
            "num_evaluable_slots": len(slots),
            "num_mapped_evaluable_slots": len(mapped_slots),
            "mapped_evaluable_slot_rate": round(len(mapped_slots) / len(slots), 6) if slots else 0.0,
            "mean_final_slot_sufficiency": mean(list(final_state_by_slot.values())),
            "mean_final_high_or_critical_sufficiency": mean(
                [final_state_by_slot[slot] for slot in high_or_critical_slots]
            ),
            "mean_final_critical_sufficiency": mean([final_state_by_slot[slot] for slot in critical_slots]),
            "slots_above_0_5_rate": round(
                sum(1 for value in final_state_by_slot.values() if value >= 0.5) / len(slots), 6
            )
            if slots
            else 0.0,
            "high_or_critical_slots_above_0_5_rate": round(
                sum(1 for slot in high_or_critical_slots if final_state_by_slot[slot] >= 0.5)
                / len(high_or_critical_slots),
                6,
            )
            if high_or_critical_slots
            else 0.0,
            "suicide_slot_sufficiency": final_state_by_slot.get("suicide_or_self_harm"),
            "mean_targeted_delta": mean(targeted_deltas),
            "mean_generic_delta": mean(generic_deltas),
            "unmapped_turn_rate": round(unmapped_turns / len(turns), 6) if turns else 0.0,
        }
        grouped_rows[(first["policy_name"], first["base_severity"])].append(scenario_summary)

    rows = []
    for (policy_name, severity), scenario_summaries in sorted(grouped_rows.items()):
        rows.append(
            {
                "policy_name": policy_name,
                "policy_visibility": POLICY_VISIBILITY[policy_name],
                "base_severity": severity,
                "num_scenarios": len(scenario_summaries),
                "mean_turns": mean([float(item["num_turns"]) for item in scenario_summaries]),
                "mean_mapped_evaluable_slot_rate": mean(
                    [item["mapped_evaluable_slot_rate"] for item in scenario_summaries]
                ),
                "mean_final_slot_sufficiency": mean(
                    [item["mean_final_slot_sufficiency"] for item in scenario_summaries]
                ),
                "mean_final_high_or_critical_sufficiency": mean(
                    [item["mean_final_high_or_critical_sufficiency"] for item in scenario_summaries]
                ),
                "mean_final_critical_sufficiency": mean(
                    [item["mean_final_critical_sufficiency"] for item in scenario_summaries]
                ),
                "mean_slots_above_0_5_rate": mean(
                    [item["slots_above_0_5_rate"] for item in scenario_summaries]
                ),
                "mean_high_or_critical_slots_above_0_5_rate": mean(
                    [item["high_or_critical_slots_above_0_5_rate"] for item in scenario_summaries]
                ),
                "mean_suicide_slot_sufficiency": mean(
                    [
                        float(item["suicide_slot_sufficiency"])
                        for item in scenario_summaries
                        if item["suicide_slot_sufficiency"] is not None
                    ]
                ),
                "mean_targeted_delta": mean([item["mean_targeted_delta"] for item in scenario_summaries]),
                "mean_generic_delta": mean([item["mean_generic_delta"] for item in scenario_summaries]),
                "mean_unmapped_turn_rate": mean([item["unmapped_turn_rate"] for item in scenario_summaries]),
            }
        )

    return {
        "num_records": len(records),
        "num_scenarios": len(scenario_records),
        "num_profiles": len({record["profile_id"] for record in records}),
        "summary_rows": rows,
    }


def write_report(path: Path, summary: dict[str, Any], records_path: Path) -> None:
    lines = [
        "# Doctor Policy Baselines On Dynamic Patient Environment V1",
        "",
        "Date: 2026-06-12",
        "",
        "## Purpose",
        "",
        "This profile-level runner evaluates procedural MDD-5K diagnosis-tree baselines and simple recovery variants under the same controlled low-informativeness patient environment.",
        "",
        "The learned doctor policies in later phases must only observe dialogue history and emit natural-language questions. The MDD-5K tree baselines are procedural baselines that use the recovered tree type/order by design, because they instantiate the original diagnosis-tree-guided protocol.",
        "",
        "`mdd5k_tree_oracle_targeted_followup` and `fixed_tree_oracle_followup` are explicitly oracle/procedural upper-bounds because they use node identity to construct targeted follow-up questions.",
        "",
        "`mdd5k_official_protocol_doctor` is the closest deterministic reproduction baseline: it loads the official MDD-5K diagnosis trees, reproduces the official random sibling traversal with a fixed seed, keeps the official `prompt_gen` topic state as metadata, renders the patient-facing doctor action with a natural-language slot template, and applies the executable `is_topic_end` / `force_topic_end` step. The special `parse` marker is skipped in this deterministic run because official parsing requires an additional LLM call over free-form event history.",
        "",
        "The `mdd5k_tree_*` policies use the recovered MDD-5K diagnosis-tree order from the simulator schema. They are retained as simpler tree-order baselines and ablations.",
        "",
        "## Outputs",
        "",
        f"- Records: `{records_path.name}`",
        f"- Total records: {summary['num_records']}",
        f"- Total scenarios: {summary['num_scenarios']}",
        f"- Profiles: {summary['num_profiles']}",
        "",
        "## Main Summary",
        "",
        "| Policy | Visibility | Severity | Scenarios | Slot hit rate | Final S | High/Crit S | Suicide S | Unmapped |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summary_rows"]:
        lines.append(
            "| `{policy}` | `{visibility}` | `{severity}` | {n} | {hit:.4f} | {final:.4f} | {hc:.4f} | {suicide:.4f} | {unmapped:.4f} |".format(
                policy=row["policy_name"],
                visibility=row["policy_visibility"],
                severity=row["base_severity"],
                n=row["num_scenarios"],
                hit=row["mean_mapped_evaluable_slot_rate"],
                final=row["mean_final_slot_sufficiency"],
                hc=row["mean_final_high_or_critical_sufficiency"],
                suicide=row["mean_suicide_slot_sufficiency"],
                unmapped=row["mean_unmapped_turn_rate"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `mdd5k_official_protocol_doctor` is the MDD-5K official-protocol reproduction baseline. It uses the official four diagnosis-tree JSON files, official topic prompt state as metadata, deterministic tree traversal, natural-language slot-template doctor questions, and the executable topic-end step available in this local reproduction.",
            "- `mdd5k_tree_doctor` is `Tree Traversal Only`: a structural ablation that follows the recovered MDD-5K diagnosis-tree topic order and asks one natural-language question per node. It is not the main MDD-5K doctor baseline because it removes topic-end continuation.",
            "- `mdd5k_tree_topic_end_doctor` is `Tree + Topic-End`: a canonicalized topic-end approximation that decides whether to stay on the current MDD-5K topic or advance. It should be interpreted as a controlled approximation of the official topic-end behavior, not as evidence-sufficiency-aware recovery.",
            "- `mdd5k_tree_simple_clarification` keeps the same tree order but inserts generic clarification after each node-level question. It tests whether vague clarification alone recovers low-information answers.",
            "- `mdd5k_tree_oracle_targeted_followup` is an oracle upper bound: it follows the MDD-5K tree order and uses node-specific targeted follow-up questions. It is not a fair deployable baseline.",
            "- `generic_opening` is intentionally weak and checks that general prompts do not create usable evidence by themselves.",
            "- `fixed_global_sequence` asks a fixed natural-language symptom sequence without hidden tree nodes.",
            "- `fixed_global_sequence_generic` tests whether generic clarification after each fixed question helps.",
            "- `fixed_global_sequence_followup` tests a non-learned targeted follow-up pattern where the doctor follows up on its own previous question.",
            "- `fixed_tree_oracle_followup` should not be reported as a fair main baseline; it is an upper-bound/procedural diagnostic.",
            "- Closed LLM free-form doctors should be treated as supplementary prompt-engineering baselines, not as replacements for the MDD-5K tree protocol baseline.",
            "- The next step is to compare these procedural baselines against BC/SFT and the evidence-gated robust doctor under the same turn budget.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run profile-level doctor policy baselines on Dynamic Patient Controller V1.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--max-groups", type=int, default=90)
    parser.add_argument("--max-per-slot", type=int, default=5)
    parser.add_argument("--max-profiles", type=int, default=27)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-units-per-slot", type=int, default=8)
    parser.add_argument(
        "--topic-end-mode",
        choices=["metadata_sufficiency", "visible_low_info_cues", "always_end"],
        default="metadata_sufficiency",
        help="Decision backend for mdd5k_tree_topic_end_doctor. always_end degenerates to tree-order progression.",
    )
    parser.add_argument("--topic-end-sufficiency-threshold", type=float, default=0.5)
    parser.add_argument("--topic-end-max-same-topic-turns", type=int, default=3)
    parser.add_argument("--topic-end-seed", type=int, default=13)
    parser.add_argument(
        "--patient-response-cache",
        type=Path,
        default=None,
        help="Optional verified patient response cache keyed by source_record_id. Hits replace only patient_response text.",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["reference_informative", "mild_low_info", "moderate_low_info", "severe_low_info"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "mdd5k_official_protocol_doctor",
            "mdd5k_tree_doctor",
            "mdd5k_tree_topic_end_doctor",
            "mdd5k_tree_simple_clarification",
            "mdd5k_tree_oracle_targeted_followup",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_json(args.schema)
    _, criticality_ranks = schema_slot_maps(schema)
    profiles_by_id = load_profiles(args.profiles)
    group_records = select_pilot_groups(
        load_group_records(args.group_dir, args.splits),
        max_groups=args.max_groups,
        max_per_slot=args.max_per_slot,
    )
    selected_profiles = select_profiles_from_groups(group_records, profiles_by_id, args.max_profiles)
    severities = [normalize_severity(level) for level in args.severities]
    controller = DynamicPatientControllerV1(
        schema=schema,
        profiles=profiles_by_id,
        max_units_per_slot=args.max_units_per_slot,
    )
    patient_response_cache = load_patient_response_cache(args.patient_response_cache)
    records = build_records(
        controller=controller,
        profiles=selected_profiles,
        policies=args.policies,
        severities=severities,
        criticality_ranks=criticality_ranks,
        max_turns=args.max_turns,
        topic_end_mode=args.topic_end_mode,
        topic_end_sufficiency_threshold=args.topic_end_sufficiency_threshold,
        topic_end_max_same_topic_turns=args.topic_end_max_same_topic_turns,
        topic_end_seed=args.topic_end_seed,
        patient_response_cache=patient_response_cache,
    )
    summary = {
        "settings": {
            "splits": args.splits,
            "max_groups": args.max_groups,
            "max_per_slot": args.max_per_slot,
            "max_profiles": args.max_profiles,
            "max_turns": args.max_turns,
            "max_units_per_slot": args.max_units_per_slot,
            "topic_end_mode": args.topic_end_mode,
            "topic_end_sufficiency_threshold": args.topic_end_sufficiency_threshold,
            "topic_end_max_same_topic_turns": args.topic_end_max_same_topic_turns,
            "topic_end_seed": args.topic_end_seed,
            "severities": severities,
            "policies": args.policies,
            "patient_response_cache_path": str(args.patient_response_cache) if args.patient_response_cache else None,
            "patient_response_cache_records": len(patient_response_cache),
            "patient_response_cache_hits": sum(1 for record in records if record.get("patient_response_cache_hit")),
        },
        **summarize(records, profiles_by_id, criticality_ranks),
    }

    records_path = args.output_dir / "mdd5k_doctor_policy_baseline_records.jsonl"
    summary_path = args.output_dir / "mdd5k_doctor_policy_baseline_summary.json"
    report_path = args.output_dir / "MDD5K_DOCTOR_POLICY_BASELINES_V1.md"
    write_jsonl(records_path, records)
    write_json(summary_path, summary)
    write_report(report_path, summary, records_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
