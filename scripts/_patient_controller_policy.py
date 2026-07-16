from __future__ import annotations

from typing import Any

from _patient_controller_base import (
    ANAPHORA_CUES,
    contains_any,
    normalize_severity,
)
from _patient_controller_disclosure import doctor_recovery_quality
from _patient_controller_state import (
    DynamicPatientControllerV3,
    initial_disclosure_readiness,
    trait_for_profile,
    update_disclosure_readiness,
)


RAPPORT_OR_PERMISSION_CUES = [
    "慢慢来",
    "没关系",
    "不用着急",
    "可以先",
    "愿意",
    "如果可以",
    "方便",
    "不需要说细节",
    "不用说细节",
    "先从",
    "从你愿意说的",
    "我们可以",
    "我只是想了解",
    "我想确认",
    "你觉得可以",
]

OPEN_NARRATIVE_PATIENT_RESPONSES = {
    "fully_cooperative": "可以，我会尽量说清楚。你可以先问你最需要了解的部分。",
    "random_disclosure": "可以，我可以先说一点。只是有些地方我可能一时还不太想展开。",
    "mild_low_info": "可以，我可以先慢慢说一点。最近整体确实不太舒服，但具体从哪说起我还有点乱。",
    "moderate_low_info": "我可以试着说一点，不过现在很多事情我自己也说不太清楚，可能只能先大概讲讲。",
    "severe_low_info": "我知道你是在问我，但这个现在说起来有点难。我可能需要先缓一缓，只能先说我最近状态确实不太好。",
    "reference_informative": "可以，我会尽量慢慢说清楚。",
}


def is_rapport_or_permission_turn(text: str) -> bool:
    return contains_any(text or "", RAPPORT_OR_PERMISSION_CUES)


class DynamicPatientControllerV31(DynamicPatientControllerV3):
    """PCV3.1: V3 plus rapport/open-narrative handling and clearer disclosure metadata.

    The controller still owns evidence disclosure. This class only changes routing for
    supportive-but-low-specificity turns so they are not treated as ordinary unmapped
    failures, and it exposes withheld/forbidden metadata for constrained LLM realizers.
    """

    def _route_target(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
        doctor_question: str,
        dialogue_history: list[dict[str, str]] | None,
    ) -> tuple[str | None, dict[str, Any], str]:
        target, pred, routing_source = super()._route_target(
            profile,
            state,
            doctor_question,
            dialogue_history,
        )
        if target:
            pred.setdefault("pcv3_1_route_type", "slot_mapped")
            return target, pred, routing_source

        if state.get("last_target_slot") and is_rapport_or_permission_turn(doctor_question):
            target = state["last_target_slot"]
            pred["simulator_internal_target_node"] = target
            pred["target_tree_node"] = target
            pred["query_interpreter_status"] = "rapport_or_permission_fallback_to_previous_target"
            pred["query_interpreter_confidence"] = "medium"
            pred["pcv3_1_route_type"] = "rapport_previous_slot"
            return target, pred, "rapport_or_permission_to_previous_target"

        if is_rapport_or_permission_turn(doctor_question):
            pred["query_interpreter_status"] = "rapport_or_permission_open_narrative"
            pred["query_interpreter_confidence"] = "low"
            pred["pcv3_1_route_type"] = "rapport_open_narrative"
            return None, pred, "rapport_or_permission_open_narrative"

        pred.setdefault("pcv3_1_route_type", "unmapped")
        return None, pred, routing_source

    def _open_narrative_response(
        self,
        *,
        profile_id: str,
        profile: dict[str, Any],
        doctor_question: str,
        severity: str,
        state: dict[str, Any],
        previous_readiness: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response_text = OPEN_NARRATIVE_PATIENT_RESPONSES.get(
            severity,
            "可以，我可以先慢慢说一点，但现在还不太知道怎么讲清楚。",
        )
        quality = "supportive" if is_rapport_or_permission_turn(doctor_question) else "neutral"
        updated_readiness, readiness_delta = update_disclosure_readiness(
            previous_readiness=previous_readiness,
            response_type="rapport_open_narrative",
            doctor_quality=quality,
            prior_boundary_refusal=False,
            asked_before=0,
            is_targeted_followup=False,
            is_generic_clarification=False,
        )
        # A rapport/opening turn should not be penalized as unmapped. It costs a turn
        # and reveals no evidence, but can make later disclosure slightly easier.
        if quality == "supportive":
            updated_readiness = min(0.95, round(updated_readiness + 0.02, 4))
            readiness_delta["rapport_opening_bonus"] = 0.02
            readiness_delta["total_delta"] = round(float(readiness_delta.get("total_delta", 0.0)) + 0.02, 4)
        state["disclosure_readiness"] = updated_readiness
        state["turn_index"] = int(state.get("turn_index") or 0) + 1
        return {
            "patient_response": response_text,
            "base_severity": severity,
            "dynamic_stage": "rapport_open_narrative",
            "low_info_category": "rapport_open_narrative_no_evidence",
            "low_info_cause": "rapport_open_narrative",
            "response_type": "rapport_open_narrative",
            "controller_version": "dynamic_profile_grounded_controller_v3_1",
            "profile_id": profile_id,
            "case_id": profile.get("case_id"),
            "diagnoses": profile.get("diagnoses"),
            "icd_codes": profile.get("icd_codes"),
            "active_tree_type": profile.get("primary_tree_type"),
            "doctor_question": doctor_question,
            "target_tree_node": None,
            "target_node_role": "open_narrative_no_slot",
            "target_node_visibility": "simulator_internal_not_doctor_visible",
            "routing_source": "rapport_or_permission_open_narrative",
            "query_interpreter": {
                "query_interpreter_status": "rapport_or_permission_open_narrative",
                "pcv3_1_route_type": "rapport_open_narrative",
            },
            "asked_count_for_slot_before": 0,
            "asked_count_for_slot_after": 0,
            "is_targeted_followup": False,
            "is_generic_clarification": False,
            "is_rapport_or_permission_turn": True,
            "topic_responsiveness": 0.0,
            "information_retention": 0.0,
            "clarity": 0.55,
            "g_target": 0.0,
            "previous_g_target_for_slot": 0.0,
            "delta_g_target_for_slot": 0.0,
            "cumulative_slot_sufficiency": 0.0,
            "previous_cumulative_slot_sufficiency": 0.0,
            "delta_cumulative_slot_sufficiency": 0.0,
            "target_slot_evidence_unit_count": 0,
            "retained_unit_ids": [],
            "weakened_unit_ids": [],
            "removed_unit_ids": [],
            "withheld_unit_ids": [],
            "forbidden_unit_ids": [],
            "retained_profile_unit_ids": [],
            "weakened_profile_unit_ids": [],
            "withheld_profile_unit_ids": [],
            "forbidden_profile_unit_ids": [],
            "observed_evidence_units": [],
            "disclosed_profile_unit_ids_before": [],
            "disclosed_profile_unit_ids_after": [],
            "disclosure_readiness_before": round(previous_readiness, 4),
            "disclosure_readiness_after": updated_readiness,
            "delta_disclosure_readiness": round(updated_readiness - previous_readiness, 4),
            "disclosure_readiness_delta_components": readiness_delta,
            "validity": {
                "label_preserved": True,
                "no_new_clinical_fact": True,
                "profile_grounded": True,
                "stateful_disclosure": True,
                "doctor_node_hidden": True,
                "theory_constrained_stress_environment": True,
                "rapport_turn_not_penalized_as_unmapped": True,
            },
            "realizer": {
                "type": "deterministic_rule_based",
                "model": "none",
                "version": "dynamic_profile_grounded_controller_v3_1",
            },
        }, state

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
        severity = normalize_severity(base_severity)
        if state.get("disclosure_readiness") is None:
            trait = trait_for_profile(str(profile.get("profile_id") or profile_id))
            state["disclosure_readiness"] = initial_disclosure_readiness(trait=trait, severity=severity)
        previous_readiness = float(state.get("disclosure_readiness") or 0.50)

        target, pred, routing_source = self._route_target(profile, state, doctor_question, dialogue_history)
        if not target and pred.get("pcv3_1_route_type") == "rapport_open_narrative":
            return self._open_narrative_response(
                profile_id=profile_id,
                profile=profile,
                doctor_question=doctor_question,
                severity=severity,
                state=state,
                previous_readiness=previous_readiness,
            )

        response, state = super().step(
            profile_id=profile_id,
            doctor_question=doctor_question,
            base_severity=base_severity,
            state=state,
            dialogue_history=dialogue_history,
        )
        response["controller_version"] = "dynamic_profile_grounded_controller_v3_1"
        response.setdefault("query_interpreter", {})["pcv3_1_route_type"] = pred.get("pcv3_1_route_type")
        response["is_rapport_or_permission_turn"] = is_rapport_or_permission_turn(doctor_question)
        response["pcv3_1_routing_source"] = routing_source

        # Current profiles do not mark permanently forbidden evidence units inside a slot.
        # In-profile units not disclosed this turn are withheld; diagnosis labels and any
        # facts absent from the profile remain forbidden to any LLM realizer.
        response["withheld_unit_ids"] = list(response.get("removed_unit_ids") or [])
        response["withheld_profile_unit_ids"] = [
            str(unit.get("profile_unit_id") or unit.get("unit_id"))
            for unit in (response.get("observed_evidence_units") or [])
            if unit.get("unit_id") in set(response.get("removed_unit_ids") or [])
        ]
        response["forbidden_unit_ids"] = []
        response["forbidden_profile_unit_ids"] = []
        response.setdefault("validity", {})["withheld_forbidden_metadata_v3_1"] = True
        response.setdefault("realizer", {})["version"] = "dynamic_profile_grounded_controller_v3_1"
        return response, state
