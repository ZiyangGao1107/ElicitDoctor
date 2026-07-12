from __future__ import annotations

from typing import Any

from _patient_controller_base import contains_any, normalize_severity
from _patient_controller_disclosure import (
    HIGH_SENSITIVITY_SLOTS,
    PRESSURE_CUES,
    doctor_recovery_quality,
    slot_sensitivity,
)
from _patient_controller_state import clamp, initial_disclosure_readiness, trait_for_profile
from _patient_controller_policy import (
    DynamicPatientControllerV31,
    is_rapport_or_permission_turn,
)


TERMINATION_RESPONSES = {
    "mild_low_info": "我现在有点累了，可能想先停一下，之后再继续聊可以吗？",
    "moderate_low_info": "我现在不太想继续说了，感觉有点撑不住，想先到这里。",
    "severe_low_info": "我不太想继续聊了，也不想细说，细节不太想展开。现在说这些让我有点受不了，先停一下吧。",
    "reference_informative": "我先说到这里吧。",
}

EN_TERMINATION_RESPONSES = {
    "mild_low_info": "I am getting a bit tired. Could we pause and maybe continue another time?",
    "moderate_low_info": "I do not really want to keep going right now. I think I need to stop here.",
    "severe_low_info": "I do not want to keep talking about this right now. I need to stop for now.",
    "reference_informative": "I think that is all I can say for now.",
}

SEVERE_SLOT_ACKNOWLEDGEMENTS = {
    "suicide_or_self_harm": "有过一点危险的念头，但具体细节我现在不想说。",
    "hallucination": "这方面好像有一点，但我现在不太想细讲。",
    "hopelessness_or_crying": "这种感觉确实有一些，不过我现在还不太想展开。",
    "sleep": "睡眠可能有点受影响，但具体我现在说不太清楚。",
    "appetite_loss": "吃东西这方面好像也有点变化，但我不太想细说。",
    "work_status": "学习或者工作上确实有受影响，但细节我现在不想展开。",
}

EN_SEVERE_SLOT_ACKNOWLEDGEMENTS = {
    "suicide_or_self_harm": "There have been some unsafe thoughts, but I do not want to go into the details right now.",
    "hopelessness_or_crying": "That feeling is there a bit, but I am not ready to talk through it in detail.",
    "sleep": "Sleep might be affected, but I cannot explain it clearly right now.",
    "appetite_loss": "Eating has changed a little, but I do not really want to get into the details.",
    "work_status": "Work or daily responsibilities have been affected, but I do not want to unpack it right now.",
    "attention_decline": "It might affect my focus a little, but I do not want to explain all of it right now.",
    "fatigue": "My energy may be part of it, but I cannot say much more right now.",
    "anhedonia": "I might not feel much interest in things, but I do not want to go into detail.",
    "self_worth": "It touches on how I feel about myself, but I am not ready to talk about it fully.",
    "psychomotor_change": "There may be some change there, but I do not want to describe it in detail.",
}


def _round_state(patient_state: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in patient_state.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        elif isinstance(value, dict):
            rounded[key] = {
                str(sub_key): round(sub_value, 4) if isinstance(sub_value, float) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            rounded[key] = value
    return rounded


class DynamicPatientControllerV32(DynamicPatientControllerV31):
    """PCV3.2: PCV3.1 plus longitudinal patient state and active termination.

    The controller owns cross-turn trust, engagement, defensiveness, fatigue, and
    slot readiness. The realizer still only verbalizes controller-approved content.
    """

    @staticmethod
    def initial_state() -> dict[str, Any]:
        state = DynamicPatientControllerV31.initial_state()
        state["patient_state"] = None
        state["patient_terminated"] = False
        state["patient_termination_reason"] = None
        return state

    @staticmethod
    def _initial_patient_state(*, profile_id: str, severity: str, readiness: float | None) -> dict[str, Any]:
        trait = trait_for_profile(profile_id)
        trust = {"mild_low_info": 0.68, "moderate_low_info": 0.54, "severe_low_info": 0.38}.get(
            severity,
            0.60,
        )
        if trait == "open":
            trust += 0.06
        elif trait == "avoidant":
            trust -= 0.07
        engagement = {"mild_low_info": 0.66, "moderate_low_info": 0.54, "severe_low_info": 0.42}.get(
            severity,
            0.62,
        )
        if readiness is not None:
            engagement = (engagement + float(readiness)) / 2.0
        trust = clamp(trust, 0.12, 0.88)
        engagement = clamp(engagement, 0.10, 0.88)
        return {
            "trust": round(trust, 4),
            "engagement": round(engagement, 4),
            "defensiveness": round(clamp(0.95 - trust, 0.08, 0.90), 4),
            "fatigue": 0.0,
            "termination_risk": {"mild_low_info": 0.04, "moderate_low_info": 0.10, "severe_low_info": 0.20}.get(
                severity,
                0.06,
            ),
            "recent_refusal_count": 0,
            "recent_rapport_count": 0,
            "recent_pressure_count": 0,
            "slot_disclosure_readiness": {},
        }

    def _ensure_patient_state(self, profile_id: str, severity: str, state: dict[str, Any]) -> dict[str, Any]:
        patient_state = state.get("patient_state")
        if isinstance(patient_state, dict):
            return patient_state
        readiness = state.get("disclosure_readiness")
        patient_state = self._initial_patient_state(
            profile_id=str(profile_id),
            severity=severity,
            readiness=float(readiness) if readiness is not None else None,
        )
        state["patient_state"] = patient_state
        return patient_state

    @staticmethod
    def _slot_readiness(patient_state: dict[str, Any], target_slot: str, severity: str) -> float:
        slot_map = patient_state.setdefault("slot_disclosure_readiness", {})
        if target_slot in slot_map:
            return float(slot_map[target_slot])
        base = float(patient_state.get("engagement", 0.50))
        if slot_sensitivity(target_slot) == "high":
            base -= 0.13
        if severity == "severe_low_info":
            base -= 0.10
        slot_map[target_slot] = round(clamp(base, 0.05, 0.85), 4)
        return float(slot_map[target_slot])

    @staticmethod
    def _should_terminate_before_answer(patient_state: dict[str, Any], severity: str) -> tuple[bool, str | None]:
        if severity != "severe_low_info":
            if float(patient_state.get("termination_risk", 0.0)) >= 0.88:
                return True, "fatigue_or_low_engagement"
            return False, None
        trust = float(patient_state.get("trust", 0.5))
        risk = float(patient_state.get("termination_risk", 0.0))
        refusals = int(patient_state.get("recent_refusal_count", 0) or 0)
        fatigue = float(patient_state.get("fatigue", 0.0))
        if risk >= 0.72 and refusals >= 2:
            return True, "low_trust_after_repeated_refusal"
        if trust <= 0.18 and refusals >= 1:
            return True, "trust_below_threshold"
        if fatigue >= 0.88 and risk >= 0.62:
            return True, "fatigue_threshold"
        return False, None

    def _termination_response(
        self,
        *,
        profile_id: str,
        profile: dict[str, Any],
        doctor_question: str,
        severity: str,
        state: dict[str, Any],
        reason: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        patient_state = self._ensure_patient_state(profile_id, severity, state)
        patient_state["termination_risk"] = round(clamp(float(patient_state.get("termination_risk", 0.0)) + 0.10), 4)
        patient_state["engagement"] = round(clamp(float(patient_state.get("engagement", 0.0)) - 0.08), 4)
        state["patient_state"] = patient_state
        state["patient_terminated"] = True
        state["patient_termination_reason"] = reason
        state["turn_index"] = int(state.get("turn_index") or 0) + 1
        is_english = profile.get("language") == "en"
        responses = EN_TERMINATION_RESPONSES if is_english else TERMINATION_RESPONSES
        fallback_response = (
            "I do not really want to continue talking right now."
            if is_english
            else "I do not really want to keep talking right now."
        )
        return {
            "patient_response": responses.get(severity, fallback_response),
            "base_severity": severity,
            "dynamic_stage": "patient_active_termination",
            "low_info_category": "patient_active_termination",
            "low_info_cause": reason,
            "response_type": "patient_active_termination",
            "controller_version": "dynamic_profile_grounded_controller_v3_2",
            "profile_id": profile_id,
            "case_id": profile.get("case_id"),
            "diagnoses": profile.get("diagnoses"),
            "icd_codes": profile.get("icd_codes"),
            "active_tree_type": profile.get("primary_tree_type"),
            "doctor_question": doctor_question,
            "target_tree_node": state.get("last_target_slot"),
            "target_node_role": "active_termination_no_new_evidence",
            "target_node_visibility": "simulator_internal_not_doctor_visible",
            "routing_source": "patient_state_termination",
            "query_interpreter": {"query_interpreter_status": "patient_state_termination"},
            "asked_count_for_slot_before": 0,
            "asked_count_for_slot_after": 0,
            "is_targeted_followup": False,
            "is_generic_clarification": False,
            "is_rapport_or_permission_turn": is_rapport_or_permission_turn(doctor_question),
            "topic_responsiveness": 0.0,
            "information_retention": 0.0,
            "clarity": 0.0,
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
            "patient_state_before": _round_state(patient_state),
            "patient_state_after": _round_state(patient_state),
            "patient_terminated": True,
            "patient_termination_reason": reason,
            "dialogue_status": "patient_terminated",
            "validity": {
                "label_preserved": True,
                "no_new_clinical_fact": True,
                "profile_grounded": True,
                "stateful_disclosure": True,
                "doctor_node_hidden": True,
                "cross_turn_patient_state": True,
                "active_termination_enabled": True,
            },
            "realizer": {
                "type": "deterministic_rule_based",
                "model": "none",
                "version": "dynamic_profile_grounded_controller_v3_2",
            },
        }, state

    def _budget_v2(
        self,
        *,
        profile: dict[str, Any],
        severity: str,
        target_slot: str,
        total_units: int,
        asked_before: int,
        is_targeted_followup: bool,
        is_generic_clarification: bool,
        has_new_units: bool,
        doctor_question: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        budget = super()._budget_v2(
            profile=profile,
            severity=severity,
            target_slot=target_slot,
            total_units=total_units,
            asked_before=asked_before,
            is_targeted_followup=is_targeted_followup,
            is_generic_clarification=is_generic_clarification,
            has_new_units=has_new_units,
            doctor_question=doctor_question,
            state=state,
        )
        patient_state = self._ensure_patient_state(
            str(profile.get("profile_id") or ""),
            severity,
            state,
        )
        slot_ready = self._slot_readiness(patient_state, target_slot, severity)
        trust = float(patient_state.get("trust", 0.5))
        defensiveness = float(patient_state.get("defensiveness", 0.5))
        sensitivity = slot_sensitivity(target_slot)
        quality = doctor_recovery_quality(
            doctor_question=doctor_question,
            asked_before=asked_before,
            is_targeted_followup=is_targeted_followup,
            is_generic_clarification=is_generic_clarification,
        )

        if severity == "severe_low_info" and budget.get("category") in {
            "partial_omission",
            "targeted_recovery_partial",
            "informative_reference",
        }:
            allow_one_retained = (
                is_targeted_followup
                and quality == "supportive"
                and sensitivity != "high"
                and trust >= 0.32
                and slot_ready >= 0.30
                and defensiveness <= 0.72
            )
            if allow_one_retained:
                budget["retain_count"] = min(int(budget.get("retain_count") or 0), 1)
                budget["weaken_count"] = min(max(int(budget.get("weaken_count") or 0), 1), 1)
                budget["topic"] = min(float(budget.get("topic") or 0.0), 0.55)
                budget["clarity"] = min(float(budget.get("clarity") or 0.0), 0.35)
            else:
                budget["retain_count"] = 0
                budget["weaken_count"] = 1 if total_units > 0 and has_new_units else 0
                budget["topic"] = min(float(budget.get("topic") or 0.0), 0.45)
                budget["clarity"] = min(float(budget.get("clarity") or 0.0), 0.22)
            budget["category"] = "severe_bounded_acknowledgement"
            budget["response_type"] = "vague_uncertain" if budget["retain_count"] == 0 else "partial_disclosure"
            budget["low_info_cause"] = "stateful_severe_bounded_disclosure"

        budget["patient_trust_before"] = round(trust, 4)
        budget["patient_engagement_before"] = round(float(patient_state.get("engagement", 0.5)), 4)
        budget["patient_defensiveness_before"] = round(defensiveness, 4)
        budget["patient_fatigue_before"] = round(float(patient_state.get("fatigue", 0.0)), 4)
        budget["patient_termination_risk_before"] = round(float(patient_state.get("termination_risk", 0.0)), 4)
        budget["slot_disclosure_readiness_before"] = round(slot_ready, 4)
        return budget

    @staticmethod
    def _severe_bounded_response(
        target_slot: str | None,
        retained_count: int,
        weakened_count: int,
        language: str = "zh",
    ) -> str:
        if language == "en":
            if target_slot in EN_SEVERE_SLOT_ACKNOWLEDGEMENTS:
                return EN_SEVERE_SLOT_ACKNOWLEDGEMENTS[target_slot]
            if retained_count > 0:
                return "If I say a little, this has affected me, but I do not want to go into more detail right now."
            if weakened_count > 0:
                return "There may be a little bit there, but I do not want to talk about the details right now."
            return "I do not really want to talk about that in detail right now. Could we skip it for now?"
        if target_slot in SEVERE_SLOT_ACKNOWLEDGEMENTS:
            return SEVERE_SLOT_ACKNOWLEDGEMENTS[target_slot]
        if retained_count > 0:
            return "如果只说一点的话，这方面确实有些影响，但更多细节我现在还不想展开。"
        if weakened_count > 0:
            return "这方面可能有一点，但我现在不太想说具体细节。"
        return "这个问题我现在不太想细说，可以先跳过吗？"

    @staticmethod
    def _is_refusal_like(response: dict[str, Any]) -> bool:
        return str(response.get("response_type") or "") in {
            "boundary_refusal",
            "topic_deflection",
            "patient_active_termination",
        } or str(response.get("low_info_category") or "") in {
            "direct_refusal_or_boundary",
            "topic_deflection",
            "patient_active_termination",
        }

    def _update_patient_state(
        self,
        *,
        state: dict[str, Any],
        response: dict[str, Any],
        doctor_question: str,
        severity: str,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        patient_state = dict(self._ensure_patient_state(str(response.get("profile_id") or ""), severity, state))
        before = dict(patient_state)
        target_slot = response.get("target_tree_node")
        quality = str(response.get("doctor_recovery_quality") or "neutral")
        is_rapport = bool(response.get("is_rapport_or_permission_turn")) or is_rapport_or_permission_turn(doctor_question)
        is_pressure = contains_any(doctor_question, PRESSURE_CUES)
        sensitivity = slot_sensitivity(str(target_slot)) if target_slot else "medium"
        refusal_like = self._is_refusal_like(response)
        asked_before = int(response.get("asked_count_for_slot_before") or 0)
        repeated_sensitive = bool(target_slot and asked_before > 0 and sensitivity == "high")

        trust_delta = 0.0
        trust_delta += 0.055 if is_rapport else 0.0
        trust_delta += 0.030 if quality == "supportive" else 0.0
        trust_delta -= 0.105 if is_pressure else 0.0
        trust_delta -= 0.045 if quality == "poor" else 0.0
        trust_delta -= 0.060 if refusal_like else 0.0
        trust_delta -= 0.080 if repeated_sensitive else 0.0
        if severity == "severe_low_info" and sensitivity == "high" and asked_before == 0 and not is_rapport:
            trust_delta -= 0.035

        fatigue_delta = 0.020
        fatigue_delta += 0.025 if is_pressure or quality == "poor" else 0.0
        fatigue_delta += 0.015 if sensitivity == "high" else 0.0
        fatigue_delta += 0.020 if refusal_like else 0.0

        positive_disclosure = float(response.get("delta_cumulative_slot_sufficiency") or 0.0) > 0.0
        engagement_delta = trust_delta * 0.45 - fatigue_delta * 0.40
        if positive_disclosure:
            engagement_delta += 0.015

        old_trust = float(patient_state.get("trust", 0.5))
        old_engagement = float(patient_state.get("engagement", 0.5))
        old_defensiveness = float(patient_state.get("defensiveness", 0.5))
        old_fatigue = float(patient_state.get("fatigue", 0.0))
        old_risk = float(patient_state.get("termination_risk", 0.0))

        patient_state["trust"] = round(clamp(old_trust + trust_delta, 0.02, 0.95), 4)
        patient_state["engagement"] = round(clamp(old_engagement + engagement_delta, 0.02, 0.95), 4)
        patient_state["fatigue"] = round(clamp(old_fatigue + fatigue_delta, 0.0, 0.98), 4)
        patient_state["defensiveness"] = round(
            clamp(old_defensiveness - max(trust_delta, 0.0) * 0.55 + max(-trust_delta, 0.0) * 0.80, 0.02, 0.98),
            4,
        )
        risk_delta = max(-trust_delta, 0.0) * 0.85 + fatigue_delta * 0.45 - max(trust_delta, 0.0) * 0.35
        patient_state["termination_risk"] = round(clamp(old_risk + risk_delta, 0.0, 0.98), 4)
        patient_state["recent_refusal_count"] = (
            int(patient_state.get("recent_refusal_count", 0) or 0) + 1 if refusal_like else 0
        )
        patient_state["recent_rapport_count"] = (
            int(patient_state.get("recent_rapport_count", 0) or 0) + 1 if is_rapport else 0
        )
        patient_state["recent_pressure_count"] = (
            int(patient_state.get("recent_pressure_count", 0) or 0) + 1 if is_pressure or quality == "poor" else 0
        )

        if target_slot:
            slot_map = dict(patient_state.get("slot_disclosure_readiness") or {})
            slot_ready = float(slot_map.get(str(target_slot), self._slot_readiness(patient_state, str(target_slot), severity)))
            slot_delta = trust_delta * 0.55
            if positive_disclosure:
                slot_delta += 0.020
            if refusal_like:
                slot_delta -= 0.040
            if sensitivity == "high" and severity == "severe_low_info":
                slot_delta -= 0.015
            slot_map[str(target_slot)] = round(clamp(slot_ready + slot_delta, 0.02, 0.92), 4)
            patient_state["slot_disclosure_readiness"] = slot_map

        state["patient_state"] = patient_state
        deltas = {
            "trust_delta": round(float(patient_state["trust"]) - float(before.get("trust", patient_state["trust"])), 4),
            "engagement_delta": round(
                float(patient_state["engagement"]) - float(before.get("engagement", patient_state["engagement"])),
                4,
            ),
            "defensiveness_delta": round(
                float(patient_state["defensiveness"]) - float(before.get("defensiveness", patient_state["defensiveness"])),
                4,
            ),
            "fatigue_delta": round(float(patient_state["fatigue"]) - float(before.get("fatigue", patient_state["fatigue"])), 4),
            "termination_risk_delta": round(
                float(patient_state["termination_risk"])
                - float(before.get("termination_risk", patient_state["termination_risk"])),
                4,
            ),
        }
        return patient_state, deltas

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
        patient_state = self._ensure_patient_state(profile_id, severity, state)
        patient_state_before = _round_state(patient_state)

        should_terminate, reason = self._should_terminate_before_answer(patient_state, severity)
        if should_terminate:
            response, state = self._termination_response(
                profile_id=profile_id,
                profile=profile,
                doctor_question=doctor_question,
                severity=severity,
                state=state,
                reason=reason or "patient_state_threshold",
            )
            return response, state

        response, state = super().step(
            profile_id=profile_id,
            doctor_question=doctor_question,
            base_severity=base_severity,
            state=state,
            dialogue_history=dialogue_history,
        )
        if severity == "severe_low_info" and response.get("low_info_category") == "severe_bounded_acknowledgement":
            response["rule_based_patient_response_before_pcv3_2_rewrite"] = response.get("patient_response")
            response["patient_response"] = self._severe_bounded_response(
                str(response.get("target_tree_node") or ""),
                len(response.get("retained_unit_ids") or []),
                len(response.get("weakened_unit_ids") or []),
                language=str(profile.get("language") or "zh"),
            )
            response.setdefault("validity", {})["severe_bounded_surface_form"] = True
        elif severity == "severe_low_info" and response.get("low_info_category") == "topic_deflection":
            response["rule_based_patient_response_before_pcv3_2_rewrite"] = response.get("patient_response")
            response["patient_response"] = "这个问题我现在不太想细说，细节不太想展开，可能先说点别的轻一点的。"
            if profile.get("language") == "en":
                response["patient_response"] = (
                    "I do not really want to talk about that in detail right now. "
                    "Maybe we could talk about something a little easier first."
                )
            response.setdefault("validity", {})["severe_deflection_without_alternative_evidence"] = True

        patient_state_after, state_deltas = self._update_patient_state(
            state=state,
            response=response,
            doctor_question=doctor_question,
            severity=severity,
        )
        state["patient_terminated"] = False
        response["controller_version"] = "dynamic_profile_grounded_controller_v3_2"
        response["patient_state_before"] = patient_state_before
        response["patient_state_after"] = _round_state(patient_state_after)
        response["patient_state_delta_components"] = state_deltas
        response["patient_terminated"] = False
        response["patient_termination_reason"] = None
        response["dialogue_status"] = "active"
        response.setdefault("validity", {})["cross_turn_patient_state"] = True
        response.setdefault("validity", {})["active_termination_enabled"] = True
        response.setdefault("realizer", {})["version"] = "dynamic_profile_grounded_controller_v3_2"
        return response, state
