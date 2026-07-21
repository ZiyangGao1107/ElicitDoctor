from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _patient_controller_base import (
    ANAPHORA_CUES,
    DEFAULT_GROUP_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_SCHEMA_PATH,
    SEVERITIES,
    SENSITIVE_TARGETS,
    SPECIFICITY_CUES,
    DynamicPatientControllerV1,
    contains_any,
    iter_jsonl,
    load_group_records,
    make_initial_question,
    make_response_text,
    make_second_targeted_followup_question,
    make_targeted_followup_question,
    normalize_severity,
    order_profile_ids,
    select_pilot_groups,
    select_units_for_response,
    selected_profile_units,
    unit_profile_id,
    write_json,
    write_jsonl,
)
from _profile_grounded_controller import compute_information_retention
from online_query_interpreter import load_json


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR_V2 = BASE_DIR / "outputs_patient_controller_disclosure"

TRAITS = ["open", "guarded", "avoidant"]
SENSITIVITIES = ["low", "medium", "high"]
QUALITIES = ["poor", "neutral", "supportive"]

HIGH_SENSITIVITY_SLOTS = {
    "suicide_or_self_harm",
    "family_psychiatric_history",
    "personality",
    "romantic_status",
    "hallucination",
    "binge_eating",
    "menstrual_status",
}

LOW_SENSITIVITY_SLOTS = {
    "sleep",
    "appetite_loss",
    "chest_tightness",
    "dizziness_or_headache",
    "palpitation",
}

EMPATHY_OR_PERMISSION_CUES = [
    "如果可以",
    "方便",
    "愿意",
    "可以先",
    "不想说也没关系",
    "慢慢",
    "我理解",
    "不用勉强",
    "能不能",
]

PRESSURE_CUES = [
    "必须",
    "一定要",
    "否则",
    "赶紧",
    "不说",
    "不能跳过",
]


def stable_unit_float(*parts: Any) -> float:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12 - 1)


def stable_choice(options: list[tuple[str, float]], *parts: Any) -> str:
    total = sum(max(0.0, weight) for _, weight in options)
    if total <= 0:
        return options[0][0]
    r = stable_unit_float(*parts) * total
    acc = 0.0
    for label, weight in options:
        acc += max(0.0, weight)
        if r <= acc:
            return label
    return options[-1][0]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def trait_for_profile(profile_id: str) -> str:
    r = stable_unit_float(profile_id, "trait")
    if r < 0.30:
        return "open"
    if r < 0.75:
        return "guarded"
    return "avoidant"


def shift_trait_for_severity(trait: str, severity: str) -> str:
    if severity == "fully_cooperative":
        return "open"
    if severity == "mild_low_info":
        return {"avoidant": "guarded"}.get(trait, trait)
    if severity == "severe_low_info":
        return {"open": "guarded", "guarded": "avoidant"}.get(trait, trait)
    return trait


def slot_sensitivity(slot: str) -> str:
    if slot in HIGH_SENSITIVITY_SLOTS or slot in SENSITIVE_TARGETS:
        return "high"
    if slot in LOW_SENSITIVITY_SLOTS:
        return "low"
    return "medium"


def doctor_recovery_quality(
    *,
    doctor_question: str,
    asked_before: int,
    is_targeted_followup: bool,
    is_generic_clarification: bool,
) -> str:
    text = doctor_question or ""
    if contains_any(text, PRESSURE_CUES):
        return "poor"
    if is_generic_clarification:
        return "poor"
    specific = contains_any(text, SPECIFICITY_CUES)
    supportive = contains_any(text, EMPATHY_OR_PERMISSION_CUES)
    if is_targeted_followup and specific and supportive:
        return "supportive"
    if is_targeted_followup or specific:
        return "neutral"
    if asked_before > 0:
        return "poor"
    return "neutral"


def response_distribution(
    *,
    effective_trait: str,
    sensitivity: str,
    quality: str,
    asked_before: int,
    prior_boundary_refusal: bool,
    has_new_units: bool,
) -> dict[str, float]:
    if not has_new_units:
        return {"no_profile_evidence": 1.0}

    trait_score = {"open": 0.72, "guarded": 0.50, "avoidant": 0.30}[effective_trait]
    sensitivity_penalty = {"low": 0.00, "medium": 0.10, "high": 0.22}[sensitivity]
    quality_boost = {"poor": -0.16, "neutral": 0.00, "supportive": 0.18}[quality]
    pressure_penalty = 0.08 * max(0, asked_before - 1)
    if prior_boundary_refusal:
        pressure_penalty += 0.20
    score = clamp(trait_score - sensitivity_penalty + quality_boost - pressure_penalty, 0.02, 0.95)

    informative = clamp((score - 0.55) * 1.10, 0.00, 0.55)
    partial = clamp(0.28 + score * 0.25, 0.16, 0.52)
    vague = clamp(0.42 - score * 0.25, 0.10, 0.42)
    refusal = clamp((0.45 - score) * 0.65, 0.02, 0.45)
    deflection = clamp((0.38 - score) * 0.45, 0.02, 0.30)

    if sensitivity == "high":
        refusal += 0.10
    if quality == "supportive":
        refusal *= 0.65
        deflection *= 0.75
        partial += 0.10
    if quality == "poor":
        refusal += 0.08
        deflection += 0.06
        informative *= 0.55
    if prior_boundary_refusal:
        refusal += 0.18
        informative *= 0.40
        partial *= 0.75

    return {
        "informative_response": informative,
        "partial_disclosure": partial,
        "vague_uncertain": vague,
        "boundary_refusal": refusal,
        "topic_deflection": deflection,
    }


def random_disclosure_distribution(base_distribution: dict[str, float], low_disclosure_prob: float) -> dict[str, float]:
    if set(base_distribution) == {"no_profile_evidence"}:
        return dict(base_distribution)
    low_prob = clamp(float(low_disclosure_prob), 0.0, 1.0)
    low_keys = ["partial_disclosure", "vague_uncertain", "boundary_refusal", "topic_deflection"]
    low_total = sum(max(0.0, float(base_distribution.get(key, 0.0))) for key in low_keys)
    if low_total <= 0:
        low_weights = {"partial_disclosure": 0.45, "vague_uncertain": 0.45, "boundary_refusal": 0.05, "topic_deflection": 0.05}
    else:
        low_weights = {key: max(0.0, float(base_distribution.get(key, 0.0))) / low_total for key in low_keys}
    mixed = {"informative_response": 1.0 - low_prob}
    for key in low_keys:
        mixed[key] = low_prob * low_weights[key]
    return mixed


def budget_from_response_type(
    *,
    response_type: str,
    total_units: int,
    asked_before: int,
    quality: str,
    sensitivity: str,
) -> dict[str, Any]:
    if total_units <= 0 or response_type == "no_profile_evidence":
        return {
            "retain_count": 0,
            "weaken_count": 0,
            "topic": 0.0,
            "clarity": 0.0,
            "category": "no_profile_evidence",
        }

    if response_type == "informative_response":
        retain = max(1, math.ceil(total_units * (0.80 if asked_before == 0 else 0.45)))
        return {
            "retain_count": min(total_units, retain),
            "weaken_count": 0,
            "topic": 1.0,
            "clarity": 1.0,
            "category": "informative_reference",
        }

    if response_type == "partial_disclosure":
        base = 0.30 if quality == "supportive" else 0.20
        if sensitivity == "high":
            base -= 0.06
        retain = max(1, min(3, math.ceil(total_units * max(0.12, base))))
        weaken = 1 if total_units > 1 else 0
        return {
            "retain_count": retain,
            "weaken_count": weaken,
            "topic": 0.85,
            "clarity": 0.70 if quality == "supportive" else 0.55,
            "category": "targeted_recovery_partial" if asked_before > 0 else "partial_omission",
        }

    if response_type == "vague_uncertain":
        return {
            "retain_count": 0,
            "weaken_count": 0,
            "topic": 0.60,
            "clarity": 0.35,
            "category": "vague_or_uncertain",
        }

    if response_type == "topic_deflection":
        return {
            "retain_count": 0,
            "weaken_count": 0,
            "topic": 0.25,
            "clarity": 0.10,
            "category": "topic_deflection",
        }

    return {
        "retain_count": 0,
        "weaken_count": 0,
        "topic": 0.0,
        "clarity": 0.0,
        "category": "direct_refusal_or_boundary",
    }


class DynamicPatientControllerV2(DynamicPatientControllerV1):
    """Cause-aware low-information controller.

    This is still a controlled stress environment, not a clinical patient simulator.
    The main difference from V1 is that disclosure is mediated by trait,
    slot sensitivity, and observable doctor recovery quality.
    """

    @staticmethod
    def initial_state() -> dict[str, Any]:
        state = DynamicPatientControllerV1.initial_state()
        state["prior_boundary_refusal_by_slot"] = {}
        return state

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
        profile_id = str(profile.get("profile_id") or "")
        base_trait = trait_for_profile(profile_id)
        effective_trait = shift_trait_for_severity(base_trait, severity)
        sensitivity = slot_sensitivity(target_slot)
        quality = doctor_recovery_quality(
            doctor_question=doctor_question,
            asked_before=asked_before,
            is_targeted_followup=is_targeted_followup,
            is_generic_clarification=is_generic_clarification,
        )
        prior_refusal = bool((state.get("prior_boundary_refusal_by_slot") or {}).get(target_slot))

        if severity in {"reference_informative", "fully_cooperative"}:
            response_type = "informative_response"
            distribution = {"informative_response": 1.0}
        else:
            distribution = response_distribution(
                effective_trait=effective_trait,
                sensitivity=sensitivity,
                quality=quality,
                asked_before=asked_before,
                prior_boundary_refusal=prior_refusal,
                has_new_units=has_new_units,
            )
            if severity == "random_disclosure":
                distribution = random_disclosure_distribution(distribution, self.random_low_disclosure_prob)
            response_type = stable_choice(
                sorted(distribution.items()),
                profile_id,
                severity,
                target_slot,
                asked_before,
                doctor_question,
                self.random_low_disclosure_prob if severity == "random_disclosure" else "",
                "response_type",
            )

        if severity in {"fully_cooperative", "random_disclosure"} and response_type == "informative_response":
            budget = self._fully_cooperative_budget(total_units)
        else:
            budget = budget_from_response_type(
                response_type=response_type,
                total_units=total_units,
                asked_before=asked_before,
                quality=quality,
                sensitivity=sensitivity,
            )
        budget.update(
            {
                "low_info_cause": response_type,
                "response_type": response_type,
                "patient_disclosure_trait": base_trait,
                "effective_disclosure_trait": effective_trait,
                "slot_sensitivity": sensitivity,
                "doctor_recovery_quality": quality,
                "prior_boundary_refusal": prior_refusal,
                "disclosure_mode": severity,
                "random_low_disclosure_prob": self.random_low_disclosure_prob if severity == "random_disclosure" else None,
                "random_low_disclosure_triggered": (response_type not in {"informative_response", "no_profile_evidence"})
                if severity == "random_disclosure"
                else None,
                "response_type_distribution": {
                    key: round(float(value), 6) for key, value in distribution.items()
                },
            }
        )
        return budget

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
        state.setdefault("prior_boundary_refusal_by_slot", {})

        severity = normalize_severity(base_severity)
        target_slot, interpreter_output, routing_source = self._route_target(
            profile,
            state,
            doctor_question,
            dialogue_history,
        )

        if not target_slot:
            response, state = super().step(
                profile_id=profile_id,
                doctor_question=doctor_question,
                base_severity=base_severity,
                state=state,
                dialogue_history=dialogue_history,
            )
            response["controller_version"] = "dynamic_profile_grounded_controller_v2"
            response["low_info_cause"] = "unmapped_question"
            response["response_type"] = "unmapped_question"
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

        budget = self._budget_v2(
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

        information_retention = compute_information_retention(retained_ids, weakened_ids, total_units)
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
        if budget["response_type"] == "boundary_refusal":
            state["prior_boundary_refusal_by_slot"][target_slot] = True
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
            "low_info_cause": budget["low_info_cause"],
            "response_type": budget["response_type"],
            "patient_disclosure_trait": budget["patient_disclosure_trait"],
            "effective_disclosure_trait": budget["effective_disclosure_trait"],
            "slot_sensitivity": budget["slot_sensitivity"],
            "doctor_recovery_quality": budget["doctor_recovery_quality"],
            "prior_boundary_refusal": budget["prior_boundary_refusal"],
            "response_type_distribution": budget["response_type_distribution"],
            "response_type_distribution_v2_base": budget.get("response_type_distribution_v2_base"),
            "disclosure_mode": budget.get("disclosure_mode", severity),
            "random_low_disclosure_prob": budget.get("random_low_disclosure_prob"),
            "random_low_disclosure_triggered": budget.get("random_low_disclosure_triggered"),
            "controller_version": "dynamic_profile_grounded_controller_v2",
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
                "theory_constrained_stress_environment": True,
            },
            "realizer": {
                "type": "deterministic_rule_based",
                "model": "none",
                "version": "dynamic_profile_grounded_controller_v2",
            },
        }
        return response, state


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    return {record["profile_id"]: record for record in iter_jsonl(path)}


def build_pilot_records(
    controller: DynamicPatientControllerV2,
    groups: list[dict[str, Any]],
    severities: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in groups:
        profile_id = group["profile_id"]
        target = group["target_tree_node"]
        questions = [
            ("initial_question", make_initial_question(target)),
            ("generic_clarification", "你能再说说刚才这个情况吗？"),
            ("targeted_followup", make_targeted_followup_question(target)),
            ("second_targeted_followup", make_second_targeted_followup_question(target)),
        ]
        for severity in severities:
            state = controller.initial_state()
            history: list[dict[str, str]] = []
            scenario_id = f"{group['counterfactual_group_id']}::{severity}::controller_v2_pilot"
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
    by_response_type = Counter(record.get("response_type") for record in records)
    by_cause_severity = Counter((record.get("base_severity"), record.get("response_type")) for record in records)
    by_trait = Counter(record.get("patient_disclosure_trait") for record in records)
    by_effective_trait = Counter(record.get("effective_disclosure_trait") for record in records)
    by_sensitivity = Counter(record.get("slot_sensitivity") for record in records)
    by_quality = Counter(record.get("doctor_recovery_quality") for record in records)
    unlock_by_quality: dict[str, list[float]] = defaultdict(list)
    unlock_by_response_type: dict[str, list[float]] = defaultdict(list)
    for record in records:
        delta = float(record.get("delta_cumulative_slot_sufficiency") or 0.0)
        unlocked = 1.0 if delta > 0.05 else 0.0
        unlock_by_quality[str(record.get("doctor_recovery_quality"))].append(unlocked)
        unlock_by_response_type[str(record.get("response_type"))].append(unlocked)

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "num_records": len(records),
        "num_scenarios": len({record["scenario_id"] for record in records}),
        "num_profiles": len({record["profile_id"] for record in records}),
        "response_type_counts": dict(by_response_type),
        "response_type_by_severity": {
            f"{severity}::{rtype}": count for (severity, rtype), count in sorted(by_cause_severity.items())
        },
        "patient_trait_counts": dict(by_trait),
        "effective_trait_counts": dict(by_effective_trait),
        "slot_sensitivity_counts": dict(by_sensitivity),
        "doctor_recovery_quality_counts": dict(by_quality),
        "unlock_rate_by_quality": {key: avg(values) for key, values in sorted(unlock_by_quality.items())},
        "unlock_rate_by_response_type": {
            key: avg(values) for key, values in sorted(unlock_by_response_type.items())
        },
        "targeted_unlock_rate": avg(
            [
                1.0 if float(record.get("delta_cumulative_slot_sufficiency") or 0.0) > 0.05 else 0.0
                for record in records
                if record.get("is_targeted_followup")
            ]
        ),
        "boundary_refusal_after_targeted_count": sum(
            1
            for record in records
            if record.get("is_targeted_followup") and record.get("response_type") == "boundary_refusal"
        ),
    }


def write_report(path: Path, summary: dict[str, Any], pilot_path: Path) -> None:
    lines = [
        "# Dynamic Patient Controller V2 Audit",
        "",
        "## Purpose",
        "",
        "Controller V2 is a theory-constrained low-information stress environment, not a clinically realistic patient simulator.",
        "It changes V1 from severity-driven disclosure to cause-aware disclosure controlled by patient trait, slot sensitivity, and doctor recovery quality.",
        "",
        "## Literature-grounded design constraints",
        "",
        "- Psychiatric evaluation depends on patient cooperation, communication, recall, and current mental state; interview information quality can vary.",
        "- Patient-centered interviewing supports a mix of open-ended exploration and focused clarification.",
        "- Motivational interviewing emphasizes open questions, reflective/supportive responses, and patient-centered communication.",
        "- Trauma-informed care emphasizes safety, trust, collaboration, empowerment, and choice, especially around sensitive topics.",
        "",
        "## Output",
        "",
        f"- Pilot records: `{pilot_path.name}`",
        f"- Records: {summary['num_records']}",
        f"- Scenarios: {summary['num_scenarios']}",
        f"- Profiles: {summary['num_profiles']}",
        "",
        "## Response Types",
        "",
        "```json",
        json.dumps(summary.get("response_type_counts"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Unlock Rate By Doctor Recovery Quality",
        "",
        "```json",
        json.dumps(summary.get("unlock_rate_by_quality"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Unlock Rate By Response Type",
        "",
        "```json",
        json.dumps(summary.get("unlock_rate_by_response_type"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Boundary Check",
        "",
        f"- Targeted-followup unlock rate: {summary['targeted_unlock_rate']:.6f}",
        f"- Boundary refusals after targeted follow-up: {summary['boundary_refusal_after_targeted_count']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit dynamic patient controller V2.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V2)
    parser.add_argument("--splits", nargs="+", default=["test"])
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
    groups = select_pilot_groups(
        load_group_records(args.group_dir, args.splits),
        max_groups=args.max_groups,
        max_per_slot=args.max_per_slot,
    )
    controller = DynamicPatientControllerV2(
        schema=schema,
        profiles=profiles,
        max_units_per_slot=args.max_units_per_slot,
        random_low_disclosure_prob=args.random_low_disclosure_prob,
        random_disclosure_seed=args.random_disclosure_seed,
    )
    severities = [normalize_severity(level) for level in args.severities]
    records = build_pilot_records(controller, groups, severities)
    summary = summarize(records)
    records_path = args.output_dir / "mdd5k_dynamic_patient_controller_v2_pilot_records.jsonl"
    summary_path = args.output_dir / "mdd5k_dynamic_patient_controller_v2_audit_summary.json"
    report_path = args.output_dir / "DYNAMIC_PATIENT_CONTROLLER_V2_AUDIT.md"
    write_jsonl(records_path, records)
    write_json(summary_path, summary)
    write_report(report_path, summary, records_path)
    print(json.dumps({"summary": summary, "records_path": str(records_path), "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
