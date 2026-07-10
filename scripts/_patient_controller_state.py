from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _patient_controller_base import (
    DEFAULT_GROUP_DIR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_SCHEMA_PATH,
    SEVERITIES,
    iter_jsonl,
    make_initial_question,
    make_second_targeted_followup_question,
    make_targeted_followup_question,
    normalize_severity,
    select_pilot_groups,
    write_json,
    write_jsonl,
)
from _patient_controller_disclosure import (
    DynamicPatientControllerV2,
    budget_from_response_type,
    doctor_recovery_quality,
    load_group_records,
    load_profiles,
    response_distribution,
    slot_sensitivity,
    stable_choice,
    trait_for_profile,
)
from online_query_interpreter import load_json


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR_V3 = BASE_DIR / "outputs_dynamic_patient_controller_v3"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def initial_disclosure_readiness(*, trait: str, severity: str) -> float:
    base = {"open": 0.72, "guarded": 0.50, "avoidant": 0.32}.get(trait, 0.50)
    if severity == "mild_low_info":
        base += 0.05
    elif severity == "severe_low_info":
        base -= 0.08
    return round(clamp(base, 0.12, 0.88), 4)


def readiness_adjusted_distribution(distribution: dict[str, float], readiness: float) -> dict[str, float]:
    """Shift V2 response probabilities with a cumulative disclosure-readiness state."""
    if set(distribution) == {"no_profile_evidence"}:
        return distribution
    centered = clamp(readiness, 0.05, 0.95) - 0.50
    adjusted = dict(distribution)
    positive_factor = 1.0 + centered * 0.70
    low_info_factor = 1.0 - centered * 0.65
    refusal_factor = 1.0 - centered * 0.95
    adjusted["informative_response"] = max(0.0, adjusted.get("informative_response", 0.0) * positive_factor)
    adjusted["partial_disclosure"] = max(0.0, adjusted.get("partial_disclosure", 0.0) * positive_factor)
    adjusted["vague_uncertain"] = max(0.0, adjusted.get("vague_uncertain", 0.0) * low_info_factor)
    adjusted["topic_deflection"] = max(0.0, adjusted.get("topic_deflection", 0.0) * low_info_factor)
    adjusted["boundary_refusal"] = max(0.0, adjusted.get("boundary_refusal", 0.0) * refusal_factor)
    return adjusted


def update_disclosure_readiness(
    *,
    previous_readiness: float,
    response_type: str,
    doctor_quality: str,
    prior_boundary_refusal: bool,
    asked_before: int,
    is_targeted_followup: bool,
    is_generic_clarification: bool,
) -> tuple[float, dict[str, float]]:
    quality_delta = {"supportive": 0.055, "neutral": 0.0, "poor": -0.055}.get(doctor_quality, 0.0)
    response_delta = {
        "informative_response": 0.030,
        "partial_disclosure": 0.020,
        "vague_uncertain": -0.020,
        "topic_deflection": -0.040,
        "boundary_refusal": -0.085,
        "no_profile_evidence": -0.010,
        "unmapped_question": -0.040,
    }.get(response_type, 0.0)
    repetition_delta = 0.0
    if asked_before >= 2:
        repetition_delta -= 0.025 * min(3, asked_before - 1)
    if is_generic_clarification:
        repetition_delta -= 0.025
    if prior_boundary_refusal and is_targeted_followup:
        repetition_delta -= 0.090
    boundary_repair_delta = 0.0
    if prior_boundary_refusal and doctor_quality == "supportive" and not is_targeted_followup:
        boundary_repair_delta += 0.035
    total_delta = quality_delta + response_delta + repetition_delta + boundary_repair_delta
    updated = round(clamp(previous_readiness + total_delta, 0.05, 0.95), 4)
    return updated, {
        "quality_delta": round(quality_delta, 4),
        "response_delta": round(response_delta, 4),
        "repetition_delta": round(repetition_delta, 4),
        "boundary_repair_delta": round(boundary_repair_delta, 4),
        "total_delta": round(total_delta, 4),
    }


class DynamicPatientControllerV3(DynamicPatientControllerV2):
    """V2 plus cumulative patient disclosure readiness.

    The readiness state models whether previous interaction quality makes future
    disclosure easier or harder. It is simulator metadata only and is not visible
    to the doctor policy.
    """

    @staticmethod
    def initial_state() -> dict[str, Any]:
        state = DynamicPatientControllerV2.initial_state()
        state["disclosure_readiness"] = None
        state["disclosure_readiness_by_slot"] = {}
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
        sensitivity = slot_sensitivity(target_slot)
        quality = doctor_recovery_quality(
            doctor_question=doctor_question,
            asked_before=asked_before,
            is_targeted_followup=is_targeted_followup,
            is_generic_clarification=is_generic_clarification,
        )
        prior_refusal = bool((state.get("prior_boundary_refusal_by_slot") or {}).get(target_slot))
        readiness = state.get("disclosure_readiness")
        if readiness is None:
            readiness = initial_disclosure_readiness(trait=base_trait, severity=severity)
            state["disclosure_readiness"] = readiness
        readiness = float(readiness)

        if severity == "reference_informative":
            response_type = "informative_response"
            distribution = {"informative_response": 1.0}
            base_distribution = dict(distribution)
        else:
            effective_trait = self._effective_trait_from_readiness(base_trait, readiness)
            base_distribution = response_distribution(
                effective_trait=effective_trait,
                sensitivity=sensitivity,
                quality=quality,
                asked_before=asked_before,
                prior_boundary_refusal=prior_refusal,
                has_new_units=has_new_units,
            )
            distribution = readiness_adjusted_distribution(base_distribution, readiness)
            response_type = stable_choice(
                sorted(distribution.items()),
                profile_id,
                severity,
                target_slot,
                asked_before,
                doctor_question,
                readiness,
                "response_type_v3",
            )

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
                "effective_disclosure_trait": self._effective_trait_from_readiness(base_trait, readiness),
                "slot_sensitivity": sensitivity,
                "doctor_recovery_quality": quality,
                "prior_boundary_refusal": prior_refusal,
                "disclosure_readiness_before": round(readiness, 4),
                "response_type_distribution_v2_base": {
                    key: round(float(value), 6) for key, value in base_distribution.items()
                },
                "response_type_distribution": {
                    key: round(float(value), 6) for key, value in distribution.items()
                },
            }
        )
        return budget

    @staticmethod
    def _effective_trait_from_readiness(base_trait: str, readiness: float) -> str:
        if readiness >= 0.68:
            return "open"
        if readiness <= 0.35:
            return "avoidant"
        if base_trait == "open" and readiness < 0.48:
            return "guarded"
        if base_trait == "avoidant" and readiness > 0.58:
            return "guarded"
        return "guarded" if base_trait not in {"open", "avoidant"} else base_trait

    def step(
        self,
        *,
        profile_id: str,
        doctor_question: str,
        base_severity: str,
        state: dict[str, Any] | None = None,
        dialogue_history: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = dict(state or self.initial_state())
        profile = self.profiles.get(profile_id)
        severity = normalize_severity(base_severity)
        if profile is not None and state.get("disclosure_readiness") is None:
            trait = trait_for_profile(str(profile.get("profile_id") or profile_id))
            state["disclosure_readiness"] = initial_disclosure_readiness(trait=trait, severity=severity)

        previous_readiness = float(state.get("disclosure_readiness") or 0.50)
        response, state = super().step(
            profile_id=profile_id,
            doctor_question=doctor_question,
            base_severity=base_severity,
            state=state,
            dialogue_history=dialogue_history,
        )
        updated_readiness, readiness_delta = update_disclosure_readiness(
            previous_readiness=previous_readiness,
            response_type=str(response.get("response_type") or ""),
            doctor_quality=str(response.get("doctor_recovery_quality") or "neutral"),
            prior_boundary_refusal=bool(response.get("prior_boundary_refusal")),
            asked_before=int(response.get("asked_count_for_slot_before") or 0),
            is_targeted_followup=bool(response.get("is_targeted_followup")),
            is_generic_clarification=bool(response.get("is_generic_clarification")),
        )
        state["disclosure_readiness"] = updated_readiness
        target_slot = response.get("target_tree_node")
        if target_slot:
            readiness_by_slot = dict(state.get("disclosure_readiness_by_slot") or {})
            readiness_by_slot[str(target_slot)] = updated_readiness
            state["disclosure_readiness_by_slot"] = readiness_by_slot

        response["controller_version"] = "dynamic_profile_grounded_controller_v3"
        response["disclosure_readiness_before"] = round(previous_readiness, 4)
        response["disclosure_readiness_after"] = updated_readiness
        response["delta_disclosure_readiness"] = round(updated_readiness - previous_readiness, 4)
        response["disclosure_readiness_delta_components"] = readiness_delta
        response.setdefault("validity", {})["cumulative_disclosure_readiness"] = True
        response.setdefault("realizer", {})["version"] = "dynamic_profile_grounded_controller_v3"
        return response, state


def build_pilot_records(
    controller: DynamicPatientControllerV3,
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
            scenario_id = f"{group['counterfactual_group_id']}::{severity}::controller_v3_pilot"
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
    by_readiness_bucket = Counter(readiness_bucket(record.get("disclosure_readiness_before")) for record in records)
    by_severity = Counter(record.get("base_severity") for record in records)
    deltas_by_quality: dict[str, list[float]] = defaultdict(list)
    deltas_by_response: dict[str, list[float]] = defaultdict(list)
    unlock_by_readiness: dict[str, list[float]] = defaultdict(list)
    for record in records:
        delta_r = float(record.get("delta_disclosure_readiness") or 0.0)
        deltas_by_quality[str(record.get("doctor_recovery_quality"))].append(delta_r)
        deltas_by_response[str(record.get("response_type"))].append(delta_r)
        bucket = readiness_bucket(record.get("disclosure_readiness_before"))
        unlocked = 1.0 if float(record.get("delta_cumulative_slot_sufficiency") or 0.0) > 0.05 else 0.0
        unlock_by_readiness[bucket].append(unlocked)

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "num_records": len(records),
        "num_scenarios": len({record["scenario_id"] for record in records}),
        "num_profiles": len({record["profile_id"] for record in records}),
        "response_type_counts": dict(by_response_type),
        "base_severity_counts": dict(by_severity),
        "readiness_bucket_counts": dict(by_readiness_bucket),
        "mean_delta_readiness_by_quality": {key: avg(value) for key, value in sorted(deltas_by_quality.items())},
        "mean_delta_readiness_by_response_type": {key: avg(value) for key, value in sorted(deltas_by_response.items())},
        "unlock_rate_by_readiness_bucket": {key: avg(value) for key, value in sorted(unlock_by_readiness.items())},
    }


def readiness_bucket(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if score < 0.35:
        return "low"
    if score < 0.60:
        return "mid"
    return "high"


def write_report(path: Path, summary: dict[str, Any], pilot_path: Path) -> None:
    lines = [
        "# Dynamic Patient Controller V3 Audit",
        "",
        "## Purpose",
        "",
        "Controller V3 extends V2 with cumulative Patient Disclosure Readiness, a hidden simulator state that changes future disclosure probabilities.",
        "It is intended to support trajectory/value-model experiments where early supportive questions can have delayed benefit.",
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
        "## Readiness Dynamics",
        "",
        "```json",
        json.dumps(
            {
                "readiness_bucket_counts": summary.get("readiness_bucket_counts"),
                "mean_delta_readiness_by_quality": summary.get("mean_delta_readiness_by_quality"),
                "mean_delta_readiness_by_response_type": summary.get("mean_delta_readiness_by_response_type"),
                "unlock_rate_by_readiness_bucket": summary.get("unlock_rate_by_readiness_bucket"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit dynamic patient controller V3.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V3)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--max-groups", type=int, default=90)
    parser.add_argument("--max-per-slot", type=int, default=5)
    parser.add_argument("--max-units-per-slot", type=int, default=8)
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
    controller = DynamicPatientControllerV3(schema=schema, profiles=profiles, max_units_per_slot=args.max_units_per_slot)
    severities = [normalize_severity(level) for level in args.severities]
    records = build_pilot_records(controller, groups, severities)
    summary = summarize(records)
    records_path = args.output_dir / "mdd5k_dynamic_patient_controller_v3_pilot_records.jsonl"
    summary_path = args.output_dir / "mdd5k_dynamic_patient_controller_v3_audit_summary.json"
    report_path = args.output_dir / "DYNAMIC_PATIENT_CONTROLLER_V3_AUDIT.md"
    write_jsonl(records_path, records)
    write_json(summary_path, summary)
    write_report(report_path, summary, records_path)
    print(json.dumps({"summary": summary, "records_path": str(records_path), "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
