from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from _patient_controller_base import (
    DEFAULT_GROUP_DIR,
    DEFAULT_PROFILE_PATH,
    DEFAULT_SCHEMA_PATH,
    DynamicPatientControllerV1,
    load_group_records,
    load_profiles,
    make_initial_question,
    make_second_targeted_followup_question,
    make_targeted_followup_question,
    normalize_severity,
    select_pilot_groups,
)
from _patient_controller_disclosure import DynamicPatientControllerV2
from _patient_controller_state import DynamicPatientControllerV3
from _patient_controller_policy import DynamicPatientControllerV31
from patient_controller import DynamicPatientControllerV32
from online_query_interpreter import load_json
from _doctor_request_prompts import POLICY_PROMPTS, build_messages, select_profiles_from_groups
from _doctor_policy_baselines import (
    GLOBAL_CORE_SEQUENCE,
    POLICY_VISIBILITY as DOCTOR_POLICY_VISIBILITY,
    evaluable_slots,
    schema_slot_maps,
    summarize,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_llm_doctor_online_replay"

DOCTOR_POLICY_VISIBILITY.update(
    {policy_name: policy["visibility"] for policy_name, policy in POLICY_PROMPTS.items()}
)

LOW_INFO_CUES = [
    "说不清",
    "说不出口",
    "不太想",
    "不知道",
    "不清楚",
    "跳过",
    "差不多",
    "暂时",
    "不想展开",
    "有点乱",
    "不好说",
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def clean_doctor_question(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^[-*\d\.\)、\s]+", "", text)
    text = text.splitlines()[0].strip() if text else ""
    if not text:
        return "你最近最困扰你的情况是什么？"
    if not text.endswith(("?", "？")):
        text = text.rstrip("。.!！") + "？"
    return text


def load_cached_outputs(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    outputs: dict[str, str] = {}
    for record in iter_jsonl(path):
        request_id = record.get("request_id")
        content = record.get("doctor_question") or record.get("output") or record.get("content")
        if not request_id or not content:
            continue
        outputs[str(request_id)] = clean_doctor_question(str(content))
    return outputs


def load_patient_realizer_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        source_record_id = record.get("source_record_id")
        response = record.get("patient_response")
        if not source_record_id or not response:
            continue
        cache[str(source_record_id)] = record
    return cache


def apply_patient_realizer(
    *,
    response: dict[str, Any],
    record_id: str,
    mode: str,
    cache: dict[str, dict[str, Any]],
    cache_policy: str,
) -> dict[str, Any]:
    if mode == "rule":
        response["patient_realizer_mode"] = "rule"
        response["patient_realizer_cache_hit"] = False
        return response

    cached = cache.get(record_id)
    if not cached:
        if cache_policy == "error":
            raise KeyError(f"Missing verified patient realizer cache for record_id={record_id}")
        response["patient_realizer_mode"] = "rule_fallback_missing_verified_cache"
        response["patient_realizer_cache_hit"] = False
        response.setdefault("realizer", {})["fallback_reason"] = "missing_verified_cache"
        return response

    realized = dict(response)
    realized["rule_based_patient_response"] = response.get("patient_response")
    realized["patient_response"] = cached.get("patient_response")
    realized["patient_realizer_mode"] = "verified_llm_cache"
    realized["patient_realizer_cache_hit"] = True
    realized["patient_response_realizer"] = {
        "type": "verified_llm_cache",
        "request_id": cached.get("request_id"),
        "provider": cached.get("provider"),
        "model": cached.get("model"),
        "prompt_protocol_version": cached.get("prompt_protocol_version"),
        "history_mode": cached.get("history_mode"),
        "low_info_category": cached.get("low_info_category"),
        "mean_allowed_coverage": cached.get("mean_allowed_coverage"),
        "warnings": cached.get("warnings") or [],
    }
    realized.setdefault("validity", {})["verified_llm_realizer_cache"] = True
    realized.setdefault("realizer", {}).update(
        {
            "type": "verified_llm_cache",
            "model": cached.get("model"),
            "provider": cached.get("provider"),
            "version": cached.get("prompt_protocol_version") or "llm_patient_realizer_v3_1",
        }
    )
    return realized


def slot_keywords(schema: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for slot in schema.get("slots", []):
        result[slot["slot"]] = list(slot.get("question_keywords") or [])
    return result


def infer_slot_from_visible_question(question: str, keywords_by_slot: dict[str, list[str]]) -> str | None:
    question = question or ""
    best_slot = None
    best_score = 0
    for slot, keywords in keywords_by_slot.items():
        score = sum(1 for keyword in keywords if keyword and keyword in question)
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot if best_score > 0 else None


def low_info_from_text(response: str) -> bool:
    return any(cue in (response or "") for cue in LOW_INFO_CUES)


def next_global_slot(turn_index: int) -> str:
    return GLOBAL_CORE_SEQUENCE[min(turn_index, len(GLOBAL_CORE_SEQUENCE) - 1)]


def scripted_question(
    *,
    policy_name: str,
    turn_index: int,
    history: list[dict[str, str]],
    keywords_by_slot: dict[str, list[str]],
) -> str:
    if turn_index == 0:
        if policy_name == "closed_llm_evidence_aware":
            return "最近最困扰你的情绪、睡眠或学习工作影响是什么？"
        return "你最近最困扰你的情况是什么？"

    if policy_name == "closed_llm_evidence_aware" and history:
        last_turn = history[-1]
        previous_slot = infer_slot_from_visible_question(last_turn.get("doctor_utterance", ""), keywords_by_slot)
        if previous_slot and low_info_from_text(last_turn.get("patient_utterance", "")):
            if turn_index % 2 == 1:
                return make_targeted_followup_question(previous_slot)
            return make_second_targeted_followup_question(previous_slot)

    slot_offset = turn_index if policy_name == "closed_llm_general" else max(0, turn_index - 1)
    return make_initial_question(next_global_slot(slot_offset))


def build_request_record(
    *,
    profile: dict[str, Any],
    severity: str,
    policy_name: str,
    turn_index: int,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    request_id = f"{profile['profile_id']}::{severity}::{policy_name}::turn_{turn_index}"
    history_snapshot = [dict(turn) for turn in history]
    return {
        "request_id": request_id,
        "policy_name": policy_name,
        "policy_visibility": POLICY_PROMPTS[policy_name]["visibility"],
        "profile_id": profile["profile_id"],
        "case_id": profile.get("case_id"),
        "base_severity": severity,
        "turn_index": turn_index,
        "dialogue_history": history_snapshot,
        "messages": build_messages(policy_name, history_snapshot),
        "expected_output": {"doctor_question": "one natural-language Chinese question"},
        "doctor_visible_fields": ["dialogue_history"],
        "hidden_eval_metadata_not_for_model": [
            "profile_id",
            "case_id",
            "base_severity",
            "diagnoses",
            "icd_codes",
            "primary_tree_type",
            "active_tree_slots",
            "slot_profiles",
            "simulator_internal_target_node",
            "controller_metadata",
        ],
    }


def choose_doctor_question(
    *,
    provider: str,
    request: dict[str, Any],
    cached_outputs: dict[str, str],
    missing_output_policy: str,
    history: list[dict[str, str]],
    keywords_by_slot: dict[str, list[str]],
) -> tuple[str | None, str]:
    request_id = request["request_id"]
    if provider == "cached":
        if request_id in cached_outputs:
            return cached_outputs[request_id], "cached"
        if missing_output_policy == "error":
            raise KeyError(f"Missing cached doctor output for request_id={request_id}")
        if missing_output_policy == "stop":
            return None, "missing_stop"

    question = scripted_question(
        policy_name=request["policy_name"],
        turn_index=int(request["turn_index"]),
        history=history,
        keywords_by_slot=keywords_by_slot,
    )
    source = "scripted_smoke" if provider == "scripted" else "scripted_fallback"
    return clean_doctor_question(question), source


def run_online_replay(
    *,
    controller: DynamicPatientControllerV1 | DynamicPatientControllerV2 | DynamicPatientControllerV3 | DynamicPatientControllerV31 | DynamicPatientControllerV32,
    profiles: list[dict[str, Any]],
    severities: list[str],
    policies: list[str],
    max_turns: int,
    provider: str,
    cached_outputs: dict[str, str],
    missing_output_policy: str,
    keywords_by_slot: dict[str, list[str]],
    patient_realizer_mode: str = "rule",
    patient_realizer_cache: dict[str, dict[str, Any]] | None = None,
    patient_realizer_cache_policy: str = "fallback",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    turn_records: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []
    pending_requests: list[dict[str, Any]] = []
    patient_realizer_cache = patient_realizer_cache or {}

    for profile in profiles:
        profile_id = profile["profile_id"]
        for severity in severities:
            for policy_name in policies:
                state = controller.initial_state()
                history: list[dict[str, str]] = []
                scenario_id = f"{profile_id}::{severity}::{policy_name}::{provider}"
                for turn_index in range(max_turns):
                    request = build_request_record(
                        profile=profile,
                        severity=severity,
                        policy_name=policy_name,
                        turn_index=turn_index,
                        history=history,
                    )
                    request_records.append(request)
                    doctor_question, question_source = choose_doctor_question(
                        provider=provider,
                        request=request,
                        cached_outputs=cached_outputs,
                        missing_output_policy=missing_output_policy,
                        history=history,
                        keywords_by_slot=keywords_by_slot,
                    )
                    if doctor_question is None:
                        pending_requests.append(request)
                        break

                    response, state = controller.step(
                        profile_id=profile_id,
                        doctor_question=doctor_question,
                        base_severity=severity,
                        state=state,
                        dialogue_history=history,
                    )
                    record_id = f"{scenario_id}::turn_{turn_index}"
                    response = apply_patient_realizer(
                        response=response,
                        record_id=record_id,
                        mode=patient_realizer_mode,
                        cache=patient_realizer_cache,
                        cache_policy=patient_realizer_cache_policy,
                    )
                    history.append(
                        {
                            "doctor_utterance": doctor_question,
                            "patient_utterance": response["patient_response"],
                        }
                    )
                    turn_records.append(
                        {
                            "record_id": record_id,
                            "scenario_id": scenario_id,
                            "request_id": request["request_id"],
                            "profile_id": profile_id,
                            "case_id": profile.get("case_id"),
                            "diagnoses": profile.get("diagnoses"),
                            "icd_codes": profile.get("icd_codes"),
                            "policy_name": policy_name,
                            "policy_visibility": POLICY_PROMPTS[policy_name]["visibility"],
                            "llm_provider": provider,
                            "doctor_question_source": question_source,
                            "base_severity": severity,
                            "turn_index": turn_index,
                            "question_type": "llm_generated_question",
                            "doctor_question": doctor_question,
                            **response,
                        }
                    )
                    if response.get("patient_terminated") or state.get("patient_terminated"):
                        break
    return turn_records, request_records, pending_requests


def summarize_request_safety(requests: list[dict[str, Any]]) -> dict[str, Any]:
    banned_terms = [
        "target_tree_node",
        "simulator_internal_target_node",
        "active_tree_slots",
        "slot_profiles",
        "diagnoses",
        "icd_codes",
        "g_target",
        "controller_metadata",
    ]
    leakage_counts = {term: 0 for term in banned_terms}
    for request in requests:
        prompt_text = "\n".join(message.get("content", "") for message in request.get("messages", []))
        for term in banned_terms:
            if term in prompt_text:
                leakage_counts[term] += 1
    return {
        "num_requests": len(requests),
        "prompt_leakage_counts": leakage_counts,
        "has_prompt_leakage": any(value > 0 for value in leakage_counts.values()),
    }


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    records_path: Path,
    requests_path: Path,
    pending_path: Path,
) -> None:
    lines = [
        "# LLM Doctor Online Replay V1",
        "",
        "Date: 2026-06-11",
        "",
        "## Purpose",
        "",
        "This runner executes online doctor-patient replay where the doctor sees only natural-language dialogue history and outputs one natural-language question per turn.",
        "",
        "The patient environment internally maps each question to a simulator node for response generation and evaluation. Hidden nodes, labels, profile slots, and controller metadata are not copied into model-visible messages.",
        "",
        "## Outputs",
        "",
        f"- Replay records: `{records_path.name}`",
        f"- Model-visible requests: `{requests_path.name}`",
        f"- Pending requests: `{pending_path.name}`",
        f"- Total replay records: {summary['num_records']}",
        f"- Total scenarios: {summary['num_scenarios']}",
        f"- Profiles: {summary['num_profiles']}",
        f"- Prompt leakage detected: {summary['request_safety']['has_prompt_leakage']}",
        "",
        "## Main Summary",
        "",
        "| Policy | Severity | Scenarios | Slot hit rate | Final S | High/Crit S | Suicide S | Unmapped |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summary_rows"]:
        lines.append(
            "| `{policy}` | `{severity}` | {n} | {hit:.4f} | {final:.4f} | {hc:.4f} | {suicide:.4f} | {unmapped:.4f} |".format(
                policy=row["policy_name"],
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
            "- `provider=scripted` is only a local smoke test for the online replay loop, not a closed-LLM result.",
            "- `provider=cached` should be used for real LLM/API outputs; the input cache must map `request_id` to `doctor_question`.",
            "- Every turn request is saved so API calls can be audited and replayed.",
            "- The key paper-facing comparison should use cached real LLM outputs or a trained candidate policy, not the scripted smoke provider.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-style online doctor replay on Dynamic Patient Controller V1/V2/V3.")
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
    parser.add_argument("--patient-controller-version", choices=["v1", "v2", "v3", "v3_1", "v3_2"], default="v1")
    parser.add_argument("--provider", choices=["scripted", "cached"], default="scripted")
    parser.add_argument("--model-output-path", type=Path, default=None)
    parser.add_argument("--missing-output-policy", choices=["error", "stop", "scripted"], default="error")
    parser.add_argument("--patient-realizer-mode", choices=["rule", "verified_cache"], default="rule")
    parser.add_argument("--patient-realizer-cache-path", type=Path, default=None)
    parser.add_argument("--patient-realizer-cache-policy", choices=["fallback", "error"], default="fallback")
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["mild_low_info", "moderate_low_info", "severe_low_info"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["closed_llm_general", "closed_llm_evidence_aware"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_json(args.schema)
    _, criticality_ranks = schema_slot_maps(schema)
    keywords_by_slot = slot_keywords(schema)
    profiles_by_id = load_profiles(args.profiles)
    groups = select_pilot_groups(
        load_group_records(args.group_dir, args.splits),
        max_groups=args.max_groups,
        max_per_slot=args.max_per_slot,
    )
    profiles = select_profiles_from_groups(groups, profiles_by_id, args.max_profiles)
    severities = [normalize_severity(level) for level in args.severities]
    cached_outputs = load_cached_outputs(args.model_output_path)
    patient_realizer_cache = load_patient_realizer_cache(args.patient_realizer_cache_path)
    if args.patient_realizer_mode == "verified_cache" and not patient_realizer_cache and args.patient_realizer_cache_policy == "error":
        raise FileNotFoundError(f"No verified patient realizer cache loaded from {args.patient_realizer_cache_path}")
    controller_cls = {
        "v1": DynamicPatientControllerV1,
        "v2": DynamicPatientControllerV2,
        "v3": DynamicPatientControllerV3,
        "v3_1": DynamicPatientControllerV31,
        "v3_2": DynamicPatientControllerV32,
    }[args.patient_controller_version]
    controller = controller_cls(
        schema=schema,
        profiles=profiles_by_id,
        max_units_per_slot=args.max_units_per_slot,
    )

    records, requests, pending = run_online_replay(
        controller=controller,
        profiles=profiles,
        severities=severities,
        policies=args.policies,
        max_turns=args.max_turns,
        provider=args.provider,
        cached_outputs=cached_outputs,
        missing_output_policy=args.missing_output_policy,
        keywords_by_slot=keywords_by_slot,
        patient_realizer_mode=args.patient_realizer_mode,
        patient_realizer_cache=patient_realizer_cache,
        patient_realizer_cache_policy=args.patient_realizer_cache_policy,
    )
    summary = {
        "settings": {
            "splits": args.splits,
            "max_groups": args.max_groups,
            "max_per_slot": args.max_per_slot,
            "max_profiles": args.max_profiles,
            "max_turns": args.max_turns,
            "max_units_per_slot": args.max_units_per_slot,
            "patient_controller_version": args.patient_controller_version,
            "provider": args.provider,
            "model_output_path": str(args.model_output_path) if args.model_output_path else None,
            "missing_output_policy": args.missing_output_policy,
            "patient_realizer_mode": args.patient_realizer_mode,
            "patient_realizer_cache_path": str(args.patient_realizer_cache_path) if args.patient_realizer_cache_path else None,
            "patient_realizer_cache_records": len(patient_realizer_cache),
            "patient_realizer_cache_policy": args.patient_realizer_cache_policy,
            "severities": severities,
            "policies": args.policies,
        },
        **summarize(records, profiles_by_id, criticality_ranks),
        "request_safety": summarize_request_safety(requests),
        "num_pending_requests": len(pending),
        "evaluable_slot_count_mean": mean(
            [float(len(evaluable_slots(profile, criticality_ranks))) for profile in profiles]
        ),
    }

    records_path = args.output_dir / "mdd5k_llm_doctor_online_replay_records.jsonl"
    requests_path = args.output_dir / "mdd5k_llm_doctor_online_replay_requests.jsonl"
    pending_path = args.output_dir / "mdd5k_llm_doctor_online_replay_pending_requests.jsonl"
    summary_path = args.output_dir / "mdd5k_llm_doctor_online_replay_summary.json"
    report_path = args.output_dir / "LLM_DOCTOR_ONLINE_REPLAY_V1.md"

    write_jsonl(records_path, records)
    write_jsonl(requests_path, requests)
    write_jsonl(pending_path, pending)
    write_json(summary_path, summary)
    write_report(
        report_path,
        summary=summary,
        records_path=records_path,
        requests_path=requests_path,
        pending_path=pending_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
