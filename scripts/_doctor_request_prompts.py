from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _patient_controller_base import (
    DEFAULT_GROUP_DIR,
    DEFAULT_PROFILE_PATH,
    load_group_records,
    load_profiles,
    normalize_severity,
    select_pilot_groups,
)
try:
    from prepare_reward_centered_grpo_training_data import render_prompt as render_reward_training_prompt
except ModuleNotFoundError:
    render_reward_training_prompt = None

try:
    from build_boundary_aware_preference_pairs_v5_patient_v2 import render_prompt as render_boundary_aware_v5_prompt
except ModuleNotFoundError:
    render_boundary_aware_v5_prompt = None


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_llm_doctor_requests"

POLICY_PROMPTS = {
    "closed_llm_general": {
        "visibility": "doctor_visible_only",
        "system": (
            "你是一名进行精神心理初诊访谈的医生。"
            "你的任务是在多轮对话中主动收集与抑郁及相关风险有关的信息。"
            "每次只输出下一句要问患者的问题，不要做诊断，不要总结，不要解释。"
            "问题要自然、简洁、非评判。"
        ),
    },
    "closed_llm_evidence_aware": {
        "visibility": "doctor_visible_only",
        "system": (
            "你是一名进行精神心理初诊访谈的医生。"
            "你的任务是在多轮对话中主动收集与抑郁及相关风险有关的信息。"
            "如果患者上一轮回答含糊、回避、过短或信息不足，你应该针对刚才缺失的细节追问，"
            "例如频率、持续时间、严重程度、功能影响、风险意图或具体例子。"
            "如果已有回答较充分，则继续询问尚未覆盖的重要症状或风险。"
            "每次只输出下一句要问患者的问题，不要做诊断，不要总结，不要解释。"
            "问题要自然、简洁、非评判。"
        ),
    },
    "reward_trained_nobelief": {
        "visibility": "doctor_visible_only",
        "system": "",
    },
    "boundary_aware_v5_patient_v2": {
        "visibility": "doctor_visible_only",
        "system": "",
    },
    "reward_centered_v6_patient_v2": {
        "visibility": "doctor_visible_only",
        "system": "",
    },
}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def render_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "当前还没有对话历史。"
    lines = []
    for idx, turn in enumerate(history, start=1):
        lines.append(f"{idx}. 医生：{turn.get('doctor_utterance', '')}")
        lines.append(f"   患者：{turn.get('patient_utterance', '')}")
    return "\n".join(lines)


def render_history_en(history: list[dict[str, str]]) -> str:
    if not history:
        return "No prior dialogue yet."
    lines = []
    for idx, turn in enumerate(history, start=1):
        lines.append(f"{idx}. Doctor: {turn.get('doctor_utterance', '')}")
        lines.append(f"   Patient: {turn.get('patient_utterance', '')}")
    return "\n".join(lines)


def render_v6_history(history: list[dict[str, str]], max_turns: int = 8) -> str:
    if not history:
        return "（尚无对话历史）"
    lines = []
    for idx, turn in enumerate(history[-max_turns:], start=1):
        doctor = str(turn.get("doctor_utterance") or "").strip()
        patient = str(turn.get("patient_utterance") or "").strip()
        lines.append(f"{idx}. 医生：{doctor}")
        lines.append(f"   患者：{patient}")
    return "\n".join(lines)


def render_reward_centered_v6_prompt(history: list[dict[str, str]], max_turns: int = 8) -> str:
    history_text = render_v6_history(history, max_turns=max_turns)
    return (
        "你是一名精神心理主动问诊 doctor agent。\n"
        "任务：根据可见对话历史提出下一句自然、具体、安全的问诊问题，目标是在低信息回答下恢复临床证据。\n"
        "要求：\n"
        "1. 只输出一句医生下一问，不输出诊断。\n"
        "2. 如果患者刚才含糊、只说了一点、转移话题，应优先用低压力方式澄清同一证据维度。\n"
        "3. 如果患者明确表示不想谈，应先尊重边界，可做最小必要确认；连续拒绝后允许转到相邻、较不敏感的问题。\n"
        "4. 不要机械重复上一句，也不要把不充分回答当成已经收集到充分证据。\n\n"
        "当前可见对话历史：\n"
        f"{history_text}\n\n"
        "请输出下一句医生问题："
    )


def build_closed_llm_messages_en(policy_name: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    if policy_name == "closed_llm_evidence_aware":
        system = (
            "You are a mental-health intake doctor. Your task is to actively gather "
            "information relevant to depression screening across a multi-turn dialogue. "
            "If the patient's previous answer was vague, avoidant, too short, or insufficient, "
            "ask a focused follow-up about frequency, duration, severity, functional impact, "
            "or a concrete example. Ask exactly one natural, concise, non-judgmental question. "
            "Do not diagnose, summarize, explain, or mention hidden evaluation slots."
        )
    else:
        system = (
            "You are a mental-health intake doctor. Your task is to actively gather "
            "information relevant to depression screening across a multi-turn dialogue. "
            "Ask exactly one natural, concise, non-judgmental question. Do not diagnose, "
            "summarize, explain, or mention hidden evaluation slots."
        )
    user = (
        "Based only on the visible dialogue history below, generate the doctor's next question.\n\n"
        f"{render_history_en(history)}\n\n"
        "Output requirement: write exactly one English question, with no numbering, diagnosis, explanation, or extra text."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_messages(policy_name: str, history: list[dict[str, str]], language: str = "zh") -> list[dict[str, str]]:
    if policy_name == "reward_centered_v6_patient_v2":
        prompt = render_reward_centered_v6_prompt(history, max_turns=8)
        return [{"role": "user", "content": prompt}]
    if policy_name == "reward_trained_nobelief":
        if render_reward_training_prompt is None:
            raise ModuleNotFoundError(
                "prepare_reward_centered_grpo_training_data is required for reward_trained_nobelief."
            )
        prompt = render_reward_training_prompt(
            {"dialogue_history": history, "belief_before": {}},
            history_turns=8,
            include_belief=False,
        )
        return [{"role": "user", "content": prompt}]
    if policy_name == "boundary_aware_v5_patient_v2":
        boundary_history = [
            {
                "doctor_question": turn.get("doctor_utterance", ""),
                "patient_response": turn.get("patient_utterance", ""),
            }
            for turn in history
        ]
        if render_boundary_aware_v5_prompt is None:
            raise ModuleNotFoundError(
                "build_boundary_aware_preference_pairs_v5_patient_v2 is required for boundary_aware_v5_patient_v2."
            )
        prompt = render_boundary_aware_v5_prompt(boundary_history, max_history_turns=8)
        return [{"role": "user", "content": prompt}]

    if language == "en":
        return build_closed_llm_messages_en(policy_name, history)

    policy = POLICY_PROMPTS[policy_name]
    user = (
        "请根据以下对话历史，生成下一句医生问题。\n\n"
        f"{render_history(history)}\n\n"
        "输出要求：只输出一句中文问题，不要包含编号、解释、诊断或额外文本。"
    )
    return [
        {"role": "system", "content": policy["system"]},
        {"role": "user", "content": user},
    ]


def build_initial_requests(
    *,
    profiles: list[dict[str, Any]],
    severities: list[str],
    policies: list[str],
    language: str = "zh",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for profile in profiles:
        for severity in severities:
            for policy_name in policies:
                request_id = f"{profile['profile_id']}::{severity}::{policy_name}::turn_0"
                records.append(
                    {
                        "request_id": request_id,
                        "policy_name": policy_name,
                        "policy_visibility": POLICY_PROMPTS[policy_name]["visibility"],
                        "profile_id": profile["profile_id"],
                        "case_id": profile.get("case_id"),
                        "base_severity": severity,
                        "turn_index": 0,
                        "dialogue_history": [],
                        "messages": build_messages(policy_name, [], language=language),
                        "expected_output": {
                            "doctor_question": f"one natural-language {'English' if language == 'en' else 'Chinese'} question",
                        },
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
                        ],
                    }
                )
    return records


def write_protocol(path: Path, request_path: Path, num_requests: int) -> None:
    lines = [
        "# LLM Doctor Baseline Request Protocol V1",
        "",
        "Date: 2026-06-11",
        "",
        "## Purpose",
        "",
        "This file defines the request format for small/closed LLM dynamic doctor baselines.",
        "",
        "The LLM doctor must only see dialogue history and instructions. It must not see diagnosis-tree nodes, hidden profile slots, diagnosis labels, or controller metadata.",
        "",
        "## Request File",
        "",
        f"- JSONL: `{request_path.name}`",
        f"- Initial requests: {num_requests}",
        "",
        "Each request contains:",
        "",
        "- `messages`: model-visible chat messages.",
        "- `dialogue_history`: model-visible history, repeated for inspection.",
        "- `profile_id`, `case_id`, and `base_severity`: evaluation metadata only, not model-visible fields.",
        "- `hidden_eval_metadata_not_for_model`: fields that must never be included in an LLM prompt.",
        "",
        "## Policies",
        "",
        "| Policy | Visibility | Purpose |",
        "|---|---|---|",
        "| `closed_llm_general` | doctor_visible_only | Test whether a strong/general LLM naturally asks useful active-diagnosis questions. |",
        "| `closed_llm_evidence_aware` | doctor_visible_only | Test whether prompt-level evidence-sufficiency guidance is enough without training. |",
        "",
        "## Replay Requirement",
        "",
        "For online evaluation, the runner should repeat this loop:",
        "",
        "```text",
        "LLM sees dialogue_history only",
        "LLM outputs doctor_question",
        "V1 environment internally maps question to simulator node",
        "patient controller returns patient_response + hidden metadata",
        "append doctor_question/patient_response to dialogue_history",
        "build next request",
        "```",
        "",
        "No hidden field may be copied into `messages`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare initial requests for small/closed LLM doctor baselines.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--max-groups", type=int, default=90)
    parser.add_argument("--max-per-slot", type=int, default=5)
    parser.add_argument("--max-profiles", type=int, default=27)
    parser.add_argument("--dataset-prefix", default="mdd5k")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
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
    profiles_by_id = load_profiles(args.profiles)
    groups = select_pilot_groups(
        load_group_records(args.group_dir, args.splits),
        max_groups=args.max_groups,
        max_per_slot=args.max_per_slot,
    )
    profiles = select_profiles_from_groups(groups, profiles_by_id, args.max_profiles)
    severities = [normalize_severity(level) for level in args.severities]
    requests = build_initial_requests(
        profiles=profiles,
        severities=severities,
        policies=args.policies,
        language=args.language,
    )

    request_path = args.output_dir / f"{args.dataset_prefix}_llm_doctor_initial_requests.jsonl"
    protocol_path = args.output_dir / "LLM_DOCTOR_BASELINE_REQUEST_PROTOCOL_V1.md"
    write_jsonl(request_path, requests)
    write_protocol(protocol_path, request_path, len(requests))
    print(
        json.dumps(
            {
                "num_requests": len(requests),
                "num_profiles": len(profiles),
                "dataset_prefix": args.dataset_prefix,
                "language": args.language,
                "severities": severities,
                "policies": args.policies,
                "request_path": str(request_path),
                "protocol_path": str(protocol_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
