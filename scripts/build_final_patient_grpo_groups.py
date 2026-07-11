from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_grpo_groups"


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("patient_verify_hard_error")
        or row.get("patient_hard_error")
        or row.get("hard_error")
    )


def reward_value(row: dict[str, Any], reward_source: str) -> float:
    if reward_source == "immediate_delta":
        return safe_float(row.get("delta_cumulative_slot_sufficiency"))
    if reward_source == "cumulative_after":
        return safe_float(row.get("cumulative_slot_sufficiency"))
    if reward_source == "g_target":
        return safe_float(row.get("g_target"))
    if reward_source == "action_value_total":
        return safe_float(row.get("action_value_total_gain"))
    raise ValueError(f"Unsupported reward_source={reward_source!r}")


def load_value_predictions(path: Path | None, value_field: str) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {
        "record_id": {},
        "source_record_id": {},
        "state_candidate": {},
        "action_value_record_id": {},
    }
    if not path:
        return maps
    for row in iter_jsonl(path):
        value = safe_float(row.get(value_field), default=float("nan"))
        if value != value:
            continue
        record_id = str(row.get("record_id") or "")
        if record_id:
            maps["record_id"][record_id] = value
        metadata = row.get("metadata") or {}
        source_record_id = str(row.get("source_record_id") or metadata.get("source_record_id") or "")
        if source_record_id:
            maps["source_record_id"][source_record_id] = value
        source_state_id = str(row.get("source_state_id") or metadata.get("source_state_id") or row.get("state_id") or "")
        candidate_index = str(row.get("candidate_index") or metadata.get("candidate_index") or "")
        if source_state_id and candidate_index:
            maps["state_candidate"][f"{source_state_id}::{candidate_index}"] = value
            maps["action_value_record_id"][f"{source_state_id}::candidate_{candidate_index}"] = value
    return maps


def value_prediction_for(row: dict[str, Any], maps: dict[str, dict[str, float]]) -> float | None:
    record_id = str(row.get("record_id") or "")
    if record_id in maps["source_record_id"]:
        return maps["source_record_id"][record_id]
    if record_id in maps["record_id"]:
        return maps["record_id"][record_id]
    if record_id in maps["action_value_record_id"]:
        return maps["action_value_record_id"][record_id]
    state_id = str(row.get("source_state_id") or row.get("scenario_id") or "")
    candidate_index = str(row.get("candidate_index") or "")
    if state_id and candidate_index:
        key = f"{state_id}::{candidate_index}"
        if key in maps["state_candidate"]:
            return maps["state_candidate"][key]
        action_key = f"{state_id}::candidate_{candidate_index}"
        if action_key in maps["action_value_record_id"]:
            return maps["action_value_record_id"][action_key]
    return None


def visible_prompt(row: dict[str, Any]) -> str:
    state_identity = row.get("source_state_identity") or {}
    history = row.get("dialogue_history") or state_identity.get("visible_history") or []
    if not history:
        return (
            "你是一个研究场景中的精神心理主动问诊医生。"
            "请只输出下一句自然、具体、安全的医生问题。\n\n"
            "可见问诊历史：暂无对话历史。\n\n下一句医生问题："
        )
    lines: list[str] = []
    for idx, turn in enumerate(history[-12:], start=max(1, len(history) - 11)):
        doctor = clean_text(turn.get("doctor") or turn.get("doctor_utterance"))
        patient = clean_text(turn.get("patient") or turn.get("patient_utterance"))
        lines.append(f"{idx}. 医生：{doctor}")
        lines.append(f"   患者：{patient}")
    return (
        "你是一个研究场景中的精神心理主动问诊医生。"
        "请只输出下一句自然、具体、安全的医生问题。\n\n"
        f"可见问诊历史：\n{chr(10).join(lines)}\n\n下一句医生问题："
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GRPO groups from same-state candidate rollout records."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reward-source",
        choices=["immediate_delta", "cumulative_after", "g_target", "action_value_total"],
        default="immediate_delta",
    )
    parser.add_argument("--value-predictions", type=Path, default=None)
    parser.add_argument("--value-field", default="prediction")
    parser.add_argument("--base-reward-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--require-value-predictions", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--allow-rule", action="store_true")
    parser.add_argument("--allow-zero-reward-margin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    value_maps = load_value_predictions(args.value_predictions, args.value_field)
    if args.value_weight != 0.0 and not args.value_predictions:
        raise ValueError("--value-predictions is required when --value-weight is non-zero.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for row in iter_jsonl(args.records):
        counters["records_seen"] += 1
        mode = str(row.get("patient_realizer_mode") or "")
        if not args.allow_non_verified:
            if mode != "verified_llm_cache" and not (args.allow_rule and mode == "rule"):
                counters["skip_non_verified_patient"] += 1
                continue
        if hard_error(row):
            counters["skip_hard_error"] += 1
            continue
        state_id = str(row.get("source_state_id") or row.get("scenario_id") or "")
        question = clean_text(row.get("doctor_question"))
        if not state_id:
            counters["skip_missing_state_id"] += 1
            continue
        if not question:
            counters["skip_empty_question"] += 1
            continue
        grouped[state_id].append(row)

    groups: list[dict[str, Any]] = []
    for state_id, rows in sorted(grouped.items()):
        responses = []
        seen_questions: set[str] = set()
        for row in sorted(rows, key=lambda item: int(item.get("candidate_index") or 0)):
            question = clean_text(row.get("doctor_question"))
            if question in seen_questions:
                counters["skip_duplicate_question"] += 1
                continue
            seen_questions.add(question)
            base_reward = reward_value(row, args.reward_source)
            predicted_value = value_prediction_for(row, value_maps)
            if args.value_weight != 0.0 and predicted_value is None:
                counters["skip_missing_value_prediction"] += 1
                if args.require_value_predictions:
                    continue
                predicted_value = 0.0
            final_reward = args.base_reward_weight * base_reward
            if predicted_value is not None:
                final_reward += args.value_weight * predicted_value
            responses.append(
                {
                    "text": question,
                    "reward": round(final_reward, 6),
                    "metadata": {
                        "record_id": row.get("record_id"),
                        "request_id": row.get("request_id"),
                        "source_state_id": state_id,
                        "candidate_index": row.get("candidate_index"),
                        "base_severity": row.get("base_severity"),
                        "turn_index": row.get("turn_index"),
                        "patient_response": row.get("patient_response"),
                        "patient_realizer_mode": row.get("patient_realizer_mode"),
                        "response_type": row.get("response_type"),
                        "target_tree_node_for_audit_only": row.get("target_tree_node"),
                        "delta_cumulative_slot_sufficiency": row.get("delta_cumulative_slot_sufficiency"),
                        "cumulative_slot_sufficiency": row.get("cumulative_slot_sufficiency"),
                        "g_target": row.get("g_target"),
                        "action_value_total_gain": row.get("action_value_total_gain"),
                        "base_reward": round(base_reward, 6),
                        "value_prediction": round(predicted_value, 6) if predicted_value is not None else None,
                        "base_reward_weight": args.base_reward_weight,
                        "value_weight": args.value_weight,
                        "reward_source": args.reward_source,
                    },
                }
            )
        if args.max_candidates > 0 and len(responses) > args.max_candidates:
            ranked = sorted(responses, key=lambda item: safe_float(item.get("reward")), reverse=True)
            responses = ranked[: args.max_candidates]
        if len(responses) < args.min_candidates:
            counters["skip_too_few_candidates"] += 1
            continue
        rewards = [safe_float(item.get("reward")) for item in responses]
        if not args.allow_zero_reward_margin and max(rewards) - min(rewards) <= 1e-9:
            counters["skip_zero_reward_margin"] += 1
            continue
        first = rows[0]
        groups.append(
            {
                "id": f"final_patient_candidate_group::{state_id}",
                "prompt": visible_prompt(first),
                "responses": responses,
                "metadata": {
                    "source_state_id": state_id,
                    "candidate_count": len(responses),
                    "base_severity": first.get("base_severity"),
                    "turn_index": first.get("turn_index"),
                    "profile_id": first.get("profile_id"),
                    "reward_source": args.reward_source,
                    "same_state_boundary": "same source_state_id from final-patient state bank",
                    "final_patient_setting": True,
                },
            }
        )

    groups_path = args.output_dir / "final_patient_candidate_grpo_groups.jsonl"
    write_jsonl(groups_path, groups)
    margins = []
    for group in groups:
        rewards = [safe_float(resp.get("reward")) for resp in group.get("responses") or []]
        if rewards:
            margins.append(max(rewards) - min(rewards))
    summary = {
        "settings": {
            "records": str(args.records),
            "reward_source": args.reward_source,
            "value_predictions": str(args.value_predictions) if args.value_predictions else None,
            "value_field": args.value_field,
            "base_reward_weight": args.base_reward_weight,
            "value_weight": args.value_weight,
            "require_value_predictions": args.require_value_predictions,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "require_verified": not args.allow_non_verified,
            "allow_rule": args.allow_rule,
            "require_reward_margin": not args.allow_zero_reward_margin,
        },
        "counters": dict(counters),
        "states_seen": len(grouped),
        "value_prediction_records_loaded": sum(len(values) for values in value_maps.values()),
        "groups": len(groups),
        "candidate_count_distribution": {
            str(k): v for k, v in sorted(Counter(len(group.get("responses") or []) for group in groups).items())
        },
        "severity_distribution": dict(
            sorted(Counter(str((group.get("metadata") or {}).get("base_severity")) for group in groups).items())
        ),
        "turn_index_distribution": dict(
            sorted(
                Counter(str((group.get("metadata") or {}).get("turn_index")) for group in groups).items(),
                key=lambda item: int(item[0]),
            )
        )
        if groups
        else {},
        "mean_reward_margin": round(sum(margins) / len(margins), 6) if margins else 0.0,
        "max_reward_margin": round(max(margins), 6) if margins else 0.0,
        "groups_path": str(groups_path),
    }
    write_json(args.output_dir / "final_patient_candidate_grpo_group_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
