from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_same_state_grpo_groups_v1"

SYSTEM_PROMPT = (
    "You are a research doctor agent for active mental-health interviewing. "
    "Given only the visible dialogue history, ask the next natural, specific, "
    "safe doctor question that is most likely to recover missing canonical "
    "clinical evidence. Output only the next doctor question. Do not diagnose, "
    "summarize, or mention internal labels, slots, evidence units, trees, rewards, "
    "or patient simulator metadata."
)


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


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Use LABEL=OUTPUT_DIR for --source.")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Empty source label.")
    return label, Path(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("patient_verify_hard_error")
        or row.get("patient_hard_error")
        or row.get("hard_error")
    )


def records_path(output_dir: Path) -> Path:
    return output_dir / "mdd5k_llm_doctor_online_replay_records.jsonl"


def recovery_path(output_dir: Path) -> Path:
    return (
        output_dir
        / "tree_aligned_canonical_recovery"
        / "tree_aligned_canonical_evidence_recovery_rows.jsonl"
    )


def load_final_scores(output_dir: Path, metric_name: str) -> dict[str, float]:
    path = recovery_path(output_dir)
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    for row in iter_jsonl(path):
        if row.get("metric_name") != metric_name:
            continue
        scenario_id = str(row.get("scenario_id") or "")
        if scenario_id:
            scores[scenario_id] = safe_float(row.get("tree_aligned_canonical_final_s"))
    return scores


def visible_history_key(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "doctor": clean_text(item.get("doctor")),
            "patient": clean_text(item.get("patient")),
        }
        for item in history
    ]


def make_state_identity(record: dict[str, Any], history: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "profile_id": str(record.get("profile_id") or ""),
        "case_id": str(record.get("case_id") or ""),
        "base_severity": str(record.get("base_severity") or ""),
        "policy_name": str(record.get("policy_name") or ""),
        "policy_visibility": str(record.get("policy_visibility") or ""),
        "turn_index": int(record.get("turn_index") or 0),
        "visible_history": visible_history_key(history),
    }


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no visible dialogue history yet)"
    lines: list[str] = []
    start = max(0, len(history) - 12)
    for idx, item in enumerate(history[start:], start=start + 1):
        lines.append(f"Turn {idx} Doctor: {clean_text(item.get('doctor'))}")
        lines.append(f"Turn {idx} Patient: {clean_text(item.get('patient'))}")
    return "\n".join(lines)


def make_prompt(history: list[dict[str, str]]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Visible dialogue history:\n"
        f"{format_history(history)}\n\n"
        "Next doctor question:"
    )


def reward_for_record(
    record: dict[str, Any],
    *,
    final_scores: dict[str, float],
    reward_source: str,
) -> float:
    if reward_source == "final_recovery":
        return final_scores.get(str(record.get("scenario_id") or ""), 0.0)
    if reward_source == "immediate_delta":
        return safe_float(record.get("delta_cumulative_slot_sufficiency"))
    if reward_source == "cumulative_after":
        return safe_float(record.get("cumulative_slot_sufficiency"))
    raise ValueError(f"Unsupported reward_source={reward_source!r}")


def collect_candidates_for_source(
    *,
    label: str,
    output_dir: Path,
    metric_name: str,
    reward_source: str,
    require_verified: bool,
    max_turn_index: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = records_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    final_scores = load_final_scores(output_dir, metric_name)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for record in iter_jsonl(path):
        counters["records_seen"] += 1
        scenario_id = str(record.get("scenario_id") or "")
        if not scenario_id:
            counters["skip_missing_scenario"] += 1
            continue
        if require_verified and record.get("patient_realizer_mode") != "verified_llm_cache":
            counters["skip_non_verified_patient"] += 1
            continue
        if hard_error(record):
            counters["skip_hard_error"] += 1
            continue
        turn_index = record.get("turn_index")
        if not isinstance(turn_index, int):
            counters["skip_missing_turn_index"] += 1
            continue
        if max_turn_index is not None and turn_index > max_turn_index:
            counters["skip_after_max_turn"] += 1
            continue
        question = clean_text(record.get("doctor_question"))
        patient = clean_text(record.get("patient_response"))
        if not question or not patient:
            counters["skip_empty_question_or_response"] += 1
            continue
        if reward_source == "final_recovery" and scenario_id not in final_scores:
            counters["skip_missing_final_score"] += 1
            continue
        by_scenario[scenario_id].append(record)

    candidates: list[dict[str, Any]] = []
    for scenario_id, records in sorted(by_scenario.items()):
        history: list[dict[str, str]] = []
        for record in sorted(records, key=lambda item: int(item.get("turn_index") or 0)):
            state_identity = make_state_identity(record, history)
            state_hash = stable_hash(state_identity)
            question = clean_text(record.get("doctor_question"))
            patient = clean_text(record.get("patient_response"))
            candidates.append(
                {
                    "state_hash": state_hash,
                    "state_identity": state_identity,
                    "prompt": make_prompt(history),
                    "source_label": label,
                    "source_output_dir": str(output_dir),
                    "scenario_id": scenario_id,
                    "profile_id": record.get("profile_id"),
                    "case_id": record.get("case_id"),
                    "base_severity": record.get("base_severity"),
                    "policy_name": record.get("policy_name"),
                    "turn_index": record.get("turn_index"),
                    "doctor_question": question,
                    "patient_response": patient,
                    "reward": reward_for_record(
                        record,
                        final_scores=final_scores,
                        reward_source=reward_source,
                    ),
                    "record_id": record.get("record_id"),
                    "request_id": record.get("request_id"),
                    "response_type": record.get("response_type"),
                    "target_tree_node_for_audit_only": record.get("target_tree_node"),
                    "delta_cumulative_slot_sufficiency": record.get("delta_cumulative_slot_sufficiency"),
                    "cumulative_slot_sufficiency": record.get("cumulative_slot_sufficiency"),
                    "patient_realizer_mode": record.get("patient_realizer_mode"),
                }
            )
            history.append({"doctor": question, "patient": patient})

    counters["scenarios_used"] = len(by_scenario)
    counters["candidates_collected"] = len(candidates)
    counters["final_scores_loaded"] = len(final_scores)
    return candidates, dict(counters)


def build_groups(
    candidates: list[dict[str, Any]],
    *,
    min_candidates: int,
    require_reward_margin: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["state_hash"])].append(candidate)

    groups: list[dict[str, Any]] = []
    for state_hash, rows in sorted(grouped.items()):
        responses = []
        seen_questions: set[str] = set()
        state_identity = rows[0]["state_identity"]
        prompt = rows[0]["prompt"]
        for row in sorted(rows, key=lambda item: (str(item["source_label"]), str(item["record_id"]))):
            question = clean_text(row.get("doctor_question"))
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            responses.append(
                {
                    "text": question,
                    "reward": round(float(row["reward"]), 6),
                    "metadata": {
                        "source_label": row.get("source_label"),
                        "source_output_dir": row.get("source_output_dir"),
                        "scenario_id": row.get("scenario_id"),
                        "record_id": row.get("record_id"),
                        "request_id": row.get("request_id"),
                        "patient_response": row.get("patient_response"),
                        "response_type": row.get("response_type"),
                        "target_tree_node_for_audit_only": row.get("target_tree_node_for_audit_only"),
                        "delta_cumulative_slot_sufficiency": row.get("delta_cumulative_slot_sufficiency"),
                        "cumulative_slot_sufficiency": row.get("cumulative_slot_sufficiency"),
                        "patient_realizer_mode": row.get("patient_realizer_mode"),
                    },
                }
            )
        if len(responses) < min_candidates:
            continue
        rewards = [float(item["reward"]) for item in responses]
        if require_reward_margin and max(rewards) - min(rewards) <= 1e-9:
            continue
        groups.append(
            {
                "id": f"final_patient_same_state::{state_hash}",
                "prompt": prompt,
                "responses": responses,
                "metadata": {
                    "state_hash": state_hash,
                    "state_identity": state_identity,
                    "candidate_count": len(responses),
                    "turn_index": state_identity.get("turn_index"),
                    "base_severity": state_identity.get("base_severity"),
                    "profile_id": state_identity.get("profile_id"),
                    "same_state_boundary": "exact profile/severity/policy/turn/visible-history match",
                    "final_patient_setting": True,
                    "doctor_visible_only": True,
                },
            }
        )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict same-visible-state GRPO groups from final-patient "
            "online baseline records. This safely generalizes turn-0 grouping: "
            "later turns are grouped only when the visible dialogue history is exact."
        )
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric-name", default="keyword_supported_only")
    parser.add_argument(
        "--reward-source",
        choices=["final_recovery", "immediate_delta", "cumulative_after"],
        default="final_recovery",
    )
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-turn-index", type=int, default=None)
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--allow-zero-reward-margin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for label, output_dir in args.source:
        candidates, summary = collect_candidates_for_source(
            label=label,
            output_dir=output_dir,
            metric_name=args.metric_name,
            reward_source=args.reward_source,
            require_verified=not args.allow_non_verified,
            max_turn_index=args.max_turn_index,
        )
        source_summaries[label] = summary
        all_candidates.extend(candidates)

    groups = build_groups(
        all_candidates,
        min_candidates=args.min_candidates,
        require_reward_margin=not args.allow_zero_reward_margin,
    )

    groups_path = args.output_dir / "final_patient_same_state_grpo_groups.jsonl"
    write_jsonl(groups_path, groups)

    candidate_counts = Counter(len(group.get("responses") or []) for group in groups)
    severity_counts = Counter(str((group.get("metadata") or {}).get("base_severity")) for group in groups)
    turn_counts = Counter(str((group.get("metadata") or {}).get("turn_index")) for group in groups)
    reward_margins = []
    for group in groups:
        rewards = [safe_float(resp.get("reward")) for resp in group.get("responses") or []]
        if rewards:
            reward_margins.append(max(rewards) - min(rewards))

    summary = {
        "settings": {
            "metric_name": args.metric_name,
            "reward_source": args.reward_source,
            "min_candidates": args.min_candidates,
            "max_turn_index": args.max_turn_index,
            "require_verified": not args.allow_non_verified,
            "require_reward_margin": not args.allow_zero_reward_margin,
            "same_state_boundary": "exact profile/severity/policy/turn/visible-history match",
        },
        "source_summaries": source_summaries,
        "candidates_collected": len(all_candidates),
        "groups": len(groups),
        "candidate_count_distribution": {str(k): v for k, v in sorted(candidate_counts.items())},
        "severity_distribution": dict(sorted(severity_counts.items())),
        "turn_index_distribution": dict(sorted(turn_counts.items(), key=lambda item: int(item[0]))),
        "mean_reward_margin": round(sum(reward_margins) / len(reward_margins), 6) if reward_margins else 0.0,
        "max_reward_margin": round(max(reward_margins), 6) if reward_margins else 0.0,
        "groups_path": str(groups_path),
    }
    write_json(args.output_dir / "final_patient_same_state_grpo_group_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
