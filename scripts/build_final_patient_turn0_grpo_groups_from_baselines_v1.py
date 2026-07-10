from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_turn0_grpo_groups_v1"

PROMPT = (
    "你是一名精神心理主动问诊 doctor agent。\n"
    "任务：在一次新的低信息患者问诊开始时，提出第一句自然、具体、安全的医生问题，"
    "以最大化后续 canonical clinical evidence recovery。\n"
    "要求：只输出一句医生问题；不要诊断；不要提到内部标签、症状槽位、证据单元或患者模拟器。"
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
        raise argparse.ArgumentTypeError("Empty label.")
    return label, Path(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def turn0_rows_for_source(
    *,
    label: str,
    output_dir: Path,
    metric_name: str,
    require_verified: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = records_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    final_scores = load_final_scores(output_dir, metric_name)
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        counters["records_seen"] += 1
        if row.get("turn_index") != 0:
            continue
        counters["turn0_seen"] += 1
        if require_verified and row.get("patient_realizer_mode") != "verified_llm_cache":
            counters["skip_non_verified_patient"] += 1
            continue
        if hard_error(row):
            counters["skip_hard_error"] += 1
            continue
        question = str(row.get("doctor_question") or "").strip()
        if not question:
            counters["skip_empty_question"] += 1
            continue
        scenario_id = str(row.get("scenario_id") or "")
        if scenario_id not in final_scores:
            counters["skip_missing_final_score"] += 1
            continue
        rows.append(
            {
                "source_label": label,
                "source_output_dir": str(output_dir),
                "scenario_id": scenario_id,
                "profile_id": row.get("profile_id"),
                "case_id": row.get("case_id"),
                "base_severity": row.get("base_severity"),
                "policy_name": row.get("policy_name"),
                "doctor_question": question,
                "patient_response": row.get("patient_response"),
                "final_recovery_score": final_scores[scenario_id],
                "delta_cumulative_slot_sufficiency": safe_float(
                    row.get("delta_cumulative_slot_sufficiency")
                ),
                "response_type": row.get("response_type"),
                "target_tree_node_for_audit_only": row.get("target_tree_node"),
                "record_id": row.get("record_id"),
                "request_id": row.get("request_id"),
            }
        )
    counters["rows_used"] = len(rows)
    counters["final_scores_loaded"] = len(final_scores)
    return rows, dict(counters)


def make_group_prompt(base_severity: str) -> str:
    return f"{PROMPT}\n\n当前患者信息：base_severity={base_severity}。暂无对话历史。\n\n下一句医生问题："


def build_groups(
    source_rows: list[dict[str, Any]],
    *,
    min_candidates: int,
    reward_source: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        profile_id = str(row.get("profile_id") or "")
        severity = str(row.get("base_severity") or "")
        if profile_id and severity:
            grouped[(profile_id, severity)].append(row)

    groups: list[dict[str, Any]] = []
    for (profile_id, severity), rows in sorted(grouped.items()):
        responses = []
        seen_text: set[str] = set()
        for row in sorted(rows, key=lambda item: str(item.get("source_label") or "")):
            text = str(row.get("doctor_question") or "").strip()
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            if reward_source == "final_recovery":
                reward = safe_float(row.get("final_recovery_score"))
            elif reward_source == "first_turn_delta":
                reward = safe_float(row.get("delta_cumulative_slot_sufficiency"))
            else:
                raise ValueError(f"Unsupported reward_source={reward_source!r}")
            responses.append(
                {
                    "text": text,
                    "reward": round(float(reward), 6),
                    "metadata": {
                        "source_label": row.get("source_label"),
                        "source_output_dir": row.get("source_output_dir"),
                        "scenario_id": row.get("scenario_id"),
                        "record_id": row.get("record_id"),
                        "request_id": row.get("request_id"),
                        "policy_name": row.get("policy_name"),
                        "patient_response": row.get("patient_response"),
                        "response_type": row.get("response_type"),
                        "final_recovery_score": row.get("final_recovery_score"),
                        "delta_cumulative_slot_sufficiency": row.get("delta_cumulative_slot_sufficiency"),
                        "target_tree_node_for_audit_only": row.get("target_tree_node_for_audit_only"),
                        "reward_source": reward_source,
                    },
                }
            )
        if len(responses) < min_candidates:
            continue
        rewards = [float(item["reward"]) for item in responses]
        if max(rewards) - min(rewards) <= 1e-9:
            continue
        groups.append(
            {
                "id": f"final_patient_turn0::{profile_id}::{severity}",
                "prompt": make_group_prompt(severity),
                "responses": responses,
                "metadata": {
                    "profile_id": profile_id,
                    "base_severity": severity,
                    "turn_index": 0,
                    "candidate_count": len(responses),
                    "reward_source": reward_source,
                    "same_state_boundary": "turn0_empty_history_only",
                    "final_patient_setting": True,
                    "doctor_visible_only": True,
                },
            }
        )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict turn-0 same-state GRPO groups from final-patient baseline outputs."
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric-name", default="keyword_supported_only")
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--reward-source", choices=["final_recovery", "first_turn_delta"], default="final_recovery")
    parser.add_argument("--allow-non-verified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    source_summaries = {}
    for label, output_dir in args.source:
        rows, summary = turn0_rows_for_source(
            label=label,
            output_dir=output_dir,
            metric_name=args.metric_name,
            require_verified=not args.allow_non_verified,
        )
        source_summaries[label] = summary
        all_rows.extend(rows)
    groups = build_groups(
        all_rows,
        min_candidates=args.min_candidates,
        reward_source=args.reward_source,
    )
    groups_path = args.output_dir / "final_patient_turn0_grpo_groups.jsonl"
    write_jsonl(groups_path, groups)
    candidate_counts = Counter(len(group.get("responses") or []) for group in groups)
    severity_counts = Counter(str((group.get("metadata") or {}).get("base_severity")) for group in groups)
    reward_margins = []
    for group in groups:
        rewards = [safe_float(resp.get("reward")) for resp in group.get("responses") or []]
        if rewards:
            reward_margins.append(max(rewards) - min(rewards))
    summary = {
        "sources": {label: str(path) for label, path in args.source},
        "source_summaries": source_summaries,
        "settings": {
            "metric_name": args.metric_name,
            "require_verified": not args.allow_non_verified,
            "min_candidates": args.min_candidates,
            "reward_source": args.reward_source,
            "same_state_boundary": "only turn0 empty-history states are grouped across models",
        },
        "input_turn0_rows": len(all_rows),
        "groups": len(groups),
        "candidate_count_distribution": {str(k): v for k, v in sorted(candidate_counts.items())},
        "severity_counts": dict(severity_counts),
        "mean_reward_margin": round(sum(reward_margins) / len(reward_margins), 6) if reward_margins else 0.0,
        "groups_path": str(groups_path),
    }
    write_json(args.output_dir / "final_patient_turn0_grpo_group_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
