from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_state_bank_v1"

SYSTEM_PROMPT = (
    "你是一个研究场景中的精神心理主动问诊医生。"
    "你只能根据可见医患对话历史提出下一句自然、具体、安全的医生问题，"
    "目标是逐步恢复缺失的规范化临床证据。"
    "只输出下一句医生问题；不要诊断、总结、安慰过长，"
    "不要提到内部标签、症状槽位、证据单元、奖励、患者模拟器或树结构。"
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


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


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


def history_key(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "doctor": clean_text(turn.get("doctor_utterance")),
            "patient": clean_text(turn.get("patient_utterance")),
        }
        for turn in history
    ]


def state_identity(record: dict[str, Any], history: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "profile_id": str(record.get("profile_id") or ""),
        "case_id": str(record.get("case_id") or ""),
        "base_severity": str(record.get("base_severity") or ""),
        "policy_name": str(record.get("policy_name") or ""),
        "policy_visibility": str(record.get("policy_visibility") or ""),
        "turn_index": int(record.get("turn_index") or 0),
        "visible_history": history_key(history),
    }


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "暂无对话历史。"
    lines: list[str] = []
    for idx, turn in enumerate(history[-12:], start=max(1, len(history) - 11)):
        doctor = clean_text(turn.get("doctor_utterance"))
        patient = clean_text(turn.get("patient_utterance"))
        lines.append(f"{idx}. 医生：{doctor}")
        lines.append(f"   患者：{patient}")
    return "\n".join(lines)


def build_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请根据下面的可见问诊历史，生成下一句医生问题。\n"
                "要求：只输出一句问题；温和、具体、可追问；不要暴露内部标签。\n\n"
                f"可见问诊历史：\n{format_history(history)}\n\n"
                "下一句医生问题："
            ),
        },
    ]


def initial_controller_state() -> dict[str, Any]:
    return {
        "turn_index": 0,
        "last_target_slot": None,
        "asked_count_by_slot": {},
        "disclosed_profile_unit_ids_by_slot": {},
        "last_g_target_by_slot": {},
        "last_cumulative_coverage_by_slot": {},
        "prior_boundary_refusal_by_slot": {},
        "disclosure_readiness": None,
        "disclosure_readiness_by_slot": {},
        "patient_state": None,
        "patient_terminated": False,
        "patient_termination_reason": None,
    }


def controller_state_before(record: dict[str, Any], running_state: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(running_state)
    state["turn_index"] = int(record.get("turn_index") or 0)
    if record.get("disclosure_readiness_before") is not None:
        state["disclosure_readiness"] = safe_float(record.get("disclosure_readiness_before"))
    if isinstance(record.get("patient_state_before"), dict):
        state["patient_state"] = copy.deepcopy(record.get("patient_state_before"))
    state["patient_terminated"] = False
    state["patient_termination_reason"] = None
    return state


def update_controller_state_from_record(record: dict[str, Any], running_state: dict[str, Any]) -> None:
    target_slot = str(record.get("target_tree_node") or "")
    if target_slot:
        running_state.setdefault("asked_count_by_slot", {})[target_slot] = int(
            record.get("asked_count_for_slot_after") or 0
        )
        running_state.setdefault("disclosed_profile_unit_ids_by_slot", {})[target_slot] = list(
            record.get("disclosed_profile_unit_ids_after") or []
        )
        running_state.setdefault("last_g_target_by_slot", {})[target_slot] = safe_float(record.get("g_target"))
        running_state.setdefault("last_cumulative_coverage_by_slot", {})[target_slot] = safe_float(
            record.get("cumulative_slot_sufficiency")
        )
        running_state["last_target_slot"] = target_slot
        if record.get("low_info_category") == "direct_refusal_or_boundary":
            running_state.setdefault("prior_boundary_refusal_by_slot", {})[target_slot] = True
        if record.get("disclosure_readiness_after") is not None:
            running_state.setdefault("disclosure_readiness_by_slot", {})[target_slot] = safe_float(
                record.get("disclosure_readiness_after")
            )
    if record.get("disclosure_readiness_after") is not None:
        running_state["disclosure_readiness"] = safe_float(record.get("disclosure_readiness_after"))
    if isinstance(record.get("patient_state_after"), dict):
        running_state["patient_state"] = copy.deepcopy(record.get("patient_state_after"))
    running_state["patient_terminated"] = bool(record.get("patient_terminated"))
    running_state["patient_termination_reason"] = record.get("patient_termination_reason")
    running_state["turn_index"] = int(record.get("turn_index") or 0) + 1


def build_states_for_source(
    *,
    label: str,
    output_dir: Path,
    metric_name: str,
    require_verified: bool,
    max_turn_index: int | None,
    min_final_score: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = records_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    final_scores = load_final_scores(output_dir, metric_name)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for row in iter_jsonl(path):
        counters["records_seen"] += 1
        if require_verified and row.get("patient_realizer_mode") != "verified_llm_cache":
            counters["skip_non_verified_patient"] += 1
            continue
        if hard_error(row):
            counters["skip_hard_error"] += 1
            continue
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            counters["skip_missing_scenario"] += 1
            continue
        if min_final_score is not None:
            score = final_scores.get(scenario_id)
            if score is None:
                counters["skip_missing_final_score"] += 1
                continue
            if score < min_final_score:
                counters["skip_low_final_score"] += 1
                continue
        turn_index = row.get("turn_index")
        if not isinstance(turn_index, int):
            counters["skip_missing_turn_index"] += 1
            continue
        if max_turn_index is not None and turn_index > max_turn_index:
            counters["skip_after_max_turn"] += 1
            continue
        question = clean_text(row.get("doctor_question"))
        patient = clean_text(row.get("patient_response"))
        if not question or not patient:
            counters["skip_empty_question_or_response"] += 1
            continue
        by_scenario[scenario_id].append(row)

    states: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(by_scenario.items()):
        history: list[dict[str, str]] = []
        running_state = initial_controller_state()
        final_score = final_scores.get(scenario_id)
        for row in sorted(rows, key=lambda item: int(item.get("turn_index") or 0)):
            identity = state_identity(row, history)
            state_hash = stable_hash(identity)
            state_before = controller_state_before(row, running_state)
            states.append(
                {
                    "state_id": f"final_patient_state::{state_hash}",
                    "state_hash": state_hash,
                    "state_identity": identity,
                    "controller_state_before": state_before,
                    "dialogue_history": list(history),
                    "messages": build_messages(history),
                    "source_label": label,
                    "source_output_dir": str(output_dir),
                    "scenario_id": scenario_id,
                    "profile_id": row.get("profile_id"),
                    "case_id": row.get("case_id"),
                    "base_severity": row.get("base_severity"),
                    "policy_name": row.get("policy_name"),
                    "turn_index": row.get("turn_index"),
                    "reference_doctor_question": clean_text(row.get("doctor_question")),
                    "reference_patient_response": clean_text(row.get("patient_response")),
                    "reference_record_id": row.get("record_id"),
                    "reference_request_id": row.get("request_id"),
                    "reference_response_type": row.get("response_type"),
                    "target_tree_node_for_audit_only": row.get("target_tree_node"),
                    "scenario_final_recovery": final_score,
                    "delta_cumulative_slot_sufficiency": row.get("delta_cumulative_slot_sufficiency"),
                    "cumulative_slot_sufficiency": row.get("cumulative_slot_sufficiency"),
                    "patient_realizer_mode": row.get("patient_realizer_mode"),
                }
            )
            history.append(
                {
                    "doctor_utterance": clean_text(row.get("doctor_question")),
                    "patient_utterance": clean_text(row.get("patient_response")),
                }
            )
            update_controller_state_from_record(row, running_state)

    counters["scenarios_used"] = len(by_scenario)
    counters["states_built"] = len(states)
    counters["final_scores_loaded"] = len(final_scores)
    return states, dict(counters)


def dedupe_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[str(state["state_id"])].append(state)

    result: list[dict[str, Any]] = []
    for state_id, rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda item: (
                -safe_float(item.get("scenario_final_recovery")),
                str(item.get("source_label") or ""),
                str(item.get("reference_record_id") or ""),
            ),
        )
        base = dict(rows[0])
        base["source_refs"] = [
            {
                "source_label": row.get("source_label"),
                "source_output_dir": row.get("source_output_dir"),
                "scenario_id": row.get("scenario_id"),
                "reference_record_id": row.get("reference_record_id"),
                "reference_request_id": row.get("reference_request_id"),
                "reference_doctor_question": row.get("reference_doctor_question"),
                "scenario_final_recovery": row.get("scenario_final_recovery"),
                "delta_cumulative_slot_sufficiency": row.get("delta_cumulative_slot_sufficiency"),
            }
            for row in rows
        ]
        base["num_source_refs"] = len(rows)
        result.append(base)
    return result


def build_candidate_requests(
    states: list[dict[str, Any]],
    *,
    candidates_per_state: int,
    method: str,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for state in states:
        for candidate_index in range(candidates_per_state):
            request_id = f"{state['state_id']}::cand_{candidate_index:02d}"
            requests.append(
                {
                    "request_id": request_id,
                    "task_name": "final_patient_same_state_doctor_candidate",
                    "method": method,
                    "state_id": state["state_id"],
                    "state_hash": state["state_hash"],
                    "candidate_index": candidate_index,
                    "policy_name": state.get("policy_name"),
                    "profile_id": state.get("profile_id"),
                    "case_id": state.get("case_id"),
                    "base_severity": state.get("base_severity"),
                    "turn_index": state.get("turn_index"),
                    "messages": state.get("messages") or [],
                    "expected_output": {"doctor_question": "one natural-language Chinese question"},
                    "metadata": {
                        "state_id": state["state_id"],
                        "source_refs": state.get("source_refs") or [],
                        "reference_doctor_question": state.get("reference_doctor_question"),
                        "reference_patient_response": state.get("reference_patient_response"),
                        "scenario_final_recovery": state.get("scenario_final_recovery"),
                    },
                }
            )
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a verified final-patient visible-state bank for same-state candidate rollout."
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric-name", default="keyword_supported_only")
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--max-turn-index", type=int, default=None)
    parser.add_argument("--min-final-score", type=float, default=None)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--candidates-per-state", type=int, default=4)
    parser.add_argument("--candidate-method", default="same_state_sampled_doctor_candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_states: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for label, output_dir in args.source:
        states, summary = build_states_for_source(
            label=label,
            output_dir=output_dir,
            metric_name=args.metric_name,
            require_verified=not args.allow_non_verified,
            max_turn_index=args.max_turn_index,
            min_final_score=args.min_final_score,
        )
        source_summaries[label] = summary
        all_states.extend(states)

    states = dedupe_states(all_states)
    states = sorted(
        states,
        key=lambda item: (
            -safe_float(item.get("scenario_final_recovery")),
            int(item.get("turn_index") or 0),
            str(item.get("state_id")),
        ),
    )
    if args.max_states > 0:
        states = states[: args.max_states]

    requests = build_candidate_requests(
        states,
        candidates_per_state=max(1, args.candidates_per_state),
        method=args.candidate_method,
    )

    state_path = args.output_dir / "final_patient_state_bank.jsonl"
    request_path = args.output_dir / "final_patient_same_state_candidate_requests.jsonl"
    write_jsonl(state_path, states)
    write_jsonl(request_path, requests)

    summary = {
        "settings": {
            "metric_name": args.metric_name,
            "require_verified": not args.allow_non_verified,
            "max_turn_index": args.max_turn_index,
            "min_final_score": args.min_final_score,
            "max_states": args.max_states,
            "candidates_per_state": args.candidates_per_state,
            "candidate_method": args.candidate_method,
        },
        "source_summaries": source_summaries,
        "raw_states": len(all_states),
        "deduped_states": len(states),
        "candidate_requests": len(requests),
        "turn_index_distribution": dict(
            sorted(Counter(str(state.get("turn_index")) for state in states).items(), key=lambda item: int(item[0]))
        ),
        "severity_distribution": dict(sorted(Counter(str(state.get("base_severity")) for state in states).items())),
        "source_ref_count_distribution": {
            str(k): v for k, v in sorted(Counter(int(state.get("num_source_refs") or 0) for state in states).items())
        },
        "state_path": str(state_path),
        "candidate_request_path": str(request_path),
    }
    write_json(args.output_dir / "final_patient_state_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
