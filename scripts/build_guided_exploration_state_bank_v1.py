from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_final_patient_state_bank_from_online_records import (
    build_states_for_source,
    clean_text,
    dedupe_states,
    parse_source,
    safe_float,
    write_json,
    write_jsonl,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_guided_exploration_state_bank_v1"

STRATEGY_SPECS: dict[str, dict[str, str]] = {
    "direct": {
        "name": "direct symptom probe",
        "instruction": (
            "Ask one concrete, clinically relevant symptom question. "
            "Be specific, but do not interrogate, accuse, diagnose, or list multiple symptoms."
        ),
    },
    "gentle_low_burden": {
        "name": "gentle low-burden probe",
        "instruction": (
            "Ask one low-pressure question that can be answered with a small amount of information. "
            "Avoid demanding sensitive details, dates, causes, or multiple symptoms."
        ),
    },
    "bridge_low_sensitivity": {
        "name": "bridge through a lower-sensitivity area",
        "instruction": (
            "Do not ask the sensitive target directly yet. Ask an adjacent lower-sensitivity question "
            "that could naturally lead toward the target in later turns."
        ),
    },
    "permission_before_sensitive": {
        "name": "permission before sensitive probe",
        "instruction": (
            "If the topic may be sensitive, first ask permission and make clear that a minimal answer is acceptable. "
            "Then ask at most one carefully bounded question."
        ),
    },
    "repair_after_refusal": {
        "name": "repair after avoidance",
        "instruction": (
            "If the recent patient answer sounded hesitant, vague, defensive, or avoidant, explicitly lower pressure "
            "and switch to a simpler adjacent question. Do not repeat the same sensitive question."
        ),
    },
    "choice_based_concrete": {
        "name": "choice-based concrete probe",
        "instruction": (
            "Ask one concrete question with two or three easy choices, including an option to skip. "
            "The choices should reduce burden while still moving the clinical interview forward."
        ),
    },
}


SYSTEM_PROMPT_ZH = (
    "You are a doctor in a research psychiatric interview. "
    "Generate exactly one natural Chinese doctor question. "
    "Do not output explanations, labels, diagnoses, summaries, scores, or internal simulator terms."
)

SYSTEM_PROMPT_EN = (
    "You are a doctor in a research psychiatric interview. "
    "Generate exactly one natural English doctor question. "
    "Do not output explanations, labels, diagnoses, summaries, scores, or internal simulator terms."
)


def visible_history_text(history: list[dict[str, Any]], *, max_turns: int = 12) -> str:
    if not history:
        return "No previous visible dialogue."
    start = max(1, len(history) - max_turns + 1)
    lines: list[str] = []
    for idx, turn in enumerate(history[-max_turns:], start=start):
        doctor = clean_text(turn.get("doctor") or turn.get("doctor_utterance"))
        patient = clean_text(turn.get("patient") or turn.get("patient_utterance"))
        lines.append(f"Turn {idx} Doctor: {doctor}")
        lines.append(f"Turn {idx} Patient: {patient}")
    return "\n".join(lines)


def build_strategy_messages(
    state: dict[str, Any],
    strategy: str,
    variant_index: int,
    *,
    language: str,
) -> list[dict[str, str]]:
    spec = STRATEGY_SPECS[strategy]
    history = state.get("dialogue_history") or []
    reference = clean_text(state.get("reference_doctor_question"))
    prior_patient = clean_text(state.get("reference_patient_response"))
    output_language = "English" if language == "en" else "Chinese"
    user = {
        "task": "generate_next_doctor_question",
        "output_language": output_language,
        "output_contract": [
            "Return exactly one doctor question.",
            "Do not mention strategy names, symptom-slot names, controller rules, rewards, or evidence units.",
            "Do not ask several questions in one sentence.",
            "Do not make a diagnosis or summarize the case.",
        ],
        "strategy": {
            "id": strategy,
            "name": spec["name"],
            "instruction": spec["instruction"],
            "variant_index": variant_index,
        },
        "visible_dialogue_history": visible_history_text(history),
        "reference_question_for_context_only": reference,
        "reference_patient_response_for_context_only": prior_patient,
        "next_doctor_question": "",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT_ZH},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def parse_strategy_list(raw: str) -> list[str]:
    strategies: list[str] = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in STRATEGY_SPECS:
            raise argparse.ArgumentTypeError(
                f"Unknown strategy={key!r}; choices={','.join(sorted(STRATEGY_SPECS))}"
            )
        strategies.append(key)
    if not strategies:
        raise argparse.ArgumentTypeError("At least one strategy is required.")
    return strategies


def turn_bucket(turn_index: Any) -> str:
    turn = int(turn_index or 0)
    if turn <= 3:
        return "early"
    if turn <= 9:
        return "mid"
    if turn <= 15:
        return "late"
    return "very_late"


def balanced_limit(states: list[dict[str, Any]], max_states: int) -> list[dict[str, Any]]:
    if max_states <= 0 or len(states) <= max_states:
        return states
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        key = (str(state.get("base_severity") or "unknown"), turn_bucket(state.get("turn_index")))
        buckets[key].append(state)
    for key in buckets:
        buckets[key] = sorted(
            buckets[key],
            key=lambda item: (
                -safe_float(item.get("scenario_final_recovery")),
                str(item.get("state_id") or ""),
            ),
        )

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < max_states and keys:
        key = keys[cursor % len(keys)]
        rows = buckets[key]
        if rows:
            selected.append(rows.pop(0))
        keys = [item for item in keys if buckets[item]]
        cursor += 1
    return selected


def build_candidate_requests(
    states: list[dict[str, Any]],
    *,
    strategies: list[str],
    variants_per_strategy: int,
    method_prefix: str,
    language: str,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for state in states:
        candidate_index = 0
        for strategy in strategies:
            for variant_index in range(max(1, variants_per_strategy)):
                request_id = f"{state['state_id']}::cand_{candidate_index:02d}_{strategy}_v{variant_index}"
                method = f"{method_prefix}_{strategy}"
                requests.append(
                    {
                        "request_id": request_id,
                        "task_name": "final_patient_guided_same_state_doctor_candidate",
                        "method": method,
                        "strategy": strategy,
                        "strategy_name": STRATEGY_SPECS[strategy]["name"],
                        "strategy_variant_index": variant_index,
                        "state_id": state["state_id"],
                        "state_hash": state["state_hash"],
                        "candidate_index": candidate_index,
                        "policy_name": state.get("policy_name"),
                        "profile_id": state.get("profile_id"),
                        "case_id": state.get("case_id"),
                        "base_severity": state.get("base_severity"),
                        "turn_index": state.get("turn_index"),
                        "messages": build_strategy_messages(
                            state,
                            strategy,
                            variant_index,
                            language=language,
                        ),
                        "expected_output": {
                            "doctor_question": f"one natural {'English' if language == 'en' else 'Chinese'} question"
                        },
                        "metadata": {
                            "state_id": state["state_id"],
                            "strategy": strategy,
                            "strategy_variant_index": variant_index,
                            "source_refs": state.get("source_refs") or [],
                            "reference_doctor_question": state.get("reference_doctor_question"),
                            "reference_patient_response": state.get("reference_patient_response"),
                            "scenario_final_recovery": state.get("scenario_final_recovery"),
                            "same_state_exploration": True,
                        },
                    }
                )
                candidate_index += 1
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a same-state exploration state bank and strategy-aware doctor candidate requests. "
            "This only creates doctor-query candidate requests; patient responses still need the final "
            "PCV3.2 realizer/verifier pipeline."
        )
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric-name", default="keyword_supported_only")
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--max-turn-index", type=int, default=18)
    parser.add_argument("--min-final-score", type=float, default=None)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--state-shard-index", type=int, default=0)
    parser.add_argument("--state-shard-count", type=int, default=1)
    parser.add_argument(
        "--strategies",
        type=parse_strategy_list,
        default=parse_strategy_list(
            "direct,gentle_low_burden,bridge_low_sensitivity,"
            "permission_before_sensitive,repair_after_refusal,choice_based_concrete"
        ),
    )
    parser.add_argument("--variants-per-strategy", type=int, default=1)
    parser.add_argument("--candidate-method-prefix", default="guided_exploration_v1")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_shard_count <= 0:
        raise ValueError("--state-shard-count must be positive.")
    if args.state_shard_index < 0 or args.state_shard_index >= args.state_shard_count:
        raise ValueError("--state-shard-index must be in [0, state_shard_count).")
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

    deduped = dedupe_states(all_states)
    deduped = sorted(
        deduped,
        key=lambda item: (
            str(item.get("base_severity") or ""),
            turn_bucket(item.get("turn_index")),
            str(item.get("state_id") or ""),
        ),
    )
    sharded = [
        state
        for idx, state in enumerate(deduped)
        if idx % args.state_shard_count == args.state_shard_index
    ]
    selected = balanced_limit(sharded, args.max_states)
    requests = build_candidate_requests(
        selected,
        strategies=args.strategies,
        variants_per_strategy=max(1, args.variants_per_strategy),
        method_prefix=args.candidate_method_prefix,
        language=args.language,
    )

    state_path = args.output_dir / "guided_exploration_state_bank.jsonl"
    request_path = args.output_dir / "guided_exploration_doctor_candidate_requests.jsonl"
    write_jsonl(state_path, selected)
    write_jsonl(request_path, requests)

    summary = {
        "settings": {
            "metric_name": args.metric_name,
            "require_verified": not args.allow_non_verified,
            "max_turn_index": args.max_turn_index,
            "min_final_score": args.min_final_score,
            "max_states": args.max_states,
            "state_shard_index": args.state_shard_index,
            "state_shard_count": args.state_shard_count,
            "strategies": args.strategies,
            "variants_per_strategy": max(1, args.variants_per_strategy),
            "candidate_method_prefix": args.candidate_method_prefix,
            "language": args.language,
        },
        "source_summaries": source_summaries,
        "raw_states": len(all_states),
        "deduped_states": len(deduped),
        "sharded_states": len(sharded),
        "selected_states": len(selected),
        "candidate_requests": len(requests),
        "severity_distribution": dict(Counter(str(state.get("base_severity")) for state in selected)),
        "turn_bucket_distribution": dict(Counter(turn_bucket(state.get("turn_index")) for state in selected)),
        "turn_index_distribution": dict(
            sorted(Counter(str(state.get("turn_index")) for state in selected).items(), key=lambda item: int(item[0]))
        )
        if selected
        else {},
        "strategy_distribution": dict(Counter(str(request.get("strategy")) for request in requests)),
        "state_path": str(state_path),
        "candidate_request_path": str(request_path),
        "purpose": (
            "Create direct/gentle/bridge/permission/repair/choice candidate branches for later final-patient "
            "rollout and delayed value labeling. RFV-v2 trajectories should not be used as sources."
        ),
    }
    write_json(args.output_dir / "guided_exploration_state_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
