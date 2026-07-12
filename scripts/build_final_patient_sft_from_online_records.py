from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_sft_data"
DEFAULT_DATASET_PREFIX = "mdd5k"

SYSTEM_PROMPT = (
    "你是一名用于研究场景的精神心理主动问诊 doctor agent。"
    "你的任务是根据可见的医患对话历史，提出下一句自然、具体、安全的医生问题，"
    "以逐步收集诊断相关证据。不要给出诊断结论，不要暴露任何内部标签、症状槽位、"
    "诊断树节点或患者模拟器信息。"
)

EN_SYSTEM_PROMPT = (
    "You are a doctor agent for a research mental-health screening setting. "
    "Your task is to ask the next natural, specific, safe clinician question based only on the visible doctor-patient dialogue history. "
    "The goal is to gradually gather clinically relevant evidence. Do not diagnose, summarize at length, or reveal any internal labels, symptom slots, evidence units, simulator metadata, or tree structure."
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


def stable_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Use LABEL=OUTPUT_DIR for --source.")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Empty source label.")
    return label, Path(path)


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "暂无对话历史。"
    lines: list[str] = []
    for idx, turn in enumerate(history[-12:], start=1):
        doctor = str(turn.get("doctor_utterance") or "").strip()
        patient = str(turn.get("patient_utterance") or "").strip()
        lines.append(f"{idx}. 医生：{doctor}")
        lines.append(f"   患者：{patient}")
    return "\n".join(lines)


def make_user_prompt(history: list[dict[str, str]]) -> str:
    return (
        "请只根据下面的可见问诊历史，生成下一句医生问题。\n"
        "要求：\n"
        "1. 只输出一句医生下一问。\n"
        "2. 不要诊断、总结或安慰过长。\n"
        "3. 如果患者刚才回答含糊、回避或信息不足，优先温和追问同一主题。\n"
        "4. 如果某一主题已经比较充分，可以转向仍未明确的相关症状或风险。\n"
        "5. 不要提到内部标签、证据单元、症状槽位或患者模拟器。\n\n"
        f"可见对话历史：\n{format_history(history)}\n\n"
        "下一句医生问题："
    )


def format_history_for_language(history: list[dict[str, str]], language: str) -> str:
    if language != "en":
        return format_history(history)
    if not history:
        return "No dialogue history yet."
    lines: list[str] = []
    for idx, turn in enumerate(history[-12:], start=1):
        doctor = str(turn.get("doctor_utterance") or "").strip()
        patient = str(turn.get("patient_utterance") or "").strip()
        lines.append(f"{idx}. Doctor: {doctor}")
        lines.append(f"   Patient: {patient}")
    return "\n".join(lines)


def make_user_prompt_for_language(history: list[dict[str, str]], language: str) -> str:
    if language != "en":
        return make_user_prompt(history)
    return (
        "Based only on the visible screening dialogue below, generate the next doctor question.\n"
        "Requirements:\n"
        "1. Output exactly one next doctor question.\n"
        "2. Do not diagnose, summarize, or over-reassure.\n"
        "3. If the patient was vague, avoidant, or low-information, gently follow up on the same topic.\n"
        "4. If a topic is covered well enough, move to another relevant symptom, risk, or functioning area.\n"
        "5. Do not mention internal labels, evidence units, symptom slots, or the simulator.\n\n"
        f"Visible dialogue history:\n{format_history_for_language(history, language)}\n\n"
        "Next doctor question:"
    )


def source_paths(output_dir: Path, dataset_prefix: str = DEFAULT_DATASET_PREFIX) -> tuple[Path, Path]:
    records = output_dir / f"{dataset_prefix}_llm_doctor_online_replay_records.jsonl"
    recovery = (
        output_dir
        / "tree_aligned_canonical_recovery"
        / "tree_aligned_canonical_evidence_recovery_rows.jsonl"
    )
    return records, recovery


def load_recovery_scores(path: Path, metric_name: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    for row in iter_jsonl(path):
        if row.get("metric_name") != metric_name:
            continue
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            continue
        score = stable_float(row.get("tree_aligned_canonical_final_s"))
        if score is not None:
            scores[scenario_id] = score
    return scores


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("patient_verify_hard_error")
        or row.get("patient_hard_error")
        or row.get("hard_error")
    )


def build_examples_for_source(
    *,
    label: str,
    output_dir: Path,
    metric_name: str,
    min_final_score: float | None,
    min_delta_sufficiency: float | None,
    max_turn_index: int | None,
    require_verified: bool,
    dataset_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_path, recovery_path = source_paths(output_dir, dataset_prefix=dataset_prefix)
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    recovery_scores = load_recovery_scores(recovery_path, metric_name)

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for row in iter_jsonl(records_path):
        counters["records_seen"] += 1
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            counters["skip_missing_scenario"] += 1
            continue
        if require_verified and row.get("patient_realizer_mode") != "verified_llm_cache":
            counters["skip_non_verified_patient"] += 1
            continue
        if hard_error(row):
            counters["skip_hard_error"] += 1
            continue
        turn_index = row.get("turn_index")
        if max_turn_index is not None and isinstance(turn_index, int) and turn_index > max_turn_index:
            counters["skip_after_max_turn"] += 1
            continue
        if min_delta_sufficiency is not None:
            delta = stable_float(row.get("delta_cumulative_slot_sufficiency"), default=0.0)
            if delta is None or delta < min_delta_sufficiency:
                counters["skip_low_delta_sufficiency"] += 1
                continue
        if min_final_score is not None:
            final_score = recovery_scores.get(scenario_id)
            if final_score is None:
                counters["skip_missing_recovery_score"] += 1
                continue
            if final_score < min_final_score:
                counters["skip_low_final_score"] += 1
                continue
        question = str(row.get("doctor_question") or "").strip()
        response = str(row.get("patient_response") or "").strip()
        if not question or not response:
            counters["skip_empty_question_or_response"] += 1
            continue
        by_scenario[scenario_id].append(row)

    examples: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(by_scenario.items()):
        rows = sorted(rows, key=lambda item: int(item.get("turn_index") or 0))
        history: list[dict[str, str]] = []
        final_score = recovery_scores.get(scenario_id)
        for row in rows:
            question = str(row.get("doctor_question") or "").strip()
            patient = str(row.get("patient_response") or "").strip()
            language = str(row.get("language") or "zh")
            example_id = f"{label}::{row.get('record_id') or scenario_id + '::turn_' + str(row.get('turn_index'))}"
            examples.append(
                {
                    "id": example_id,
                    "messages": [
                        {"role": "system", "content": EN_SYSTEM_PROMPT if language == "en" else SYSTEM_PROMPT},
                        {"role": "user", "content": make_user_prompt_for_language(history, language)},
                        {"role": "assistant", "content": question},
                    ],
                    "metadata": {
                        "source_label": label,
                        "source_output_dir": str(output_dir),
                        "record_id": row.get("record_id"),
                        "request_id": row.get("request_id"),
                        "scenario_id": scenario_id,
                        "profile_id": row.get("profile_id"),
                        "case_id": row.get("case_id"),
                        "base_severity": row.get("base_severity"),
                        "turn_index": row.get("turn_index"),
                        "response_type": row.get("response_type"),
                        "patient_realizer_mode": row.get("patient_realizer_mode"),
                        "language": language,
                        "final_recovery_metric": metric_name,
                        "scenario_final_recovery": final_score,
                        "delta_cumulative_slot_sufficiency": row.get(
                            "delta_cumulative_slot_sufficiency"
                        ),
                        "cumulative_slot_sufficiency": row.get("cumulative_slot_sufficiency"),
                        "target_tree_node_for_audit_only": row.get("target_tree_node"),
                    },
                }
            )
            history.append({"doctor_utterance": question, "patient_utterance": patient})
    counters["scenarios_used"] = len(by_scenario)
    counters["examples_built"] = len(examples)
    counters["recovery_scores_loaded"] = len(recovery_scores)
    return examples, dict(counters)


def split_examples(
    examples: list[dict[str, Any]],
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    cutoff = int(dev_ratio * 10_000)
    for example in examples:
        scenario_id = str((example.get("metadata") or {}).get("scenario_id") or example.get("id"))
        bucket = stable_hash(f"{seed}::{scenario_id}") % 10_000
        if bucket < cutoff:
            dev.append(example)
        else:
            train.append(example)
    return train, dev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final-patient doctor SFT data from online verified patient records."
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--metric-name", default="keyword_supported_only")
    parser.add_argument("--min-final-score", type=float, default=None)
    parser.add_argument("--min-delta-sufficiency", type=float, default=None)
    parser.add_argument("--max-turn-index", type=int, default=None)
    parser.add_argument("--allow-non-verified", action="store_true")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_examples: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    seen_ids: set[str] = set()

    for label, output_dir in args.source:
        examples, summary = build_examples_for_source(
            label=label,
            output_dir=output_dir,
            metric_name=args.metric_name,
            min_final_score=args.min_final_score,
            min_delta_sufficiency=args.min_delta_sufficiency,
            max_turn_index=args.max_turn_index,
            require_verified=not args.allow_non_verified,
            dataset_prefix=args.dataset_prefix,
        )
        source_summaries[label] = summary
        for example in examples:
            if example["id"] in seen_ids:
                continue
            seen_ids.add(example["id"])
            all_examples.append(example)

    train, dev = split_examples(all_examples, dev_ratio=args.dev_ratio, seed=args.seed)
    train_path = args.output_dir / f"{args.dataset_prefix}_final_patient_doctor_sft_train.jsonl"
    dev_path = args.output_dir / f"{args.dataset_prefix}_final_patient_doctor_sft_dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    summary = {
        "sources": {label: str(path) for label, path in args.source},
        "source_summaries": source_summaries,
        "settings": {
            "dataset_prefix": args.dataset_prefix,
            "metric_name": args.metric_name,
            "min_final_score": args.min_final_score,
            "min_delta_sufficiency": args.min_delta_sufficiency,
            "max_turn_index": args.max_turn_index,
            "require_verified": not args.allow_non_verified,
            "dev_ratio": args.dev_ratio,
            "seed": args.seed,
        },
        "examples_total": len(all_examples),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
    }
    write_json(args.output_dir / "final_patient_sft_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
