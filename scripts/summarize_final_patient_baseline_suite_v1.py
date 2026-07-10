from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "closed_evidence",
    "qwen_base",
    "qwen_sft_r16",
    "qwen_grpo_v6_300",
    "qwen_grpo_v6_full1500",
    "qwen_valueaug_full1500",
    "qwen_grpo_rfv2_ckpt1600",
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_metrics(path: Path) -> dict[str, float | None]:
    values = {"mild": None, "moderate": None, "severe": None, "mean": None}
    if not path.exists():
        return values
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_severity: dict[str, float] = {}
    for row in rows:
        severity = row.get("base_severity")
        value = row.get("mean_tree_aligned_canonical_final_s")
        if isinstance(value, (int, float)):
            by_severity[str(severity)] = float(value)
    values["mild"] = by_severity.get("mild_low_info")
    values["moderate"] = by_severity.get("moderate_low_info")
    values["severe"] = by_severity.get("severe_low_info")
    nums = [v for v in (values["mild"], values["moderate"], values["severe"]) if isinstance(v, float)]
    values["mean"] = sum(nums) / len(nums) if nums else None
    return values


def read_record_audit(records_path: Path) -> dict[str, Any]:
    audit = {
        "records": line_count(records_path) or 0,
        "patient_realizer_mode_counts": {},
        "verified_only": False,
        "hard_error_rows": 0,
        "max_turn_index": None,
    }
    if not records_path.exists():
        return audit
    modes: Counter[str] = Counter()
    hard = 0
    max_turn = None
    for row in iter_jsonl(records_path):
        modes[str(row.get("patient_realizer_mode"))] += 1
        if row.get("patient_verify_hard_error") or row.get("patient_hard_error") or row.get("hard_error"):
            hard += 1
        turn_index = row.get("turn_index")
        if isinstance(turn_index, int):
            max_turn = turn_index if max_turn is None else max(max_turn, turn_index)
    audit["patient_realizer_mode_counts"] = dict(modes)
    audit["verified_only"] = bool(modes) and set(modes) == {"verified_llm_cache"}
    audit["hard_error_rows"] = hard
    audit["max_turn_index"] = max_turn
    return audit


def summarize_model(*, tag: str, model: str, phase_dir: Path) -> dict[str, Any]:
    out_dir = phase_dir / f"outputs_{tag}_{model}"
    records_path = out_dir / "mdd5k_llm_doctor_online_replay_records.jsonl"
    pending_path = out_dir / "mdd5k_llm_doctor_online_replay_pending_requests.jsonl"
    cache_path = out_dir / "online_patient_work" / "current_verified_patient_cache.jsonl"
    summary_path = out_dir / "pcv32_keyword_supported_only.json"
    metrics = read_metrics(summary_path)
    audit = read_record_audit(records_path)
    entry = {
        "model": model,
        "output_dir": str(out_dir),
        "exists": out_dir.exists(),
        "summary_complete": summary_path.exists(),
        "pending": line_count(pending_path),
        "cache_records": line_count(cache_path),
        "keyword_supported_only": metrics,
        **audit,
    }
    entry["complete_verified_no_hard_errors"] = bool(
        entry["summary_complete"] and entry["verified_only"] and entry["hard_error_rows"] == 0
    )
    return entry


def format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


def write_markdown(path: Path, *, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| rank | model | mild | moderate | severe | mean | records | pending | verified_only | hard_errors | status |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    completed = [
        row
        for row in rows
        if isinstance((row.get("keyword_supported_only") or {}).get("mean"), (int, float))
    ]
    rank_by_model = {
        row["model"]: idx
        for idx, row in enumerate(
            sorted(completed, key=lambda item: (item["keyword_supported_only"]["mean"] or -1), reverse=True),
            start=1,
        )
    }
    for row in rows:
        metrics = row.get("keyword_supported_only") or {}
        status = "done" if row.get("summary_complete") else ("running" if row.get("exists") else "not_started")
        lines.append(
            "| {rank} | {model} | {mild} | {moderate} | {severe} | {mean} | {records} | {pending} | {verified} | {hard} | {status} |".format(
                rank=rank_by_model.get(row["model"], ""),
                model=row["model"],
                mild=format_score(metrics.get("mild")),
                moderate=format_score(metrics.get("moderate")),
                severe=format_score(metrics.get("severe")),
                mean=format_score(metrics.get("mean")),
                records=row.get("records", ""),
                pending="" if row.get("pending") is None else row.get("pending"),
                verified=row.get("verified_only"),
                hard=row.get("hard_error_rows", ""),
                status=status,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PCV3.2 online final-patient baseline suite outputs.")
    parser.add_argument("--tag", action="append", required=True, help="Suite tag, e.g. pcv32_online_final_patient_baseline_turn24_20260709_after_freeze")
    parser.add_argument("--turn", action="append", type=int, default=None, help="Optional turn label matching each --tag.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--phase-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    turns = args.turn or []
    if turns and len(turns) != len(args.tag):
        raise ValueError("--turn must be provided once per --tag, or omitted.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suites = []
    for idx, tag in enumerate(args.tag):
        rows = [summarize_model(tag=tag, model=model, phase_dir=args.phase_dir) for model in models]
        turn = turns[idx] if turns else None
        suite = {
            "tag": tag,
            "turn": turn,
            "models": rows,
            "complete_models": sum(1 for row in rows if row["summary_complete"]),
            "verified_complete_models": sum(1 for row in rows if row["complete_verified_no_hard_errors"]),
            "all_complete": all(row["summary_complete"] for row in rows),
            "all_complete_verified_no_hard_errors": all(row["complete_verified_no_hard_errors"] for row in rows),
        }
        suites.append(suite)
        suffix = f"turn{turn}" if turn is not None else tag
        write_markdown(
            args.output_dir / f"{suffix}_baseline_summary.md",
            title=f"Final Patient Baseline {suffix}",
            rows=rows,
        )
    result = {
        "suites": suites,
        "models": models,
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "final_patient_baseline_suite_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
