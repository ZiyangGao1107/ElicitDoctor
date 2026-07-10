from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_verified_candidate_rollout"


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


def hard_error(record: dict[str, Any]) -> bool:
    return bool(
        record.get("hard_error")
        or record.get("patient_hard_error")
        or record.get("patient_verify_hard_error")
        or record.get("has_forbidden_leak")
        or record.get("forbidden_evidence_leak")
    )


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        source_record_id = record.get("source_record_id")
        patient_response = record.get("patient_response")
        if not source_record_id or not patient_response:
            continue
        cache[str(source_record_id)] = record
    return cache


def apply_cache_to_record(record: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["rule_based_patient_response"] = record.get("patient_response")
    result["patient_response"] = cached.get("patient_response")
    result["patient_realizer_mode"] = "verified_llm_cache"
    result["patient_realizer_cache_hit"] = True
    result["patient_response_realizer"] = {
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
    result.setdefault("validity", {})["verified_llm_realizer_cache"] = True
    result.setdefault("realizer", {}).update(
        {
            "type": "verified_llm_cache",
            "model": cached.get("model"),
            "provider": cached.get("provider"),
            "version": cached.get("prompt_protocol_version") or "llm_patient_realizer_pcv3_2",
        }
    )
    result["patient_verify_warnings"] = cached.get("warnings") or []
    result["patient_verify_source_request_id"] = cached.get("request_id")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply verified LLM patient cache to same-state candidate rollout records."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--drop-hard-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache)

    verified_records: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for record in iter_jsonl(args.records):
        counters["records_seen"] += 1
        record_id = str(record.get("record_id") or "")
        cached = cache.get(record_id)
        if not cached:
            counters["missing_cache"] += 1
            missing_records.append(record)
            if args.require_all:
                continue
            continue
        if args.drop_hard_errors and hard_error(cached):
            counters["skip_cache_hard_error"] += 1
            continue
        verified = apply_cache_to_record(record, cached)
        if args.drop_hard_errors and hard_error(verified):
            counters["skip_verified_hard_error"] += 1
            continue
        verified_records.append(verified)
        counters["verified_records"] += 1

    record_path = args.output_dir / "final_patient_candidate_verified_rollout_records.jsonl"
    missing_path = args.output_dir / "final_patient_candidate_missing_verified_cache_records.jsonl"
    write_jsonl(record_path, verified_records)
    write_jsonl(missing_path, missing_records)
    summary = {
        "settings": {
            "records": str(args.records),
            "cache": str(args.cache),
            "require_all": args.require_all,
            "drop_hard_errors": args.drop_hard_errors,
        },
        "cache_records_loaded": len(cache),
        "records": len(verified_records),
        "missing_records": len(missing_records),
        "counters": dict(counters),
        "patient_realizer_mode_distribution": dict(
            Counter(str(row.get("patient_realizer_mode")) for row in verified_records)
        ),
        "hard_error_rows": sum(1 for row in verified_records if hard_error(row)),
        "record_path": str(record_path),
        "missing_path": str(missing_path),
    }
    write_json(args.output_dir / "final_patient_candidate_verified_rollout_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
