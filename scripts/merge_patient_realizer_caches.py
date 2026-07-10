from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _patient_realizer_io import iter_jsonl


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cache_key(record: dict[str, Any]) -> str:
    request_id = str(record.get("request_id") or "").strip()
    if request_id:
        return request_id
    source_record_id = str(record.get("source_record_id") or "").strip()
    return f"{source_record_id}::llm_patient_realizer" if source_record_id else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge verified patient-realizer cache shards by request id.")
    parser.add_argument("--cache-path", type=Path, action="append", default=[])
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--prefer-later", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: dict[str, dict[str, Any]] = {}
    source_counts = Counter()
    skipped = Counter()
    duplicate_count = 0

    for path in args.cache_path:
        if not path or not path.exists() or path.stat().st_size == 0:
            skipped[f"missing_or_empty:{path}"] += 1
            continue
        for record in iter_jsonl(path):
            key = cache_key(record)
            if not key:
                skipped["missing_key"] += 1
                continue
            if key in merged:
                duplicate_count += 1
                if not args.prefer_later:
                    continue
            merged[key] = record
            source_counts[str(record.get("realizer_source") or "unknown")] += 1

    records = [merged[key] for key in sorted(merged)]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_path, records)
    summary = {
        "cache_paths": [str(path) for path in args.cache_path],
        "output_path": str(args.output_path),
        "num_records": len(records),
        "duplicates_seen": duplicate_count,
        "prefer_later": args.prefer_later,
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
    }
    write_json(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
