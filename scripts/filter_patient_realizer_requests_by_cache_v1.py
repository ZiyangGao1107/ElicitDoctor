from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_llm_patient_realizer_requests_v1 import iter_jsonl


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_cached_request_ids(paths: list[Path]) -> set[str]:
    cached: set[str] = set()
    for path in paths:
        if not path or not path.exists() or path.stat().st_size == 0:
            continue
        for record in iter_jsonl(path):
            request_id = str(record.get("request_id") or "").strip()
            source_record_id = str(record.get("source_record_id") or "").strip()
            if request_id:
                cached.add(request_id)
            if source_record_id:
                cached.add(f"{source_record_id}::llm_patient_realizer")
    return cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep patient-realizer requests not already covered by verified cache.")
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, action="append", default=[])
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_ids = load_cached_request_ids(args.cache_path)
    kept: list[dict[str, Any]] = []
    skipped = Counter()
    total = 0
    for request in iter_jsonl(args.request_path):
        total += 1
        request_id = str(request.get("request_id") or "").strip()
        if request_id and request_id in cached_ids:
            skipped["already_cached"] += 1
            continue
        kept.append(request)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_path, kept)
    summary = {
        "request_path": str(args.request_path),
        "cache_paths": [str(path) for path in args.cache_path],
        "output_path": str(args.output_path),
        "total_requests": total,
        "kept_requests": len(kept),
        "skipped": dict(skipped),
        "cached_request_ids": len(cached_ids),
    }
    write_json(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
