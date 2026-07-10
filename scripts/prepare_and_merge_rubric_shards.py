from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    if not path.exists():
        return
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


def request_id(record: dict[str, Any]) -> str:
    return str(record.get("request_id") or "")


def load_by_request_id(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    records: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for path in paths:
        count = 0
        for record in iter_jsonl(path) or []:
            rid = request_id(record)
            if not rid:
                continue
            records.setdefault(rid, record)
            count += 1
        counts[str(path)] = count
    return records, counts


def prepare(args: argparse.Namespace) -> None:
    requests = list(iter_jsonl(args.request_path) or [])
    existing, source_counts = load_by_request_id([args.output_path])
    todo = [record for record in requests if request_id(record) not in existing]
    args.shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [args.shard_dir / f"requests_shard_{idx}.jsonl" for idx in range(args.num_shards)]
    shard_counts = [0 for _ in range(args.num_shards)]
    handles = [path.open("w", encoding="utf-8", newline="\n") for path in shard_paths]
    try:
        for idx, record in enumerate(todo):
            shard_idx = idx % args.num_shards
            handles[shard_idx].write(json.dumps(record, ensure_ascii=False) + "\n")
            shard_counts[shard_idx] += 1
    finally:
        for handle in handles:
            handle.close()

    write_json(
        args.shard_dir / "prepare_summary.json",
        {
            "request_path": str(args.request_path),
            "output_path": str(args.output_path),
            "total_requests": len(requests),
            "existing_unique_outputs": len(existing),
            "todo": len(todo),
            "num_shards": args.num_shards,
            "shard_counts": dict(zip([str(path) for path in shard_paths], shard_counts)),
            "source_counts": source_counts,
        },
    )


def merge(args: argparse.Namespace) -> None:
    requests = list(iter_jsonl(args.request_path) or [])
    request_order = [request_id(record) for record in requests]
    shard_outputs = [args.shard_dir / f"outputs_shard_{idx}.jsonl" for idx in range(args.num_shards)]
    merged, source_counts = load_by_request_id([args.output_path] + shard_outputs)

    missing = [rid for rid in request_order if rid and rid not in merged]
    duplicate_count = sum(source_counts.values()) - len(merged)
    if missing and not args.allow_incomplete:
        write_json(
            args.shard_dir / "merge_failed_summary.json",
            {
                "request_path": str(args.request_path),
                "output_path": str(args.output_path),
                "num_requests": len(request_order),
                "num_unique_outputs": len(merged),
                "missing_count": len(missing),
                "missing_sample": missing[:20],
                "duplicate_count": duplicate_count,
                "source_counts": source_counts,
            },
        )
        raise SystemExit(f"Refusing to merge incomplete rubric outputs: missing={len(missing)}")

    ordered_records = [merged[rid] for rid in request_order if rid in merged]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = args.output_path.with_suffix(args.output_path.suffix + f".pre_shard_merge_{timestamp}.bak")
    temp_path = args.output_path.with_suffix(args.output_path.suffix + f".tmp_shard_merge_{timestamp}")

    if args.output_path.exists():
        shutil.copy2(args.output_path, backup_path)
    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in ordered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(args.output_path)

    write_json(
        args.shard_dir / "merge_summary.json",
        {
            "request_path": str(args.request_path),
            "output_path": str(args.output_path),
            "backup_path": str(backup_path) if backup_path.exists() else None,
            "num_requests": len(request_order),
            "num_unique_outputs": len(merged),
            "written_records": len(ordered_records),
            "missing_count": len(missing),
            "missing_sample": missing[:20],
            "duplicate_count": duplicate_count,
            "source_counts": source_counts,
            "allow_incomplete": args.allow_incomplete,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and merge sharded patient realizer rubric outputs.")
    parser.add_argument("--mode", choices=["prepare", "merge"], required=True)
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.mode == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
