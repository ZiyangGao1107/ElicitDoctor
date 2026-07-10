from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_llm_patient_realizer_requests_v1 import iter_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VERIFICATION_RECORDS = (
    BASE_DIR
    / "outputs_llm_patient_realizer_v3_1"
    / "mdd5k_patient_realizer_verification_records_llm_outputs.jsonl"
)
DEFAULT_REQUEST_PATH = (
    BASE_DIR
    / "outputs_llm_patient_realizer_v3_1"
    / "mdd5k_llm_patient_realizer_requests.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_llm_patient_realizer_v3_1"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_request_meta(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["request_id"]): record for record in iter_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cache of verifier-accepted LLM patient responses.")
    parser.add_argument("--verification-records", type=Path, default=DEFAULT_VERIFICATION_RECORDS)
    parser.add_argument("--request-path", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-warned", action="store_true", help="Include accepted records even if they have readability warnings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_meta = load_request_meta(args.request_path)
    cache_records: list[dict[str, Any]] = []
    skipped = Counter()
    severity_counter = Counter()
    warning_counter = Counter()

    for record in iter_jsonl(args.verification_records):
        request_id = str(record.get("request_id"))
        request = request_meta.get(request_id, {})
        warnings = record.get("warnings") or []
        if not record.get("accepted"):
            skipped["not_accepted"] += 1
            continue
        if warnings and not args.include_warned:
            skipped["accepted_but_warned"] += 1
            warning_counter.update(warnings)
            continue
        severity_counter.update([str(record.get("base_severity"))])
        warning_counter.update(warnings)
        cache_records.append(
            {
                "source_record_id": record.get("source_record_id"),
                "request_id": request_id,
                "scenario_id": record.get("scenario_id"),
                "profile_id": request.get("profile_id"),
                "case_id": request.get("case_id"),
                "policy_name": record.get("policy_name"),
                "base_severity": record.get("base_severity"),
                "turn_index": record.get("turn_index") or request.get("turn_index"),
                "target_tree_node": record.get("target_tree_node"),
                "low_info_category": record.get("low_info_category"),
                "patient_response": record.get("patient_response"),
                "realizer_source": "llm_verified",
                "provider": record.get("provider"),
                "model": record.get("model"),
                "prompt_protocol_version": request.get("prompt_protocol_version"),
                "history_mode": request.get("history_mode"),
                "accepted": True,
                "warnings": warnings,
                "mean_allowed_coverage": record.get("mean_allowed_coverage"),
                "g_target": record.get("g_target"),
            }
        )

    suffix = "include_warned" if args.include_warned else "clean_only"
    cache_path = args.output_dir / f"mdd5k_verified_patient_response_cache_{suffix}.jsonl"
    summary_path = args.output_dir / f"mdd5k_verified_patient_response_cache_summary_{suffix}.json"
    summary = {
        "verification_records": str(args.verification_records),
        "request_path": str(args.request_path),
        "cache_path": str(cache_path),
        "include_warned": args.include_warned,
        "num_cached": len(cache_records),
        "skipped": dict(skipped),
        "by_severity": dict(severity_counter),
        "warnings_seen": dict(warning_counter),
    }
    write_jsonl(cache_path, cache_records)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
