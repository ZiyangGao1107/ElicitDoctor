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


def by_request_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record.get("request_id")): record for record in iter_jsonl(path) if record.get("request_id") is not None}


def by_source_record_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_record_id")): record
        for record in iter_jsonl(path)
        if record.get("source_record_id") is not None
    }


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def cache_record(
    *,
    request: dict[str, Any],
    verification: dict[str, Any] | None,
    patient_response: str,
    source: str,
    original_request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "source_record_id": request.get("source_record_id"),
        "request_id": original_request_id or request.get("request_id"),
        "scenario_id": request.get("scenario_id"),
        "profile_id": request.get("profile_id"),
        "case_id": request.get("case_id"),
        "policy_name": request.get("policy_name"),
        "base_severity": request.get("base_severity"),
        "turn_index": request.get("turn_index"),
        "target_tree_node": request.get("target_tree_node"),
        "low_info_category": request.get("low_info_category"),
        "patient_response": patient_response,
        "realizer_source": source,
        "provider": None if verification is None else verification.get("provider"),
        "model": None if verification is None else verification.get("model"),
        "prompt_protocol_version": request.get("prompt_protocol_version"),
        "history_mode": request.get("history_mode"),
        "accepted": True if verification is not None else None,
        "warnings": [] if verification is None else (verification.get("warnings") or []),
        "hard_errors": [] if verification is None else (verification.get("hard_errors") or []),
        "mean_allowed_coverage": None if verification is None else verification.get("mean_allowed_coverage"),
        "g_target": (request.get("hidden_verifier_metadata_not_for_realizer") or {}).get("g_target"),
    }


def accepted(verification: dict[str, Any] | None, include_warned: bool) -> bool:
    if not verification or not verification.get("accepted"):
        return False
    if verification.get("warnings") and not include_warned:
        return False
    return not verification.get("hard_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge primary, repair, and rule fallback patient realizer cache.")
    parser.add_argument("--primary-request-path", type=Path, required=True)
    parser.add_argument("--primary-verification-records", type=Path, required=True)
    parser.add_argument("--repair-request-path", type=Path, default=None)
    parser.add_argument("--repair-verification-records", type=Path, default=None)
    parser.add_argument("--rule-verification-records", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-warned", action="store_true")
    parser.add_argument(
        "--fallback-to-rule",
        action="store_true",
        help="Fill remaining failed records with source_rule_based_patient_response for 100% cache coverage.",
    )
    parser.add_argument("--dataset-prefix", default="mdd5k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_requests = by_request_id(args.primary_request_path)
    primary_verifications = by_request_id(args.primary_verification_records)

    repair_requests = by_request_id(args.repair_request_path) if args.repair_request_path else {}
    repair_verifications = by_request_id(args.repair_verification_records) if args.repair_verification_records else {}
    rule_verifications = by_request_id(args.rule_verification_records) if args.rule_verification_records else {}

    repair_by_original: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for repair_id, repair_request in repair_requests.items():
        original_id = str(repair_request.get("repair_of_request_id") or "").strip()
        verification = repair_verifications.get(repair_id)
        if original_id and verification:
            repair_by_original[original_id] = (repair_request, verification)

    cache_records: list[dict[str, Any]] = []
    source_counter = Counter()
    skipped = Counter()

    for request_id, request in primary_requests.items():
        primary_verification = primary_verifications.get(request_id)
        if accepted(primary_verification, args.include_warned):
            cache_records.append(
                cache_record(
                    request=request,
                    verification=primary_verification,
                    patient_response=clean_text(primary_verification.get("patient_response")),
                    source="llm_verified_primary",
                )
            )
            source_counter["llm_verified_primary"] += 1
            continue

        repair_pair = repair_by_original.get(request_id)
        if repair_pair:
            repair_request, repair_verification = repair_pair
            if accepted(repair_verification, args.include_warned):
                cache_records.append(
                    cache_record(
                        request=request,
                        verification=repair_verification,
                        patient_response=clean_text(repair_verification.get("patient_response")),
                        source="llm_verified_repair1",
                        original_request_id=request_id,
                    )
                )
                source_counter["llm_verified_repair1"] += 1
                continue
            skipped["repair_failed"] += 1
        else:
            skipped["no_repair_record"] += 1

        if args.fallback_to_rule:
            fallback_response = clean_text(request.get("source_rule_based_patient_response"))
            if fallback_response:
                rule_verification = rule_verifications.get(request_id)
                if rule_verifications and not accepted(rule_verification, args.include_warned):
                    skipped["rule_fallback_failed_verifier"] += 1
                    continue
                cache_records.append(
                    cache_record(
                        request=request,
                        verification=rule_verification,
                        patient_response=(
                            clean_text(rule_verification.get("patient_response"))
                            if rule_verification
                            else fallback_response
                        ),
                        source=(
                            "rule_verified_fallback_after_failed_llm_or_repair"
                            if rule_verification
                            else "rule_fallback_after_failed_llm_or_repair"
                        ),
                    )
                )
                if rule_verification:
                    source_counter["rule_verified_fallback_after_failed_llm_or_repair"] += 1
                else:
                    source_counter["rule_fallback_after_failed_llm_or_repair"] += 1
            else:
                skipped["missing_rule_fallback_response"] += 1
        else:
            skipped["failed_without_fallback"] += 1

    suffix = "include_warned" if args.include_warned else "clean_only"
    if args.fallback_to_rule:
        suffix += "_with_rule_fallback"
    cache_path = args.output_dir / f"{args.dataset_prefix}_verified_patient_response_cache_repair_{suffix}.jsonl"
    summary_path = args.output_dir / f"{args.dataset_prefix}_verified_patient_response_cache_repair_summary_{suffix}.json"
    summary = {
        "dataset_prefix": args.dataset_prefix,
        "primary_request_path": str(args.primary_request_path),
        "primary_verification_records": str(args.primary_verification_records),
        "repair_request_path": str(args.repair_request_path) if args.repair_request_path else None,
        "repair_verification_records": str(args.repair_verification_records) if args.repair_verification_records else None,
        "rule_verification_records": str(args.rule_verification_records) if args.rule_verification_records else None,
        "cache_path": str(cache_path),
        "num_primary_requests": len(primary_requests),
        "num_cached": len(cache_records),
        "coverage": round(len(cache_records) / len(primary_requests), 6) if primary_requests else 0.0,
        "source_counts": dict(source_counter),
        "skipped": dict(skipped),
        "include_warned": args.include_warned,
        "fallback_to_rule": args.fallback_to_rule,
    }
    write_jsonl(cache_path, cache_records)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
