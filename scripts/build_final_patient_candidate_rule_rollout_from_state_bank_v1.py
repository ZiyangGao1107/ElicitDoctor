from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_dynamic_patient_controller_v1 import DEFAULT_GROUP_DIR, DEFAULT_PROFILE_PATH, DEFAULT_SCHEMA_PATH
from build_dynamic_patient_controller_v1 import load_profiles
from build_dynamic_patient_controller_v3_2 import DynamicPatientControllerV32
from online_query_interpreter import load_json


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_candidate_rule_rollout_v1"


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


def clean_question(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    if not text.endswith(("?", "？")):
        text = text.rstrip("。.!！?？") + "？"
    return text


def load_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in iter_jsonl(path) if row.get(key) is not None}


def hard_error(row: dict[str, Any]) -> bool:
    return bool(
        row.get("patient_verify_hard_error")
        or row.get("patient_hard_error")
        or row.get("hard_error")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay same-state doctor candidates through the PCV3.2 rule controller. "
            "This produces trajectory records that can then be passed through the "
            "LLM patient realizer/verifier pipeline."
        )
    )
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--candidate-requests", type=Path, required=True)
    parser.add_argument("--candidate-outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--require-all-outputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema = load_json(args.schema)
    profiles = load_profiles(args.profiles)
    controller = DynamicPatientControllerV32(schema=schema, profiles=profiles, max_units_per_slot=8)

    states = load_by_key(args.state_bank, "state_id")
    requests = list(iter_jsonl(args.candidate_requests))
    outputs = load_by_key(args.candidate_outputs, "request_id")

    records: list[dict[str, Any]] = []
    missing_requests: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen_state_candidates: set[str] = set()

    for request in requests:
        request_id = str(request.get("request_id") or "")
        state_id = str(request.get("state_id") or "")
        state = states.get(state_id)
        if not request_id or not state:
            counters["skip_missing_state"] += 1
            continue
        output = outputs.get(request_id)
        if not output:
            counters["missing_candidate_output"] += 1
            missing_requests.append(request)
            if args.require_all_outputs:
                continue
            continue

        question = clean_question(output.get("doctor_question") or output.get("output") or output.get("content"))
        if not question:
            counters["skip_empty_question"] += 1
            continue
        profile_id = str(state.get("profile_id") or request.get("profile_id") or "")
        severity = str(state.get("base_severity") or request.get("base_severity") or "")
        if profile_id not in profiles:
            counters["skip_unknown_profile"] += 1
            continue

        controller_state_before = state.get("controller_state_before") or {}
        dialogue_history = state.get("dialogue_history") or []
        response, controller_state_after = controller.step(
            profile_id=profile_id,
            doctor_question=question,
            base_severity=severity,
            state=controller_state_before,
            dialogue_history=dialogue_history,
        )
        record_id = f"{state_id}::{request_id.rsplit('::', 1)[-1]}::rule"
        record = {
            "record_id": record_id,
            "scenario_id": state_id,
            "request_id": request_id,
            "source_state_id": state_id,
            "source_state_hash": state.get("state_hash"),
            "candidate_index": request.get("candidate_index"),
            "candidate_method": request.get("method"),
            "profile_id": profile_id,
            "case_id": state.get("case_id"),
            "diagnoses": profiles[profile_id].get("diagnoses"),
            "icd_codes": profiles[profile_id].get("icd_codes"),
            "policy_name": state.get("policy_name"),
            "policy_visibility": "doctor_visible_only",
            "llm_provider": output.get("provider"),
            "doctor_question_source": "same_state_candidate_output",
            "base_severity": severity,
            "turn_index": state.get("turn_index"),
            "question_type": "same_state_candidate_question",
            "doctor_question": question,
            "candidate_raw_output": output.get("raw_output"),
            "candidate_model": output.get("model"),
            "candidate_adapter": output.get("adapter"),
            "reference_doctor_question": state.get("reference_doctor_question"),
            "reference_patient_response": state.get("reference_patient_response"),
            "source_state_identity": state.get("state_identity"),
            "dialogue_history": state.get("dialogue_history") or [],
            "source_refs": state.get("source_refs") or [],
            "controller_state_before_replay": controller_state_before,
            "controller_state_after_replay": controller_state_after,
            **response,
        }
        record["patient_realizer_mode"] = "rule"
        record["patient_realizer_cache_hit"] = False
        if hard_error(record):
            counters["hard_error"] += 1
            continue
        records.append(record)
        seen_state_candidates.add(f"{state_id}::{request.get('candidate_index')}")
        counters["records_built"] += 1
        if args.max_records > 0 and len(records) >= args.max_records:
            break

    record_path = args.output_dir / "final_patient_candidate_rule_rollout_records.jsonl"
    missing_path = args.output_dir / "final_patient_candidate_missing_output_requests.jsonl"
    write_jsonl(record_path, records)
    write_jsonl(missing_path, missing_requests)

    summary = {
        "settings": {
            "state_bank": str(args.state_bank),
            "candidate_requests": str(args.candidate_requests),
            "candidate_outputs": str(args.candidate_outputs),
            "max_records": args.max_records,
            "require_all_outputs": args.require_all_outputs,
        },
        "states_loaded": len(states),
        "candidate_requests_loaded": len(requests),
        "candidate_outputs_loaded": len(outputs),
        "records": len(records),
        "missing_outputs": len(missing_requests),
        "unique_state_candidates": len(seen_state_candidates),
        "counters": dict(counters),
        "severity_distribution": dict(Counter(str(row.get("base_severity")) for row in records)),
        "turn_index_distribution": dict(
            sorted(Counter(str(row.get("turn_index")) for row in records).items(), key=lambda item: int(item[0]))
        ),
        "record_path": str(record_path),
        "missing_output_request_path": str(missing_path),
    }
    write_json(args.output_dir / "final_patient_candidate_rule_rollout_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
