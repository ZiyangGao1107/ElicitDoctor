from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_llm_patient_realizer_requests_v1 import clean_text, iter_jsonl, units_by_ids, write_json, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_PATH = BASE_DIR / "outputs_llm_doctor_online_replay_v1" / "mdd5k_llm_doctor_online_replay_records.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_llm_patient_realizer_rubric_v3_1"

RUBRIC_DIMENSIONS = {
    "grounding": {
        "question": "Is the patient response faithful to allowed/weakened evidence, without unsupported facts or contradictions?",
        "score_1": "Major hallucination, contradiction, or diagnosis/risk assertion not supported by allowed/weakened evidence.",
        "score_3": "Mostly grounded but contains mild unsupported inference, over-specific wording, or ambiguous factual expansion.",
        "score_5": "Fully grounded; no new clinical facts; accurately preserves uncertainty and evidence scope.",
    },
    "disclosure_control": {
        "question": "Does the response respect retained/weakened/withheld/forbidden disclosure boundaries?",
        "score_1": "Leaks withheld/forbidden evidence or gives far more detail than the controller allowed.",
        "score_3": "Mostly respects boundaries but one detail is too explicit or weakened evidence is overstated.",
        "score_5": "Exactly respects allowed/weakened evidence and does not reveal withheld/forbidden details.",
    },
    "avoidance_quality": {
        "question": "If the response avoids, is the avoidance natural, bounded, and recoverable by good follow-up?",
        "score_1": "Inappropriate dead-end refusal, mechanical refusal, or avoids ordinary low-sensitivity questions.",
        "score_3": "Somewhat plausible avoidance but generic, repetitive, or not clearly recoverable.",
        "score_5": "Clinically/dialogically plausible boundary or vagueness with enough openness for supportive follow-up.",
    },
    "query_responsiveness": {
        "question": "Does the response answer the doctor's current question at the intended level without going off-topic?",
        "score_1": "Off-topic or ignores the question; response seems unrelated to the target topic.",
        "score_3": "Partially responsive but vague in a way not justified by setting or loses the doctor question focus.",
        "score_5": "Clearly responsive to the question while preserving the intended low-information level.",
    },
    "dialogue_naturalness": {
        "question": "Does the response sound like a patient rather than a doctor, therapist, checklist, or metadata summary?",
        "score_1": "Robotic, list-like, therapist-like, or obviously template-generated.",
        "score_3": "Readable but somewhat generic, repetitive, or unnatural in emotional tone.",
        "score_5": "Natural first-person patient language with coherent tone and concise conversational flow.",
    },
}

ERROR_TAGS = [
    "unsupported_fact",
    "profile_contradiction",
    "diagnosis_leakage",
    "withheld_evidence_leakage",
    "forbidden_evidence_leakage",
    "over_disclosure",
    "over_avoidance",
    "dead_end_refusal",
    "under_answered_direct_question",
    "off_topic",
    "therapist_like",
    "template_like",
    "persona_inconsistency",
    "too_verbose",
]


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    metadata = {
        "retained_units": units_by_ids(record, record.get("retained_unit_ids") or []),
        "weakened_units": units_by_ids(record, record.get("weakened_unit_ids") or []),
        "withheld_units": units_by_ids(record, record.get("withheld_unit_ids") or record.get("removed_unit_ids") or []),
        "removed_units": units_by_ids(record, record.get("removed_unit_ids") or []),
        "forbidden_units": units_by_ids(record, record.get("forbidden_unit_ids") or []),
    }
    user = {
        "task": "Evaluate one patient-simulator response using the rubric. Return only JSON.",
        "doctor_question": clean_text(record.get("doctor_question")),
        "patient_response": clean_text(record.get("patient_response")),
        "rule_based_patient_response": clean_text(record.get("rule_based_patient_response")),
        "patient_realizer_mode": record.get("patient_realizer_mode"),
        "controller_version": record.get("controller_version"),
        "base_severity": record.get("base_severity"),
        "target_tree_node": record.get("target_tree_node"),
        "low_info_category": record.get("low_info_category"),
        "controller_response_type": record.get("response_type"),
        "is_rapport_or_permission_turn": bool(record.get("is_rapport_or_permission_turn")),
        "evidence_available_to_realizer": {
            "allowed_retained": [unit.get("unit_text") for unit in metadata["retained_units"]],
            "allowed_weakened": [unit.get("unit_text") for unit in metadata["weakened_units"]],
        },
        "evidence_hidden_from_realizer_but_visible_to_judge": {
            "withheld": [unit.get("unit_text") for unit in metadata["withheld_units"]],
            "removed": [unit.get("unit_text") for unit in metadata["removed_units"]],
            "forbidden": [unit.get("unit_text") for unit in metadata["forbidden_units"]],
        },
        "rubric": RUBRIC_DIMENSIONS,
        "allowed_error_tags": ERROR_TAGS,
        "output_format": {
            "scores": {key: "integer 1-5" for key in RUBRIC_DIMENSIONS},
            "overall_score": "integer 1-5",
            "pass_fail": "pass | soft_fail | hard_fail",
            "error_tags": "list of allowed_error_tags",
            "brief_rationale": "one concise Chinese paragraph",
        },
        "hard_fail_rules": [
            "Any withheld/forbidden evidence leakage is hard_fail.",
            "Any unsupported suicide/self-harm, diagnosis, duration, or behavior claim is hard_fail.",
            "A patient response that sounds like a clinician explaining the case is at most soft_fail.",
        ],
    }
    system = (
        "You are an expert evaluator for a controlled psychiatric patient simulator. "
        "Judge the patient response, not the doctor. Be strict about evidence grounding and disclosure control. "
        "Return only valid JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def build_requests(records: list[dict[str, Any]], max_requests: int | None, seed: int | None) -> list[dict[str, Any]]:
    candidates = [record for record in records if record.get("doctor_question") and record.get("patient_response")]
    if seed is not None:
        random.Random(seed).shuffle(candidates)
    if max_requests is not None:
        candidates = candidates[:max_requests]
    requests = []
    for record in candidates:
        requests.append(
            {
                "request_id": f"{record.get('record_id')}::patient_realizer_rubric_judge",
                "task_name": "mdd5k_patient_realizer_rubric_judge",
                "prompt_protocol_version": "patient_realizer_rubric_v3_1",
                "source_record_id": record.get("record_id"),
                "scenario_id": record.get("scenario_id"),
                "profile_id": record.get("profile_id"),
                "policy_name": record.get("policy_name"),
                "base_severity": record.get("base_severity"),
                "target_tree_node": record.get("target_tree_node"),
                "patient_realizer_mode": record.get("patient_realizer_mode"),
                "messages": build_messages(record),
                "expected_output": {
                    "scores": "grounding/disclosure_control/avoidance_quality/query_responsiveness/dialogue_naturalness 1-5",
                    "pass_fail": "pass | soft_fail | hard_fail",
                    "error_tags": ERROR_TAGS,
                },
            }
        )
    return requests


def write_rubric(path: Path) -> None:
    lines = [
        "# Patient Realizer Rubric V3.1",
        "",
        "This rubric evaluates the patient simulator response, not the doctor policy.",
        "",
        "## Dimensions",
        "",
    ]
    for name, item in RUBRIC_DIMENSIONS.items():
        lines.extend(
            [
                f"### {name}",
                "",
                item["question"],
                "",
                f"- 1: {item['score_1']}",
                f"- 3: {item['score_3']}",
                f"- 5: {item['score_5']}",
                "",
            ]
        )
    lines.extend(["## Error Tags", "", "`" + "`, `".join(ERROR_TAGS) + "`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare closed-LLM rubric judge requests for patient-simulator responses.")
    parser.add_argument("--trajectory-path", type=Path, default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(args.trajectory_path))
    max_requests = args.max_requests if args.max_requests > 0 else None
    requests = build_requests(records, max_requests=max_requests, seed=args.sample_seed)
    request_path = args.output_dir / "mdd5k_patient_realizer_rubric_judge_requests.jsonl"
    summary_path = args.output_dir / "mdd5k_patient_realizer_rubric_judge_request_summary.json"
    rubric_path = args.output_dir / "PATIENT_REALIZER_RUBRIC_V3_1.md"
    write_jsonl(request_path, requests)
    write_rubric(rubric_path)
    summary = {
        "trajectory_path": str(args.trajectory_path),
        "request_path": str(request_path),
        "rubric_path": str(rubric_path),
        "num_source_records": len(records),
        "num_requests": len(requests),
        "by_realizer_mode": dict(Counter(str(record.get("patient_realizer_mode")) for record in records)),
        "prompt_protocol_version": "patient_realizer_rubric_v3_1",
        "sample_seed": args.sample_seed,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
