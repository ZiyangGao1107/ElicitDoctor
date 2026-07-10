from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_llm_patient_realizer_requests_v1 import clean_text, iter_jsonl, units_by_ids, write_json, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_PATH = BASE_DIR / "outputs_llm_doctor_online_replay_v1" / "mdd5k_llm_doctor_online_replay_records.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_llm_patient_realizer_rubric_v3_2"
PROMPT_PROTOCOL_VERSION = "patient_realizer_rubric_v3_2"


RUBRIC_DIMENSIONS = {
    "grounding": {
        "question": "Is the patient response faithful to the allowed retained/weakened evidence, without unsupported facts, contradictions, diagnosis claims, or invented risk details?",
        "score_1": "Major hallucination or contradiction; invents symptom/risk/duration/behavior not supported by allowed evidence; or contradicts the hidden profile.",
        "score_2": "Contains a clinically important unsupported inference or over-specific detail, but the main topic is still related to the allowed evidence.",
        "score_3": "Mostly grounded, with minor unsupported wording, mild over-specificity, or ambiguous expansion.",
        "score_4": "Grounded with only harmless paraphrase-level inference; preserves the intended uncertainty and severity.",
        "score_5": "Fully grounded; no new clinical facts; accurately preserves retained/weakened evidence scope and uncertainty.",
    },
    "disclosure_control": {
        "question": "Does the response respect retained/weakened/withheld/removed/forbidden disclosure boundaries?",
        "score_1": "Leaks withheld/removed/forbidden evidence or gives severe/risk details that the controller did not allow.",
        "score_2": "Clear over-disclosure: reveals multiple specifics where only vague acknowledgement was allowed.",
        "score_3": "Mostly respects boundaries, but one detail is too explicit or weakened evidence is stated too strongly.",
        "score_4": "Respects boundaries with only minor wording that slightly strengthens weakened evidence.",
        "score_5": "Exactly respects allowed/weakened evidence and does not reveal hidden or forbidden details.",
    },
    "avoidance_quality": {
        "question": "When the controller intends low-information or avoidance, is the avoidance natural, bounded, and recoverable by supportive follow-up?",
        "score_1": "Dead-end refusal, mechanical refusal, hostile shutdown, or refusal of a normal low-sensitivity question without reason.",
        "score_2": "Mostly blocks the conversation; gives almost no foothold for a gentle follow-up.",
        "score_3": "Plausible but generic/repetitive avoidance; recoverability is weak or unclear.",
        "score_4": "Natural boundary/vagueness with a small clue or emotional cue that a good doctor can follow up on.",
        "score_5": "Clinically plausible guardedness: bounded, human, non-leaky, and clearly openable by warmth/trust/permission.",
    },
    "query_responsiveness": {
        "question": "Does the response answer the doctor's current question at the intended information level without being off-topic?",
        "score_1": "Off-topic, ignores the question, or answers a different clinical area without a conversational bridge.",
        "score_2": "Weakly related but mostly evasive or misrouted; the doctor cannot tell whether the question was understood.",
        "score_3": "Partially responsive but under-answers more than the controller setting justifies.",
        "score_4": "Responsive to the asked topic while staying appropriately vague or bounded.",
        "score_5": "Clearly answers the asked question at the allowed level; topic, tone, and amount of information match the controller.",
    },
    "dialogue_naturalness": {
        "question": "Does the response sound like a patient in an interview rather than a doctor, therapist, checklist, metadata summary, or policy explanation?",
        "score_1": "Robotic, template-like, clinician-like, list-like, or explains simulator policy/control metadata.",
        "score_2": "Understandable but unnatural, over-formal, repetitive, or emotionally implausible.",
        "score_3": "Readable patient-like response with some generic or scripted feel.",
        "score_4": "Natural first-person patient language with coherent emotion and concise conversational flow.",
        "score_5": "Highly natural patient utterance: context-aware, concise, human, and clinically plausible.",
    },
}


ERROR_TAGS = [
    "unsupported_fact",
    "profile_contradiction",
    "diagnosis_leakage",
    "self_harm_or_suicide_detail_leakage",
    "withheld_evidence_leakage",
    "removed_evidence_leakage",
    "forbidden_evidence_leakage",
    "over_disclosure",
    "over_avoidance",
    "dead_end_refusal",
    "under_answered_direct_question",
    "off_topic",
    "reasonable_topic_shift",
    "therapist_like",
    "template_like",
    "persona_inconsistency",
    "too_verbose",
    "metadata_leakage",
]


def unit_texts(record: dict[str, Any], field: str) -> list[str]:
    return [clean_text(unit.get("unit_text")) for unit in units_by_ids(record, record.get(field) or []) if clean_text(unit.get("unit_text"))]


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    retained = unit_texts(record, "retained_unit_ids")
    weakened = unit_texts(record, "weakened_unit_ids")
    withheld = unit_texts(record, "withheld_unit_ids") or unit_texts(record, "removed_unit_ids")
    removed = unit_texts(record, "removed_unit_ids")
    forbidden = unit_texts(record, "forbidden_unit_ids")
    hidden_observed = unit_texts(record, "hidden_observed_unit_ids")
    controller_state = {
        "controller_version": record.get("controller_version"),
        "base_severity": record.get("base_severity"),
        "target_tree_node": record.get("target_tree_node"),
        "low_info_category": record.get("low_info_category"),
        "controller_response_type": record.get("response_type"),
        "is_rapport_or_permission_turn": bool(record.get("is_rapport_or_permission_turn")),
        "patient_realizer_mode": record.get("patient_realizer_mode"),
    }
    user = {
        "task": "Evaluate one controlled patient-simulator response. This is a rubric judge, not a deterministic verifier. Return only valid JSON.",
        "evaluation_scope": [
            "Judge the patient response, not the doctor question.",
            "A low-information response can be good if the controller intended guardedness.",
            "Do not reward verbosity. Reward faithful, bounded, recoverable patient behavior.",
            "Distinguish true off-topic from a reasonable patient-like topic shift. If it is a clinically plausible bridge and still answers the question, use reasonable_topic_shift rather than off_topic.",
            "The deterministic verifier handles exact string leakage; here you should judge semantic leakage, unsupported facts, naturalness, and conversational quality.",
        ],
        "doctor_question": clean_text(record.get("doctor_question")),
        "patient_response": clean_text(record.get("patient_response")),
        "rule_based_patient_response": clean_text(record.get("rule_based_patient_response")),
        "controller_state": controller_state,
        "evidence_available_to_realizer": {
            "allowed_retained_direct": retained,
            "allowed_weakened_vague_or_softened": weakened,
        },
        "evidence_hidden_from_realizer_but_visible_to_judge": {
            "withheld": withheld,
            "removed": removed,
            "forbidden": forbidden,
            "hidden_observed": hidden_observed,
        },
        "scoring_instructions": {
            "score_range": "Use integer scores 1, 2, 3, 4, 5 for each dimension.",
            "overall_score": "Integer 1-5. Overall should be capped at 2 for hard_fail, capped at 3 for major soft_fail, otherwise reflect the five dimensions.",
            "pass_fail": {
                "pass": "Response is usable for controlled patient simulation.",
                "soft_fail": "Usable for analysis but has a noticeable quality problem.",
                "hard_fail": "Unsafe/unusable because it leaks hidden evidence, hallucinates important clinical facts, contradicts the profile, or breaks patient persona.",
            },
        },
        "rubric": RUBRIC_DIMENSIONS,
        "allowed_error_tags": ERROR_TAGS,
        "hard_fail_rules": [
            "Semantic leakage of withheld, removed, hidden_observed, or forbidden evidence is hard_fail unless the same information is also explicitly present in allowed_retained_direct or allowed_weakened_vague_or_softened.",
            "Any unsupported self-harm/suicide detail, diagnosis, exact duration, behavior, medication, hospitalization, or trauma detail is hard_fail.",
            "Contradicting the hidden profile is hard_fail.",
            "Speaking as a clinician, evaluator, simulator, policy, or metadata source is hard_fail if severe; otherwise at most soft_fail.",
        ],
        "output_format": {
            "scores": {key: "integer 1-5" for key in RUBRIC_DIMENSIONS},
            "overall_score": "integer 1-5",
            "pass_fail": "pass | soft_fail | hard_fail",
            "error_tags": "list of allowed_error_tags",
            "brief_rationale": "one concise Chinese paragraph",
        },
    }
    system = (
        "You are a strict expert judge for a controlled psychiatric patient simulator. "
        "Evaluate whether the patient response is grounded, disclosure-controlled, natural, and responsive. "
        "Be especially strict about semantic leakage and unsupported clinical facts. "
        "Return only valid JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def selected_records(
    records: list[dict[str, Any]],
    *,
    max_requests: int | None,
    max_per_bucket: int | None,
    seed: int | None,
    require_realizer_mode: set[str] | None,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record.get("doctor_question")
        and record.get("patient_response")
        and (not require_realizer_mode or str(record.get("patient_realizer_mode")) in require_realizer_mode)
    ]
    rng = random.Random(seed)
    if max_per_bucket and max_per_bucket > 0:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in candidates:
            buckets[(str(record.get("base_severity")), str(record.get("patient_realizer_mode")))].append(record)
        selected: list[dict[str, Any]] = []
        for key in sorted(buckets):
            bucket = buckets[key]
            rng.shuffle(bucket)
            selected.extend(bucket[:max_per_bucket])
        rng.shuffle(selected)
        candidates = selected
    elif seed is not None:
        rng.shuffle(candidates)
    if max_requests is not None:
        candidates = candidates[:max_requests]
    return candidates


def build_requests(
    records: list[dict[str, Any]],
    *,
    max_requests: int | None,
    max_per_bucket: int | None,
    seed: int | None,
    require_realizer_mode: set[str] | None,
) -> list[dict[str, Any]]:
    requests = []
    for record in selected_records(
        records,
        max_requests=max_requests,
        max_per_bucket=max_per_bucket,
        seed=seed,
        require_realizer_mode=require_realizer_mode,
    ):
        requests.append(
            {
                "request_id": f"{record.get('record_id')}::patient_realizer_rubric_judge_v3_2",
                "task_name": "mdd5k_patient_realizer_rubric_judge",
                "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
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
        "# Patient Realizer Rubric V3.2",
        "",
        "This rubric evaluates the patient simulator response, not the doctor policy.",
        "It complements deterministic verifier gates; it does not replace them.",
        "",
        "## Dimensions",
        "",
    ]
    for name, item in RUBRIC_DIMENSIONS.items():
        lines.extend([f"### {name}", "", item["question"], ""])
        for score in range(1, 6):
            lines.append(f"- {score}: {item[f'score_{score}']}")
        lines.append("")
    lines.extend(["## Error Tags", "", "`" + "`, `".join(ERROR_TAGS) + "`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare closed-LLM rubric v3.2 judge requests for patient-simulator responses.")
    parser.add_argument("--trajectory-path", type=Path, default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-requests", type=int, default=200, help="<=0 means all selected records.")
    parser.add_argument("--max-per-bucket", type=int, default=0, help="Optional cap per (base_severity, patient_realizer_mode) before max-requests.")
    parser.add_argument("--sample-seed", type=int, default=31)
    parser.add_argument(
        "--require-realizer-mode",
        default="",
        help="Comma-separated patient_realizer_mode values to keep, e.g. verified_llm_cache. Empty keeps all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(args.trajectory_path))
    max_requests = args.max_requests if args.max_requests > 0 else None
    max_per_bucket = args.max_per_bucket if args.max_per_bucket > 0 else None
    require_realizer_mode = {item.strip() for item in args.require_realizer_mode.split(",") if item.strip()} or None
    requests = build_requests(
        records,
        max_requests=max_requests,
        max_per_bucket=max_per_bucket,
        seed=args.sample_seed,
        require_realizer_mode=require_realizer_mode,
    )
    request_path = args.output_dir / "mdd5k_patient_realizer_rubric_judge_requests.jsonl"
    summary_path = args.output_dir / "mdd5k_patient_realizer_rubric_judge_request_summary.json"
    rubric_path = args.output_dir / "PATIENT_REALIZER_RUBRIC_V3_2.md"
    write_jsonl(request_path, requests)
    write_rubric(rubric_path)
    summary = {
        "trajectory_path": str(args.trajectory_path),
        "request_path": str(request_path),
        "rubric_path": str(rubric_path),
        "num_source_records": len(records),
        "num_requests": len(requests),
        "by_source_realizer_mode": dict(Counter(str(record.get("patient_realizer_mode")) for record in records)),
        "by_request_realizer_mode": dict(Counter(str(request.get("patient_realizer_mode")) for request in requests)),
        "by_request_severity": dict(Counter(str(request.get("base_severity")) for request in requests)),
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "sample_seed": args.sample_seed,
        "max_requests": args.max_requests,
        "max_per_bucket": args.max_per_bucket,
        "require_realizer_mode": sorted(require_realizer_mode) if require_realizer_mode else [],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
