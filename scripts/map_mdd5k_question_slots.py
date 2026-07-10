from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = BASE_DIR / "schemas" / "mdd5k_symptom_slot_schema.json"
DEFAULT_INPUT_PATH = (
    BASE_DIR
    / "outputs_mdd5k_normalized"
    / "mdd5k_normalized_dialogue_variants.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_question_slot_mapping"

CRITICALITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

FRAME_CUES = [
    "我想问一下",
    "想问一下",
    "我想了解",
    "想进一步了解",
    "我想确认",
    "我想知道",
    "接下来想问",
    "能跟我",
    "能不能跟我",
    "可以跟我",
    "请你",
]

QUESTION_MARK_PATTERN = re.compile(r"[^。！？!?]*[！？!?]")

BROAD_KEYWORD_FILTERS = {
    "school_or_study_status": {
        "broad_keywords": {"学习", "学校", "上课", "学业"},
        "support_keywords": {
            "成绩",
            "作业",
            "考试",
            "注意力",
            "集中",
            "完成",
            "困难",
            "影响",
            "下降",
            "逃避",
            "跟上",
            "效率",
            "班主任",
            "老师",
        },
    },
    "work_status": {
        "broad_keywords": {"工作", "上班"},
        "support_keywords": {
            "同事",
            "老板",
            "请假",
            "离职",
            "绩效",
            "完成",
            "困难",
            "影响",
            "压力",
            "效率",
            "迟到",
            "缺勤",
        },
    },
    "romantic_status": {
        "broad_keywords": {"关系", "情感"},
        "support_keywords": {
            "恋爱",
            "婚恋",
            "伴侣",
            "男朋友",
            "女朋友",
            "丈夫",
            "妻子",
            "分手",
            "感情",
        },
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    return re.sub(r"\s+", "", text)


def extract_question_focus(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    cue_positions = []
    for cue in FRAME_CUES:
        pos = text.rfind(cue)
        if pos >= 0:
            cue_positions.append(pos)
    if cue_positions:
        return text[min(cue_positions) :].strip()

    question_sentences = QUESTION_MARK_PATTERN.findall(text)
    if question_sentences:
        return "".join(question_sentences).strip()

    return text


def recover_tree_metadata(dialogue: dict[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    for turn in dialogue.get("dialogue_turns", []):
        text_parts.append(str(turn.get("doctor_utterance") or ""))
        text_parts.append(str(turn.get("patient_utterance") or ""))
    group_text = "\n".join(text_parts)
    age, age_group, age_confidence, age_source = recover_age(group_text)
    gender, gender_confidence, gender_source = recover_gender(group_text)
    tree_type = tree_type_from(gender, age_group)
    if tree_type == "unknown":
        tree_confidence = "low"
    elif "low" in {age_confidence, gender_confidence}:
        tree_confidence = "low"
    elif "medium" in {age_confidence, gender_confidence}:
        tree_confidence = "medium"
    else:
        tree_confidence = "high"

    return {
        "age": age,
        "age_group": age_group,
        "age_confidence": age_confidence,
        "age_source": age_source,
        "gender": gender,
        "gender_confidence": gender_confidence,
        "gender_source": gender_source,
        "tree_type": tree_type,
        "tree_type_confidence": tree_confidence,
        "metadata_source": "recovered_from_normalized_dialogue_variant",
    }


def recover_age(text: str) -> tuple[int | None, str, str, str]:
    explicit = re.search(r"(\d{1,3})\s*岁", text)
    if explicit:
        age = int(explicit.group(1))
        return age, "teen" if age < 18 else "adult", "high", "explicit_age"

    teen_terms = [
        "高一",
        "高二",
        "高三",
        "初一",
        "初二",
        "初三",
        "学生",
        "班主任",
        "老师",
        "父母知情",
        "学校",
        "学业",
    ]
    adult_terms = [
        "上班",
        "工作",
        "同事",
        "老板",
        "丈夫",
        "妻子",
        "婚姻",
        "婚恋",
        "孩子",
        "职场",
    ]
    teen_score = sum(text.count(term) for term in teen_terms)
    adult_score = sum(text.count(term) for term in adult_terms)
    if teen_score > adult_score and teen_score > 0:
        return None, "teen", "medium", "school_stage_terms"
    if adult_score > teen_score and adult_score > 0:
        return None, "adult", "medium", "adult_role_terms"
    return None, "unknown", "low", "unknown"


def recover_gender(text: str) -> tuple[str, str, str]:
    female_terms = ["女性", "女生", "女孩", "女儿", "妻子", "月经"]
    male_terms = ["男性", "男生", "男孩", "儿子", "丈夫"]
    female_score = sum(text.count(term) for term in female_terms)
    male_score = sum(text.count(term) for term in male_terms)
    if female_score > male_score and female_score > 0:
        source = "menstrual_slot" if "月经" in text else "gender_role_terms"
        confidence = "high" if any(term in text for term in ["女性", "女生", "月经"]) else "medium"
        return "female", confidence, source
    if male_score > female_score and male_score > 0:
        confidence = "high" if any(term in text for term in ["男性", "男生"]) else "medium"
        return "male", confidence, "gender_role_terms"
    return "unknown", "low", "unknown"


def tree_type_from(gender: str, age_group: str) -> str:
    if gender in {"female", "male"} and age_group in {"teen", "adult"}:
        return f"{gender}_{age_group}"
    return "unknown"


def active_slot_names(
    tree_type: str,
    tree_space: dict[str, list[str]],
    all_slot_names: list[str],
) -> tuple[list[str], str]:
    active = tree_space.get(tree_type) or []
    if active:
        return list(active), "tree_type_specific"
    return list(all_slot_names), "unknown_tree_union_fallback"


def match_slot(question: str, slot_def: dict[str, Any]) -> dict[str, Any] | None:
    normalized_question = normalize_for_match(question)
    matched = []
    for keyword in slot_def.get("question_keywords", []):
        normalized_keyword = normalize_for_match(keyword)
        if not normalized_keyword:
            continue
        start = normalized_question.find(normalized_keyword)
        if start >= 0:
            matched.append({"keyword": keyword, "start": start})

    if not matched:
        return None

    slot_name = slot_def["slot"]
    filter_rule = BROAD_KEYWORD_FILTERS.get(slot_name)
    if filter_rule:
        matched_keywords = {item["keyword"] for item in matched}
        broad_keywords = filter_rule["broad_keywords"]
        support_keywords = filter_rule["support_keywords"]
        has_only_broad_keywords = matched_keywords <= broad_keywords
        has_support_keyword = any(
            normalize_for_match(keyword) in normalized_question
            for keyword in support_keywords
        )
        if has_only_broad_keywords and not has_support_keyword:
            return None

    matched.sort(key=lambda item: (item["start"], -len(item["keyword"])))
    return {
        "slot": slot_name,
        "zh_name": slot_def.get("zh_name"),
        "criticality": slot_def.get("criticality", "medium"),
        "evidence_unit_hints": slot_def.get("evidence_unit_hints", []),
        "matched_keywords": [item["keyword"] for item in matched],
        "first_match_char": matched[0]["start"],
        "rule_score": len(matched),
    }


def rank_slot_match(match: dict[str, Any]) -> tuple[int, int, int, str]:
    criticality = CRITICALITY_RANK.get(match.get("criticality", "medium"), 2)
    return (
        int(match.get("first_match_char", 10**9)),
        -int(match.get("rule_score", 0)),
        -criticality,
        match.get("slot", ""),
    )


def map_question(question: str, active_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for slot_def in active_slots:
        matched = match_slot(question, slot_def)
        if matched:
            matches.append(matched)
    matches.sort(key=rank_slot_match)
    return matches


def build_mapping_records(input_path: Path, schema: dict[str, Any]):
    all_slots = schema.get("slots", [])
    slots_by_name = {slot["slot"]: slot for slot in all_slots}
    all_slot_names = [slot["slot"] for slot in all_slots]
    tree_space = schema.get("tree_space", {})

    for dialogue in iter_jsonl(input_path):
        tree_metadata = recover_tree_metadata(dialogue)
        active_names, active_slot_source = active_slot_names(
            tree_metadata["tree_type"], tree_space, all_slot_names
        )
        active_slots = [
            slots_by_name[name]
            for name in active_names
            if name in slots_by_name
        ]

        for turn in dialogue.get("dialogue_turns", []):
            doctor_utterance = turn.get("doctor_utterance", "")
            question_focus_text = extract_question_focus(doctor_utterance)
            target_slots = map_question(question_focus_text, active_slots)
            primary_slot = target_slots[0]["slot"] if target_slots else None
            if not target_slots:
                mapping_status = "unmapped"
            elif len(target_slots) == 1:
                mapping_status = "mapped"
            else:
                mapping_status = "multi_slot"

            yield {
                "dialogue_id": dialogue.get("dialogue_id"),
                "case_id": dialogue.get("case_id"),
                "variant_id": dialogue.get("variant_id"),
                "diagnosis": dialogue.get("diagnosis"),
                "diagnoses": dialogue.get("diagnoses", []),
                "icd_code": dialogue.get("icd_code"),
                "icd_codes": dialogue.get("icd_codes", []),
                "tree_metadata": tree_metadata,
                "active_tree_type": tree_metadata["tree_type"],
                "active_slot_source": active_slot_source,
                "active_tree_slots": active_names,
                "turn_id": turn.get("turn_id"),
                "doctor_utterance": doctor_utterance,
                "question_focus_text": question_focus_text,
                "patient_utterance": turn.get("patient_utterance", ""),
                "target_slots": target_slots,
                "primary_slot": primary_slot,
                "mapping_status": mapping_status,
            }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_slot_distribution_csv(path: Path, slot_counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slot", "turn_count"])
        for slot, count in slot_counts.most_common():
            writer.writerow([slot, count])


def summarize(records: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    total_turns = len(records)
    mapped_records = [r for r in records if r["mapping_status"] != "unmapped"]
    multi_slot_records = [r for r in records if r["mapping_status"] == "multi_slot"]
    unique_questions = Counter(r["doctor_utterance"] for r in records)
    unique_mapped_questions = {
        r["doctor_utterance"] for r in records if r["mapping_status"] != "unmapped"
    }

    slot_counts = Counter()
    primary_slot_counts = Counter()
    criticality_counts = Counter()
    tree_type_counts = Counter()
    active_slot_source_counts = Counter()
    tree_confidence_counts = Counter()
    for record in records:
        tree_type_counts[record["active_tree_type"]] += 1
        active_slot_source_counts[record["active_slot_source"]] += 1
        tree_confidence_counts[
            record["tree_metadata"].get("tree_type_confidence", "unknown")
        ] += 1
        if record["primary_slot"]:
            primary_slot_counts[record["primary_slot"]] += 1
        for slot_match in record["target_slots"]:
            slot_counts[slot_match["slot"]] += 1
            criticality_counts[slot_match.get("criticality", "medium")] += 1

    unmapped_questions = Counter(
        r["doctor_utterance"] for r in records if r["mapping_status"] == "unmapped"
    )

    def pct(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    return {
        "schema_name": schema.get("schema_name"),
        "schema_version": schema.get("version"),
        "n_tree_types": len(schema.get("tree_space", {})),
        "n_union_slots": len(schema.get("slots", [])),
        "total_turns": total_turns,
        "mapped_turns": len(mapped_records),
        "unmapped_turns": total_turns - len(mapped_records),
        "multi_slot_turns": len(multi_slot_records),
        "mapping_coverage": pct(len(mapped_records), total_turns),
        "multi_slot_rate": pct(len(multi_slot_records), total_turns),
        "unique_doctor_questions": len(unique_questions),
        "unique_mapped_doctor_questions": len(unique_mapped_questions),
        "unique_question_coverage": pct(
            len(unique_mapped_questions), len(unique_questions)
        ),
        "tree_type_counts": dict(tree_type_counts.most_common()),
        "tree_type_confidence_counts": dict(tree_confidence_counts.most_common()),
        "active_slot_source_counts": dict(active_slot_source_counts.most_common()),
        "slot_turn_counts": dict(slot_counts.most_common()),
        "primary_slot_counts": dict(primary_slot_counts.most_common()),
        "criticality_counts": dict(criticality_counts.most_common()),
        "top_unmapped_doctor_questions": [
            {"doctor_utterance": question, "count": count}
            for question, count in unmapped_questions.most_common(30)
        ],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MDD-5K Tree-Aware Question-to-Slot Mapping Report",
        "",
        "## Scope",
        "",
        "- Input: normalized MDD-5K dialogue variants.",
        "- Method: recover `tree_type`, activate the corresponding MDD-5K diagnosis-tree slots, extract question-focus text, and run deterministic keyword alignment within the active tree.",
        "- Purpose: create an executable bridge from doctor questions to MDD-5K diagnosis-tree nodes for the controlled low-informativeness patient environment.",
        "- Note: `evidence_unit_hints` are mapping/extraction hints, not evidence-gate supervision labels.",
        "",
        "## Coverage",
        "",
        f"- Union tree slots: {summary['n_union_slots']}",
        f"- Tree types: {summary['n_tree_types']}",
        f"- Total turns: {summary['total_turns']}",
        f"- Mapped turns: {summary['mapped_turns']} ({summary['mapping_coverage']:.2%})",
        f"- Unmapped turns: {summary['unmapped_turns']}",
        f"- Multi-slot turns: {summary['multi_slot_turns']} ({summary['multi_slot_rate']:.2%})",
        f"- Unique doctor questions: {summary['unique_doctor_questions']}",
        f"- Unique mapped doctor questions: {summary['unique_mapped_doctor_questions']} ({summary['unique_question_coverage']:.2%})",
        "",
        "## Tree-Type Recovery",
        "",
        "| Tree Type | Turn Count |",
        "|---|---:|",
    ]
    for tree_type, count in summary["tree_type_counts"].items():
        lines.append(f"| {tree_type} | {count} |")

    lines.extend(
        [
            "",
            "## Active Slot Source",
            "",
            "| Source | Turn Count |",
            "|---|---:|",
        ]
    )
    for source, count in summary["active_slot_source_counts"].items():
        lines.append(f"| {source} | {count} |")

    lines.extend(
        [
            "",
            "## Slot Distribution",
            "",
            "| Slot | Turn Count |",
            "|---|---:|",
        ]
    )
    for slot, count in summary["slot_turn_counts"].items():
        lines.append(f"| {slot} | {count} |")

    lines.extend(
        [
            "",
            "## Top Unmapped Questions",
            "",
            "| Count | Doctor Utterance |",
            "|---:|---|",
        ]
    )
    for item in summary["top_unmapped_doctor_questions"]:
        utterance = item["doctor_utterance"].replace("|", "\\|")
        lines.append(f"| {item['count']} | {utterance} |")

    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "The mapped tree-node IDs should be used by the patient-side controller to decide which required diagnostic elements are being requested at each turn. For known tree types, the controller should operate only over active tree slots. For `unknown` tree type, the current script uses a union-tree fallback and flags it via `active_slot_source`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map MDD-5K doctor questions to active diagnosis-tree slots."
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_json(args.schema)
    slots = schema.get("slots", [])
    if not slots:
        raise ValueError(f"No slots found in schema: {args.schema}")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(build_mapping_records(args.input, schema))
    summary = summarize(records, schema)

    mapping_path = args.output_dir / "mdd5k_question_slot_map_rule.jsonl"
    summary_path = args.output_dir / "mdd5k_question_slot_mapping_summary.json"
    distribution_path = args.output_dir / "mdd5k_question_slot_distribution.csv"
    report_path = args.output_dir / "MDD5K_QUESTION_SLOT_MAPPING_REPORT.md"

    write_jsonl(mapping_path, records)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    write_slot_distribution_csv(
        distribution_path, Counter(summary["slot_turn_counts"])
    )
    write_report(report_path, summary)

    print(f"Wrote {len(records)} mapped turn records to {mapping_path}")
    print(f"Coverage: {summary['mapped_turns']}/{summary['total_turns']} = {summary['mapping_coverage']:.2%}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
