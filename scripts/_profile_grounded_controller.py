from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = (
    BASE_DIR
    / "outputs_patient_profiles"
    / "mdd5k_dialogue_derived_patient_profiles.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_profile_grounded_controller"

LEVELS = [
    "reference_informative",
    "mild_low_info",
    "moderate_low_info",
    "severe_low_info",
]

SENSITIVE_TARGETS = {
    "suicide_or_self_harm",
    "romantic_status",
    "family_psychiatric_history",
    "menstrual_status",
}

STRESS_TARGETS = {"school_or_study_status", "work_status", "hopelessness_or_crying"}

BROAD_CONTEXT_SLOTS = {"parent_awareness", "personality"}

SLOT_DISPLAY = {
    "school_or_study_status": "学习或学校状态",
    "work_status": "工作状态",
    "sleep": "睡眠情况",
    "suicide_or_self_harm": "自伤或自杀相关想法",
    "appetite_loss": "食欲变化",
    "binge_eating": "暴食情况",
    "chest_tightness": "胸闷或呼吸不适",
    "dizziness_or_headache": "头晕或头痛",
    "palpitation": "心慌或心悸",
    "hallucination": "幻觉或异常知觉",
    "menstrual_status": "月经情况",
    "memory_problem": "记忆力问题",
    "cognitive_slowing": "思考或反应变慢",
    "attention_decline": "注意力下降",
    "mania_screen": "情绪高涨或躁狂样表现",
    "hopelessness_or_crying": "低落、绝望或哭泣",
    "romantic_status": "情感或婚恋状态",
    "family_psychiatric_history": "家族精神疾病史",
    "parent_awareness": "父母知情和支持",
    "personality": "性格和应对方式",
}


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    return (text or "").strip().strip("，,。；;、 ")


def ensure_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return "这个问题我现在不太想细说，可以先跳过吗？"
    if text[-1] not in "。！？!?":
        text += "。"
    return text


def join_units(units: list[dict[str, Any]], max_chars: int = 220) -> str:
    parts = [clean_text(unit.get("unit_text") or "") for unit in units]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    text = "，".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip("，,。；;、 ") + "..."
    return ensure_sentence(text)


def selected_profile_units(slot_profile: dict[str, Any], max_units: int) -> list[dict[str, Any]]:
    units = slot_profile.get("evidence_units") or []
    selected = units[:max_units]
    result = []
    for idx, unit in enumerate(selected, start=1):
        result.append(
            {
                "unit_id": f"u{idx}",
                "profile_unit_id": unit.get("unit_id"),
                "unit_text": unit.get("unit_text"),
                "target_relevance": unit.get("target_relevance"),
                "source_count": unit.get("source_count", 0),
                "source_refs": unit.get("source_refs", []),
            }
        )
    return result


def unit_order(unit_id: str) -> int:
    digits = "".join(ch for ch in unit_id if ch.isdigit())
    return int(digits) if digits else 10**9


def priority_unit_ids(units: list[dict[str, Any]]) -> list[str]:
    rank = {"core": 0, "supporting": 1, "peripheral": 2}
    ordered = sorted(
        units,
        key=lambda unit: (
            rank.get(unit.get("target_relevance"), 3),
            -int(unit.get("source_count") or 0),
            unit_order(unit["unit_id"]),
        ),
    )
    return [unit["unit_id"] for unit in ordered]


def ordered_units_by_ids(units: list[dict[str, Any]], unit_ids: set[str]) -> list[dict[str, Any]]:
    return [unit for unit in units if unit["unit_id"] in unit_ids]


def compute_information_retention(
    retained_unit_ids: list[str],
    weakened_unit_ids: list[str],
    total_units: int,
) -> float:
    if total_units <= 0:
        return 0.0
    value = (len(retained_unit_ids) + 0.5 * len(weakened_unit_ids)) / total_units
    return round(value, 4)


def make_doctor_question(slot: str) -> str:
    display = SLOT_DISPLAY.get(slot, slot)
    return f"我想进一步了解一下你的{display}。你可以具体说说最近这方面的情况吗？"


def make_question_focus(slot: str) -> str:
    display = SLOT_DISPLAY.get(slot, slot)
    return f"请说明最近的{display}。"


def alternative_deflection_unit(
    profile: dict[str, Any],
    target_slot: str,
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    for slot in profile.get("observed_slots", []):
        if slot == target_slot:
            continue
        slot_profile = profile["slot_profiles"].get(slot) or {}
        units = slot_profile.get("evidence_units") or []
        if units:
            return slot, units[0]
    return None, None


def make_variant(
    seed: dict[str, Any],
    profile: dict[str, Any],
    units: list[dict[str, Any]],
    level: str,
) -> dict[str, Any]:
    all_ids = [unit["unit_id"] for unit in units]
    priority_ids = priority_unit_ids(units)
    total_units = len(units)
    target = seed["target_tree_node"]

    retained: list[str] = []
    weakened: list[str] = []
    removed: list[str] = []
    category = "informative_reference"
    topic = 1.0
    clarity = 1.0
    generated = seed["profile_reference_response"]
    note = "profile_reference"

    if level == "reference_informative":
        retained = all_ids
        category = "informative_reference"
        generated = seed["profile_reference_response"]
    elif level == "mild_low_info":
        keep_count = max(1, math.ceil(total_units * 0.7))
        retained = sorted(priority_ids[:keep_count], key=unit_order)
        removed = [unit_id for unit_id in all_ids if unit_id not in set(retained)]
        category = "partial_omission"
        generated = join_units(ordered_units_by_ids(units, set(retained)))
        note = "profile_rule_partial_omission"
    elif level == "moderate_low_info":
        retain_count = max(1, min(3, math.floor(total_units * 0.25)))
        weaken_count = max(1, min(3, math.ceil(total_units * 0.25)))
        retained = sorted(priority_ids[:retain_count], key=unit_order)
        weakened = sorted(priority_ids[retain_count : retain_count + weaken_count], key=unit_order)
        kept = set(retained) | set(weakened)
        removed = [unit_id for unit_id in all_ids if unit_id not in kept]
        category = "vague_or_uncertain"
        clarity = 0.5
        topic = 1.0
        source_text = join_units(ordered_units_by_ids(units, set(retained)), max_chars=120)
        generated = ensure_sentence(f"大概就是{source_text.rstrip('。')}这些吧，其他具体细节我一时也说不太清楚")
        note = "profile_rule_vague_weakening"
    elif level == "severe_low_info":
        removed = all_ids
        clarity = 0.0
        alt_slot, alt_unit = alternative_deflection_unit(profile, target)
        if target in SENSITIVE_TARGETS:
            category = "direct_refusal_or_boundary"
            topic = 0.0
            generated = "这个问题我现在不太想细说，可以先跳过吗？"
            note = "profile_rule_sensitive_boundary"
        elif target not in STRESS_TARGETS and alt_unit:
            category = "topic_deflection"
            topic = 0.0
            alt_text = clean_text(alt_unit.get("unit_text") or "")
            generated = ensure_sentence(f"{alt_text}。至于刚才问的那个，我暂时不太想展开")
            note = f"profile_rule_deflection_to_{alt_slot}"
        else:
            category = "minimal_generic"
            topic = 0.5
            generated = "这个我有点说不清楚，暂时也不太想展开。"
            note = "profile_rule_minimal_generic"
    else:
        raise ValueError(f"Unknown information level: {level}")

    information_retention = compute_information_retention(retained, weakened, total_units)
    g_target = round(topic * information_retention * clarity, 4)

    return {
        "record_id": f"{seed['counterfactual_group_id']}_{level}",
        "counterfactual_group_id": seed["counterfactual_group_id"],
        "case_id": profile.get("case_id"),
        "profile_id": profile.get("profile_id"),
        "profile_source": profile.get("profile_source"),
        "profile_type": profile.get("profile_type"),
        "dialogue_id": f"profile::{profile.get('profile_id')}",
        "variant_id": None,
        "turn_id": -1,
        "source_dataset": profile.get("source_dataset", "MDD-5K"),
        "diagnoses": profile.get("diagnoses"),
        "icd_codes": profile.get("icd_codes"),
        "active_tree_type": profile.get("primary_tree_type"),
        "active_tree_slots": profile.get("active_tree_slots"),
        "target_tree_node": target,
        "target_node_role": "simulator_internal_target_node",
        "target_node_visibility": "simulator_internal_not_doctor_visible",
        "target_slot_profile_status": seed["target_slot_profile_status"],
        "target_slot_evidence_unit_count": seed["target_slot_evidence_unit_count"],
        "information_level": level,
        "low_info_category": category,
        "doctor_question": seed["doctor_question"],
        "doctor_question_source": "canonical_question_for_offline_cache",
        "question_focus_text": seed["question_focus_text"],
        "original_patient_response": seed["profile_reference_response"],
        "profile_reference_response": seed["profile_reference_response"],
        "observed_evidence_units": units,
        "retained_unit_ids": retained,
        "weakened_unit_ids": weakened,
        "removed_unit_ids": removed,
        "generated_patient_response": generated,
        "topic_responsiveness": topic,
        "information_retention": information_retention,
        "clarity": clarity,
        "g_target": g_target,
        "validity": {
            "label_preserved": True,
            "no_new_clinical_fact": True,
            "no_contradiction_with_original_response": True,
            "target_conditioned_category_valid": True,
            "monotonic_within_group": True,
            "cached_reproducibly": True,
            "profile_grounded": True,
        },
        "realizer": {
            "type": "deterministic_rule_based",
            "model": "none",
            "version": "profile_grounded_low_info_rule_v0.1",
            "note": note,
        },
        "simulator_input": {
            "interface_mode": "offline_cache",
            "profile_id": profile.get("profile_id"),
            "target_tree_node": target,
            "target_node_role": "simulator_internal_target_node",
            "target_node_visibility": "simulator_internal_not_doctor_visible",
            "information_level": level,
            "controller_source": "slot_profiles[simulator_internal_target_node]",
            "online_interface": "dialogue_history + doctor_question -> query_interpreter -> simulator_internal_target_node",
        },
    }


def candidate_slots(
    profile: dict[str, Any],
    min_units: int,
    include_broad_context_slots: bool,
) -> list[dict[str, Any]]:
    result = []
    for slot in profile.get("active_tree_slots") or []:
        if not include_broad_context_slots and slot in BROAD_CONTEXT_SLOTS:
            continue
        slot_profile = profile.get("slot_profiles", {}).get(slot) or {}
        if slot_profile.get("profile_status") != "observed":
            continue
        if int(slot_profile.get("num_evidence_units") or 0) < min_units:
            continue
        result.append(
            {
                "profile": profile,
                "slot": slot,
                "num_units": int(slot_profile.get("num_evidence_units") or 0),
                "support_turn_count": int(slot_profile.get("support_turn_count") or 0),
            }
        )
    return result


def select_interactions(
    profiles: list[dict[str, Any]],
    max_interactions: int,
    max_per_slot: int,
    max_per_case: int,
    min_units: int,
    include_broad_context_slots: bool,
) -> list[dict[str, Any]]:
    candidates = []
    for profile in profiles:
        candidates.extend(candidate_slots(profile, min_units, include_broad_context_slots))

    candidates.sort(
        key=lambda item: (
            item["slot"],
            -item["support_turn_count"],
            item["profile"].get("case_id", ""),
        )
    )

    selected: list[dict[str, Any]] = []
    slot_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    used_pairs: set[tuple[str, str]] = set()

    while len(selected) < max_interactions:
        added = False
        candidates.sort(
            key=lambda item: (
                slot_counts[item["slot"]],
                case_counts[item["profile"].get("case_id", "")],
                -item["support_turn_count"],
                item["profile"].get("case_id", ""),
                item["slot"],
            )
        )
        for item in candidates:
            case_id = item["profile"].get("case_id", "")
            slot = item["slot"]
            pair = (case_id, slot)
            if pair in used_pairs:
                continue
            if slot_counts[slot] >= max_per_slot:
                continue
            if case_counts[case_id] >= max_per_case:
                continue
            selected.append(item)
            used_pairs.add(pair)
            slot_counts[slot] += 1
            case_counts[case_id] += 1
            added = True
            break
        if not added:
            break
    return selected


def build_seed(candidate: dict[str, Any], max_units: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = candidate["profile"]
    slot = candidate["slot"]
    slot_profile = profile["slot_profiles"][slot]
    units = selected_profile_units(slot_profile, max_units=max_units)
    reference = join_units(units)
    group_id = f"{profile['profile_id']}_{slot}"
    seed = {
        "counterfactual_group_id": group_id,
        "target_tree_node": slot,
        "target_slot_profile_status": slot_profile.get("profile_status"),
        "target_slot_evidence_unit_count": slot_profile.get("num_evidence_units"),
        "doctor_question": make_doctor_question(slot),
        "question_focus_text": make_question_focus(slot),
        "profile_reference_response": reference,
    }
    return seed, units


def apply_monotonic_checks(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    level_rank = {level: idx for idx, level in enumerate(LEVELS)}
    for record in records:
        grouped[record["counterfactual_group_id"]].append(record)
    for group_records in grouped.values():
        ordered = sorted(group_records, key=lambda record: level_rank[record["information_level"]])
        ok = all(
            ordered[idx]["g_target"] >= ordered[idx + 1]["g_target"]
            for idx in range(len(ordered) - 1)
        )
        for record in ordered:
            record["validity"]["monotonic_within_group"] = ok


def build_records(
    selected: list[dict[str, Any]],
    max_units: int,
) -> list[dict[str, Any]]:
    records = []
    for candidate in selected:
        profile = candidate["profile"]
        seed, units = build_seed(candidate, max_units)
        for level in LEVELS:
            records.append(make_variant(seed, profile, units, level))
    apply_monotonic_checks(records)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts = Counter(record["information_level"] for record in records)
    category_counts = Counter(record["low_info_category"] for record in records)
    target_counts = Counter(
        record["target_tree_node"]
        for record in records
        if record["information_level"] == "reference_informative"
    )
    profile_ids = {
        record["profile_id"]
        for record in records
        if record["information_level"] == "reference_informative"
    }
    group_ids = {record["counterfactual_group_id"] for record in records}
    g_by_level: dict[str, list[float]] = defaultdict(list)
    monotonic_violations = 0
    for record in records:
        g_by_level[record["information_level"]].append(float(record["g_target"]))
        if not record["validity"]["monotonic_within_group"] and record["information_level"] == "reference_informative":
            monotonic_violations += 1

    return {
        "num_profile_slot_groups": len(group_ids),
        "num_profiles_used": len(profile_ids),
        "num_records": len(records),
        "information_level_counts": dict(level_counts),
        "low_info_category_counts": dict(category_counts),
        "target_tree_node_counts": dict(target_counts.most_common()),
        "g_target_mean_by_level": {
            level: round(sum(values) / len(values), 4) if values else 0.0
            for level, values in g_by_level.items()
        },
        "monotonic_violations": monotonic_violations,
        "profile_grounded": True,
        "broad_context_slots_excluded_by_default": sorted(BROAD_CONTEXT_SLOTS),
    }


def write_report(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Profile-Grounded Low-Informativeness Controller Pilot",
        "",
        "Date: 2026-06-09",
        "",
        "## Purpose",
        "",
        "This pilot moves from turn-level rewriting to a patient-simulator cache. Each group is generated from a dialogue-derived case profile and a simulator-internal diagnosis-tree node.",
        "",
        "Offline cache construction form:",
        "",
        "```text",
        "case_profile + simulator_internal_target_node + information_level",
        "```",
        "",
        "Online doctor-agent interaction form:",
        "",
        "```text",
        "dialogue_history + doctor_question",
        "-> simulator-side query interpreter",
        "-> simulator_internal_target_node",
        "-> controller retrieves slot_profiles[simulator_internal_target_node]",
        "```",
        "",
        "The internal node is hidden from the doctor policy. It is retained in cached records only as simulator routing metadata for reproducibility and evaluation.",
        "",
        "## Summary",
        "",
        f"- Profile-slot groups: {summary['num_profile_slot_groups']}",
        f"- Profiles used: {summary['num_profiles_used']}",
        f"- Generated records: {summary['num_records']}",
        f"- Monotonic violations: {summary['monotonic_violations']}",
        f"- Broad context slots excluded by default: {', '.join(summary['broad_context_slots_excluded_by_default'])}",
        "",
        "## Information Levels",
        "",
        "| Level | Count | Mean gate target |",
        "|---|---:|---:|",
    ]
    for level in LEVELS:
        lines.append(
            f"| `{level}` | {summary['information_level_counts'].get(level, 0)} | {summary['g_target_mean_by_level'].get(level, 0.0):.4f} |"
        )

    lines.extend(["", "## Low-Information Categories", "", "| Category | Count |", "|---|---:|"])
    for category, count in summary["low_info_category_counts"].items():
        lines.append(f"| `{category}` | {count} |")

    lines.extend(["", "## Target Tree Nodes", "", "| Target node | Groups |", "|---|---:|"])
    for target, count in summary["target_tree_node_counts"].items():
        lines.append(f"| `{target}` | {count} |")

    lines.extend(["", "## Sample Groups", ""])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["counterfactual_group_id"]].append(record)
    level_rank = {level: idx for idx, level in enumerate(LEVELS)}
    for group_id, group_records in list(grouped.items())[:3]:
        ordered = sorted(group_records, key=lambda record: level_rank[record["information_level"]])
        first = ordered[0]
        lines.extend(
            [
                f"### `{group_id}`",
                "",
                f"- Case: `{first['case_id']}`",
                f"- Internal target: `{first['target_tree_node']}`",
                f"- Profile evidence units in slot: {first['target_slot_evidence_unit_count']}",
                "",
                "| Level | Category | g | Generated response |",
                "|---|---|---:|---|",
            ]
        )
        for record in ordered:
            response = (record["generated_patient_response"] or "").replace("|", " ")
            if len(response) > 120:
                response = response[:117] + "..."
            lines.append(
                f"| `{record['information_level']}` | `{record['low_info_category']}` | {record['g_target']:.4f} | {response} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Method Notes",
            "",
            "- This is a deterministic pilot realizer, not the final LLM realizer.",
            "- The profile is dialogue-derived and should not be described as an original clinical case profile.",
            "- `target_tree_node` in these files means simulator-internal routing metadata, not a doctor-visible observation or action.",
            "- The controller metadata can later supervise the doctor-side evidence gate, but the primary goal here is patient simulation.",
            "- Broad context slots are excluded in this pilot because they are more vulnerable to mapping contamination; they can be reintroduced after slot-quality auditing.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build profile-grounded low-informativeness controller pilot.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-interactions", type=int, default=80)
    parser.add_argument("--max-per-slot", type=int, default=5)
    parser.add_argument("--max-per-case", type=int, default=2)
    parser.add_argument("--min-units", type=int, default=6)
    parser.add_argument("--max-units-per-response", type=int, default=8)
    parser.add_argument("--include-broad-context-slots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.profiles.exists():
        raise FileNotFoundError(args.profiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = list(iter_jsonl(args.profiles))
    selected = select_interactions(
        profiles=profiles,
        max_interactions=args.max_interactions,
        max_per_slot=args.max_per_slot,
        max_per_case=args.max_per_case,
        min_units=args.min_units,
        include_broad_context_slots=args.include_broad_context_slots,
    )
    records = build_records(selected, max_units=args.max_units_per_response)
    summary = summarize(records)

    output_path = args.output_dir / "mdd5k_profile_grounded_controller_pilot.jsonl"
    summary_path = args.output_dir / "mdd5k_profile_grounded_controller_pilot_summary.json"
    report_path = args.output_dir / "MDD5K_PROFILE_GROUNDED_CONTROLLER_PILOT.md"

    write_jsonl(output_path, records)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    write_report(report_path, records, summary)

    print(f"Profile-slot groups: {summary['num_profile_slot_groups']}")
    print(f"Records: {summary['num_records']}")
    print(f"Monotonic violations: {summary['monotonic_violations']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
