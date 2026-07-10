from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from map_mdd5k_question_slots import (
    active_slot_names,
    extract_question_focus,
    load_json,
    map_question,
    recover_tree_metadata,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = BASE_DIR / "schemas" / "mdd5k_symptom_slot_schema.json"
DEFAULT_NORMALIZED_PATH = (
    BASE_DIR
    / "outputs_mdd5k_normalized"
    / "mdd5k_normalized_dialogue_variants.jsonl"
)
DEFAULT_MAPPING_PATH = (
    BASE_DIR
    / "outputs_question_slot_mapping"
    / "mdd5k_question_slot_map_rule.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_query_interpreter"


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


def build_history(turns: list[dict[str, Any]], end_turn_id: int) -> list[dict[str, str]]:
    history = []
    for turn in turns:
        if int(turn.get("turn_id") or 0) >= end_turn_id:
            break
        history.append(
            {
                "doctor_utterance": turn.get("doctor_utterance") or "",
                "patient_utterance": turn.get("patient_utterance") or "",
            }
        )
    return history


class OnlineQueryInterpreter:
    """Map a doctor-visible natural-language question to a hidden simulator node."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.all_slots = schema.get("slots", [])
        self.slots_by_name = {slot["slot"]: slot for slot in self.all_slots}
        self.all_slot_names = [slot["slot"] for slot in self.all_slots]
        self.tree_space = schema.get("tree_space", {})

    def _tree_from_history(
        self,
        dialogue_history: list[dict[str, str]],
        doctor_question: str,
    ) -> dict[str, Any]:
        turns = list(dialogue_history)
        turns.append({"doctor_utterance": doctor_question, "patient_utterance": ""})
        return recover_tree_metadata({"dialogue_turns": turns})

    def _active_slots_for_tree(self, tree_type: str) -> tuple[list[str], str]:
        return active_slot_names(tree_type, self.tree_space, self.all_slot_names)

    def interpret(
        self,
        doctor_question: str,
        dialogue_history: list[dict[str, str]] | None = None,
        hidden_profile_tree_type: str | None = None,
    ) -> dict[str, Any]:
        dialogue_history = dialogue_history or []
        if hidden_profile_tree_type:
            tree_metadata = {
                "tree_type": hidden_profile_tree_type,
                "tree_type_confidence": "profile_hidden",
                "metadata_source": "hidden_profile_metadata",
            }
        else:
            tree_metadata = self._tree_from_history(dialogue_history, doctor_question)

        tree_type = tree_metadata.get("tree_type", "unknown")
        active_names, active_slot_source = self._active_slots_for_tree(tree_type)
        active_slots = [
            self.slots_by_name[name]
            for name in active_names
            if name in self.slots_by_name
        ]
        question_focus_text = extract_question_focus(doctor_question)
        matches = map_question(question_focus_text, active_slots)
        primary = matches[0]["slot"] if matches else None
        status = "unmapped" if not matches else "mapped" if len(matches) == 1 else "multi_slot"

        confidence = "none"
        if matches:
            if tree_metadata.get("tree_type_confidence") == "low":
                confidence = "low"
            elif len(matches) > 1:
                confidence = "medium"
            elif int(matches[0].get("rule_score") or 0) >= 2:
                confidence = "high"
            else:
                confidence = "medium"

        return {
            "doctor_question": doctor_question,
            "question_focus_text": question_focus_text,
            "simulator_internal_target_node": primary,
            "target_tree_node": primary,
            "target_node_role": "simulator_internal_target_node",
            "target_node_visibility": "simulator_internal_not_doctor_visible",
            "query_interpreter_status": status,
            "query_interpreter_confidence": confidence,
            "target_slots": matches,
            "active_tree_type": tree_type,
            "active_tree_slots": active_names,
            "active_slot_source": active_slot_source,
            "tree_metadata": tree_metadata,
            "doctor_visible_input_fields": [
                "dialogue_history",
                "doctor_question",
                "patient_response_after_environment_step",
            ],
            "hidden_environment_fields": [
                "hidden_profile_tree_type",
                "simulator_internal_target_node",
                "target_slots",
                "g_target",
                "evidence_unit_metadata",
            ],
        }


def load_gold_mapping(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for record in iter_jsonl(path):
        result[(record["dialogue_id"], int(record["turn_id"]))] = record
    return result


def update_eval_counter(
    counter: Counter[str],
    pred: dict[str, Any],
    gold: dict[str, Any],
) -> None:
    gold_primary = gold.get("primary_slot")
    pred_primary = pred.get("simulator_internal_target_node")
    counter["total"] += 1
    if gold_primary:
        counter["gold_mapped"] += 1
    if pred_primary:
        counter["pred_mapped"] += 1
    if gold_primary and pred_primary == gold_primary:
        counter["top1_agree_on_gold_mapped"] += 1
    if pred_primary == gold_primary:
        counter["exact_agree_including_unmapped"] += 1
    if pred["active_tree_type"] == gold.get("active_tree_type"):
        counter["tree_type_agree"] += 1


def pct(n: int, d: int) -> float:
    return round(n / d, 6) if d else 0.0


def summarize_counter(counter: Counter[str]) -> dict[str, Any]:
    total = counter["total"]
    gold_mapped = counter["gold_mapped"]
    return {
        "total": total,
        "gold_mapped": gold_mapped,
        "pred_mapped": counter["pred_mapped"],
        "coverage": pct(counter["pred_mapped"], total),
        "top1_agreement_on_gold_mapped": pct(counter["top1_agree_on_gold_mapped"], gold_mapped),
        "exact_agreement_including_unmapped": pct(counter["exact_agree_including_unmapped"], total),
        "tree_type_agreement": pct(counter["tree_type_agree"], total),
    }


def evaluate(
    interpreter: OnlineQueryInterpreter,
    normalized_path: Path,
    mapping_path: Path,
    max_error_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_key = load_gold_mapping(mapping_path)
    eval_records = []
    counters = {
        "hidden_profile_tree_type": Counter(),
        "history_only": Counter(),
    }
    mismatch_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for dialogue in iter_jsonl(normalized_path):
        turns = dialogue.get("dialogue_turns", [])
        for turn in turns:
            key = (dialogue["dialogue_id"], int(turn["turn_id"]))
            gold = gold_by_key.get(key)
            if not gold:
                continue
            question = turn.get("doctor_utterance") or ""
            history = build_history(turns, int(turn["turn_id"]))

            hidden_pred = interpreter.interpret(
                question,
                dialogue_history=history,
                hidden_profile_tree_type=gold.get("active_tree_type"),
            )
            history_pred = interpreter.interpret(question, dialogue_history=history)
            predictions = {
                "hidden_profile_tree_type": hidden_pred,
                "history_only": history_pred,
            }
            for mode, pred in predictions.items():
                update_eval_counter(counters[mode], pred, gold)
                status_counts[mode][pred["query_interpreter_status"]] += 1
                if (
                    len(mismatch_examples[mode]) < max_error_examples
                    and gold.get("primary_slot")
                    and pred.get("simulator_internal_target_node") != gold.get("primary_slot")
                ):
                    mismatch_examples[mode].append(
                        {
                            "dialogue_id": dialogue["dialogue_id"],
                            "turn_id": turn["turn_id"],
                            "doctor_question": question,
                            "question_focus_text": pred["question_focus_text"],
                            "gold_primary_slot": gold.get("primary_slot"),
                            "predicted_node": pred.get("simulator_internal_target_node"),
                            "gold_tree_type": gold.get("active_tree_type"),
                            "pred_tree_type": pred.get("active_tree_type"),
                            "prediction_status": pred["query_interpreter_status"],
                        }
                    )

            eval_records.append(
                {
                    "dialogue_id": dialogue["dialogue_id"],
                    "case_id": dialogue.get("case_id"),
                    "variant_id": dialogue.get("variant_id"),
                    "turn_id": turn.get("turn_id"),
                    "gold_primary_slot": gold.get("primary_slot"),
                    "gold_mapping_status": gold.get("mapping_status"),
                    "gold_active_tree_type": gold.get("active_tree_type"),
                    "doctor_question": question,
                    "hidden_profile_tree_type_prediction": {
                        "simulator_internal_target_node": hidden_pred.get(
                            "simulator_internal_target_node"
                        ),
                        "status": hidden_pred["query_interpreter_status"],
                        "confidence": hidden_pred["query_interpreter_confidence"],
                        "target_slots": hidden_pred["target_slots"],
                    },
                    "history_only_prediction": {
                        "simulator_internal_target_node": history_pred.get(
                            "simulator_internal_target_node"
                        ),
                        "status": history_pred["query_interpreter_status"],
                        "confidence": history_pred["query_interpreter_confidence"],
                        "active_tree_type": history_pred["active_tree_type"],
                    },
                    "target_node_visibility": "simulator_internal_not_doctor_visible",
                }
            )

    summary = {
        "schema_name": interpreter.schema.get("schema_name"),
        "schema_version": interpreter.schema.get("version"),
        "evaluation_gold": "existing_mdd5k_question_slot_map_rule_primary_slot",
        "notes": [
            "The gold labels are the previous deterministic tree-aware mapper outputs, not human annotations.",
            "hidden_profile_tree_type mode uses environment-hidden profile metadata and is the planned online patient-environment interface.",
            "history_only mode estimates tree type only from dialogue history and doctor question; it is diagnostic, not the preferred environment setting.",
            "Doctor policy must not receive simulator_internal_target_node, target_slots, or g_target.",
        ],
        "modes": {
            mode: {
                **summarize_counter(counter),
                "status_counts": dict(status_counts[mode].most_common()),
                "mismatch_examples": mismatch_examples[mode],
            }
            for mode, counter in counters.items()
        },
    }
    return eval_records, summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    hidden = summary["modes"]["hidden_profile_tree_type"]
    history = summary["modes"]["history_only"]
    lines = [
        "# Online Query Interpreter Evaluation",
        "",
        "Date: 2026-06-10",
        "",
        "## Purpose",
        "",
        "This evaluates the online simulator-side query interpreter:",
        "",
        "```text",
        "dialogue_history + doctor_question -> simulator_internal_target_node",
        "```",
        "",
        "The output node is environment-internal routing metadata. It is not a doctor-agent observation or action.",
        "",
        "## Modes",
        "",
        "- `hidden_profile_tree_type`: uses hidden profile tree type available to the patient environment. This is the planned online environment interface.",
        "- `history_only`: recovers tree type only from dialogue history and the current question. This is a diagnostic fallback.",
        "",
        "## Summary",
        "",
        "| Mode | Total | Coverage | Top-1 agreement on mapped turns | Exact agreement incl. unmapped | Tree-type agreement |",
        "|---|---:|---:|---:|---:|---:|",
        f"| hidden_profile_tree_type | {hidden['total']} | {hidden['coverage']:.2%} | {hidden['top1_agreement_on_gold_mapped']:.2%} | {hidden['exact_agreement_including_unmapped']:.2%} | {hidden['tree_type_agreement']:.2%} |",
        f"| history_only | {history['total']} | {history['coverage']:.2%} | {history['top1_agreement_on_gold_mapped']:.2%} | {history['exact_agreement_including_unmapped']:.2%} | {history['tree_type_agreement']:.2%} |",
        "",
        "## Status Counts",
        "",
    ]
    for mode in ["hidden_profile_tree_type", "history_only"]:
        lines.extend([f"### `{mode}`", "", "| Status | Count |", "|---|---:|"])
        for status, count in summary["modes"][mode]["status_counts"].items():
            lines.append(f"| `{status}` | {count} |")
        lines.append("")

    lines.extend(["## Mismatch Examples", ""])
    for mode in ["hidden_profile_tree_type", "history_only"]:
        lines.extend([f"### `{mode}`", "", "| Dialogue | Turn | Gold | Pred | Question |", "|---|---:|---|---|---|"])
        examples = summary["modes"][mode]["mismatch_examples"]
        if not examples:
            lines.append("| - | - | - | - | No mismatches collected. |")
        for item in examples:
            question = (item["doctor_question"] or "").replace("|", " ")
            if len(question) > 100:
                question = question[:97] + "..."
            lines.append(
                f"| `{item['dialogue_id']}` | {item['turn_id']} | `{item['gold_primary_slot']}` | `{item['predicted_node']}` | {question} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interface Boundary",
            "",
            "Doctor-visible fields:",
            "",
            "```text",
            "dialogue_history",
            "doctor_question",
            "patient_response",
            "```",
            "",
            "Simulator-internal fields:",
            "",
            "```text",
            "hidden_profile_tree_type",
            "simulator_internal_target_node",
            "target_slots",
            "g_target",
            "evidence_unit_metadata",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and evaluate the online query interpreter.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED_PATH)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-error-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_json(args.schema)
    interpreter = OnlineQueryInterpreter(schema)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, summary = evaluate(
        interpreter=interpreter,
        normalized_path=args.normalized,
        mapping_path=args.mapping,
        max_error_examples=args.max_error_examples,
    )
    predictions_path = args.output_dir / "mdd5k_online_query_interpreter_eval.jsonl"
    summary_path = args.output_dir / "mdd5k_online_query_interpreter_summary.json"
    report_path = args.output_dir / "MDD5K_ONLINE_QUERY_INTERPRETER_REPORT.md"
    write_jsonl(predictions_path, records)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    write_report(report_path, summary)

    hidden = summary["modes"]["hidden_profile_tree_type"]
    history = summary["modes"]["history_only"]
    print(f"Records: {len(records)}")
    print(
        "Hidden-profile mode agreement: "
        f"{hidden['top1_agreement_on_gold_mapped']:.2%}"
    )
    print(
        "History-only mode agreement: "
        f"{history['top1_agreement_on_gold_mapped']:.2%}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
