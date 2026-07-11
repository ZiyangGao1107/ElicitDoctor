#!/usr/bin/env python3
"""Select checkpoints for the Final Patient pipeline.

This utility is intentionally lightweight: it does not run evaluation. It reads
existing training/evaluation summaries and writes a deterministic selection
report so checkpoint choices are not made ad hoc.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def get_nested(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def metric_row_from_keyword_file(path: Path, metric_name: str) -> dict[str, Any]:
    obj = read_json(path)
    rows = obj if isinstance(obj, list) else obj.get("results", obj.get("summary", []))
    if isinstance(rows, dict):
        rows = [rows]
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("metric_name") in {None, metric_name} or row.get("metric") == metric_name:
            if any(key in row for key in ("mild", "moderate", "severe", "mean")):
                return dict(row)
    return {}


def metric_row_from_tree_summary(path: Path, metric_name: str) -> dict[str, Any]:
    obj = read_json(path)
    rows = obj.get("results", obj.get("summary", [])) if isinstance(obj, dict) else []
    for row in rows:
        if isinstance(row, dict) and row.get("metric_name") == metric_name:
            return dict(row)
    return {}


def best_eval_loss_from_trainer_state(path: Path) -> tuple[float | None, str | None]:
    obj = read_json(path)
    best_metric = as_float(obj.get("best_metric"))
    best_ckpt = obj.get("best_model_checkpoint")
    if best_metric is not None:
        return best_metric, str(best_ckpt) if best_ckpt else None
    best_loss: float | None = None
    best_step: int | None = None
    for row in obj.get("log_history", []):
        if not isinstance(row, dict):
            continue
        loss = as_float(row.get("eval_loss"))
        if loss is None:
            continue
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_step = int(row.get("step", 0) or 0)
    ckpt = None if best_step is None else str(path.parent / f"checkpoint-{best_step}")
    return best_loss, ckpt


def normalize_candidate_name_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def candidate_from_path(name: str, path: Path, metric_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "metrics": {},
        "source_files": [],
    }
    if not path.exists():
        row["status"] = "missing"
        return row

    if path.is_file():
        obj = read_json(path)
        row["source_files"].append(str(path))
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and any(key in item for key in ("mild", "moderate", "severe", "mean")):
                    row["metrics"].update({k: item.get(k) for k in ("mild", "moderate", "severe", "mean")})
                    row["raw"] = item
                    break
        elif isinstance(obj, dict):
            if "final_eval" in obj:
                row["metrics"].update(obj.get("final_eval") or {})
                row["recommended_checkpoint_path"] = obj.get("final_lora_adapter")
            if "eval_metrics" in obj:
                row["metrics"].update({f"eval_metrics.{k}": v for k, v in (obj.get("eval_metrics") or {}).items()})
            if "eval_pair_metrics" in obj:
                row["metrics"].update(
                    {f"eval_pair_metrics.{k}": v for k, v in (obj.get("eval_pair_metrics") or {}).items()}
                )
            for key in ("mild", "moderate", "severe", "mean"):
                if key in obj:
                    row["metrics"][key] = obj[key]
            row["raw"] = obj
        return row

    keyword_path = first_existing(
        [
            path / "pcv32_keyword_supported_only.json",
            path / "keyword_supported_only.json",
        ]
    )
    if keyword_path is not None:
        metric = metric_row_from_keyword_file(keyword_path, metric_name)
        row["source_files"].append(str(keyword_path))
        row["metrics"].update({k: metric.get(k) for k in ("mild", "moderate", "severe", "mean") if k in metric})
        row["raw_keyword_supported_only"] = metric

    tree_path = path / "tree_aligned_canonical_recovery" / "tree_aligned_canonical_evidence_recovery_summary.json"
    if tree_path.exists() and not row["metrics"]:
        metric = metric_row_from_tree_summary(tree_path, metric_name)
        row["source_files"].append(str(tree_path))
        row["metrics"].update({k: metric.get(k) for k in ("mild", "moderate", "severe", "mean") if k in metric})
        row["raw_keyword_supported_only"] = metric

    grpo_summary = path / "grpo_train_summary.json"
    if grpo_summary.exists():
        obj = read_json(grpo_summary)
        row["source_files"].append(str(grpo_summary))
        row["metrics"].update(obj.get("final_eval") or {})
        row["recommended_checkpoint_path"] = obj.get("final_lora_adapter")
        row["raw_train_summary"] = obj

    value_summary = path / "final_patient_rfv_value_model_train_summary.json"
    if value_summary.exists():
        obj = read_json(value_summary)
        row["source_files"].append(str(value_summary))
        row["metrics"].update({f"eval_metrics.{k}": v for k, v in (obj.get("eval_metrics") or {}).items()})
        row["metrics"].update({f"eval_pair_metrics.{k}": v for k, v in (obj.get("eval_pair_metrics") or {}).items()})
        model_file = path / "final_patient_rfv_value_model_numpy.npz"
        if model_file.exists():
            row["recommended_checkpoint_path"] = str(model_file)
        row["raw_value_summary"] = obj

    trainer_state = path / "trainer_state.json"
    if trainer_state.exists():
        loss, best_ckpt = best_eval_loss_from_trainer_state(trainer_state)
        row["source_files"].append(str(trainer_state))
        if loss is not None:
            row["metrics"]["eval_loss"] = loss
        if best_ckpt:
            row["recommended_checkpoint_path"] = best_ckpt
    final_adapter = path / "final_lora_adapter"
    if "recommended_checkpoint_path" not in row and final_adapter.exists():
        row["recommended_checkpoint_path"] = str(final_adapter)

    return row


def candidates_from_suite_summary(path: Path, metric_name: str) -> list[dict[str, Any]]:
    obj = read_json(path)
    out: list[dict[str, Any]] = []
    for suite in obj.get("suites", []):
        for model in suite.get("models", []):
            if not isinstance(model, dict):
                continue
            metric = model.get(metric_name) or model.get("keyword_supported_only") or {}
            row = {
                "name": str(model.get("model")),
                "path": str(model.get("out_dir", "")),
                "exists": True,
                "summary_complete": bool(model.get("summary_complete")),
                "verified_only": bool(model.get("verified_only")),
                "hard_error_rows": int(model.get("hard_error_rows") or 0),
                "fallback_rows": int(model.get("fallback_rows") or 0),
                "metrics": {k: metric.get(k) for k in ("mild", "moderate", "severe", "mean")},
                "source_files": [str(path)],
                "raw_suite_model": model,
            }
            out.append(row)
    return out


def metric(candidate: dict[str, Any], key: str) -> float | None:
    return as_float(candidate.get("metrics", {}).get(key))


def score_doctor_like(candidate: dict[str, Any], args: argparse.Namespace) -> tuple[float | None, list[str]]:
    reasons: list[str] = []
    if candidate.get("summary_complete") is False:
        reasons.append("summary_incomplete")
    if candidate.get("verified_only") is False:
        reasons.append("not_verified_only")
    if int(candidate.get("hard_error_rows") or 0) > 0:
        reasons.append("hard_error_rows_nonzero")
    if int(candidate.get("fallback_rows") or 0) > 0:
        reasons.append("fallback_rows_nonzero")

    mean = metric(candidate, "mean")
    severe = metric(candidate, "severe")
    if mean is None:
        reasons.append("missing_mean")
        return None, reasons
    if severe is None:
        severe = mean

    if args.baseline_mean is not None and mean < args.baseline_mean + args.min_mean_delta:
        reasons.append("below_baseline_mean_margin")
    if args.baseline_severe is not None and severe < args.baseline_severe + args.min_severe_delta:
        reasons.append("below_baseline_severe_margin")

    kl = metric(candidate, "eval_kl_loss")
    if args.max_kl_loss is not None and kl is not None and kl > args.max_kl_loss:
        reasons.append("kl_loss_too_high")
    shift = metric(candidate, "eval_mean_logp_shift")
    if args.max_abs_logp_shift is not None and shift is not None and abs(shift) > args.max_abs_logp_shift:
        reasons.append("logp_shift_too_high")

    score = mean + args.severe_weight * severe
    return score, reasons


def score_value_model(candidate: dict[str, Any], args: argparse.Namespace) -> tuple[float | None, list[str]]:
    reasons: list[str] = []
    spearman = metric(candidate, "eval_metrics.spearman")
    pair_acc = metric(candidate, "eval_pair_metrics.pair_accuracy")
    top1 = metric(candidate, "eval_pair_metrics.top1_accuracy")
    regret = metric(candidate, "eval_pair_metrics.mean_oracle_regret")

    if spearman is None:
        reasons.append("missing_spearman")
        spearman = 0.0
    if pair_acc is None:
        reasons.append("missing_pair_accuracy")
        pair_acc = 0.0
    if top1 is None:
        reasons.append("missing_top1_accuracy")
        top1 = 0.0
    if regret is None:
        reasons.append("missing_mean_oracle_regret")
        regret = 0.0

    if spearman < args.min_spearman:
        reasons.append("spearman_below_minimum")
    if pair_acc < args.min_pair_accuracy:
        reasons.append("pair_accuracy_below_minimum")

    score = spearman + pair_acc + top1 - args.oracle_regret_weight * regret
    return score, reasons


def score_sft(candidate: dict[str, Any], args: argparse.Namespace) -> tuple[float | None, list[str]]:
    if metric(candidate, "mean") is not None:
        return score_doctor_like(candidate, args)
    eval_loss = metric(candidate, "eval_loss")
    if eval_loss is None:
        return None, ["missing_eval_loss_or_online_metric"]
    return -eval_loss, []


def rank_candidates(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        if args.stage in {"doctor_eval", "grpo", "rfv"}:
            score, reject_reasons = score_doctor_like(candidate, args)
        elif args.stage == "value_model":
            score, reject_reasons = score_value_model(candidate, args)
        elif args.stage == "sft":
            score, reject_reasons = score_sft(candidate, args)
        else:
            raise ValueError(f"Unsupported stage: {args.stage}")
        out = dict(candidate)
        out["selection_score"] = score
        out["reject_reasons"] = reject_reasons
        out["eligible"] = score is not None and not reject_reasons
        ranked.append(out)
    ranked.sort(
        key=lambda row: (
            1 if row["selection_score"] is not None else 0,
            1 if row["eligible"] else 0,
            row["selection_score"] if row["selection_score"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return ranked


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Final Patient Checkpoint Selection",
        "",
        f"- Stage: `{report['stage']}`",
        f"- Recommended: `{report.get('recommended_name')}`",
        f"- Recommended path: `{report.get('recommended_checkpoint_path')}`",
        "",
        "## Decision Rule",
        "",
        report["decision_rule"],
        "",
        "## Candidates",
        "",
        "| Rank | Name | Eligible | Score | Mean | Severe | Reject reasons |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(report["ranked_candidates"], start=1):
        metrics = row.get("metrics", {})
        score = row.get("selection_score")
        lines.append(
            "| {rank} | `{name}` | {eligible} | {score} | {mean} | {severe} | {reasons} |".format(
                rank=idx,
                name=row.get("name"),
                eligible="yes" if row.get("eligible") else "no",
                score="" if score is None else f"{score:.6f}",
                mean="" if as_float(metrics.get("mean")) is None else f"{float(metrics['mean']):.6f}",
                severe="" if as_float(metrics.get("severe")) is None else f"{float(metrics['severe']):.6f}",
                reasons=", ".join(row.get("reject_reasons") or []),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decision_rule(args: argparse.Namespace) -> str:
    if args.stage == "sft":
        return (
            "Use final-patient online recovery if available. Otherwise select the SFT checkpoint with the "
            "lowest held-out eval loss, then verify it with the frozen Final Patient online evaluation before RL."
        )
    if args.stage == "value_model":
        return (
            "Select the value checkpoint by held-out same-state action ranking: positive Spearman, pairwise "
            "accuracy above random, high top-1 accuracy, and low oracle regret."
        )
    return (
        "Select GRPO/RFV/doctor checkpoints by frozen Final Patient online canonical evidence recovery. "
        f"The score is mean + {args.severe_weight} * severe, with optional baseline, KL, logp-shift, "
        "verified-only, fallback, and hard-error filters."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "value_model", "grpo", "rfv", "doctor_eval"], required=True)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=PATH or PATH. May repeat.")
    parser.add_argument("--suite-summary", type=Path, action="append", default=[], help="Baseline suite summary JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-name", default="keyword_supported_only")
    parser.add_argument("--severe-weight", type=float, default=0.5)
    parser.add_argument("--baseline-mean", type=float)
    parser.add_argument("--baseline-severe", type=float)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-severe-delta", type=float, default=0.0)
    parser.add_argument("--max-kl-loss", type=float)
    parser.add_argument("--max-abs-logp-shift", type=float)
    parser.add_argument("--min-spearman", type=float, default=0.0)
    parser.add_argument("--min-pair-accuracy", type=float, default=0.5)
    parser.add_argument("--oracle-regret-weight", type=float, default=1.0)
    args = parser.parse_args()

    candidates: list[dict[str, Any]] = []
    for raw in args.candidate:
        name, path = normalize_candidate_name_path(raw)
        candidates.append(candidate_from_path(name, path, args.metric_name))
    for suite_path in args.suite_summary:
        candidates.extend(candidates_from_suite_summary(suite_path, args.metric_name))
    if not candidates:
        raise SystemExit("No candidates provided.")

    ranked = rank_candidates(candidates, args)
    eligible = [row for row in ranked if row.get("eligible")]
    recommended = eligible[0] if eligible else ranked[0]
    report = {
        "stage": args.stage,
        "metric_name": args.metric_name,
        "decision_rule": decision_rule(args),
        "recommended_name": recommended.get("name"),
        "recommended_checkpoint_path": recommended.get("recommended_checkpoint_path", recommended.get("path")),
        "recommended_is_eligible": bool(recommended.get("eligible")),
        "ranked_candidates": ranked,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "checkpoint_selection_report.json", report)
    write_markdown(args.output_dir / "CHECKPOINT_SELECTION.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
