from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_PATH = (
    BASE_DIR / "outputs_final_patient_rfv_data_v1" / "final_patient_rfv_value_records.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_rfv_value_model_numpy_v1"


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def residual_future_target(
    record: dict[str, Any],
    *,
    any_gain_weight: float,
    boundary_penalty: float,
    deflection_penalty: float,
    vague_penalty: float,
    unmapped_penalty: float,
) -> float:
    target = safe_float(record.get("future_target_gain"))
    target += any_gain_weight * safe_float(record.get("future_any_gain"))
    for future in record.get("future_records") or []:
        response_type = str(future.get("response_type") or "")
        if response_type == "boundary_refusal":
            target -= boundary_penalty
        elif response_type == "topic_deflection":
            target -= deflection_penalty
        elif response_type == "vague_uncertain":
            target -= vague_penalty
        elif response_type in {"no_profile_evidence", "unmapped_question"}:
            target -= unmapped_penalty
    return float(target)


def row_text(record: dict[str, Any]) -> str:
    first = record.get("first_response") or {}
    metadata = record.get("metadata") or {}
    return (
        "Task: estimate residual future evidence recovery after the current doctor question.\n"
        f"Base severity: {record.get('base_severity') or ''}\n"
        f"Prefix: {record.get('prefix_name') or ''}\n"
        f"Candidate action: {record.get('candidate_action') or ''}\n"
        f"Previous low info: {bool(record.get('previous_low_info'))}\n"
        f"Prior boundary: {bool(record.get('prior_boundary_for_target'))}\n"
        f"First response type: {first.get('response_type') or ''}\n"
        f"First response quality: {first.get('doctor_recovery_quality') or ''}\n"
        f"Immediate target gain: {safe_float(record.get('immediate_target_gain')):.6f}\n"
        f"Immediate any gain: {safe_float(record.get('immediate_any_gain')):.6f}\n"
        f"Target sufficiency after first: {safe_float(record.get('target_sufficiency_after_first')):.6f}\n"
        f"Metric: {metadata.get('metric_name') or ''}\n\n"
        f"{str(record.get('value_model_input') or '').strip()}"
    ).strip()


def load_rows(
    path: Path,
    *,
    any_gain_weight: float,
    boundary_penalty: float,
    deflection_penalty: float,
    vague_penalty: float,
    unmapped_penalty: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        text = row_text(record)
        if not text:
            continue
        target = residual_future_target(
            record,
            any_gain_weight=any_gain_weight,
            boundary_penalty=boundary_penalty,
            deflection_penalty=deflection_penalty,
            vague_penalty=vague_penalty,
            unmapped_penalty=unmapped_penalty,
        )
        rows.append(
            {
                "record_id": str(record.get("record_id") or ""),
                "state_id": str(record.get("state_id") or record.get("record_id") or ""),
                "base_severity": record.get("base_severity"),
                "candidate_action": record.get("candidate_action"),
                "text": text,
                "target": target,
                "future_target_gain": safe_float(record.get("future_target_gain")),
                "future_any_gain": safe_float(record.get("future_any_gain")),
                "immediate_target_gain": safe_float(record.get("immediate_target_gain")),
                "metadata": record.get("metadata") or {},
            }
        )
    if not rows:
        raise ValueError(f"No usable value records loaded from {path}")
    return rows


def stable_hash(text: str, seed: int) -> int:
    payload = f"{seed}\0{text}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def iter_char_ngrams(text: str, min_n: int, max_n: int):
    text = text.strip()
    if not text:
        return
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for idx in range(0, len(text) - n + 1):
            yield text[idx : idx + n]


def featurize(
    text: str,
    *,
    n_features: int,
    min_n: int,
    max_n: int,
    seed: int,
) -> dict[int, float]:
    values: defaultdict[int, float] = defaultdict(float)
    count = 0
    for gram in iter_char_ngrams(text, min_n, max_n) or []:
        idx = stable_hash(gram, seed) % n_features
        values[idx] += 1.0
        count += 1
    if count <= 0:
        return {}
    norm = math.sqrt(sum(value * value for value in values.values()))
    if norm <= 0:
        return dict(values)
    return {idx: value / norm for idx, value in values.items()}


def predict_one(weights: np.ndarray, bias: float, features: dict[int, float]) -> float:
    if not features:
        return float(bias)
    total = bias
    for idx, value in features.items():
        total += float(weights[idx]) * value
    return float(total)


def split_by_state(rows: list[dict[str, Any]], eval_fraction: float, seed: int):
    states = sorted({str(row["state_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(states)
    eval_n = max(1, int(round(len(states) * eval_fraction))) if len(states) > 1 else 0
    eval_states = set(states[:eval_n])
    train = [row for row in rows if str(row["state_id"]) not in eval_states]
    eval_rows = [row for row in rows if str(row["state_id"]) in eval_states]
    if not train or not eval_rows:
        raise ValueError("Train/eval split is empty.")
    return train, eval_rows, eval_states


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def regression_metrics(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    err = preds - targets
    return {
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(np.mean(err * err))), 6),
        "pearson": round(corrcoef(targets, preds), 6),
        "spearman": round(corrcoef(rankdata(targets), rankdata(preds)), 6),
    }


def pair_metrics(rows: list[dict[str, Any]], preds: np.ndarray, min_margin: float) -> dict[str, Any]:
    by_state: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        by_state[str(row["state_id"])].append((row, float(pred)))
    total = 0
    correct = 0
    ties = 0
    for values in by_state.values():
        ordered = sorted(values, key=lambda item: float(item[0]["target"]), reverse=True)
        for i, (chosen, chosen_pred) in enumerate(ordered):
            for rejected, rejected_pred in ordered[i + 1 :]:
                margin = float(chosen["target"]) - float(rejected["target"])
                if margin < min_margin:
                    continue
                total += 1
                pred_margin = chosen_pred - rejected_pred
                if pred_margin > 0:
                    correct += 1
                elif abs(pred_margin) <= 1e-12:
                    ties += 1
    return {
        "num_eval_pairs": total,
        "pair_accuracy": round(correct / total, 6) if total else 0.0,
        "tie_rate": round(ties / total, 6) if total else 0.0,
        "min_target_margin": min_margin,
    }


def predict_rows(weights: np.ndarray, bias: float, rows: list[dict[str, Any]], config: dict[str, Any]) -> np.ndarray:
    preds = []
    for row in rows:
        features = featurize(
            row["text"],
            n_features=int(config["n_features"]),
            min_n=int(config["min_n"]),
            max_n=int(config["max_n"]),
            seed=int(config["hash_seed"]),
        )
        preds.append(predict_one(weights, bias, features))
    return np.asarray(preds, dtype=np.float64)


def write_predictions(path: Path, rows: list[dict[str, Any]], preds: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row, pred in zip(rows, preds):
            out = {
                "record_id": row.get("record_id"),
                "state_id": row.get("state_id"),
                "base_severity": row.get("base_severity"),
                "candidate_action": row.get("candidate_action"),
                "target": round(float(row["target"]), 6),
                "prediction": round(float(pred), 6),
                "future_target_gain": row.get("future_target_gain"),
                "immediate_target_gain": row.get("immediate_target_gain"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dependency-light numpy RFV value model trainer.")
    parser.add_argument("--record-path", type=Path, default=DEFAULT_RECORD_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--n-features", type=int, default=65536)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--target-clip", type=float, default=2.0)
    parser.add_argument("--any-gain-weight", type=float, default=0.0)
    parser.add_argument("--boundary-penalty", type=float, default=0.0)
    parser.add_argument("--deflection-penalty", type=float, default=0.0)
    parser.add_argument("--vague-penalty", type=float, default=0.0)
    parser.add_argument("--unmapped-penalty", type=float, default=0.0)
    parser.add_argument("--pair-min-margin", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(
        args.record_path,
        any_gain_weight=args.any_gain_weight,
        boundary_penalty=args.boundary_penalty,
        deflection_penalty=args.deflection_penalty,
        vague_penalty=args.vague_penalty,
        unmapped_penalty=args.unmapped_penalty,
    )
    train_rows, eval_rows, eval_states = split_by_state(rows, args.eval_fraction, args.seed)
    config = {
        "n_features": args.n_features,
        "min_n": args.min_n,
        "max_n": args.max_n,
        "hash_seed": args.seed,
        "target": "residual_future_value",
        "record_path": str(args.record_path),
    }

    weights = np.zeros(args.n_features, dtype=np.float32)
    bias = 0.0
    rng = random.Random(args.seed)
    trace = []
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        sqerr = 0.0
        for row_idx in order:
            row = train_rows[row_idx]
            features = featurize(
                row["text"],
                n_features=args.n_features,
                min_n=args.min_n,
                max_n=args.max_n,
                seed=args.seed,
            )
            target = max(-args.target_clip, min(args.target_clip, float(row["target"])))
            pred = predict_one(weights, bias, features)
            err = pred - target
            sqerr += err * err
            grad = max(-1.0, min(1.0, err))
            if features:
                for idx, value in features.items():
                    weights[idx] -= args.learning_rate * (grad * value + args.l2 * float(weights[idx]))
            bias -= args.learning_rate * grad
        train_pred = predict_rows(weights, bias, train_rows, config)
        eval_pred = predict_rows(weights, bias, eval_rows, config)
        train_y = np.asarray([row["target"] for row in train_rows], dtype=np.float64)
        eval_y = np.asarray([row["target"] for row in eval_rows], dtype=np.float64)
        trace.append(
            {
                "epoch": epoch,
                "mean_train_sqerr_online": round(sqerr / max(1, len(train_rows)), 6),
                "train": regression_metrics(train_y, train_pred),
                "eval": regression_metrics(eval_y, eval_pred),
            }
        )

    train_pred = predict_rows(weights, bias, train_rows, config)
    eval_pred = predict_rows(weights, bias, eval_rows, config)
    train_y = np.asarray([row["target"] for row in train_rows], dtype=np.float64)
    eval_y = np.asarray([row["target"] for row in eval_rows], dtype=np.float64)
    all_targets = np.asarray([row["target"] for row in rows], dtype=np.float64)
    summary = {
        "model_type": "hashed_char_ngram_linear_sgd_numpy",
        "num_examples": len(rows),
        "num_train_examples": len(train_rows),
        "num_eval_examples": len(eval_rows),
        "num_eval_states": len(eval_states),
        "target_mean": round(float(np.mean(all_targets)), 6),
        "target_std": round(float(np.std(all_targets)), 6),
        "target_min": round(float(np.min(all_targets)), 6),
        "target_max": round(float(np.max(all_targets)), 6),
        "train_metrics": regression_metrics(train_y, train_pred),
        "eval_metrics": regression_metrics(eval_y, eval_pred),
        "eval_pair_metrics": pair_metrics(eval_rows, eval_pred, args.pair_min_margin),
        "config": config,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "target_clip": args.target_clip,
        },
    }
    np.savez_compressed(args.output_dir / "final_patient_rfv_value_model_numpy.npz", weights=weights, bias=np.asarray([bias], dtype=np.float32))
    write_json(args.output_dir / "final_patient_rfv_value_model_config.json", config)
    write_json(args.output_dir / "final_patient_rfv_value_model_train_trace.json", trace)
    write_json(args.output_dir / "final_patient_rfv_value_model_train_summary.json", summary)
    write_predictions(args.output_dir / "final_patient_rfv_value_model_eval_predictions.jsonl", eval_rows, eval_pred)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
