from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_final_patient_rfv_value_model import (
    DEFAULT_RECORD_PATH,
    load_rows,
    pair_metrics,
    predict_rows,
    regression_metrics,
    write_json,
    write_predictions,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BASE_DIR / "outputs_final_patient_rfv_value_model"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_final_patient_value_model_scores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score RFV/action-value records with a trained lightweight final-patient value model."
    )
    parser.add_argument("--record-path", type=Path, default=DEFAULT_RECORD_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-mode",
        choices=["residual_future", "action_value_total", "immediate_only"],
        default="action_value_total",
    )
    parser.add_argument("--any-gain-weight", type=float, default=0.0)
    parser.add_argument("--boundary-penalty", type=float, default=0.0)
    parser.add_argument("--deflection-penalty", type=float, default=0.0)
    parser.add_argument("--vague-penalty", type=float, default=0.0)
    parser.add_argument("--unmapped-penalty", type=float, default=0.0)
    parser.add_argument("--pair-min-margin", type=float, default=0.01)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "final_patient_rfv_value_model_numpy.npz"
    config_path = args.model_dir / "final_patient_rfv_value_model_config.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    model = np.load(model_path)
    weights = model["weights"]
    bias = float(model["bias"][0])
    config = load_json(config_path)

    rows = load_rows(
        args.record_path,
        target_mode=args.target_mode,
        any_gain_weight=args.any_gain_weight,
        boundary_penalty=args.boundary_penalty,
        deflection_penalty=args.deflection_penalty,
        vague_penalty=args.vague_penalty,
        unmapped_penalty=args.unmapped_penalty,
    )
    preds = predict_rows(weights, bias, rows, config)
    targets = np.asarray([row["target"] for row in rows], dtype=np.float64)

    prediction_path = args.output_dir / "final_patient_value_model_predictions.jsonl"
    write_predictions(prediction_path, rows, preds)
    summary = {
        "settings": {
            "record_path": str(args.record_path),
            "model_dir": str(args.model_dir),
            "target_mode": args.target_mode,
            "pair_min_margin": args.pair_min_margin,
        },
        "records": len(rows),
        "prediction_path": str(prediction_path),
        "regression_metrics": regression_metrics(targets, preds),
        "pair_metrics": pair_metrics(rows, preds, args.pair_min_margin),
        "model_config": config,
    }
    write_json(args.output_dir / "final_patient_value_model_score_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
