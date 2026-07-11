#!/usr/bin/env python3
"""Run frozen Final Patient online evaluation for a checkpoint manifest.

The manifest is JSONL. Each row describes one checkpoint candidate:

```json
{"method":"rfv","checkpoint_name":"ckpt400","adapter_path":".../checkpoint-400"}
```

This runner evaluates every candidate with the same patient setting, same split,
same turn budget, and same replay limits. It exists to make method comparisons
checkpoint-selection comparable rather than hand-picked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(row)
    return rows


def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "checkpoint"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--phase-dir", type=Path, default=Path.cwd())
    parser.add_argument("--script", default="scripts/run_final_patient_doctor_eval_one.sh")
    parser.add_argument("--eval-splits", default="dev")
    parser.add_argument("--max-groups", type=int, default=10000)
    parser.add_argument("--max-profiles", type=int, default=108)
    parser.add_argument("--max-per-slot", type=int, default=999)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--realizer-batch-size", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phase_dir = args.phase_dir.resolve()
    rows = read_jsonl(args.manifest)
    if not rows:
        raise SystemExit("Manifest is empty.")

    suite_dir = phase_dir / f"outputs_{args.run_tag}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        method = safe_id(str(row.get("method", "method")))
        ckpt = safe_id(str(row.get("checkpoint_name", row.get("name", f"ckpt{index}"))))
        adapter_path = str(row.get("adapter_path", "")).strip()
        if not adapter_path:
            raise ValueError(f"Missing adapter_path for manifest row {index}: {row}")
        model_id = safe_id(str(row.get("model_id", f"{method}_{ckpt}")))
        out_dir = phase_dir / f"outputs_{args.run_tag}_{model_id}"
        summary_path = out_dir / "pcv32_keyword_supported_only.json"
        command = [
            "bash",
            args.script,
            "qwen_lora_custom",
            str(out_dir),
            str(args.max_turns),
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUSTOM_ADAPTER_PATH": adapter_path,
                "CUSTOM_MODEL_ID": model_id,
                "CUSTOM_MODEL_TAG": str(row.get("model_tag", f"Qwen3-8B-{model_id}-PCV32-OnlineFinalPatient")),
                "CUSTOM_PROVIDER_TAG": str(row.get("provider_tag", f"remote_qwen3_8b_{model_id}_pcv32_online")),
                "CUSTOM_MODEL_OUTPUT_FILENAME": f"{model_id}_pcv32_online_doctor_outputs.jsonl",
                "EVAL_SPLITS": args.eval_splits,
                "MAX_GROUPS": str(args.max_groups),
                "MAX_PROFILES": str(args.max_profiles),
                "MAX_PER_SLOT": str(args.max_per_slot),
                "REPLAY_BATCH_SIZE": str(args.replay_batch_size),
                "REALIZER_BATCH_SIZE": str(args.realizer_batch_size),
            }
        )
        item = {
            "index": index,
            "method": method,
            "checkpoint_name": ckpt,
            "model_id": model_id,
            "adapter_path": adapter_path,
            "output_dir": str(out_dir),
            "summary_path": str(summary_path),
            "eval_splits": args.eval_splits,
            "max_turns": args.max_turns,
            "command": command,
        }
        plan.append(item)
        if args.skip_existing and summary_path.exists():
            item["status"] = "skipped_existing"
            continue
        if args.dry_run:
            item["status"] = "dry_run"
            continue
        log_path = suite_dir / f"{model_id}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=phase_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        item["log_path"] = str(log_path)
        item["returncode"] = proc.returncode
        item["status"] = "complete" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            write_json(suite_dir / "checkpoint_eval_manifest_status.json", plan)
            raise SystemExit(f"Checkpoint evaluation failed for {model_id}; see {log_path}")

    write_json(suite_dir / "checkpoint_eval_manifest_status.json", plan)
    print(json.dumps({"run_tag": args.run_tag, "num_candidates": len(plan), "plan": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
