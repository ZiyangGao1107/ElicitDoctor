from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from call_closed_llm_for_pending_requests import (
    DEFAULT_AGENTMEMORY_ENV,
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URL_ENVS,
    call_provider,
    load_env_file,
    load_existing_outputs,
    parse_csv_values,
    resolve_api_keys,
    resolve_base_url,
)
from prepare_llm_patient_realizer_requests_v1 import iter_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PENDING_PATH = (
    BASE_DIR
    / "outputs_llm_patient_realizer_v3_1"
    / "mdd5k_llm_patient_realizer_requests.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    BASE_DIR
    / "outputs_llm_patient_realizer_v3_1"
    / "llm_patient_realizer_outputs.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a closed LLM for constrained patient-response realization.")
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "openai_responses", "anthropic", "gemini"],
        default="openai_compatible",
        help="Default uses an OpenAI-compatible chat.completions setup.",
    )
    parser.add_argument("--model", required=True, help="Provider-specific model id.")
    parser.add_argument("--pending-path", type=Path, default=DEFAULT_PENDING_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_AGENTMEMORY_ENV)
    parser.add_argument("--api-keys", default=None, help="Comma/semicolon separated keys. Prefer --env-file.")
    parser.add_argument("--api-key-envs", default=",".join(DEFAULT_API_KEY_ENVS))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--base-url-envs", default=",".join(DEFAULT_BASE_URL_ENVS))
    parser.add_argument("--limit", type=int, default=12, help="Maximum new requests to call; <=0 means all pending.")
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--anthropic-version", default="2023-06-01")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded_env_keys = load_env_file(args.env_file)
    api_key_envs = parse_csv_values(args.api_key_envs)
    base_url_envs = parse_csv_values(args.base_url_envs)
    api_keys, key_sources = resolve_api_keys(args.api_keys, api_key_envs)
    base_url, base_url_source = resolve_base_url(
        provider=args.provider,
        cli_base_url=args.base_url,
        base_url_envs=base_url_envs,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_existing_outputs(args.output_path)
    pending = list(iter_jsonl(args.pending_path) or [])
    todo = [record for record in pending if str(record.get("request_id")) not in seen]
    if args.limit > 0:
        todo = todo[: args.limit]

    summary: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "pending_path": str(args.pending_path),
        "output_path": str(args.output_path),
        "env_file": str(args.env_file) if args.env_file else None,
        "loaded_env_keys": loaded_env_keys,
        "api_key_envs": api_key_envs,
        "api_key_count": len(api_keys),
        "api_key_sources": key_sources,
        "base_url": base_url,
        "base_url_source": base_url_source,
        "pending_records": len(pending),
        "already_cached": len(seen),
        "new_calls_planned": len(todo),
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.provider in {"openai_compatible", "openai_responses"} and not api_keys:
        raise RuntimeError(
            "No API key found. Set XIAOAI_API_KEYS/OPENAI_API_KEY in --env-file "
            "or pass --api-keys explicitly."
        )

    written = 0
    with args.output_path.open("a", encoding="utf-8", newline="\n") as out:
        for call_index, record in enumerate(todo):
            request_id = str(record["request_id"])
            messages = record.get("messages") or []
            api_key = api_keys[call_index % len(api_keys)] if api_keys else None
            key_source = key_sources[call_index % len(key_sources)] if key_sources else None
            raw_text, meta = call_provider(
                args.provider,
                model=args.model,
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                anthropic_version=args.anthropic_version,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            out.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "raw_output": raw_text,
                        "provider": args.provider,
                        "model": args.model,
                        "task_name": record.get("task_name"),
                        "prompt_protocol_version": record.get("prompt_protocol_version"),
                        "history_mode": record.get("history_mode"),
                        "source_record_id": record.get("source_record_id"),
                        "policy_name": record.get("policy_name"),
                        "base_severity": record.get("base_severity"),
                        "target_tree_node": record.get("target_tree_node"),
                        "low_info_category": record.get("low_info_category"),
                        "turn_index": record.get("turn_index"),
                        "api_key_slot": (call_index % len(api_keys)) + 1 if api_keys else None,
                        "api_key_source": key_source,
                        "base_url": base_url,
                        "temperature": args.temperature,
                        "max_output_tokens": args.max_output_tokens,
                        "metadata": meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            written += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    summary["new_calls_written"] = written
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
