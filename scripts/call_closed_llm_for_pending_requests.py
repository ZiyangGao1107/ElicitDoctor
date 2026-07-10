from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PENDING_PATH = (
    BASE_DIR
    / "outputs_closed_llm_doctor_cache"
    / "mdd5k_llm_doctor_online_replay_pending_requests.jsonl"
)
DEFAULT_OUTPUT_PATH = BASE_DIR / "outputs_closed_llm_doctor_cache" / "llm_outputs.jsonl"
DEFAULT_AGENTMEMORY_ENV = Path(os.environ.get("ACTIVE_REASONING_ENV_FILE", ".env"))
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = os.environ.get(
    "OPENAI_COMPATIBLE_BASE_URL",
    "https://api.openai.com/v1",
)
DEFAULT_API_KEY_ENVS = (
    "XIAOAI_API_KEYS",
    "OPENAI_API_KEYS",
    "XIAOAI_API_KEY",
    "OPENAI_API_KEY",
)
DEFAULT_BASE_URL_ENVS = (
    "XIAOAI_API_BASE",
    "XIAOAI_BASE_URL",
    "OPENAI_BASE_URL",
    "API_BASE_URL",
)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from exc


def load_existing_outputs(path: Path) -> set[str]:
    seen = set()
    if not path.exists():
        return seen
    for record in iter_jsonl(path):
        request_id = record.get("request_id")
        if request_id:
            seen.add(str(request_id))
    return seen


def load_env_file(path: Path | None) -> list[str]:
    """Load a simple KEY=VALUE .env file without printing secret values."""
    if path is None or not path.exists():
        return []
    loaded_keys = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, value)
            loaded_keys.append(key)
    return loaded_keys


def parse_csv_values(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace("\n", ",").replace(";", ",")
    values = []
    seen = set()
    for item in normalized.split(","):
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def resolve_api_keys(cli_api_keys: str | None, api_key_envs: list[str]) -> tuple[list[str], list[str]]:
    keys = []
    key_sources = []
    for key in parse_csv_values(cli_api_keys):
        if key not in keys:
            keys.append(key)
            key_sources.append("cli")
    for env_name in api_key_envs:
        for key in parse_csv_values(os.getenv(env_name)):
            if key not in keys:
                keys.append(key)
                key_sources.append(env_name)
    return keys, key_sources


def resolve_base_url(
    *,
    provider: str,
    cli_base_url: str | None,
    base_url_envs: list[str],
) -> tuple[str | None, str | None]:
    if cli_base_url:
        return cli_base_url, "cli"
    for env_name in base_url_envs:
        value = os.getenv(env_name)
        if value:
            return value, env_name
    if provider == "openai_compatible":
        return DEFAULT_OPENAI_COMPATIBLE_BASE_URL, "default_openai_compatible"
    return None, None


def clean_question(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "```" in text:
        fenced_parts = [part.strip() for part in text.split("```") if part.strip()]
        if fenced_parts:
            text = fenced_parts[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text
    for prefix in (
        "医生：",
        "医生:",
        "问题：",
        "问题:",
        "问：",
        "问:",
        "答：",
        "答:",
        "Doctor:",
        "Question:",
        "Assistant:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.strip("`\"'“”‘’")
    if text and not text.endswith(("?", "？")):
        text = text.rstrip("。.!！") + "？"
    return text


def messages_to_plain_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)


def call_openai_compatible_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str | None,
    max_output_tokens: int,
    temperature: float | None,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    """OpenAI-compatible chat.completions call."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    text = getattr(message, "content", "") if message else ""
    usage = getattr(response, "usage", None)
    usage_obj = usage.model_dump() if hasattr(usage, "model_dump") else usage
    return text or "", {
        "usage": usage_obj,
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
    }


def call_openai_responses(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_output_tokens: int,
    temperature: float | None,
) -> tuple[str, dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.responses.create(**kwargs)
    text = getattr(response, "output_text", "") or ""
    usage = getattr(response, "usage", None)
    usage_obj = usage.model_dump() if hasattr(usage, "model_dump") else usage
    return text, {"usage": usage_obj}


def call_anthropic(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float | None,
    api_version: str,
) -> tuple[str, dict[str, Any]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    system_parts = []
    user_messages = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            user_messages.append({"role": "user" if role not in {"user", "assistant"} else role, "content": content})

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": user_messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "x-api-key": api_key,
        "anthropic-version": api_version,
        "content-type": "application/json",
    }
    with httpx.Client(timeout=90.0) as client:
        response = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    text_parts = [
        item.get("text", "")
        for item in data.get("content", [])
        if item.get("type") == "text"
    ]
    return "\n".join(text_parts), {"usage": data.get("usage"), "stop_reason": data.get("stop_reason")}


def call_gemini(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float | None,
) -> tuple[str, dict[str, Any]]:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    config_kwargs: dict[str, Any] = {"max_output_tokens": max_output_tokens}
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    response = client.models.generate_content(
        model=model,
        contents=messages_to_plain_prompt(messages),
        config=types.GenerateContentConfig(**config_kwargs),
    )
    usage = getattr(response, "usage_metadata", None)
    usage_obj = usage.model_dump() if hasattr(usage, "model_dump") else str(usage) if usage else None
    return response.text or "", {"usage": usage_obj}


def call_provider(
    provider: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    base_url: str | None,
    max_output_tokens: int,
    temperature: float | None,
    anthropic_version: str,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    if provider == "openai_compatible":
        if not api_key:
            raise RuntimeError("No API key available for openai_compatible provider")
        return call_openai_compatible_chat(
            model=model,
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if provider == "openai_responses":
        if not api_key:
            raise RuntimeError("No API key available for openai_responses provider")
        return call_openai_responses(
            model=model,
            messages=messages,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if provider == "anthropic":
        return call_anthropic(
            model=model,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            api_version=anthropic_version,
        )
    if provider == "gemini":
        return call_gemini(
            model=model,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unknown provider: {provider}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a closed LLM for pending doctor-agent replay requests.")
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "openai_responses", "anthropic", "gemini"],
        default="openai_compatible",
        help="Default uses OpenAI-compatible chat.completions.",
    )
    parser.add_argument("--model", required=True, help="Provider-specific model id. Keep explicit for reproducibility.")
    parser.add_argument("--pending-path", type=Path, default=DEFAULT_PENDING_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_AGENTMEMORY_ENV)
    parser.add_argument("--api-keys", default=None, help="Comma/semicolon separated keys. Prefer --env-file instead.")
    parser.add_argument(
        "--api-key-envs",
        default=",".join(DEFAULT_API_KEY_ENVS),
        help="Comma-separated env names searched in order.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--base-url-envs",
        default=",".join(DEFAULT_BASE_URL_ENVS),
        help="Comma-separated env names searched in order.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum new requests to call; <=0 means all pending.")
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--anthropic-version", default="2023-06-01")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--per-request-retries", type=int, default=5)
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

    summary = {
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
            raw_text = ""
            meta: dict[str, Any] = {}
            api_key = None
            key_source = None
            last_error = None
            for attempt in range(max(1, args.per_request_retries)):
                key_index = (call_index + attempt) % len(api_keys) if api_keys else 0
                api_key = api_keys[key_index] if api_keys else None
                key_source = key_sources[key_index] if key_sources else None
                try:
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
                    meta = dict(meta or {})
                    meta["request_attempt"] = attempt + 1
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt + 1 >= max(1, args.per_request_retries):
                        raise
                    time.sleep(max(args.sleep_seconds, 1.0) * (attempt + 1))
            doctor_question = clean_question(raw_text)
            out.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "doctor_question": doctor_question,
                        "raw_output": raw_text,
                        "provider": args.provider,
                        "model": args.model,
                        "prompt_policy": record.get("policy_name"),
                        "base_severity": record.get("base_severity"),
                        "turn_index": record.get("turn_index"),
                        "api_key_slot": (call_index % len(api_keys)) + 1 if api_keys else None,
                        "api_key_source": key_source,
                        "base_url": base_url,
                        "metadata": meta,
                        "last_error": str(last_error) if last_error else None,
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
