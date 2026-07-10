from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROJECT = Path(os.environ.get("ACTIVE_REASONING_PROJECT", ".")).resolve()
DEFAULT_MODEL_PATH = DEFAULT_PROJECT / "cache" / "qwen3-8b-hf-remote-code"
DEFAULT_ADAPTER_PATH = (
    DEFAULT_PROJECT
    / "AAAI_2027_Submission"
    / "phase1_dataset_construction"
    / "outputs_qwen3_doctor_sft_lora_f32_f41_2ksteps_bf16"
    / "final_lora_adapter"
)


def ensure_data1_hf_cache() -> None:
    base = DEFAULT_PROJECT / "cache"
    if base.exists():
        os.environ.setdefault("HF_HOME", str(base / "hf_home"))
        os.environ.setdefault("HF_MODULES_CACHE", str(base / "hf_modules"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(base / "transformers"))
        for key in ("HF_HOME", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


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


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record.get("request_id")) for record in iter_jsonl(path) if record.get("request_id")}


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_doctor_question(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text
    for prefix in (
        "医生：",
        "医生:",
        "问题：",
        "问题:",
        "Doctor:",
        "Question:",
        "Assistant:",
        "assistant:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.strip("`\"'“”‘’ ")
    if not text:
        return "\u60a8\u6700\u8fd1\u6700\u56f0\u6270\u60a8\u7684\u60c5\u51b5\u662f\u4ec0\u4e48\uff1f"
    if not text.endswith(("?", "？")):
        text = text.rstrip("。.!！?？") + "？"
    return text


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Qwen3 HF + LoRA doctor model for pending replay requests.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Use the base HF model without loading a LoRA/PEFT adapter.",
    )
    parser.add_argument("--provider-tag", default="remote_qwen3_8b_hf_lora_sft")
    parser.add_argument("--model-tag", default="Qwen3-8B")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new requests; <=0 means all pending.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data1_hf_cache()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = args.status_path or args.output_path.with_suffix(".status.json")

    seen = load_seen(args.output_path)
    pending = [record for record in iter_jsonl(args.input_path) if str(record.get("request_id")) not in seen]
    if args.limit > 0:
        pending = pending[: args.limit]

    status = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "model_path": str(args.model_path),
        "adapter_path": None if args.no_adapter else str(args.adapter_path),
        "already_cached": len(seen),
        "new_calls_planned": len(pending),
        "new_calls_written": 0,
        "state": "loading_model",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_status(status_path, status)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    if not args.no_adapter:
        try:
            model = PeftModel.from_pretrained(
                model,
                args.adapter_path,
                local_files_only=True,
                torch_device="cpu",
            )
        except TypeError:
            model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)

    status["state"] = "generating"
    write_status(status_path, status)

    do_sample = args.temperature > 0
    written = 0
    with args.output_path.open("a", encoding="utf-8", newline="\n") as out:
        batch_size = max(1, int(args.batch_size))
        for batch_start in range(0, len(pending), batch_size):
            batch_records = pending[batch_start : batch_start + batch_size]
            prompts = [render_prompt(tokenizer, record.get("messages") or []) for record in batch_records]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).to(device)
            start = time.time()
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            batch_seconds = round(time.time() - start, 3)
            prompt_width = inputs["input_ids"].shape[1]
            for row_idx, record in enumerate(batch_records):
                request_id = str(record["request_id"])
                generated = output_ids[row_idx, prompt_width:]
                raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
                out.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "doctor_question": clean_doctor_question(raw),
                            "raw_output": raw,
                            "provider": args.provider_tag,
                            "model": args.model_tag,
                            "adapter": None if args.no_adapter else str(args.adapter_path),
                            "prompt_policy": record.get("policy_name"),
                            "method": record.get("method"),
                            "base_severity": record.get("base_severity"),
                            "turn_index": record.get("turn_index"),
                            "metadata": {
                                "gen_seconds": round(batch_seconds / len(batch_records), 3),
                                "batch_seconds": batch_seconds,
                                "batch_size": len(batch_records),
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "max_new_tokens": args.max_new_tokens,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            if written % args.flush_every == 0:
                out.flush()
            status["new_calls_written"] = written
            status["last_request_id"] = str(batch_records[-1]["request_id"])
            status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_status(status_path, status)

    status["new_calls_written"] = written
    status["state"] = "complete"
    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_status(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
