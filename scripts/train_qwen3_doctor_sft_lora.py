from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "cache" / "qwen3-8b-hf-remote-code"))
DEFAULT_TRAIN_PATH = (
    BASE_DIR / "outputs_doctor_sft_bc_f32_f41_v1" / "mdd5k_doctor_generation_sft_train.jsonl"
)
DEFAULT_DEV_PATH = BASE_DIR / "outputs_doctor_sft_bc_f32_f41_v1" / "mdd5k_doctor_generation_sft_dev.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_qwen3_doctor_sft_lora_f32_f41_v1"


def iter_jsonl(path: Path, max_samples: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_samples is not None and idx > max_samples:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_data1_hf_cache() -> None:
    """Keep dynamic HF module/cache files under the project cache directory."""
    base = Path(os.environ.get("ACTIVE_REASONING_CACHE", BASE_DIR / "cache"))
    if base.exists():
        os.environ.setdefault("HF_HOME", str(base / "hf_home"))
        os.environ.setdefault("HF_MODULES_CACHE", str(base / "hf_modules"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(base / "transformers"))
        for key in ("HF_HOME", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


class DoctorSFTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_length: int,
        max_samples: int | None,
    ) -> None:
        self.features: list[dict[str, list[int]]] = []
        for record in iter_jsonl(path, max_samples=max_samples):
            messages = record.get("messages") or []
            if len(messages) < 3 or messages[-1].get("role") != "assistant":
                continue
            feature = self.encode_messages(messages, tokenizer, max_length)
            if feature is not None:
                self.features.append(feature)
        if not self.features:
            raise ValueError(f"No trainable samples loaded from {path}")

    @staticmethod
    def render_chat(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    @staticmethod
    def encode_messages(messages: list[dict[str, str]], tokenizer, max_length: int) -> dict[str, list[int]] | None:
        prompt_messages = messages[:-1]
        full_text = DoctorSFTDataset.render_chat(tokenizer, messages, add_generation_prompt=False)
        prompt_text = DoctorSFTDataset.render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = full_ids[len(prompt_ids) :]
        if not answer_ids:
            return None
        if len(answer_ids) >= max_length:
            input_ids = answer_ids[:max_length]
            labels = list(input_ids)
        else:
            prompt_budget = max_length - len(answer_ids)
            kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            input_ids = kept_prompt + answer_ids
            labels = [-100] * len(kept_prompt) + list(answer_ids)
        attention_mask = [1] * len(input_ids)
        if all(label == -100 for label in labels):
            return None
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class CausalLMCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for the Qwen3 doctor next-question model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data1_hf_cache()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = DoctorSFTDataset(
        args.train_path,
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
    )
    eval_dataset = DoctorSFTDataset(
        args.dev_path,
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_dev_samples,
    )

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": 2,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }
    supported_training_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "evaluation_strategy" not in supported_training_args and "eval_strategy" in supported_training_args:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    filtered_training_kwargs = {
        key: value for key, value in training_kwargs.items() if key in supported_training_args
    }
    dropped_training_kwargs = sorted(set(training_kwargs) - set(filtered_training_kwargs))
    if dropped_training_kwargs:
        print(
            json.dumps(
                {"dropped_training_args_for_compat": dropped_training_kwargs},
                ensure_ascii=False,
            )
        )
    training_args = TrainingArguments(**filtered_training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_lora_adapter"))
    summary = {
        "model_path": str(args.model_path),
        "train_path": str(args.train_path),
        "dev_path": str(args.dev_path),
        "output_dir": str(args.output_dir),
        "train_samples": len(train_dataset),
        "dev_samples": len(eval_dataset),
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    (args.output_dir / "sft_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
