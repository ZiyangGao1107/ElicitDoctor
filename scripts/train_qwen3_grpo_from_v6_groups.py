from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = Path(os.environ.get("ACTIVE_REASONING_PROJECT", ".")).resolve()
DEFAULT_MODEL_PATH = DEFAULT_PROJECT / "cache" / "qwen3-8b-hf-remote-code"
DEFAULT_SFT_ADAPTER = (
    DEFAULT_PROJECT
    / "AAAI_2027_Submission"
    / "phase1_dataset_construction"
    / "outputs_qwen3_doctor_sft_lora_f32_f41_2ksteps_bf16"
    / "final_lora_adapter"
)
DEFAULT_GROUP_DATA = (
    BASE_DIR
    / "outputs_reward_centered_v6_patient_v2_train_turn16_20260629"
    / "reward_centered_v6_grpo_groups.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs_qwen3_reward_v6_patient_v2_grpo_smoke"


def parse_save_milestones(raw: str | None, max_steps: int) -> set[int]:
    if raw is None:
        return set()
    text = raw.strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return set()
    milestones: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            step = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid --save-milestones item: {item!r}") from exc
        if step <= 0:
            raise ValueError(f"--save-milestones must contain positive steps, got {step}")
        if step <= max_steps:
            milestones.add(step)
    return milestones


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_chat(tokenizer, prompt: str, answer: str | None = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=answer is None,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=answer is None,
        )


def encode_prompt_answer(tokenizer, prompt: str, answer: str, max_length: int) -> dict[str, list[int]] | None:
    prompt_text = render_chat(tokenizer, prompt, answer=None)
    full_text = render_chat(tokenizer, prompt, answer=answer)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
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
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def make_lora_config(args: argparse.Namespace):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def load_trainable_policy(args: argparse.Namespace, dtype: torch.dtype):
    from peft import PeftModel, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    if args.sft_adapter_path and args.sft_adapter_path.exists():
        try:
            model = PeftModel.from_pretrained(
                base,
                args.sft_adapter_path,
                local_files_only=True,
                is_trainable=True,
                torch_device="cpu",
            )
        except TypeError:
            model = PeftModel.from_pretrained(base, args.sft_adapter_path, local_files_only=True)
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
    else:
        model = get_peft_model(base, make_lora_config(args))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def load_reference(args: argparse.Namespace, dtype: torch.dtype):
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    adapter_path = args.ref_adapter_path if args.ref_adapter_path else args.sft_adapter_path
    if adapter_path and adapter_path.exists():
        try:
            ref = PeftModel.from_pretrained(
                base,
                adapter_path,
                local_files_only=True,
                is_trainable=False,
                torch_device="cpu",
            )
        except TypeError:
            ref = PeftModel.from_pretrained(base, adapter_path, local_files_only=True, is_trainable=False)
    else:
        ref = base
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    return ref


def sequence_logps(model, feature: dict[str, torch.Tensor], *, length_normalize: bool) -> torch.Tensor:
    outputs = model(
        input_ids=feature["input_ids"],
        attention_mask=feature["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels = feature["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_logps = (token_logps * mask).sum(dim=-1)
    if length_normalize:
        seq_logps = seq_logps / mask.sum(dim=-1).clamp_min(1)
    return seq_logps


def reward_value(response: dict[str, Any]) -> float:
    try:
        return float(response.get("reward") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def normalize_response_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_groups(
    path: Path,
    *,
    max_groups: int,
    eval_groups: int,
    seed: int,
    min_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = []
    filter_counters: Counter[str] = Counter()
    for group in iter_jsonl(path):
        filter_counters["raw_groups_seen"] += 1
        responses = []
        seen_texts: set[str] = set()
        group_had_duplicate_text = False
        for response in group.get("responses") or []:
            text = normalize_response_text(response.get("text"))
            if not text:
                filter_counters["skip_empty_response"] += 1
                continue
            if text in seen_texts:
                filter_counters["skip_duplicate_response_text"] += 1
                group_had_duplicate_text = True
                continue
            seen_texts.add(text)
            responses.append(
                {
                    "text": text,
                    "reward": reward_value(response),
                    "metadata": response.get("metadata") or {},
                }
            )
        if group_had_duplicate_text:
            filter_counters["groups_with_duplicate_response_text"] += 1
        rewards = [reward_value(response) for response in responses]
        if len(responses) < min_candidates:
            filter_counters["skip_too_few_candidates"] += 1
            continue
        if max(rewards) - min(rewards) <= 1e-9:
            filter_counters["skip_zero_reward_margin"] += 1
            continue
        groups.append(
            {
                "id": group.get("id"),
                "prompt": group.get("prompt"),
                "metadata": group.get("metadata") or {},
                "responses": responses,
            }
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    if max_groups > 0:
        groups = groups[:max_groups]
    eval_n = min(max(0, eval_groups), max(0, len(groups) // 5))
    eval_records = groups[:eval_n]
    train_records = groups[eval_n:]

    severity_counts: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    reward_margins = []
    for group in groups:
        metadata = group.get("metadata") or {}
        severity_counts[str(metadata.get("base_severity") or "unknown")] += 1
        responses = group.get("responses") or []
        candidate_counts[len(responses)] += 1
        rewards = [reward_value(response) for response in responses]
        if rewards:
            reward_margins.append(max(rewards) - min(rewards))
    summary = {
        "input_groups": len(groups),
        "train_groups": len(train_records),
        "eval_groups": len(eval_records),
        "severity_counts": dict(severity_counts),
        "candidate_count_distribution": {str(k): v for k, v in sorted(candidate_counts.items())},
        "mean_reward_margin": round(sum(reward_margins) / len(reward_margins), 6) if reward_margins else 0.0,
        "filter_counters": dict(filter_counters),
    }
    return train_records, eval_records, summary


def select_candidate_responses(
    responses: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    if max_candidates <= 0 or len(responses) <= max_candidates:
        return responses
    ranked = sorted(responses, key=reward_value, reverse=True)
    selected = []
    left = 0
    right = len(ranked) - 1
    while len(selected) < max_candidates and left <= right:
        selected.append(ranked[left])
        left += 1
        if len(selected) >= max_candidates or left > right:
            break
        selected.append(ranked[right])
        right -= 1
    return selected


class RewardGroupDataset(Dataset):
    def __init__(
        self,
        groups: list[dict[str, Any]],
        tokenizer,
        *,
        max_length: int,
        max_candidates: int,
    ) -> None:
        self.features: list[dict[str, Any]] = []
        for group in groups:
            prompt = str(group.get("prompt") or "").strip()
            if not prompt:
                continue
            responses = select_candidate_responses(
                list(group.get("responses") or []),
                max_candidates=max_candidates,
            )
            encoded_responses = []
            seen_texts: set[str] = set()
            for response in responses:
                text = normalize_response_text(response.get("text"))
                if not text:
                    continue
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                encoded = encode_prompt_answer(tokenizer, prompt, text, max_length)
                if encoded is None:
                    continue
                encoded_responses.append(
                    {
                        "encoded": encoded,
                        "reward": reward_value(response),
                        "text": text,
                        "metadata": response.get("metadata") or {},
                    }
                )
            if len(encoded_responses) < 2:
                continue
            rewards = [item["reward"] for item in encoded_responses]
            if max(rewards) - min(rewards) <= 1e-9:
                continue
            self.features.append(
                {
                    "id": group.get("id"),
                    "responses": encoded_responses,
                    "metadata": group.get("metadata") or {},
                }
            )
        if not self.features:
            raise ValueError("No usable GRPO groups after tokenization.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class GroupCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, groups: list[dict[str, Any]]) -> dict[str, Any]:
        flat = []
        group_sizes = []
        rewards = []
        ids = []
        metadata = []
        response_texts = []
        response_metadata = []
        for group in groups:
            responses = group["responses"]
            group_sizes.append(len(responses))
            ids.append(group.get("id"))
            metadata.append(group.get("metadata") or {})
            for response in responses:
                flat.append(response["encoded"])
                rewards.append(float(response["reward"]))
                response_texts.append(response.get("text") or "")
                response_metadata.append(response.get("metadata") or {})

        max_len = max(len(item["input_ids"]) for item in flat)
        input_ids = []
        attention_mask = []
        labels = []
        for item in flat:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "flat": {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            },
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "ids": ids,
            "metadata": metadata,
            "response_texts": response_texts,
            "response_metadata": response_metadata,
        }


def move_feature(feature: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in feature.items()}


def group_advantages(
    rewards: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    mode: str,
    reward_clip: float,
) -> torch.Tensor:
    chunks = torch.split(rewards, group_sizes.tolist())
    advantages = []
    for chunk in chunks:
        if mode == "center":
            adv = chunk - chunk.mean()
        elif mode == "zscore":
            std = chunk.std(unbiased=False).clamp_min(1e-6)
            adv = (chunk - chunk.mean()) / std
        elif mode == "rank":
            order = torch.argsort(torch.argsort(chunk))
            adv = order.float()
            adv = adv - adv.mean()
            adv = adv / adv.std(unbiased=False).clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported advantage mode: {mode}")
        if reward_clip > 0:
            adv = adv.clamp(min=-reward_clip, max=reward_clip)
        advantages.append(adv)
    return torch.cat(advantages, dim=0)


def split_by_group(values: torch.Tensor, group_sizes: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.split(values, group_sizes.tolist()))


def grpo_batch_loss(
    policy,
    reference,
    batch: dict[str, Any],
    *,
    policy_device: torch.device,
    reference_device: torch.device | None,
    advantage_mode: str,
    reward_clip: float,
    kl_coef: float,
    length_normalize: bool,
    entropy_bonus: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    flat = move_feature(batch["flat"], policy_device)
    group_sizes = batch["group_sizes"].to(policy_device)
    rewards = batch["rewards"].to(policy_device)
    policy_logps = sequence_logps(policy, flat, length_normalize=length_normalize)
    advantages = group_advantages(
        rewards,
        group_sizes,
        mode=advantage_mode,
        reward_clip=reward_clip,
    ).detach()

    policy_loss = -(advantages * policy_logps).mean()
    kl_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_device)
    ref_logps = None
    if reference is not None and kl_coef > 0:
        with torch.no_grad():
            ref_flat = flat if reference_device == policy_device else move_feature(batch["flat"], reference_device)
            ref_logps = sequence_logps(reference, ref_flat, length_normalize=length_normalize)
            ref_logps = ref_logps.to(policy_device)
        kl_loss = ((policy_logps - ref_logps) ** 2).mean()

    entropy_proxy = -policy_logps.mean()
    loss = policy_loss + kl_coef * kl_loss - entropy_bonus * entropy_proxy

    with torch.no_grad():
        reward_chunks = split_by_group(rewards, group_sizes)
        logp_chunks = split_by_group(policy_logps, group_sizes)
        selected_rewards = []
        oracle_rewards = []
        random_rewards = []
        top1_matches = []
        for reward_chunk, logp_chunk in zip(reward_chunks, logp_chunks):
            policy_idx = int(torch.argmax(logp_chunk).item())
            oracle_idx = int(torch.argmax(reward_chunk).item())
            selected_rewards.append(float(reward_chunk[policy_idx].detach().cpu()))
            oracle_rewards.append(float(reward_chunk[oracle_idx].detach().cpu()))
            random_rewards.append(float(reward_chunk.mean().detach().cpu()))
            top1_matches.append(1.0 if policy_idx == oracle_idx else 0.0)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "mean_advantage_abs": float(advantages.abs().mean().detach().cpu()),
            "policy_selected_reward": sum(selected_rewards) / max(1, len(selected_rewards)),
            "oracle_reward": sum(oracle_rewards) / max(1, len(oracle_rewards)),
            "random_reward": sum(random_rewards) / max(1, len(random_rewards)),
            "top1_reward_match": sum(top1_matches) / max(1, len(top1_matches)),
            "mean_policy_logp": float(policy_logps.mean().detach().cpu()),
        }
        if ref_logps is not None:
            metrics["mean_ref_logp"] = float(ref_logps.mean().detach().cpu())
            metrics["mean_logp_shift"] = float((policy_logps - ref_logps).mean().detach().cpu())
    return loss, metrics


@torch.no_grad()
def evaluate(policy, reference, dataloader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    policy.eval()
    if reference is not None:
        reference.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(dataloader, start=1):
        if args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
            break
        _, metrics = grpo_batch_loss(
            policy,
            reference,
            batch,
            policy_device=device,
            reference_device=device if reference is not None else None,
            advantage_mode=args.advantage_mode,
            reward_clip=args.reward_clip,
            kl_coef=args.kl_coef,
            length_normalize=args.length_normalize,
            entropy_bonus=args.entropy_bonus,
        )
        batch_size = int(batch["group_sizes"].shape[0])
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
    policy.train()
    if not count:
        return {}
    return {f"eval_{key}": round(value / count, 6) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 doctor with offline GRPO-style group-relative rewards from V6 candidate groups.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft-adapter-path", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--ref-adapter-path", type=Path, default=None)
    parser.add_argument("--group-data", type=Path, default=DEFAULT_GROUP_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-groups", type=int, default=9000)
    parser.add_argument("--eval-groups", type=int, default=512)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100, help="Periodic checkpoint interval. Use 0 to disable.")
    parser.add_argument(
        "--save-milestones",
        default="",
        help="Comma-separated checkpoint steps to save in addition to --save-steps. Steps above --max-steps are ignored.",
    )
    parser.add_argument("--max-eval-batches", type=int, default=128)
    parser.add_argument("--advantage-mode", choices=["center", "zscore", "rank"], default="zscore")
    parser.add_argument("--reward-clip", type=float, default=3.0)
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    parser.add_argument("--length-normalize", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run-data", action="store_true", help="Only load/summarize groups; do not load model.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip final evaluation after saving the adapter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data1_hf_cache()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_groups, eval_groups, data_summary = load_groups(
        args.group_data,
        max_groups=args.max_groups,
        eval_groups=args.eval_groups,
        seed=args.seed,
        min_candidates=args.min_candidates,
    )
    if args.dry_run_data:
        payload = {
            "dry_run_data": True,
            "group_data": str(args.group_data),
            "output_dir": str(args.output_dir),
            "data_summary": data_summary,
            "first_train_group_id": train_groups[0].get("id") if train_groups else None,
            "first_train_response_rewards": [
                reward_value(response)
                for response in (train_groups[0].get("responses") or [])
            ]
            if train_groups
            else [],
        }
        write_json(args.output_dir / "grpo_data_dry_run_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = RewardGroupDataset(
        train_groups,
        tokenizer,
        max_length=args.max_length,
        max_candidates=args.max_candidates,
    )
    eval_dataset = (
        RewardGroupDataset(
            eval_groups,
            tokenizer,
            max_length=args.max_length,
            max_candidates=args.max_candidates,
        )
        if eval_groups
        else None
    )
    collator = GroupCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        if eval_dataset is not None
        else None
    )
    del train_groups, eval_groups
    gc.collect()

    policy = load_trainable_policy(args, dtype).to(device)
    reference = load_reference(args, dtype).to(device) if args.kl_coef > 0 else None
    trainable = [param for param in policy.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = max(0, int(args.max_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, args.max_steps),
    )
    save_milestones = parse_save_milestones(args.save_milestones, args.max_steps)

    run_summary = {
        "method": "offline_grpo_style_group_relative_policy_optimization",
        "boundary": (
            "This is the first RL policy-update step after SFT. It uses precomputed V6 candidate groups "
            "whose rewards were obtained by executing each candidate in Patient Controller V2. It is not SimPO."
        ),
        "model_path": str(args.model_path),
        "sft_adapter_path": str(args.sft_adapter_path),
        "ref_adapter_path": str(args.ref_adapter_path) if args.ref_adapter_path else str(args.sft_adapter_path),
        "group_data": str(args.group_data),
        "output_dir": str(args.output_dir),
        "data_summary": data_summary,
        "max_length": args.max_length,
        "max_candidates": args.max_candidates,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_milestones": sorted(save_milestones),
        "advantage_mode": args.advantage_mode,
        "reward_clip": args.reward_clip,
        "kl_coef": args.kl_coef,
        "entropy_bonus": args.entropy_bonus,
        "length_normalize": args.length_normalize,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(args.output_dir / "grpo_train_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))

    policy.train()
    global_step = 0
    micro_step = 0
    accum_metrics: dict[str, list[float]] = {}
    while global_step < args.max_steps:
        for batch in train_loader:
            loss, metrics = grpo_batch_loss(
                policy,
                reference,
                batch,
                policy_device=device,
                reference_device=device if reference is not None else None,
                advantage_mode=args.advantage_mode,
                reward_clip=args.reward_clip,
                kl_coef=args.kl_coef,
                length_normalize=args.length_normalize,
                entropy_bonus=args.entropy_bonus,
            )
            (loss / args.gradient_accumulation_steps).backward()
            micro_step += 1
            for key, value in metrics.items():
                accum_metrics.setdefault(key, []).append(float(value))
            if micro_step % args.gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                row = {
                    "step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    **{key: round(sum(values) / len(values), 6) for key, values in accum_metrics.items()},
                }
                print(json.dumps(row, ensure_ascii=False))
                accum_metrics = {}
            if eval_loader is not None and global_step % args.eval_steps == 0:
                eval_metrics = evaluate(policy, reference, eval_loader, device, args)
                print(json.dumps({"step": global_step, **eval_metrics}, ensure_ascii=False))
            should_save_periodic = args.save_steps > 0 and global_step % args.save_steps == 0
            should_save_milestone = global_step in save_milestones
            if should_save_periodic or should_save_milestone:
                checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                policy.save_pretrained(str(checkpoint_dir))
            if global_step >= args.max_steps:
                break

    final_dir = args.output_dir / "final_lora_adapter"
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    final_eval = {} if args.skip_final_eval else (evaluate(policy, reference, eval_loader, device, args) if eval_loader is not None else {})
    run_summary.update(
        {
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_lora_adapter": str(final_dir),
            "final_eval": final_eval,
        }
    )
    write_json(args.output_dir / "grpo_train_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
