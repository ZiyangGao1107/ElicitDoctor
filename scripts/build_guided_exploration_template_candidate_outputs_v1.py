from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def clean_question(value: Any, *, language: str = "zh") -> str:
    text = clean_text(value)
    text = re.sub(r"^(Doctor|Question|Assistant|doctor|question|assistant)\s*[:：]\s*", "", text)
    text = text.strip("`\"' ")
    if not text:
        text = "What has been bothering you the most recently?" if language == "en" else "\u6700\u8fd1\u54ea\u65b9\u9762\u6700\u8ba9\u4f60\u89c9\u5f97\u56f0\u6270\uff1f"
    if not text.endswith(("?", "\uff1f")):
        suffix = "?" if language == "en" else "\uff1f"
        text = text.rstrip("\u3002.!?\uff1f\uff01 ") + suffix
    return text


def lower_pressure_wrap(prefix: str, question: str, *, language: str = "zh") -> str:
    question = clean_question(question, language=language)
    if question.startswith(prefix):
        return question
    return prefix + question


SENSITIVE_PATTERNS: dict[str, list[str]] = {
    "suicide_or_self_harm": ["自杀", "自伤", "伤害自己", "不想活", "轻生", "结束生命"],
    "hopelessness": ["绝望", "悲观", "没有希望", "撑不下去"],
    "psychosis_or_mania": ["幻觉", "妄想", "被害", "兴奋", "情绪高涨", "睡得很少也不困"],
}

LOW_SENSITIVITY_PATTERNS: dict[str, list[str]] = {
    "sleep": ["睡", "失眠", "早醒", "睡眠"],
    "appetite": ["胃口", "食欲", "吃", "体重"],
    "energy": ["精力", "疲惫", "疲劳", "没劲"],
    "function": ["学习", "工作", "生活", "影响", "功能"],
    "duration": ["多久", "持续", "开始", "最近"],
}

AVOIDANCE_CUES = [
    "不想说",
    "不太想说",
    "没什么",
    "不知道",
    "还好",
    "算了",
    "不方便",
    "不愿意",
    "不想聊",
]

SENSITIVE_PATTERNS_EN: dict[str, list[str]] = {
    "suicide_or_self_harm": ["suicide", "self-harm", "hurt yourself", "kill yourself", "end your life", "not want to live"],
    "hopelessness": ["hopeless", "no hope", "worthless", "can't go on", "nothing to look forward"],
    "psychosis_or_mania": ["hallucination", "hearing voices", "paranoid", "mania", "manic", "too much energy"],
}

LOW_SENSITIVITY_PATTERNS_EN: dict[str, list[str]] = {
    "sleep": ["sleep", "insomnia", "wake up", "tired"],
    "appetite": ["appetite", "eating", "food", "weight"],
    "energy": ["energy", "fatigue", "tired", "exhausted"],
    "function": ["work", "school", "daily", "function", "impact", "activities"],
    "duration": ["how long", "duration", "started", "recently", "past"],
}

AVOIDANCE_CUES_EN = [
    "don't want to talk",
    "do not want to talk",
    "rather not",
    "not sure",
    "i don't know",
    "nothing much",
    "skip",
    "hard to say",
]


def infer_focus(text: str, *, language: str = "zh") -> str:
    compact = clean_text(text)
    if language == "en":
        lowered = compact.lower()
        for focus, patterns in SENSITIVE_PATTERNS_EN.items():
            if any(pattern in lowered for pattern in patterns):
                return focus
        for focus, patterns in LOW_SENSITIVITY_PATTERNS_EN.items():
            if any(pattern in lowered for pattern in patterns):
                return focus
        return "general_distress"
    for focus, patterns in SENSITIVE_PATTERNS.items():
        if any(pattern in compact for pattern in patterns):
            return focus
    for focus, patterns in LOW_SENSITIVITY_PATTERNS.items():
        if any(pattern in compact for pattern in patterns):
            return focus
    return "general_distress"


def bridge_question(focus: str, *, language: str = "zh") -> str:
    if language == "en":
        if focus == "suicide_or_self_harm":
            return "Without going into the most sensitive details, how have you been getting through the moments when your mood feels lowest?"
        if focus == "hopelessness":
            return "Have there been times recently when things felt especially hard to get through, or when you did not have much to look forward to?"
        if focus == "psychosis_or_mania":
            return "Have you noticed any recent periods when you felt very different from your usual self or hard to control?"
        if focus == "sleep":
            return "Let's start with something easier: how has your sleep changed compared with usual?"
        if focus == "appetite":
            return "Compared with usual, has your appetite or eating changed noticeably recently?"
        if focus == "energy":
            return "Have you been feeling noticeably less energy or more effort doing ordinary things recently?"
        if focus == "function":
            return "How much have these feelings been affecting work, school, or your usual daily routine?"
        if focus == "duration":
            return "Has this discomfort started recently, or has it been going on for a while?"
        return "Recently, has the main difficulty been your mood, sleep, physical energy, or how daily life is being affected?"
    if focus == "suicide_or_self_harm":
        return "先不问很敏感的细节，最近情绪最低的时候，你一般是怎么让自己撑过去的？"
    if focus == "hopelessness":
        return "最近有没有哪一段时间，你会觉得事情特别难熬，或者对接下来没什么期待？"
    if focus == "psychosis_or_mania":
        return "最近有没有出现过让你觉得和平时很不一样、自己也有点难控制的状态？"
    if focus == "sleep":
        return "我们先从容易说的开始，最近睡眠大概比平时差了多少？"
    if focus == "appetite":
        return "最近吃东西这件事和以前比，有没有明显变少、变多，或者没什么胃口？"
    if focus == "energy":
        return "最近一天里有没有明显觉得没力气、做事情比以前更费劲？"
    if focus == "function":
        return "最近这些状态对学习、工作或者日常安排影响大概到什么程度？"
    if focus == "duration":
        return "这种不舒服大概是最近才出现，还是已经持续一段时间了？"
    return "最近最让你觉得困扰的是情绪、睡眠、身体状态，还是日常生活受影响？"


def choice_question(focus: str, *, language: str = "zh") -> str:
    if language == "en":
        if focus == "suicide_or_self_harm":
            return "For this question, you can answer broadly: not at all, occasional thoughts without intent, frequent thoughts, or you would rather not answer right now?"
        if focus == "hopelessness":
            return "Would you say it has been more like occasional low mood, often feeling hopeless, or mainly stress that you can still manage?"
        if focus == "psychosis_or_mania":
            return "Recently, has it been closer to racing thoughts, unusually high energy noticed by others, hearing or seeing things others do not, or none of those?"
        if focus == "sleep":
            return "Which best fits your sleep recently: trouble falling asleep, waking during the night, waking early, sleeping more but not rested, or mostly normal?"
        if focus == "appetite":
            return "Which best fits your eating recently: much less, much more, no real change, or you would rather not say?"
        if focus == "energy":
            return "Which best fits your energy recently: mostly normal, clearly lower, up and down, or hard to tell?"
        if focus == "function":
            return "Is the impact mainly on work or school, relationships, home routines, or not much impact right now?"
        return "If you had to pick one area, is the main difficulty mood, sleep, energy, appetite, or daily functioning?"
    if focus == "suicide_or_self_harm":
        return "这个问题可以只选一个大概情况：完全没有、偶尔闪过但不会做、比较频繁，或者你现在不想回答？"
    if focus == "hopelessness":
        return "如果粗略分一下，最近更像是偶尔低落、经常觉得没希望，还是只是压力大但还能撑住？"
    if focus == "psychosis_or_mania":
        return "最近有没有更接近这几种情况：思路停不下来、别人觉得你异常兴奋、听到看到别人没有的东西，或者都没有？"
    if focus == "sleep":
        return "最近睡眠更像是哪一种：入睡困难、半夜醒、早醒、睡得多但不解乏，还是基本正常？"
    if focus == "appetite":
        return "最近吃饭更像是哪一种：明显吃少、明显吃多、没什么变化，或者不太想说？"
    if focus == "energy":
        return "最近精力更像是哪一种：基本正常、明显下降、忽高忽低，还是说不清？"
    if focus == "function":
        return "现在影响更主要在学习工作、人际相处、家里日常，还是暂时影响不大？"
    return "如果只选一个方向，最近更困扰你的是情绪、睡眠、精力、吃饭，还是日常功能？"


def permission_question(focus: str, base: str, *, language: str = "zh") -> str:
    if language == "en":
        if focus in {"suicide_or_self_harm", "psychosis_or_mania", "hopelessness"}:
            return "This may feel sensitive, and a brief yes/no or skipping it is okay: " + base
        return "I want to check one specific point, and a brief answer is enough: " + base
    if focus in {"suicide_or_self_harm", "psychosis_or_mania", "hopelessness"}:
        return (
            "接下来这个问题可能有点敏感，你可以只回答“有/没有”或者说暂时不想答："
            + base
        )
    return "我想确认一个具体点的小问题，你可以简单说大概情况：" + base


def repair_question(focus: str, prior_patient: str, *, language: str = "zh") -> str:
    if language == "en":
        lowered = clean_text(prior_patient).lower()
        avoided = any(cue in lowered for cue in AVOIDANCE_CUES_EN)
        if avoided:
            return "That's okay, we do not have to go into that part now. From an easier angle, " + bridge_question(focus, language=language)
        return "We do not need to go into details yet. " + bridge_question(focus, language=language)
    avoided = any(cue in prior_patient for cue in AVOIDANCE_CUES)
    if avoided:
        return "没关系，刚才那部分我们先不展开。换个容易一点的角度，" + bridge_question(focus)
    return "我们先不用说太细，" + bridge_question(focus)


def strategy_question(strategy: str, reference_question: str, prior_patient: str = "", *, language: str = "zh") -> str:
    base = clean_question(reference_question, language=language)
    focus = infer_focus(base + " " + prior_patient, language=language)
    if strategy == "direct":
        return base
    if strategy == "gentle_low_burden":
        if language == "en":
            return lower_pressure_wrap(
                "If you are comfortable, we can keep this brief and general: ",
                base,
                language=language,
            )
        return lower_pressure_wrap(
            "如果你愿意，我们先不用说细节，只说一点大概情况：",
            base,
            language=language,
        )
    if strategy == "bridge_low_sensitivity":
        return bridge_question(focus, language=language)
    if strategy == "permission_before_sensitive":
        return permission_question(focus, base, language=language)
    if strategy == "repair_after_refusal":
        return repair_question(focus, clean_text(prior_patient), language=language)
    if strategy == "choice_based_concrete":
        return choice_question(focus, language=language)
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic strategy-aware doctor candidate outputs from guided exploration requests."
    )
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--provider-tag", default="template_guided_exploration_candidate")
    parser.add_argument("--model-tag", default="Template-GuidedExplorationCandidate-v1")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for request in iter_jsonl(args.request_path):
        metadata = request.get("metadata") or {}
        strategy = str(request.get("strategy") or metadata.get("strategy") or "neutral")
        reference_question = clean_text(metadata.get("reference_doctor_question"))
        prior_patient = clean_text(metadata.get("reference_patient_response"))
        question = strategy_question(strategy, reference_question, prior_patient, language=args.language)
        rows.append(
            {
                "request_id": request.get("request_id"),
                "doctor_question": question,
                "raw_output": question,
                "provider": args.provider_tag,
                "model": args.model_tag,
                "adapter": None,
                "prompt_policy": request.get("policy_name"),
                "method": request.get("method"),
                "base_severity": request.get("base_severity"),
                "turn_index": request.get("turn_index"),
                "metadata": {
                    "strategy": strategy,
                    "strategy_name": request.get("strategy_name"),
                    "strategy_variant_index": request.get("strategy_variant_index"),
                    "source": "deterministic_strategy_wrapper",
                },
            }
        )
    write_jsonl(args.output_path, rows)
    print(
        json.dumps(
            {
                "request_path": str(args.request_path),
                "output_path": str(args.output_path),
                "outputs": len(rows),
                "provider": args.provider_tag,
                "model": args.model_tag,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
