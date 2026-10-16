#!/usr/bin/env python3
"""用真值探测「纯关键词规则」在情绪分类上能拿多少分（纯 CPU，零 GPU）。

动机
----
情绪识别是 25 分、是两条线里权重大的一块。判断提示词规则有没有指望，先要知道
**光靠触发词能到哪**。这个脚本把 `prompts` 里那套触发词直接当成分类器套在
全量 2354 条真值上，量出：

1. 纯关键词分类器的准确率（对比「一律答 anxiety」的零成本下限 0.3190）；
2. 每个情绪类里，有多少条的用户原话**含有该类自己的触发词**（= 触发词的召回）；
3. 哪些类一条都抓不到 —— 那些必须靠语义判断，写再多触发词也没用。

用法::

    python tools/emotion_profile/emotion_rule_probe.py
    python tools/emotion_profile/emotion_rule_probe.py --fallback neutral
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

#: 按顺序匹配，先命中先算 —— 顺序本身就是「优先级」的体现，
#: 与 prompts/emotion_profile.md 的第一步触发词表一致。
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("anger", ("烦", "烦躁", "气", "凭什么", "受不了", "摔键盘", "愤怒", "火大")),
    ("fear", ("怕", "害怕", "恐惧", "不敢", "吓")),
    ("helplessness", ("无力", "没办法", "改变不了", "只能忍着", "看不到头", "无能为力")),
    ("sadness", ("难过", "伤心", "委屈", "想哭", "低落")),
    ("loneliness", ("孤单", "孤独", "没人懂", "没人说话", "没人可以说", "一个人", "空落")),
    ("anxiety", ("担心", "焦虑", "慌", "静不下来", "怕来不及", "悬着", "紧张")),
    ("relaxed", ("轻松", "放下了", "踏实", "平静", "松了一口气", "舒服")),
    ("gratitude", ("谢谢", "感谢", "谢了")),
    ("joy", ("开心", "太好了", "期待", "高兴", "兴奋")),
    ("surprise", ("没想到", "居然", "怎么会", "竟然")),
    ("shame", ("丢脸", "都怪我", "我不配", "羞愧", "自责")),
    ("disgust", ("恶心", "反感", "嫌弃", "厌恶")),
    ("care", ("担心你", "你还好吗", "心疼")),
    ("pride", ("我做到了", "还不错", "自豪", "骄傲")),
]

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_user_turn(sample: dict) -> str:
    """取最后一条用户消息。"""
    turns = [t for t in sample.get("history", []) if t.get("role") == "user"]
    if not turns:
        return ""
    turns.sort(key=lambda t: t.get("turn_id", 0))
    return turns[-1].get("content", "") or ""


def classify(text: str, fallback: str) -> str:
    """按 RULES 顺序匹配；都不中就用 fallback。"""
    for emotion, patterns in RULES:
        for pattern in patterns:
            if pattern in text:
                return emotion
    return fallback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="纯关键词情绪分类器探针")
    parser.add_argument("--fallback", default="anxiety", choices=EMOTIONS, help="规则都不中时的兜底类")
    args = parser.parse_args(argv)

    references = read_jsonl(PROCESSED / "val_references.jsonl")
    samples = {row["id"]: row for row in read_jsonl(PROCESSED / "val_samples.jsonl")}

    texts: dict[str, str] = {}
    gold: dict[str, str] = {}
    for reference in references:
        sample = samples.get(reference["id"])
        if sample is None:
            continue
        texts[reference["id"]] = last_user_turn(sample)
        gold[reference["id"]] = (reference.get("target") or {}).get("emotion_label")

    total = len(texts)
    gold_counter = Counter(gold.values())
    majority, majority_count = gold_counter.most_common(1)[0]

    # --- 1) 纯关键词分类器的整体准确率 ---
    predicted = {key: classify(text, args.fallback) for key, text in texts.items()}
    correct = sum(1 for key in texts if predicted[key] == gold[key])
    print(f"全量 {total} 条；零成本下限（一律 {majority}）= {majority_count / total:.4f}")
    print(f"纯关键词分类器（兜底 {args.fallback}）准确率 = {correct / total:.4f}")
    print(f"  与下限之差 = {(correct - majority_count) / total:+.4f}\n")

    # --- 2) 逐类：触发词的召回与精确率 ---
    pattern_of = {emotion: patterns for emotion, patterns in RULES}

    def own_trigger(text: str, emotion: str) -> bool:
        """该类自己的触发词是否出现（不看优先级顺序）。"""
        return any(pattern in text for pattern in pattern_of.get(emotion, ()))

    print(f"{'情绪':<16}{'金标数':>7}{'自身触发':>9}{'判对':>7}{'预测数':>8}{'精确率':>9}  说明")
    print("-" * 80)
    rows = []
    for emotion in EMOTIONS:
        support = gold_counter.get(emotion, 0)
        fired = sum(1 for key in texts if gold[key] == emotion and own_trigger(texts[key], emotion))
        true_positive = sum(1 for key in texts if gold[key] == emotion and predicted[key] == emotion)
        predicted_count = sum(1 for value in predicted.values() if value == emotion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        trigger_rate = fired / support if support else 0.0
        recall = true_positive / support if support else 0.0
        note = ""
        if support and fired == 0:
            note = "自身触发词一条都没出现 → 只能靠语义判断"
        elif support and trigger_rate < 0.10:
            note = "自身触发词几乎抓不到"
        elif support and recall == 0:
            note = "有触发词但全被更高优先级的类抢走"
        rows.append((emotion, support, trigger_rate, recall, predicted_count, precision, note))

    for emotion, support, trigger_rate, recall, predicted_count, precision, note in sorted(
        rows, key=lambda item: -item[1]
    ):
        print(f"{emotion:<16}{support:>7}{trigger_rate:>9.4f}{recall:>7.4f}"
              f"{predicted_count:>8}{precision:>9.4f}  {note}")

    # --- 3) 兜底类吃了多少（anxiety 陷阱的量化） ---
    print()
    fallback_share = sum(1 for value in predicted.values() if value == args.fallback) / total
    print(f"兜底类 {args.fallback} 占了预测的 {fallback_share:.4f}；"
          f"金标里 {args.fallback} 只占 {gold_counter.get(args.fallback, 0) / total:.4f}")
    print()
    print("读法：")
    print("  · 「触发词召回」= 该类金标里，有多少条的用户原话含该类自己的触发词。")
    print("    这个数很低，说明关键词规则对此类无效，提示词再堆触发词也没用。")
    print("  · 纯关键词分类器的准确率若低于下限，说明**规则本身**不是答案，")
    print("    价值在于给模型「可判定的抓手」，而不是替代模型判断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
