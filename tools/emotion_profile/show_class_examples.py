#!/usr/bin/env python3
"""抽某个情绪类的真实样本原文，用来**写规则**（而不是拍脑袋编规则）。

为什么需要
----------
报告 §二十 发现 v3 的第二步只能到达 16 类里的 4 类，不可达样本 ≥550 条（全量的 23%）。
补规则时如果靠想象，多半会写出模型听不懂的措辞。正确做法是：
**把这一类里"第一步没命中触发词"的样本原文抽出来读，规则从真实措辞里长出来。**
v7 的 4 条规则就是这么来的（relaxed 的"松快多了/想通了"、shame 的"愧疚/掉价"等）。

用法::

    # 看 relaxed 和 mixed 里「第一步没命中」的样本
    python tools/emotion_profile/show_class_examples.py --emotion relaxed,mixed --only-no-trigger

    # 看全部 16 类里"无触发词"的占比（先知道该补哪几类）
    python tools/emotion_profile/show_class_examples.py --summary

    # 不过滤，看某个类的任意样本
    python tools/emotion_profile/show_class_examples.py --emotion pride --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emotion_rule_probe import EMOTIONS, RULES, last_user_turn  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_trigger(text: str) -> bool:
    """第一步的任一触发词是否出现。"""
    return any(pattern in text for _emotion, patterns in RULES for pattern in patterns)


def compact(text: str, width: int = 110) -> str:
    """压成一行，便于阅读。"""
    one_line = re.sub(r"\s+", " ", text).strip()
    return one_line if len(one_line) <= width else one_line[:width] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抽情绪类样本原文，用于写规则")
    parser.add_argument("--emotion", default=None, help="类名，可逗号分隔（默认全部）")
    parser.add_argument("--only-no-trigger", action="store_true", help="只看第一步没命中触发词的样本")
    parser.add_argument("--limit", type=int, default=6, help="每个类打印几条（默认 6）")
    parser.add_argument("--summary", action="store_true", help="只打印各类「无触发词」的条数与占比")
    args = parser.parse_args(argv)

    samples = {row["id"]: row for row in read_jsonl(PROCESSED / "val_samples.jsonl")}
    references = read_jsonl(PROCESSED / "val_references.jsonl")

    buckets: dict[str, list[str]] = {emotion: [] for emotion in EMOTIONS}
    total_no_trigger = 0
    for reference in references:
        sample = samples.get(reference["id"])
        if sample is None:
            continue
        emotion = (reference.get("target") or {}).get("emotion_label")
        if emotion not in buckets:
            continue
        text = last_user_turn(sample)
        no_trigger = not has_trigger(text)
        total_no_trigger += 1 if no_trigger else 0
        if args.only_no_trigger and not no_trigger:
            continue
        buckets[emotion].append(text)

    if args.summary:
        total = len(references)
        print(f"全量 {total} 条，其中「第一步没命中触发词」{total_no_trigger} 条 "
              f"（{total_no_trigger / total:.4f}）\n")
        print(f"{'情绪':<16}{'总数':>6}{'无触发词':>9}{'占该类':>9}{'占全量':>9}")
        print("-" * 52)
        rows = []
        for emotion in EMOTIONS:
            texts = buckets[emotion]
            if not texts:
                continue
            support = Counter()
            for reference in references:
                if (reference.get("target") or {}).get("emotion_label") == emotion:
                    support[emotion] += 1
            total_class = support[emotion]
            no_trigger = len(texts) if args.only_no_trigger else sum(
                1 for reference in references
                if (reference.get("target") or {}).get("emotion_label") == emotion
                and not has_trigger(last_user_turn(samples[reference["id"]]))
            )
            rows.append((emotion, total_class, no_trigger))
        for emotion, total_class, no_trigger in sorted(rows, key=lambda item: -item[2]):
            print(f"{emotion:<16}{total_class:>6}{no_trigger:>9}"
                  f"{no_trigger / total_class:>9.4f}{no_trigger / total:>9.4f}")
        print()
        print("读法：无触发词条数多、而第二步又没有该类的通路 → 那批样本结构性不可达。")
        print("      报告 §二十 就是靠这张表找到 relaxed(287) / mixed(112) / shame(53) / pride(30) 的。")
        return 0

    targets = [item.strip() for item in args.emotion.split(",")] if args.emotion else list(EMOTIONS)
    for emotion in targets:
        if emotion not in buckets:
            print(f"（未知情绪类：{emotion}）")
            continue
        texts = buckets[emotion]
        if not texts:
            print(f"\n=== {emotion}：没有符合条件的样本 ===")
            continue
        lengths = [len(text) for text in texts]
        print(f"\n{'=' * 80}\n{emotion}：{len(texts)} 条"
              f"{'（第一步未命中）' if args.only_no_trigger else ''}，"
              f"平均 {sum(lengths) / len(lengths):.0f} 字，看前 {min(args.limit, len(texts))} 条\n{'=' * 80}")
        for text in texts[:args.limit]:
            print(f"  · [{len(text):>3}字] {compact(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
