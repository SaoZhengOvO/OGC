#!/usr/bin/env python3
"""切片的代表性体检：等距切片到底代不代表全量（纯 CPU）。

背景（为什么要做这个）
--------------------
`docs/reports/role2-emotion-profile-v1-v3.md` §16 量出一个硬事实：
**val 的前 20 条（= 团队 smoke 口径）比全量容易约 25 个百分点** ——
裸模型在这 20 条上情绪 Acc 0.5000，全量实测只有 0.2464。
所以任何切片在被采用前，都要先体检「它像不像全量」。

本脚本用**与难度相关的样本特征**做代理检验（不需要跑模型）：
最后一条用户消息长度、对话轮数、金标回复长度、金标情绪分布、逐维金标留空率。
等距切片若在这些维度上都贴着全量，才有资格做决策集。

用法::

    python tools/emotion_profile/slice_representativeness.py
    python tools/emotion_profile/slice_representativeness.py --slice runs/stratified15
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
DIMS = ("personality_traits", "interests", "style")


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_user_text(sample: dict) -> str:
    """最后一条用户消息。"""
    turns = [t for t in sample.get("history", []) if t.get("role") == "user"]
    if not turns:
        return ""
    turns.sort(key=lambda t: t.get("turn_id", 0))
    return turns[-1].get("content", "") or ""


def profile_of(sample: dict) -> dict:
    """金标画像。"""
    return (sample.get("target") or {}).get("user_profile") or {}


def features(rows: list[dict]) -> dict:
    """一个切片的特征统计。"""
    last_lengths = [len(last_user_text(row)) for row in rows]
    turn_counts = [len([t for t in row.get("history", []) if t.get("role") == "user"]) for row in rows]
    ref_lengths = [len((row.get("target") or {}).get("response_text") or "") for row in rows]
    emotions = Counter((row.get("target") or {}).get("emotion_label") for row in rows)
    total = len(rows)
    empty = {dim: sum(1 for row in rows if not profile_of(row).get(dim)) / total for dim in DIMS}
    return {
        "n": total,
        "last_user_mean": statistics.fmean(last_lengths),
        "last_user_median": statistics.median(last_lengths),
        "turns_mean": statistics.fmean(turn_counts),
        "ref_len_mean": statistics.fmean(ref_lengths),
        "ref_len_median": statistics.median(ref_lengths),
        "anxiety_share": emotions.get("anxiety", 0) / total,
        "empty": empty,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="切片代表性体检")
    parser.add_argument("--slice", default=None, help="额外的切片目录（相对仓库根），可重复用逗号分隔")
    args = parser.parse_args(argv)

    samples = read_jsonl(PROCESSED / "val_samples.jsonl")
    full = features(samples)

    slices: dict[str, list[dict]] = {
        "前 20（smoke）": samples[:20],
        "前 200": samples[:200],
        "等距 200": [],
    }
    reference_by_id = {row["id"]: row for row in read_jsonl(PROCESSED / "val_references.jsonl")}
    step = len(samples) / 200
    slices["等距 200"] = [samples[int(index * step)] for index in range(200)]

    if args.slice:
        for name in args.slice.split(","):
            path = REPO_ROOT / name.strip() / "val_samples.jsonl"
            if path.exists():
                slices[f"切片 {name.strip()}"] = read_jsonl(path)

    print(f"{'切片':<20}{'n':>6}{'用户末句均长':>13}{'轮数均值':>9}{'参考均长':>9}"
          f"{'anxiety占比':>12}{'空p':>8}{'空i':>8}{'空s':>8}")
    print("-" * 96)
    rows = [("全量 2354", full)] + [(name, features(subset)) for name, subset in slices.items()]
    for name, stat in rows:
        print(f"{name:<20}{stat['n']:>6}{stat['last_user_mean']:>13.1f}{stat['turns_mean']:>9.2f}"
              f"{stat['ref_len_mean']:>9.1f}{stat['anxiety_share']:>12.4f}"
              f"{stat['empty']['personality_traits']:>8.4f}{stat['empty']['interests']:>8.4f}"
              f"{stat['empty']['style']:>8.4f}")

    print()
    print("与全量的偏离（越小越像全量）：")
    print(f"{'切片':<20}{'末句长相对差':>13}{'轮数相对差':>11}{'参考长相对差':>13}"
          f"{'anxiety差':>10}{'空s差':>9}")
    print("-" * 78)
    for name, stat in rows[1:]:
        print(f"{name:<20}"
              f"{abs(stat['last_user_mean'] - full['last_user_mean']) / full['last_user_mean']:>13.3f}"
              f"{abs(stat['turns_mean'] - full['turns_mean']) / full['turns_mean']:>11.3f}"
              f"{abs(stat['ref_len_mean'] - full['ref_len_mean']) / full['ref_len_mean']:>13.3f}"
              f"{abs(stat['anxiety_share'] - full['anxiety_share']):>10.4f}"
              f"{abs(stat['empty']['style'] - full['empty']['style']):>9.4f}")

    print()
    print("读法：这些是**与难度相关的代理特征**。前 20 条被实测证明偏易 25 个百分点，")
    print("      如果它在这些维度上也偏离全量最多，就与那条实测互相印证；")
    print("      而等距切片应当处处贴着全量 —— 这才是它配做决策集的原因。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
