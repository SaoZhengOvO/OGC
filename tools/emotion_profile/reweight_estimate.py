#!/usr/bin/env python3
"""把切片上的成绩**按全量类别分布重加权**，用来校正类别失衡。

⚠️ 先说局限（这是实测出来的，不是推测）
------------------------------------
这个工具**只能校正类别分布，校正不了难度偏差**。标定用的负结果：

    裸模型（official）在 val 前 20 条上 情绪 Acc = 0.5000
    重加权估计（按全量类别分布）        = 0.5537
    而 E1 在全量 2354 条上的实测         = 0.2464   ← 差了 0.31！

也就是说 **val 的前 20 条不只是类别偏，它系统性地更容易**（比全量高约 25 个百分点）。
重加权把类别拉平了，但样本本身"好做"这件事它拉不平。

**所以：不要把这里的估计当成全量的绝对预测。** 它的正当用途是：
在**等距切片**（`make_eval_slice.py`）这种本来就不偏的样本上，做一次小幅类别校正；
或者用来比较两个臂**在同样偏的切片上**谁更占类别便宜。

要绝对数字，只有两条路：跑全量，或者跑等距切片。

原理
----
    w_i = P_全量(该样本的金标类别) / P_切片(该样本的金标类别)
    重加权估计 = Σ w_i · hit_i / Σ w_i

区间用 bootstrap：在切片内按样本有放回重抽，重抽后重算。

用法::

    python tools/emotion_profile/reweight_estimate.py --per-sample runs/eval-v3-smoke/samples.jsonl
    python tools/emotion_profile/reweight_estimate.py --per-sample X.jsonl --metric profile
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
DIMS = ("personality_traits", "interests", "style")


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hit_of(row: dict, metric: str, dim: str | None) -> int:
    """某条样本在某指标上是否命中。"""
    if metric == "emotion":
        return 1 if row.get("emotion_hit") else 0
    gold = sorted((row.get("profile_gold") or {}).get(dim) or [])
    pred = sorted((row.get("profile_pred") or {}).get(dim) or [])
    return 1 if gold == pred else 0


def reweighted(rows: list[dict], full_dist: dict[str, float], metric: str, dim: str | None) -> float:
    """按全量类别分布重加权的估计。"""
    slice_dist = Counter(row.get("emotion_gold") for row in rows)
    total = len(rows)
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        gold_class = row.get("emotion_gold")
        slice_p = slice_dist[gold_class] / total
        full_p = full_dist.get(gold_class, 0.0)
        if slice_p <= 0:
            continue
        weight = full_p / slice_p
        numerator += weight * hit_of(row, metric, dim)
        denominator += weight
    return numerator / denominator if denominator else 0.0


def bootstrap_ci(rows, full_dist, metric, dim, *, rounds: int = 2000, seed: int = 20261016) -> tuple[float, float]:
    """bootstrap 95% 区间。"""
    rng = random.Random(seed)
    total = len(rows)
    samples = []
    for _ in range(rounds):
        resampled = [rows[rng.randrange(total)] for _ in range(total)]
        samples.append(reweighted(resampled, full_dist, metric, dim))
    samples.sort()
    return samples[int(rounds * 0.025)], samples[int(rounds * 0.975)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按全量类别分布重加权，预测全量成绩")
    parser.add_argument("--per-sample", required=True, help="src/cli.py eval --per-sample-out 的产物")
    parser.add_argument("--metric", default="emotion", choices=("emotion", "profile"), help="看情绪还是画像")
    parser.add_argument("--dim", default=None, choices=DIMS, help="--metric profile 时指定维度；不指定则三维各算一遍")
    args = parser.parse_args(argv)

    path = Path(args.per_sample)
    if not path.is_absolute():
        path = REPO_ROOT / path
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path} 是空的")

    references = read_jsonl(PROCESSED / "val_references.jsonl")
    full_counter = Counter((row.get("target") or {}).get("emotion_label") for row in references)
    full_total = sum(full_counter.values())
    full_dist = {key: value / full_total for key, value in full_counter.items()}

    slice_counter = Counter(row.get("emotion_gold") for row in rows)
    print(f"切片 {path.name}：{len(rows)} 条")
    print(f"{'金标情绪':<16}{'切片数':>7}{'切片占比':>10}{'全量占比':>10}{'权重':>9}")
    print("-" * 54)
    for emotion, count in sorted(slice_counter.items(), key=lambda item: -item[1]):
        slice_p = count / len(rows)
        full_p = full_dist.get(emotion, 0.0)
        weight = full_p / slice_p if slice_p else 0.0
        print(f"{emotion:<16}{count:>7}{slice_p:>10.4f}{full_p:>10.4f}{weight:>9.3f}")

    dims = [args.dim] if (args.metric == "profile" and args.dim) else (DIMS if args.metric == "profile" else [None])
    print()
    print(f"{'指标':<22}{'切片原始':>10}{'重加权估计':>12}{'95% 区间':>22}")
    print("-" * 68)
    for dim in dims:
        raw = sum(hit_of(row, args.metric, dim) for row in rows) / len(rows)
        estimate = reweighted(rows, full_dist, args.metric, dim)
        low, high = bootstrap_ci(rows, full_dist, args.metric, dim)
        label = "情绪 Acc" if args.metric == "emotion" else dim
        print(f"{label:<22}{raw:>10.4f}{estimate:>12.4f}   [{low:>7.4f}, {high:>7.4f}]")

    print()
    print("⚠️ 这不是全量的绝对预测：重加权只校正类别分布，校正不了难度偏差。")
    print("   标定负结果：裸模型在 val 前 20 条上 0.5000、重加权估计 0.5537，")
    print("   而 E1 全量实测只有 0.2464 —— 说明 smoke 那 20 条系统性地更容易。")
    print("   要绝对数字请跑全量或等距切片；本工具适合做小幅类别校正与臂间对比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
