"""从 val 里取一个**等距**切片，作为"比 n=20 可信、又比 n=2354 便宜"的决策集。

为什么需要它（有数据支撑的动机，见 tools/emotion_profile/LOG.md 第三轮）：

`python3 tools/eval_loop.py --limit 20` 取的是 val 的**前 20 条**，而它严重不代表全量：

| 切片 | 空p | 空i | 空s | 「一律留空」得分 |
|---|---|---|---|---|
| 前 20（smoke） | 0.6000 | 0.6500 | **0.2000** | **0.4833** |
| 全量 2354 | 0.6185 | 0.7489 | **0.5174** | **0.6283** |

style 的金标为空率差了 0.317 —— 「必须填 style」这个结论在 smoke 上的理由强度是全量的 2.6 倍。
val 内部也有位置漂移（中段 200 条的 personality 空率只有 0.39）。

所以在 smoke 上达标的规则，不一定在全量上不掉分。**等距切片**按固定步长跨全 val 取样，
对位置无偏，200 条约 8 分钟就能跑完，适合做"要不要把 v4 推上全量"的决策依据。

用法::

    python tools/emotion_profile/make_eval_slice.py --size 200
    # 产出 runs/slice200/{val_samples.jsonl,val_references.jsonl,slice_manifest.json}

    # 分层诊断切片：每类情绪固定取样，用来测「小类能不能救」
    # （注意：分层切片过采样了稀有类，它的**总体 Acc 有偏**，只能看 per-class）
    python tools/emotion_profile/make_eval_slice.py --strategy stratified --per-class 15

    # 然后直接喂给 eval_loop（--source 与 --references 都指到切片上）
    python3 tools/eval_loop.py --tag v4-slice200 \\
        --source runs/slice200/val_samples.jsonl \\
        --references runs/slice200/val_references.jsonl \\
        --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-v4

两种切片的用途分工
------------------
* ``even``（等距）：对位置无偏，**总体 Acc 可以代表全量** → 用来做"要不要改"的决策。
* ``stratified``（分层）：每类固定若干条，**总体 Acc 有偏、不能外推** →
  只用来看 per-class 的 P/R/F1，即"小类到底能不能救"。

为什么需要分层那一种：全量里 ``surprise`` 只有 9 条、``care`` 11 条，
200 条等距切片里各只有约 1 条，per-class 指标没有意义。
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


def profile_empty_rates(rows: list[dict]) -> dict[str, float]:
    """逐维金标为空率。"""
    total = len(rows)
    return {
        dim: sum(1 for row in rows if not ((row.get("target") or {}).get("user_profile") or {}).get(dim)) / total
        for dim in DIMS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 val 切片（决策集 / 诊断集）")
    parser.add_argument("--size", type=int, default=200, help="等距切片的条数（默认 200）")
    parser.add_argument(
        "--strategy",
        choices=("even", "stratified"),
        default="even",
        help="even=等距（总体无偏，决策用）；stratified=按金标情绪分层（总体有偏，只为看 per-class）",
    )
    parser.add_argument("--per-class", type=int, default=15, help="分层时每类取几条（小类全取）")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", default=None, help="输出目录（默认 runs/slice<size> 或 runs/stratified<per_class>）")
    args = parser.parse_args(argv)

    samples = read_jsonl(PROCESSED / "val_samples.jsonl")
    references = read_jsonl(PROCESSED / "val_references.jsonl")
    reference_by_id = {row["id"]: row for row in references}
    gold_emotion = {
        row["id"]: (row.get("target") or {}).get("emotion_label") for row in references
    }

    if args.strategy == "even":
        size = min(args.size, len(samples))
        step = len(samples) / size
        indices = [int(index * step) for index in range(size)]
        picked_samples = [samples[index] for index in indices]
        sampling = f"evenly spaced（步长 {step:.2f}，对位置无偏）"
        default_out = REPO_ROOT / "runs" / f"slice{size}"
        unbiased = True
    else:
        rng = random.Random(args.seed)
        by_emotion: dict[str, list[dict]] = {}
        for row in samples:
            by_emotion.setdefault(gold_emotion.get(row["id"], "<unknown>"), []).append(row)
        picked_samples = []
        for emotion in sorted(by_emotion):
            rows = by_emotion[emotion]
            picked_samples.extend(rows if len(rows) <= args.per_class else rng.sample(rows, args.per_class))
        order = {row["id"]: index for index, row in enumerate(samples)}
        picked_samples.sort(key=lambda row: order[row["id"]])
        sampling = f"stratified by gold emotion（每类 ≤{args.per_class} 条）"
        default_out = REPO_ROOT / "runs" / f"stratified{args.per_class}"
        unbiased = False

    picked_references = [reference_by_id[row["id"]] for row in picked_samples]

    out_dir = Path(args.out) if args.out else default_out
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in (("val_samples.jsonl", picked_samples), ("val_references.jsonl", picked_references)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    slice_rates = profile_empty_rates(picked_samples)
    full_rates = profile_empty_rates(samples)
    emotion_counts = Counter(gold_emotion.get(row["id"], "<unknown>") for row in picked_samples)
    full_emotion_counts = Counter(row.get("target", {}).get("emotion_label") for row in references)
    manifest = {
        "strategy": args.strategy,
        "sampling": sampling,
        "size": len(picked_samples),
        "val_total": len(samples),
        "unbiased_overall_acc": unbiased,
        "slice_empty_rates": slice_rates,
        "full_empty_rates": full_rates,
        "abs_diff": {dim: abs(slice_rates[dim] - full_rates[dim]) for dim in DIMS},
        "first_20_empty_rates": profile_empty_rates(samples[:20]),
        "emotion_counts": dict(emotion_counts.most_common()),
        "full_emotion_counts": dict(full_emotion_counts.most_common()),
        "note": (
            "等距切片：总体 Acc 可代表全量，用于决策。"
            if unbiased
            else "分层切片：过采样了稀有类，**总体 Acc 有偏、不能外推到全量**；只用来看 per-class。"
        ),
    }
    (out_dir / "slice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"切片: {out_dir}（{len(picked_samples)} 条，{sampling}）")
    print(f"{'维度':<20}{'切片':>9}{'全量':>9}{'|差|':>9}{'前20(对比)':>12}")
    for dim in DIMS:
        print(
            f"{dim:<20}{slice_rates[dim]:>9.4f}{full_rates[dim]:>9.4f}"
            f"{manifest['abs_diff'][dim]:>9.4f}{manifest['first_20_empty_rates'][dim]:>12.4f}"
        )
    worst = max(manifest["abs_diff"].values())
    print(f"\n切片与全量的最大逐维偏差: {worst:.4f}")
    print("\n情绪类分布（切片 vs 全量）：")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion:<16} 切片 {count:>4}   全量 {full_emotion_counts.get(emotion, 0):>5}")
    if not unbiased:
        print("\n⚠️ 这是分层切片，总体 Acc 有偏，不要拿它跟全量比，也不要拿它下总体结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
