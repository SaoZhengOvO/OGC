#!/usr/bin/env python3
"""独立重算评测指标，与 `src/evaluate.py` 的 score.json 对拍（纯 CPU）。

为什么需要
----------
`score.json` 里带着一条 `_caveat`：**「本地近似实现，官方评分脚本未下发；
绝对值仅供参考，请用版本间对比。」**

这条警告说的是"与官方脚本可能有差异"，但它**不覆盖"本地脚本自己有没有算错"**。
如果 `evaluate.py` 在聚合上写错了（比如三维均值算成了别的、或者把 memory_refs 也计入），
那我这一夜所有结论的地基就是歪的。所以这里**按 `COMPUTE.md` / `evaluate.py` 的口径独立重写一遍**，
拿真实产物对拍。

重算口径（与官方说明一致）：
  情绪 Acc        = 命中的条数 / 可比对条数
  画像逐维 EMA    = 「两者都为空」或「集合完全相等」的条数 / 可比对条数
  画像 Acc(三维均值) = 上面三个 EMA 的算术平均
  逐维 macro Jaccard = 空并集记 1.0，否则 |交| / |并|，再按条数平均

用法::

    python tools/emotion_profile/verify_metrics.py --tag eval-v3-smoke
    python tools/emotion_profile/verify_metrics.py --tag eval-v3-smoke --tag eval-smoke-official
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIMS = ("personality_traits", "interests", "style")


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recompute(predictions: list[dict], references: list[dict]) -> dict:
    """从头算一遍情绪与画像的指标。"""
    reference_by_id = {row["id"]: row for row in references}
    emotion_hits = 0
    dim_hits = {dim: 0 for dim in DIMS}
    jaccard_sum = {dim: 0.0 for dim in DIMS}
    compared = 0

    for prediction in predictions:
        reference = reference_by_id.get(prediction["id"])
        if reference is None:
            continue
        compared += 1
        target = reference.get("target") or {}
        if prediction.get("emotion_label") == target.get("emotion_label"):
            emotion_hits += 1

        gold_profile = target.get("user_profile") or {}
        pred_profile = prediction.get("user_profile") or {}
        for dim in DIMS:
            gold = set(gold_profile.get(dim) or [])
            pred = set(pred_profile.get(dim) or [])
            if not gold and not pred:
                dim_hits[dim] += 1
                jaccard_sum[dim] += 1.0
            elif gold == pred and gold:
                dim_hits[dim] += 1
                jaccard_sum[dim] += 1.0
            else:
                union = gold | pred
                jaccard_sum[dim] += (len(gold & pred) / len(union)) if union else 1.0

    if compared == 0:
        return {"n": 0}
    return {
        "n": compared,
        "emotion_accuracy": emotion_hits / compared,
        "profile_dimension_ema": {dim: dim_hits[dim] / compared for dim in DIMS},
        "profile_accuracy_mean_3dim": sum(dim_hits[dim] / compared for dim in DIMS) / len(DIMS),
        "macro_jaccard": {dim: jaccard_sum[dim] / compared for dim in DIMS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="独立重算指标并与 score.json 对拍")
    parser.add_argument("--tag", action="append", required=True, help="评测 tag（runs/<tag>），可重复")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="允许的绝对误差")
    args = parser.parse_args(argv)

    references_file = REPO_ROOT / "data" / "processed" / "val_references.jsonl"
    if not references_file.exists():
        raise SystemExit(f"缺 {references_file}")
    references = read_jsonl(references_file)

    failures = 0
    for tag in args.tag:
        run_dir = REPO_ROOT / "runs" / tag
        score_file = run_dir / "score.json"
        submission_file = run_dir / "submission.jsonl"
        if not score_file.exists() or not submission_file.exists():
            print(f"[跳过] {tag}：缺 score.json 或 submission.jsonl")
            continue

        mine = recompute(read_jsonl(submission_file), references)
        theirs = json.loads(score_file.read_text(encoding="utf-8"))["metrics"]

        print(f"\n=== {tag}（可比对 {mine['n']} 条）===")
        checks = [
            ("情绪 Acc", mine["emotion_accuracy"], theirs["emotion_accuracy"]),
            ("画像 Acc(三维均值)", mine["profile_accuracy_mean_3dim"], theirs["profile_accuracy_mean_3dim"]),
        ]
        for dim in DIMS:
            checks.append((
                f"  {dim} EMA",
                mine["profile_dimension_ema"][dim],
                theirs["profile_dimension_scores"][dim]["exact_match_acc"],
            ))
            checks.append((
                f"  {dim} macro-Jaccard",
                mine["macro_jaccard"][dim],
                theirs["profile_dimension_scores"][dim]["macro_jaccard"],
            ))

        print(f"  {'指标':<26}{'我独立重算':>12}{'evaluate.py':>13}{'差':>12}  判")
        print("  " + "-" * 70)
        for label, computed, reported in checks:
            diff = computed - reported
            ok = abs(diff) <= args.tolerance
            if not ok:
                failures += 1
            print(f"  {label:<26}{computed:>12.6f}{reported:>13.6f}{diff:>+12.2e}  {'✅' if ok else '❌'}")

    print()
    if failures:
        print(f"❌ 有 {failures} 项对不上 —— evaluate.py 的聚合可能与口径不一致，先查清再往下走。")
        return 1
    print("✅ 全部对上。说明 evaluate.py 的聚合与口径一致。")
    print("   注意：这只排除「本地算错」，不排除「与官方脚本有差异」——"
          "后者由 score.json 的 _caveat 声明，本地无法验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
