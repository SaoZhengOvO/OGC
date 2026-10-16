#!/usr/bin/env python3
"""格式健康度对照：把「解析失败」和「输出长度」放在一起看（纯 CPU）。

为什么需要
----------
报告 §19.2d 发现：解析失败不是随机的，它与**输出长度**强相关 ——
`mi300x-v4-smoke`（两篇提示词同时挂）里失败的两条是 307 / 381 字，
而同臂未失败样本均值只有 73 字。机制解释是**近因冲突**：
`wiring.py` 的顺序固定为 `official → emotion_profile → response`，
模型最后读到的是③那段「写 200-350 字、三到四个自然段」，
于是散文驱动压过了「必须输出 JSON object」。

这个结论有一个**可验证的预测**：如果集成跑的长度显著高于隔离跑、
且解析失败率同时上去，那么近因冲突成立；反之就要另找原因。
所以每次集成/对照跑完，都应当把这两个数一起看 —— 这个脚本就是干这个的。

用法::

    python tools/emotion_profile/format_health.py --tag v3-smoke --tag mi300x-v4-smoke
    python tools/emotion_profile/format_health.py --tag integ-slice200 --tag v8+p03-slice200
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="格式健康度对照（解析失败 + 输出长度）")
    parser.add_argument("--tag", action="append", required=True, help="评测 tag（自动补 eval- 前缀），可重复")
    args = parser.parse_args(argv)

    rows = []
    for raw_tag in args.tag:
        tag = raw_tag if raw_tag.startswith("eval-") else f"eval-{raw_tag}"
        run_dir = REPO_ROOT / "runs" / tag
        submission = run_dir / "submission.jsonl"
        if not submission.exists():
            print(f"[跳过] {tag}：没有 submission.jsonl")
            continue
        predictions = read_jsonl(submission)

        failure_ids: list[str] = []
        report_file = run_dir / "performance_report.json"
        if report_file.exists():
            failure_ids = json.loads(report_file.read_text(encoding="utf-8")).get("parse_failure_ids") or []
        failed_set = set(failure_ids)

        lengths = [len(row.get("response_text") or "") for row in predictions]
        failed_lengths = [len(row.get("response_text") or "") for row in predictions if row["id"] in failed_set]
        ok_lengths = [length for row, length in zip(predictions, lengths) if row["id"] not in failed_set]

        # 被格式残留污染的条数：回复以字段名开头
        polluted = sum(1 for row in predictions
                       if (row.get("response_text") or "").lstrip().startswith(("response_text:", '"response_text"')))

        scores = {}
        score_file = run_dir / "score.json"
        if score_file.exists():
            metrics = json.loads(score_file.read_text(encoding="utf-8"))["metrics"]
            scores = {
                "emotion": metrics["emotion_accuracy"],
                "profile": metrics["profile_accuracy_mean_3dim"],
                "bleu4": metrics["response_quality"]["bleu4"],
                "rouge_l": metrics["response_quality"]["rouge_l_f"],
            }

        rows.append({
            "tag": tag,
            "n": len(predictions),
            "failures": len(failure_ids),
            "failed_lengths": failed_lengths,
            "len_mean": statistics.fmean(lengths) if lengths else 0.0,
            "len_median": statistics.median(lengths) if lengths else 0.0,
            "ok_len_mean": statistics.fmean(ok_lengths) if ok_lengths else 0.0,
            "polluted": polluted,
            "scores": scores,
        })

    if not rows:
        raise SystemExit("没有可对照的 tag")

    print(f"{'tag':<24}{'n':>4}{'解析失败':>9}{'失败率':>8}{'长度均值':>9}{'中位':>7}"
          f"{'未失败均值':>11}{'残留污染':>9}")
    print("-" * 88)
    for row in rows:
        n = row["n"] or 0
        rate = row["failures"] / n if n else 0
        print(f"{row['tag']:<24}{n:>4}{row['failures']:>9}{rate:>8.2%}"
              f"{row['len_mean']:>9.0f}{row['len_median']:>7.0f}"
              f"{row['ok_len_mean']:>11.0f}{row['polluted']:>9}")

    print()
    print(f"{'tag':<24}{'情绪':>8}{'画像':>8}{'BLEU-4':>9}{'ROUGE-L':>9}  失败样本长度")
    print("-" * 88)
    for row in rows:
        score = row["scores"]
        print(f"{row['tag']:<24}{score.get('emotion', float('nan')):>8.4f}"
              f"{score.get('profile', float('nan')):>8.4f}"
              f"{score.get('bleu4', float('nan')):>9.4f}"
              f"{score.get('rouge_l', float('nan')):>9.4f}  {row['failed_lengths']}")

    print()
    print("读法（报告 §19.2d 的可验证预测）：")
    print("  · 若「长度均值」明显更高、且「解析失败」同时上升 → 近因冲突（散文驱动压过 JSON 契约）成立")
    print("  · 若长度不变而失败率变化 → 与长度无关，得另找原因")
    print("  · 「残留污染」= 回复以字段名开头的条数（抢救正则没命中、整段原文被当回复）")
    print("  · 硬门槛：解析失败 > 2% 即为不合格（它不是某一项差，而是一次打穿多条线）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
