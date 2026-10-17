#!/usr/bin/env python3
"""对一份正式 `submission.jsonl` 打分：先按 id 去重，再按官方口径评。

为什么需要这个工具
------------------
`deploy/participant/run_inference.py` 在**开头**用 `write_text("")` 清空
`submission.jsonl`，之后以 `open(..., "a")` 逐条追加。当同一台机器上**几乎同时**
起了两个 `run_inference`（例如两个人各跑一次 `eval_loop.py`），后启动的那个的清空
会把先启动那个的成果截断，两边再交错追加 —— 结果是同一个 `id` 出现多行，且两行内容
可能不同。

直接拿这种文件打分，分母会变成「行数」而不是「样本数」，情绪 Acc、画像 Acc 全都被
稀释。`src/evaluate.py` 的 `load_predictions()` 不会去重，所以必须在打分前处理。

本工具做三件事
--------------
1. 读 `submission.jsonl`，按 `id` **保留最后一次出现**（后写的通常是完整的）；
2. 报告总行数 / 唯一 id 数 / 重复条数 / 有冲突的 id 数；
3. 调 `src.evaluate.evaluate_predictions()` 出官方口径的指标，并可落盘 JSON。

用法
----
    python3 tools/score_submission.py \
        --submission runs/eval-cap-v3-official-full/submission.jsonl \
        --references data/processed/val_references.jsonl \
        --json-out runs/eval-cap-v3-official-full/score.json

不联网、不写曲线、不改任何数据文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# `src/evaluate.py` 里是 `from label_space import ...`（顶层模块名），所以 `src/` 也要在 path 上。
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from src.evaluate import evaluate_predictions, load_predictions, load_references  # noqa: E402


def dedupe(predictions: list[dict]) -> tuple[list[dict], dict]:
    """按 id 去重，保留最后一次出现。返回 (去重后的列表, 统计信息)。"""
    last_index: dict[str, int] = {}
    first_index: dict[str, int] = {}
    conflicts = 0
    no_id = 0

    for idx, record in enumerate(predictions):
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            no_id += 1
            continue
        if sample_id in last_index:
            prev = predictions[last_index[sample_id]]
            if json.dumps(prev, sort_keys=True) != json.dumps(record, sort_keys=True):
                conflicts += 1
        else:
            first_index[sample_id] = idx
        last_index[sample_id] = idx

    kept = [predictions[idx] for idx in sorted(last_index.values())]
    stats = {
        "rows_total": len(predictions),
        "rows_kept": len(kept),
        "unique_ids": len(last_index),
        "duplicates_dropped": len(predictions) - len(kept) - no_id,
        "conflicting_duplicates": conflicts,
        "rows_without_id": no_id,
        "first_duplicate_row": _first_duplicate_row(first_index, last_index),
    }
    return kept, stats


def _first_duplicate_row(first: dict[str, int], last: dict[str, int]) -> int | None:
    """最早一次「重复出现」发生在第几行（0-based）；没有重复返回 None。"""
    candidates = [last[sid] for sid in first if last[sid] != first[sid]]
    return min(candidates) if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 id 去重后对 submission.jsonl 打分")
    parser.add_argument("--submission", required=True, help="预测 JSONL（submission.jsonl）")
    parser.add_argument("--references", required=True, help="真值 JSONL（val_references.jsonl）")
    parser.add_argument("--json-out", default=None, help="完整指标报告输出路径（可选）")
    parser.add_argument("--per-sample-out", default=None, help="逐样本明细输出路径（可选）")
    parser.add_argument(
        "--tokenizer",
        default="char",
        choices=["char", "jieba"],
        help="BLEU/ROUGE 分词方式，默认 char（与 src/cli.py 一致）",
    )
    parser.add_argument(
        "--include-memory-refs",
        action="store_true",
        help="把 memory_refs 计入画像维度平均（官方口径未公布，默认不计）",
    )
    args = parser.parse_args(argv)

    raw = load_predictions(args.submission)
    kept, stats = dedupe(raw)
    references = load_references(args.references)

    print("=== 去重 ===")
    print(f"  总行数            {stats['rows_total']}")
    print(f"  唯一 id           {stats['unique_ids']}")
    print(f"  丢弃的重复行      {stats['duplicates_dropped']}")
    print(f"  其中内容冲突的    {stats['conflicting_duplicates']}")
    if stats["rows_without_id"]:
        print(f"  ⚠️ 缺少 id 的行    {stats['rows_without_id']}")
    if stats["first_duplicate_row"] is not None:
        print(f"  第一次重复出现在第 {stats['first_duplicate_row']} 行（0-based）")
    if stats["duplicates_dropped"] > 0:
        print("  ⚠️ 这份文件被多个进程交错写过 —— 打分前已按 id 保留最后一次出现。")

    covered = sum(1 for record in kept if record.get("id") in references)
    print()
    print("=== 覆盖 ===")
    print(f"  去重后行数        {len(kept)}")
    print(f"  真值条数          {len(references)}")
    print(f"  与真值对上的      {covered}")
    if covered < len(references):
        print(f"  ⚠️ 还差 {len(references) - covered} 条没跑到")

    report = evaluate_predictions(
        kept,
        references,
        tokenizer_name=args.tokenizer,
        include_memory_refs=args.include_memory_refs,
    )
    payload = json.loads(report.to_json())

    print()
    print("=== 官方口径指标 ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整指标已写入 {out}")
    if args.per_sample_out:
        out = Path(args.per_sample_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"去重后的预测已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
