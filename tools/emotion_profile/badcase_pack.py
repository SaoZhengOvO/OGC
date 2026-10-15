#!/usr/bin/env python3
"""把评测明细还原成「人可以一条条读」的 badcase 清单。

`src/cli.py eval --per-sample-out` 只给 id / 金标 / 预测，看不到对话原文。
这个脚本把它和 `data/processed/val_samples.jsonl` 里的真实对话拼起来，
让你能直接读「用户最后说了什么 → 金标是什么 → 模型判成了什么」。

用法::

    # 情绪判错的样本，按「金标类别」聚合，每类最多 5 条
    python tools/emotion_profile/badcase_pack.py \
        --per-sample runs/val_v1.samples.jsonl \
        --out runs/val_v1.badcase.md

    # 只看某几类，且只要情绪错了的
    python tools/emotion_profile/badcase_pack.py --per-sample X.jsonl --emotions sadness,loneliness,helplessness

    # 顺便看「该空却填」的画像错误
    python tools/emotion_profile/badcase_pack.py --per-sample X.jsonl --profile-overfill

输出是纯文本（Markdown 只用来分隔，不会被模型看到），每条包含：
样本 id、金标情绪 -> 预测情绪、画像三维对比、目标用户轮原文、上一轮用户轮（帮助判断语境）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "processed" / "val_samples.jsonl"
PROFILE_DIMS = ("personality_traits", "interests", "style")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL。"""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc


def user_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """按 turn_id 升序取出用户轮。"""
    turns = [t for t in sample.get("history", []) if t.get("role") == "user"]
    return sorted(turns, key=lambda t: t.get("turn_id", 0))


def render_entry(sample: dict[str, Any], detail: dict[str, Any], *, prev: bool) -> str:
    """渲染一条 badcase。"""
    turns = user_turns(sample)
    last = turns[-1]["content"] if turns else "(没有用户轮)"
    lines = [
        f"### {detail['id']}",
        f"- 情绪：**{detail['emotion_gold']} -> {detail['emotion_pred']}**",
    ]
    for dim in PROFILE_DIMS:
        gold = sorted(detail.get("profile_gold", {}).get(dim) or [])
        pred = sorted(detail.get("profile_pred", {}).get(dim) or [])
        if gold != pred:
            lines.append(f"- 画像 {dim}：金标 {gold or '[]'} / 预测 {pred or '[]'}")
    lines.append(f"- 目标用户轮（turn {turns[-1]['turn_id'] if turns else '?'}）：{last}")
    if prev and len(turns) >= 2:
        lines.append(f"- 上一轮用户（turn {turns[-2]['turn_id']}）：{turns[-2]['content']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成人能直接读的 badcase 清单")
    parser.add_argument("--per-sample", required=True, help="src/cli.py eval --per-sample-out 的产物")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="带 history 的验证样本")
    parser.add_argument("--out", required=True, help="输出文件（.md 或 .txt）")
    parser.add_argument("--emotions", default=None, help="只看这些金标情绪（逗号分隔）")
    parser.add_argument("--per-class", type=int, default=5, help="每个金标情绪最多列几条（默认 5）")
    parser.add_argument("--profile-overfill", action="store_true", help="只列「该空却填」的画像错误")
    parser.add_argument("--no-prev", action="store_true", help="不附上一轮用户消息")
    args = parser.parse_args(argv)

    per_sample = Path(args.per_sample)
    if not per_sample.is_absolute():
        per_sample = REPO_ROOT / per_sample
    samples_path = Path(args.samples)
    if not samples_path.is_absolute():
        samples_path = REPO_ROOT / samples_path
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out

    wanted = set(args.emotions.split(",")) if args.emotions else None
    details = list(read_jsonl(per_sample))
    lookup = {sample["id"]: sample for sample in read_jsonl(samples_path)}

    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        sample = lookup.get(detail["id"])
        if sample is None:
            continue
        if args.profile_overfill:
            for dim in PROFILE_DIMS:
                gold_nonempty = bool(detail.get("profile_gold", {}).get(dim))
                pred_nonempty = bool(detail.get("profile_pred", {}).get(dim))
                if not gold_nonempty and pred_nonempty:
                    selected[f"画像该空却填:{dim}"].append(detail)
                    break
            continue
        if detail["emotion_hit"]:
            continue
        if wanted and detail["emotion_gold"] not in wanted:
            continue
        selected[detail["emotion_gold"]].append(detail)

    chunks = [
        f"# badcase 清单（{per_sample.name}）",
        "",
        f"来源明细 {per_sample}，对话原文 {samples_path.name}，"
        f"共 {len(details)} 条明细 / 命中 {len(lookup)} 条对话样本。",
        "",
        "读法：先自己判一次「如果是我，这条会判成什么」，再看金标和模型的差异。"
        "把反复出现的错因写成规则，一次只改一处。",
        "",
    ]
    total = 0
    for group in sorted(selected):
        items = selected[group][: args.per_class]
        total += len(items)
        chunks.append(f"## {group}（共 {len(selected[group])} 条，列前 {len(items)} 条）")
        chunks.append("")
        for detail in items:
            chunks.append(render_entry(lookup[detail["id"]], detail, prev=not args.no_prev))
            chunks.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"已写入 {out}（{total} 条 badcase，{len(selected)} 个分组）")
    if not total:
        print("提示：当前明细里没有符合筛选条件的错误样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
