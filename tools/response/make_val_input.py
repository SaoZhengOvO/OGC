#!/usr/bin/env python3
"""为 ③ 号线（共情回复）生成验证输入，供快速迭代使用。

为什么还要一个自己的脚本：`tools/eval_loop.py` 已经会造输入，但那是**全队共用的裁判**，
不适合在迭代时反复改动它的行为。这个脚本只做一件事——从 val 里切出一段，
产出**不含答案**的输入 + 与之严格对齐的真值子集，让 `eval_loop.py --skip-inference`
或官方推理脚本可以拿它单独跑，不影响裁判本身。

用法
----
    python tools/response/make_val_input.py                    # 全量 2354 条
    python tools/response/make_val_input.py --limit 20         # 冒烟
    python tools/response/make_val_input.py --limit 20 --out-dir runs/resp-probe

设计约束（与 CONTRACT.md 第五条一致）
------------------------------------
* **绝不把 val 的 target 写进输入文件**——这是防泄漏，脚本会自己断言。
* 只用标准库。
* 路径一律相对仓库根推导，不含任何绝对路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_SOURCE = PROCESSED / "val_samples.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "resp-probe"

#: 输入里只允许保留这些字段；出现 target 就是泄漏，直接中止。
INPUT_KEEP = ("id", "conversation_id", "target_user_turn_id", "history")
#: 出现在输入里的禁词，命中即中止。
FORBIDDEN_KEYS = ("target", "response_text", "emotion_label", "user_profile", "memory_refs")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise SystemExit(f"{path}:{number}: 每行必须是含字符串 id 的 object")
            rows.append(row)
    if not rows:
        raise SystemExit(f"源文件为空: {path}")
    return rows


def strip_answer(row: dict) -> dict:
    """去掉答案字段，只留推理需要的部分。"""
    leaked = [key for key in FORBIDDEN_KEYS if key in row and key != "target"]
    if leaked:
        raise SystemExit(f"样本 {row['id']} 输入侧混进了答案字段 {leaked}，已中止")
    return {key: row[key] for key in INPUT_KEEP if key in row}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 ③ 号线的验证输入与对齐真值")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="val 样本（含答案）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    parser.add_argument("--limit", type=int, default=None, help="取前 N 条（默认全量）")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"找不到 {source}，请先跑: python src/cli.py build")

    rows = read_jsonl(source)
    window = rows[args.offset : (args.offset + args.limit) if args.limit else None]
    if not window:
        raise SystemExit(f"offset={args.offset} limit={args.limit} 取不到样本")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "val_input.jsonl"
    reference_path = out_dir / "val_references_subset.jsonl"

    with input_path.open("w", encoding="utf-8") as handle:
        for row in window:
            handle.write(json.dumps(strip_answer(row), ensure_ascii=False, separators=(",", ":")) + "\n")

    # 真值子集：与输入严格同序同 id，供打分对齐。
    missing = [row["id"] for row in window if not isinstance(row.get("target"), dict)]
    if missing:
        raise SystemExit(f"这些样本没有 target，无法构造真值: {missing[:5]}")
    with reference_path.open("w", encoding="utf-8") as handle:
        for row in window:
            handle.write(
                json.dumps({"id": row["id"], "target": row["target"]}, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    # 防泄漏断言：输入文件里绝不能出现任何答案字段名。
    text = input_path.read_text(encoding="utf-8")
    hit = [word for word in FORBIDDEN_KEYS if f'"{word}"' in text]
    if hit:
        raise SystemExit(f"泄漏断言失败：输入文件里出现 {hit}")
    if len(text.splitlines()) != len(window):
        raise SystemExit("行数与窗口不一致")

    print(f"验证输入   -> {input_path}  ({len(window)} 条，只含 history)")
    print(f"真值子集   -> {reference_path}")
    print(f"范围       -> offset={args.offset} limit={args.limit or '全部'}")
    print("防泄漏断言 -> 通过（输入里没有 target/response_text/emotion_label/user_profile/memory_refs）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
