#!/usr/bin/env python3
"""把 `data/processed/val_samples.jsonl` 剥成「只有输入、没有答案」的推理输入。

**为什么必须有这个脚本**：`val_samples.jsonl` 每条都带 `target`（参考答案）。
直接把整条喂给模型，等于把答案一起递过去——那跑出来的分数是假的，而且是
**静默**的假（不会报错，只会看到一个好得离谱的数）。

只保留这四个字段，并在写完后断言输出里不含任何答案字段：

    id, conversation_id, target_user_turn_id, history

用法::

    python deploy/make_val_input.py                    # -> runs/val_input.jsonl
    python deploy/make_val_input.py --limit 50         # 前 50 条（冒烟用）
    python deploy/make_val_input.py --offset 200 --limit 50 --out runs/slice.jsonl

`tools/eval_loop.py` 内部用的是同一个 `build_val_input()`，所以两边口径永远一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_loop import PROCESSED, build_val_input  # noqa: E402

DEFAULT_SOURCE = PROCESSED / "val_samples.jsonl"
DEFAULT_OUT = REPO_ROOT / "runs" / "val_input.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="剥掉验证集的答案，生成推理输入")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"带 target 的样本文件（默认 {DEFAULT_SOURCE}）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"输出路径（默认 {DEFAULT_OUT}）")
    parser.add_argument("--limit", type=int, default=None, help="最多取几条")
    parser.add_argument("--offset", type=int, default=0, help="跳过前几条")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"找不到 {args.source}，先跑 `python src/cli.py build` 展开官方数据。", file=sys.stderr)
        return 1

    count = build_val_input(args.source, args.out, args.limit, args.offset)
    print(f"写出 {count} 条 -> {args.out}")
    print("（只含 id / conversation_id / target_user_turn_id / history，已断言无答案字段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
