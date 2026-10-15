#!/usr/bin/env python3
"""把带标签的验证样本削成**不含答案**的推理输入。

角色①的 `deploy/make_val_input.py` 还没就绪（那属于部署线），所以情绪与画像线
自带一份，只依赖标准库，行为与 `tools/eval_loop.py:build_val_input` 一致：
只保留官方推理脚本需要的四个字段，并断言没有把 `target` 夹带进去。

用法::

    python tools/emotion_profile/make_val_input.py                     # 全量 2354 条
    python tools/emotion_profile/make_val_input.py --limit 200        # 迭代用小样本
    python tools/emotion_profile/make_val_input.py --out runs/val_input.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "processed" / "val_samples.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "val_input.jsonl"

INPUT_KEYS = ("id", "conversation_id", "target_user_turn_id", "history")
FORBIDDEN_KEYS = ("target", "response_text", "emotion_label", "user_profile", "memory_refs")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL，空行跳过，解析失败指出行号。"""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成不含答案的验证输入")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help=f"带标签的验证样本（默认 {DEFAULT_SOURCE.name}）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出路径（默认 {DEFAULT_OUT}）")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条（小样本迭代用）")
    parser.add_argument("--offset", type=int, default=0, help="从第 N 条开始取")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_absolute():
        source = REPO_ROOT / source
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    if not source.exists():
        raise SystemExit(f"找不到 {source}，请先运行: python src/cli.py build")

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(read_jsonl(source)):
        if index < args.offset:
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break
        payload = {key: sample[key] for key in INPUT_KEYS if key in sample}
        missing = [key for key in INPUT_KEYS if key not in payload]
        if missing:
            raise SystemExit(f"{source} 第 {index + 1} 条缺字段 {missing}")
        leaked = [key for key in FORBIDDEN_KEYS if key in payload]
        if leaked:
            raise SystemExit(f"{source} 第 {index + 1} 条的输入里混进了答案字段 {leaked}，已中止")
        rows.append(payload)

    if not rows:
        raise SystemExit(f"{source} 在 offset={args.offset} 处没有取到样本")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{out} 里出现重复 id，异常中止")
    print(f"已写入 {out}（{len(rows)} 条，仅含 {list(INPUT_KEYS)}）")
    print("下一步（模型就位后）：")
    print(f"  python GivenData/participant/run_inference.py {out.relative_to(REPO_ROOT)} runs/val_<版本>.jsonl")
    print("  python src/cli.py eval --predictions runs/val_<版本>.jsonl --per-sample-out runs/val_<版本>.samples.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
