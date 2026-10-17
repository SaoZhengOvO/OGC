"""把多台机器上的 `runs/leaderboard.tsv` 合并成一份。

为什么需要：`eval_loop.py` 只往**本机**的 `runs/leaderboard.tsv` 追加，而 `runs/`
被 .gitignore 排除。于是评测跑在哪台机器上，台账就只长在那台机器上 —— 2026-10-17
的实际情况是本地 9 行、AMD 盒子 45 行，`pack_review.py` 从本地打包，专家评审材料里
的台账就只到 `ab-rag`，而后面的 `cap-v3-official-full`（n=2354）、四臂、最终全量
一条都没有。报告里写着 0.404，台账里停在 0.255，评审一眼就能看出对不上。

这个工具**只做合并，不造行**：每条记录必须是某台机器上 `eval_loop.py` 真写出来的。
重复判定用 (time, tag)，同一秒同一 tag 视为同一条。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEADER_COLUMNS = (
    "time",
    "tag",
    "backend",
    "model",
    "script",
    "prompts",
    "n_samples",
    "n_comparable",
    "emotion_acc",
    "profile_acc",
    "bleu4",
    "rouge_l",
    "memory_f1",
)


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return list(HEADER_COLUMNS), []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        # 列数对不上说明这一行是坏行（例如磁盘写满时截断），宁可报错也不要静默丢
        if len(cells) != len(header):
            raise SystemExit(f"{path} 里有一行列数不对：{len(cells)} != {len(header)}\n{line}")
        rows.append(cells)
    return header, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="合并多台机器的 leaderboard.tsv")
    parser.add_argument("inputs", nargs="+", help="若干个 leaderboard.tsv")
    parser.add_argument("--out", required=True, help="合并结果写到这个路径")
    args = parser.parse_args(argv)

    header: list[str] | None = None
    merged: dict[tuple[str, str], list[str]] = {}
    sources: dict[str, list[str]] = {}

    for name in args.inputs:
        path = Path(name)
        if not path.exists():
            print(f"跳过不存在的 {path}", file=sys.stderr)
            continue
        head, rows = read_rows(path)
        if header is None:
            header = head
        elif head != header:
            raise SystemExit(f"{path} 的表头和第一份不一样：\n{head}\n{header}")
        for row in rows:
            time_idx = header.index("time")
            tag_idx = header.index("tag")
            key = (row[time_idx], row[tag_idx])
            merged[key] = row
            sources.setdefault(row[tag_idx], []).append(str(path))

    if header is None:
        raise SystemExit("没有任何输入可合并")

    ordered = sorted(merged.items(), key=lambda item: (item[0][0], item[0][1]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(row) for _key, row in ordered]
    # newline="\n" 不能省：Windows 上 write_text 默认会把 \n 翻成 \r\n，而 eval_loop.py
    # 写出来的是 LF。台账一旦混进 CRLF，在 Linux 上 `cut -f` 会把 \r 一起切进字段，
    # 分数列变成 "0.40399\r"，排序和比较全失效。
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"合并 {len(ordered)} 行 -> {out}")
    print(f"  来自 {len({s for v in sources.values() for s in v})} 个文件")
    dupes = [tag for tag, srcs in sources.items() if len(srcs) > 1]
    if dupes:
        print(f"  同名 tag 出现多次（可能是重跑，都保留了）：{', '.join(sorted(set(dupes)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
