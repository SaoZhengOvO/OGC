"""把队员认领的分工写进 `ROLES.md` 的「当前负责人」列。

用法::

    python tools/assign_role.py --list
    python tools/assign_role.py --line 2 --owner zhangsan
    python tools/assign_role.py --line 5A --owner lisi
    python tools/assign_role.py --line 2 --clear

设计取舍：
- **只改那张分工表的最后一个单元格**，文件里其他字节（包括换行风格）原样保留，
  这样五个人各自认领一次产生的 diff 都是一行，冲突好合。
- 纯标准库、不联网、不碰 `GivenData/`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLES_PATH = REPO_ROOT / "ROLES.md"

#: 没人认领时单元格里写什么。
UNASSIGNED = "待认领"

#: 队长在表里显示成加粗的「队长」，其他人显示成行内代码。
LEAD_ALIASES = ("队长", "lead", "LEAD")

#: 线号 -> 表格首列里用来识别这一行的前缀。
#: ⑤ 已拆成 ⑤A / ⑤B，所以不接受裸的 "5"。
LINE_MARKERS: dict[str, str] = {
    "1": "①",
    "2": "②",
    "3": "③",
    "4": "④",
    "5A": "⑤A",
    "5B": "⑤B",
}

#: 表格的列数（「线 | 提示词 | 主责 | 独占写范围 | 当前负责人」）。
COLUMN_COUNT = 5


class RolesFileError(RuntimeError):
    """`ROLES.md` 的形状和预期不符。"""


def normalize_line(raw: str) -> str:
    """把用户输入的线号规范成 `LINE_MARKERS` 的键。

    接受大小写与空格差异（``5a`` / ``5 a`` / ``5A`` 都可以）；不认识就抛 `ValueError`。
    """
    text = raw.strip().upper().replace(" ", "")
    if text in LINE_MARKERS:
        return text
    raise ValueError(
        f"不认识的线号 {raw!r}。可选：{', '.join(LINE_MARKERS)}"
        "（注意 ⑤ 已经拆成 5A 和 5B，要写全）"
    )


def render_owner(owner: str) -> str:
    """负责人单元格的统一写法。"""
    name = owner.strip()
    if not name:
        return UNASSIGNED
    if name in LEAD_ALIASES:
        return "**队长**"
    return f"`{name}`"


def split_row(line: str) -> list[str] | None:
    """把 ``| a | b | ... |`` 拆成单元格；不是这种形状就返回 None。

    返回的列表**包含**首尾两个空串，所以长度是 :data:`COLUMN_COUNT` + 2。
    """
    if not line.startswith("|"):
        return None
    cells = line.rstrip("\r\n").split("|")
    if len(cells) != COLUMN_COUNT + 2:
        return None
    return cells


def iter_role_rows(lines: list[str]) -> list[tuple[int, list[str]]]:
    """找出分工表里所有以 ①~⑤ 开头的行，返回 ``[(行号, 单元格), ...]``。"""
    rows: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        cells = split_row(line)
        if cells is None:
            continue
        first = cells[1].strip()
        if first and first[0] in "①②③④⑤":
            rows.append((index, cells))
    return rows


def apply_assignment(text: str, line: str, owner: str) -> tuple[str, str]:
    """把某一行的负责人改成 ``owner``，返回 ``(新文本, 改动前的负责人)``。

    `owner` 为空字符串表示改回「待认领」。
    """
    marker = LINE_MARKERS[normalize_line(line)]
    lines = text.splitlines(keepends=True)
    rows = iter_role_rows(lines)
    if not rows:
        raise RolesFileError(f"{ROLES_PATH} 里找不到分工表（期望有以 ①~⑤ 开头的行）")

    for index, cells in rows:
        if not cells[1].strip().startswith(marker):
            continue
        previous = cells[COLUMN_COUNT].strip()
        cells[COLUMN_COUNT] = f" {render_owner(owner)} "
        # `split_row` 吃掉了行尾，这里原样补回去——漏掉它会把这一行和下一行粘起来。
        ending = lines[index][len(lines[index].rstrip("\r\n")) :]
        lines[index] = "|".join(cells) + ending
        return "".join(lines), previous

    known = "、".join(cells[1].strip() for _, cells in rows)
    raise RolesFileError(f"分工表里没有 {marker} 这一行；现有：{known}")


def describe(text: str) -> list[tuple[str, str]]:
    """返回 ``[(线, 当前负责人), ...]``，给 ``--list`` 用。"""
    lines = text.splitlines(keepends=True)
    rows = iter_role_rows(lines)
    if not rows:
        raise RolesFileError(f"{ROLES_PATH} 里找不到分工表")
    return [(cells[1].strip(), cells[COLUMN_COUNT].strip()) for _, cells in rows]


def _read_raw(path: Path) -> str:
    """按原样读入，不做换行翻译——保住 CRLF，避免整文件 diff。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_raw(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="登记/查看 ROLES.md 的分工负责人")
    parser.add_argument("--line", help="线号：1 2 3 4 5A 5B")
    parser.add_argument("--owner", default="", help="GitHub 用户名；写「队长」会显示成加粗的队长")
    parser.add_argument("--clear", action="store_true", help="把该线改回「待认领」")
    parser.add_argument("--list", action="store_true", help="只打印当前认领情况，不改文件")
    parser.add_argument("--path", default=str(ROLES_PATH), help="要改的文件（默认仓库根的 ROLES.md）")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        print(f"找不到 {path}", file=sys.stderr)
        return 1

    try:
        text = _read_raw(path)
        if args.list or (not args.line and not args.clear):
            print(f"{path} 的当前分工：")
            for line_name, owner in describe(text):
                print(f"  {line_name:<18} {owner}")
            if not args.list:
                print("\n（没给 --line，只做了查看。要登记请加 --line 与 --owner）")
            return 0

        if not args.line:
            print("--clear 也要一起给 --line", file=sys.stderr)
            return 2

        owner = "" if args.clear else args.owner
        marker = LINE_MARKERS[normalize_line(args.line)]
        updated, previous = apply_assignment(text, args.line, owner)
    except (ValueError, RolesFileError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    if updated == text:
        print(f"{marker} 的负责人已经是「{render_owner(owner)}」，没有改动。")
        return 0

    _write_raw(path, updated)
    print(f"{marker}：{previous or UNASSIGNED} -> {render_owner(owner)}")
    print(f"已写入 {path}")
    message = f"认领：{render_owner(owner)} -> {marker}" if owner else f"释放：{marker} 改回{UNASSIGNED}"
    print(
        "\n别忘了提交：\n"
        f'  git add "{path.name}" && git commit -m "{message}" && git push'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
