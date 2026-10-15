"""`tools/assign_role.py` 的回归测试。

重点盯两件事：
- **只动一个单元格**：五个人各自认领一次，产生的 diff 都该是一行。
- **换行风格不变**：`ROLES.md` 在 Windows 工作区里是 CRLF，改完不能整文件重写。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import assign_role  # noqa: E402  （要先把 tools/ 挂进 sys.path）


SAMPLE = (
    "# 角色分工与提示词（ROLES）\r\n"
    "\r\n"
    "## 五条线\r\n"
    "\r\n"
    "| 线 | 提示词 | 主责评分项 | 独占写文件范围 | 当前负责人 |\r\n"
    "|---|---|---|---|---|\r\n"
    "| ① 部署与评测 | [a](a) | 推理性能 | `deploy/` | **队长** |\r\n"
    "| ② 提示词·情绪与画像 | [b](b) | 情感识别 | `prompts/` | 待认领 |\r\n"
    "| ⑤A RAG·融合 | [c](c) | 融合层 | `rag/fusion.py` | 待认领 |\r\n"
    "| ⑤B 专家评审文档 | [d](d) | 方案完整性 | `docs/reports/` | **队长** |\r\n"
    "\r\n"
    "尾巴一句话。\r\n"
)


class NormalizeLineTest(unittest.TestCase):
    def test_accepts_case_and_space_variants(self) -> None:
        for raw in ("5a", "5A", "5 a", " 5A "):
            self.assertEqual(assign_role.normalize_line(raw), "5A")

    def test_rejects_bare_five(self) -> None:
        # ⑤ 已经拆成 ⑤A / ⑤B，裸写 5 会被判歧义。
        with self.assertRaises(ValueError):
            assign_role.normalize_line("5")

    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            assign_role.normalize_line("9")


class RenderOwnerTest(unittest.TestCase):
    def test_lead_is_bold(self) -> None:
        self.assertEqual(assign_role.render_owner("队长"), "**队长**")
        self.assertEqual(assign_role.render_owner("LEAD"), "**队长**")

    def test_blank_is_unassigned(self) -> None:
        self.assertEqual(assign_role.render_owner(""), assign_role.UNASSIGNED)
        self.assertEqual(assign_role.render_owner("   "), assign_role.UNASSIGNED)

    def test_plain_name_is_code_span(self) -> None:
        self.assertEqual(assign_role.render_owner("zhangsan"), "`zhangsan`")


class SplitRowTest(unittest.TestCase):
    def test_rejects_prose(self) -> None:
        self.assertIsNone(assign_role.split_row("这不是表格\r\n"))

    def test_rejects_wrong_column_count(self) -> None:
        self.assertIsNone(assign_role.split_row("| 只有 | 两列 |\r\n"))

    def test_accepts_five_column_row(self) -> None:
        cells = assign_role.split_row("| ① 部署与评测 | [a](a) | x | y | **队长** |\r\n")
        self.assertIsNotNone(cells)
        assert cells is not None
        self.assertEqual(len(cells), assign_role.COLUMN_COUNT + 2)
        self.assertEqual(cells[1].strip(), "① 部署与评测")
        self.assertEqual(cells[assign_role.COLUMN_COUNT].strip(), "**队长**")


class ApplyAssignmentTest(unittest.TestCase):
    def test_changes_only_the_target_cell(self) -> None:
        updated, previous = assign_role.apply_assignment(SAMPLE, "2", "zhangsan")
        self.assertEqual(previous, "待认领")
        self.assertIn("| ② 提示词·情绪与画像 | [b](b) | 情感识别 | `prompts/` | `zhangsan` |", updated)
        # 别的行一个字都没动
        self.assertIn("| ① 部署与评测 | [a](a) | 推理性能 | `deploy/` | **队长** |", updated)
        self.assertIn("| ⑤A RAG·融合 | [c](c) | 融合层 | `rag/fusion.py` | 待认领 |", updated)
        self.assertTrue(updated.endswith("尾巴一句话。\r\n"))

    def test_preserves_crlf(self) -> None:
        updated, _ = assign_role.apply_assignment(SAMPLE, "5A", "lisi")
        self.assertNotIn("\n", updated.replace("\r\n", ""))
        self.assertEqual(updated.count("\r\n"), SAMPLE.count("\r\n"))

    def test_distinguishes_5a_from_5b(self) -> None:
        updated, _ = assign_role.apply_assignment(SAMPLE, "5A", "lisi")
        self.assertIn("| ⑤A RAG·融合 | [c](c) | 融合层 | `rag/fusion.py` | `lisi` |", updated)
        self.assertIn("| ⑤B 专家评审文档 | [d](d) | 方案完整性 | `docs/reports/` | **队长** |", updated)

    def test_clear_restores_unassigned(self) -> None:
        updated, previous = assign_role.apply_assignment(SAMPLE, "1", "")
        self.assertEqual(previous, "**队长**")
        self.assertIn("| ① 部署与评测 | [a](a) | 推理性能 | `deploy/` | 待认领 |", updated)

    def test_is_idempotent(self) -> None:
        once, _ = assign_role.apply_assignment(SAMPLE, "2", "zhangsan")
        twice, _ = assign_role.apply_assignment(once, "2", "zhangsan")
        self.assertEqual(once, twice)

    def test_missing_line_raises(self) -> None:
        text = SAMPLE.replace("| ② 提示词·情绪与画像 | [b](b) | 情感识别 | `prompts/` | 待认领 |\r\n", "")
        with self.assertRaises(assign_role.RolesFileError):
            assign_role.apply_assignment(text, "2", "zhangsan")

    def test_table_missing_raises(self) -> None:
        with self.assertRaises(assign_role.RolesFileError):
            assign_role.apply_assignment("# 只有标题\r\n", "2", "zhangsan")


class DescribeTest(unittest.TestCase):
    def test_lists_every_row(self) -> None:
        rows = assign_role.describe(SAMPLE)
        self.assertEqual(
            rows,
            [
                ("① 部署与评测", "**队长**"),
                ("② 提示词·情绪与画像", "待认领"),
                ("⑤A RAG·融合", "待认领"),
                ("⑤B 专家评审文档", "**队长**"),
            ],
        )


class RealRolesFileTest(unittest.TestCase):
    """对仓库里真实的 ROLES.md 做只读体检（不写文件）。"""

    def setUp(self) -> None:
        self.path = REPO_ROOT / "ROLES.md"
        if not self.path.is_file():  # pragma: no cover - 仓库里一定有
            self.skipTest("ROLES.md 不存在")

    def test_has_six_lines_including_split_five(self) -> None:
        rows = assign_role.describe(assign_role._read_raw(self.path))
        markers = [line.split()[0] for line, _ in rows]
        self.assertEqual(markers, ["①", "②", "③", "④", "⑤A", "⑤B"])

    def test_every_line_is_either_owned_or_unassigned(self) -> None:
        for line, owner in assign_role.describe(assign_role._read_raw(self.path)):
            with self.subTest(line=line):
                self.assertTrue(owner == assign_role.UNASSIGNED or owner.startswith(("`", "**")), owner)


if __name__ == "__main__":
    unittest.main()
