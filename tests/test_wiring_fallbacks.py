"""接线层降级台账的测试。

为什么值得单独测：`wiring.build_messages` 的契约是「永不抛异常」，于是融合层
每一条样本都抛异常时，日志里只会看到 `rag.fusion=ok`（模块能 import），
结果却整趟走官方组包 —— 「日志正常、分数悄悄变低」。这类问题在只有 5 次
测评机会的赛制里排不出来，所以降级必须留痕、且痕迹要能被测试按住。
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_DIR = REPO_ROOT / "deploy" / "participant"
if str(PARTICIPANT_DIR) not in sys.path:
    sys.path.insert(0, str(PARTICIPANT_DIR))

import wiring  # noqa: E402


def _sample() -> dict:
    return {
        "id": "val_000001_t1",
        "conversation": [{"role": "user", "content": "我最近有点累"}],
    }


def _conversation_text(sample: dict) -> str:
    return "用户：我最近有点累"


class _RaisingFusion:
    """有 build_messages，但一调用就抛。"""

    @staticmethod
    def build_messages(sample, retrieved, system_prompt=None):
        raise RuntimeError("融合层炸了")


class _BadShapeFusion:
    """能调用，但返回的 messages 不合官方契约（content 不是字符串）。"""

    @staticmethod
    def build_messages(sample, retrieved, system_prompt=None):
        return [{"role": "system", "content": None}, {"role": "user", "content": "x"}]


class _GoodFusion:
    @staticmethod
    def build_messages(sample, retrieved, system_prompt=None):
        return [
            {"role": "system", "content": system_prompt or "s"},
            {"role": "user", "content": "用户：我最近有点累"},
        ]


class FallbackLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        wiring.FALLBACK_COUNTS.clear()
        wiring.FALLBACK_REASONS.clear()

    def tearDown(self) -> None:
        wiring.FALLBACK_COUNTS.clear()
        wiring.FALLBACK_REASONS.clear()

    def _run(self, module):
        with mock.patch.object(wiring, "_import", return_value=module), \
                mock.patch.object(wiring, "retrieve_memories", return_value=[]):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                produced = wiring.build_messages(_sample(), "SYS", _conversation_text)
            return produced, buffer.getvalue()

    def test_raising_fusion_is_counted_not_silent(self) -> None:
        produced, printed = self._run(_RaisingFusion)
        self.assertEqual(wiring.FALLBACK_COUNTS["call:rag.fusion.build_messages"], 1)
        self.assertIn("[warn] 接线降级", printed)
        self.assertIn("RuntimeError", wiring.FALLBACK_REASONS["call:rag.fusion.build_messages"])
        self.assertEqual(produced[0]["role"], "system")

    def test_unusable_messages_are_counted(self) -> None:
        produced, printed = self._run(_BadShapeFusion)
        self.assertEqual(wiring.FALLBACK_COUNTS["unusable:rag.fusion.build_messages"], 1)
        self.assertIn("[warn]", printed)
        self.assertEqual(produced[0]["role"], "system")

    def test_good_fusion_records_nothing(self) -> None:
        produced, printed = self._run(_GoodFusion)
        self.assertEqual(wiring.fallback_stats()["total"], 0)
        self.assertNotIn("[warn]", printed)
        self.assertEqual(produced[0]["content"], "SYS")

    def test_same_reason_prints_once_but_keeps_counting(self) -> None:
        _, first = self._run(_RaisingFusion)
        _, second = self._run(_RaisingFusion)
        self.assertIn("[warn]", first)
        self.assertNotIn("[warn]", second, "同一原因不该每条样本都刷屏")
        self.assertEqual(wiring.FALLBACK_COUNTS["call:rag.fusion.build_messages"], 2)

    def test_stats_shape_is_report_ready(self) -> None:
        self._run(_RaisingFusion)
        stats = wiring.fallback_stats()
        self.assertEqual(set(stats), {"counts", "reasons", "total"})
        self.assertEqual(stats["total"], 1)
        self.assertIsInstance(stats["counts"], dict)


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        wiring.FALLBACK_COUNTS.clear()
        wiring.FALLBACK_REASONS.clear()

    def tearDown(self) -> None:
        wiring.FALLBACK_COUNTS.clear()
        wiring.FALLBACK_REASONS.clear()

    def test_zero_fallbacks_says_clean(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            wiring.log_fallback_summary()
        self.assertIn("降级 0 次", buffer.getvalue())

    def test_nonzero_fallbacks_warns_with_counts(self) -> None:
        wiring._note_fallback("call:rag.fusion.build_messages", "RuntimeError: x")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            wiring.log_fallback_summary()
        text = buffer.getvalue()
        self.assertIn("[warn]", text)
        self.assertIn("call:rag.fusion.build_messages=1", text)


if __name__ == "__main__":
    unittest.main()
