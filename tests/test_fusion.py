"""`rag/fusion.py` 的回归测试（角色⑤A）。

重点锁三件事，它们出错都是**静默掉分**：

1. 截断不能把目标轮丢掉（丢了模型就答非所问）。
2. 检索结果里的自泄漏与 val 泄漏必须被拦下（漏一次整套验证分数作废）。
3. 拿不到提示词时必须抛异常让接线层回退，而不是拼出一条空系统提示去推理。

集成那两条（`TestWiringSeam`）验的是角色①的 `deploy/participant/wiring.py` 与本模块的
接缝——两边任何一边改了签名，这里会红。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag import fusion  # noqa: E402


def make_history(turn_count: int, *, body: str = "这是一段测试用的对话内容，长度足够触发检查。") -> list[dict]:
    """造 `turn_count` 轮 user/assistant 交替的历史。"""
    turns = []
    for turn_id in range(1, turn_count + 1):
        turns.append(
            {
                "turn_id": turn_id,
                "role": "user" if turn_id % 2 else "assistant",
                "content": f"{body}（第 {turn_id} 轮）",
            }
        )
    return turns


def make_sample(turn_count: int = 9, **overrides) -> dict:
    sample = {
        "id": "test1_000001_t9",
        "conversation_id": "test1_conv_000001",
        "target_user_turn_id": 9,
        "history": make_history(turn_count),
    }
    sample.update(overrides)
    return sample


class FusionTestBase(unittest.TestCase):
    """把 `PROMPT_DIR` 指到临时目录，避免测试依赖仓库里 prompts/ 的真实内容。"""

    def setUp(self) -> None:
        self._saved_prompt_dir = fusion.PROMPT_DIR
        self._temp = tempfile.TemporaryDirectory()
        fusion.PROMPT_DIR = Path(self._temp.name)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        fusion.PROMPT_DIR = self._saved_prompt_dir
        self._temp.cleanup()

    def write_prompt(self, name: str, text: str = "按规则输出 JSON。") -> None:
        (Path(self._temp.name) / f"{name}.md").write_text(text, encoding="utf-8")


class TestBuildMessages(FusionTestBase):
    def test_returns_system_then_user(self) -> None:
        messages = fusion.build_messages(make_sample(), [], "系统提示")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], "系统提示")

    def test_system_prompt_is_used_verbatim(self) -> None:
        self.write_prompt("response", "队友的规则")
        messages = fusion.build_messages(make_sample(), [], "官方提示")
        # 传了就用传进来的，不做二次拼接（否则会把队友规则注入两遍）
        self.assertEqual(messages[0]["content"], "官方提示")
        self.assertNotIn("队友的规则", messages[0]["content"])

    def test_composes_from_prompt_files_when_not_given(self) -> None:
        self.write_prompt("emotion_profile", "情绪规则")
        self.write_prompt("response", "回复规则")
        messages = fusion.build_messages(make_sample(), [])
        self.assertIn("情绪规则", messages[0]["content"])
        self.assertIn("回复规则", messages[0]["content"])
        self.assertLess(
            messages[0]["content"].index("情绪规则"),
            messages[0]["content"].index("回复规则"),
        )

    def test_raises_unavailable_without_any_prompt(self) -> None:
        with self.assertRaises(fusion.FusionUnavailable):
            fusion.build_messages(make_sample(), [])

    def test_blank_system_prompt_is_treated_as_missing(self) -> None:
        with self.assertRaises(fusion.FusionUnavailable):
            fusion.build_messages(make_sample(), [], "   \n  ")

    def test_disabled_by_env_flag(self) -> None:
        saved = os.environ.get(fusion.FUSION_FLAG)
        os.environ[fusion.FUSION_FLAG] = "0"
        try:
            with self.assertRaises(fusion.FusionDisabled):
                fusion.build_messages(make_sample(), [], "系统提示")
        finally:
            if saved is None:
                os.environ.pop(fusion.FUSION_FLAG, None)
            else:
                os.environ[fusion.FUSION_FLAG] = saved

    def test_user_content_matches_official_shape(self) -> None:
        messages = fusion.build_messages(make_sample(), [], "系统提示")
        body = messages[1]["content"]
        self.assertIn("样本 ID: test1_000001_t9", body)
        self.assertIn("对话历史:", body)
        self.assertIn("user: ", body)
        self.assertIn("assistant: ", body)


class TestTruncation(FusionTestBase):
    def test_single_turn_truncation_keeps_head_and_tail(self) -> None:
        long_text = "开头" + "中" * 5000 + "结尾"
        result = fusion.truncate_middle(long_text, 10, 10)
        self.assertTrue(result.startswith("开头"))
        self.assertTrue(result.endswith("结尾"))
        self.assertIn("省略", result)
        self.assertLess(len(result), len(long_text))

    def test_short_text_untouched(self) -> None:
        self.assertEqual(fusion.truncate_middle("短", 10, 10), "短")

    def test_target_turn_survives_turn_limit(self) -> None:
        # 目标轮是最新的那一轮；限制轮数时必须留着它
        sample = make_sample(turn_count=30)
        turns, target = fusion._normalize_history(sample)
        kept = fusion.select_turns(turns, target, max_turns=4)
        self.assertEqual(len(kept), 4)
        self.assertEqual(fusion._turn_id(kept[-1]), target)

    def test_target_turn_survives_character_budget(self) -> None:
        sample = make_sample(turn_count=20)
        turns, target = fusion._normalize_history(sample)
        kept = fusion.select_turns(turns, target, max_total_chars=200)
        self.assertTrue(kept, "预算再紧也必须留下目标轮")
        self.assertEqual(fusion._turn_id(kept[-1]), target)

    def test_budget_drops_earliest_first(self) -> None:
        sample = make_sample(turn_count=20)
        turns, target = fusion._normalize_history(sample)
        kept = fusion.select_turns(turns, target, max_total_chars=200)
        self.assertLess(len(kept), len(turns))
        # 丢掉的一定是最早的那些
        kept_ids = [fusion._turn_id(t) for t in kept]
        self.assertNotIn(1, kept_ids)
        self.assertEqual(kept_ids, sorted(kept_ids))

    def test_build_messages_keeps_target_turn_text(self) -> None:
        # 目标轮就是最后一轮：轮数上限（24）与字符预算都不许把它挤掉。
        sample = make_sample(
            history=make_history(39), target_user_turn_id=39, id="test1_000001_t39"
        )
        messages = fusion.build_messages(sample, [], "系统提示")
        self.assertIn("第 39 轮", messages[1]["content"])
        self.assertNotIn("第 1 轮", messages[1]["content"])

    def test_turns_after_target_are_never_included(self) -> None:
        history = make_history(12)  # 目标轮设成 6，7~12 轮是「未来」
        sample = make_sample(history=history, target_user_turn_id=6, id="test1_000001_t6")
        messages = fusion.build_messages(sample, [], "系统提示")
        body = messages[1]["content"]
        self.assertIn("第 6 轮", body)
        self.assertNotIn("第 7 轮", body)
        self.assertNotIn("第 12 轮", body)


class TestRetrievalFiltering(FusionTestBase):
    def _retrieved(self, **overrides) -> dict:
        item = {"text": "参考回复正文", "score": 0.9, "source_id": "train_conv_000123"}
        item.update(overrides)
        return item

    def test_normal_item_is_used(self) -> None:
        messages = fusion.build_messages(
            make_sample(), [self._retrieved()], "系统提示"
        )
        self.assertIn("参考回复正文", messages[1]["content"])

    def test_self_retrieval_is_dropped(self) -> None:
        sample = make_sample()
        item = self._retrieved(source_id=sample["conversation_id"])
        prepared, dropped = fusion._usable_retrieved([item], sample)
        self.assertEqual(prepared, [])
        self.assertEqual(dropped, 1)

    def test_val_retrieval_is_dropped(self) -> None:
        # 角色④的纪律是语料只取 train；这里再拦一道，漏一次的代价是验证分数作废
        item = self._retrieved(source_id="val_conv_000001")
        prepared, dropped = fusion._usable_retrieved([item], make_sample())
        self.assertEqual(prepared, [])
        self.assertEqual(dropped, 1)

    def test_item_without_text_is_dropped(self) -> None:
        prepared, dropped = fusion._usable_retrieved(
            [{"score": 0.5, "source_id": "train_conv_000002"}, {"text": "   "}],
            make_sample(),
        )
        self.assertEqual(prepared, [])
        self.assertEqual(dropped, 2)

    def test_non_list_retrieved_is_tolerated(self) -> None:
        self.assertEqual(fusion._usable_retrieved(None, make_sample()), ([], 0))
        self.assertEqual(fusion._usable_retrieved("oops", make_sample()), ([], 0))

    def test_empty_retrieval_produces_no_section(self) -> None:
        messages = fusion.build_messages(make_sample(), [], "系统提示")
        self.assertNotIn("相似情境", messages[1]["content"])


class TestLeakAssertion(FusionTestBase):
    def test_clean_messages_pass(self) -> None:
        sample = make_sample(turn_count=9)
        messages = fusion.build_messages(sample, [], "系统提示")
        self.assertEqual(fusion.assert_no_leak(sample, messages), [])

    def test_detects_content_after_target_turn(self) -> None:
        history = make_history(9)
        sample = make_sample(history=history, target_user_turn_id=6)
        leaked = fusion._turn_content(history[7])  # turn_id=8，在目标轮 6 之后
        poisoned = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "对话历史:\n" + leaked},
        ]
        problems = fusion.assert_no_leak(sample, poisoned)
        self.assertTrue(any("泄漏" in p for p in problems), problems)

    def test_detects_missing_target_turn(self) -> None:
        sample = make_sample(turn_count=9)
        truncated = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "只剩早期上下文，目标轮被截断了"},
        ]
        problems = fusion.assert_no_leak(sample, truncated)
        self.assertTrue(any("目标轮" in p for p in problems), problems)

    def test_detects_bad_role_and_non_string_content(self) -> None:
        sample = make_sample()
        problems = fusion.assert_no_leak(
            sample, [{"role": "tool", "content": "x"}, {"role": "user", "content": 42}]
        )
        self.assertTrue(any("角色非法" in p for p in problems), problems)
        self.assertTrue(any("不是字符串" in p for p in problems), problems)

    def test_empty_messages_reported(self) -> None:
        self.assertTrue(fusion.assert_no_leak(make_sample(), []))


class TestContextStats(FusionTestBase):
    def test_reports_sane_numbers(self) -> None:
        stats = fusion.context_stats(make_sample(turn_count=40))
        self.assertEqual(stats["turns_total"], 40)
        self.assertLessEqual(stats["turns_kept"], 40)
        self.assertTrue(stats["target_turn_kept"])
        self.assertGreaterEqual(stats["chars_total"], stats["chars_kept"])

    def test_tolerates_broken_sample(self) -> None:
        stats = fusion.context_stats({"id": "x"})
        self.assertEqual(stats["turns_total"], 0)
        self.assertIsNone(stats["target_turn_kept"])


class TestTurnHelpers(FusionTestBase):
    def test_content_and_text_both_accepted(self) -> None:
        self.assertEqual(fusion._turn_content({"content": "a"}), "a")
        self.assertEqual(fusion._turn_content({"text": "b"}), "b")
        self.assertEqual(fusion._turn_content({"content": "", "text": "b"}), "b")
        self.assertEqual(fusion._turn_content({}), "")
        self.assertEqual(fusion._turn_content("nonsense"), "")

    def test_role_defaults_to_user(self) -> None:
        self.assertEqual(fusion._turn_role({"role": "assistant"}), "assistant")
        self.assertEqual(fusion._turn_role({"role": "system"}), "user")
        self.assertEqual(fusion._turn_role({}), "user")


class TestWiringSeam(FusionTestBase):
    """角色①的接线层 ↔ 本模块的接缝。

    注意这里要**同时**接管两边的提示词目录：`wiring.PROMPT_DIR` 决定接线层自己拼出
    什么系统提示，`fusion.PROMPT_DIR` 决定融合层在没人给它提示词时读哪里。生产环境里
    两者指向同一个物理目录（镜像里都是 `/root/participant/prompts`）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO_ROOT / "deploy" / "participant"))
        import wiring  # type: ignore

        cls.wiring = wiring

    def setUp(self) -> None:
        super().setUp()
        self._saved_wiring_prompt_dir = self.wiring.PROMPT_DIR
        patcher = mock.patch.object(self.wiring, "PROMPT_DIR", Path(self._temp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_wiring_passes_its_system_prompt_into_fusion(self) -> None:
        """激活融合后，系统提示必须仍然带上官方提示词（不能被丢掉）。"""
        self.write_prompt("response", "队友的回复规则")
        sample = make_sample()
        # 复刻 run_inference.py 的调用次序：先由接线层拼出系统提示（官方 + 队友块）……
        system_prompt = self.wiring.compose_system_prompt("官方 SYSTEM_PROMPT")
        # ……再连同官方的对话渲染回调一起交给接线层装配。
        messages = self.wiring.build_messages(
            sample, system_prompt, lambda s: "官方格式的对话历史"
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("官方 SYSTEM_PROMPT", messages[0]["content"])
        self.assertIn("队友的回复规则", messages[0]["content"])
        # 融合层接管了 user 部分（不再是 wiring 的官方格式占位串）
        self.assertIn("对话历史:", messages[1]["content"])
        self.assertNotEqual(messages[1]["content"], "官方格式的对话历史")

    def test_wiring_falls_back_when_fusion_unavailable(self) -> None:
        """融合层导入不到 → 必须退回官方组包，而不是崩掉。"""
        sample = make_sample()
        with mock.patch.object(self.wiring, "_import", lambda name: None):
            messages = self.wiring.build_messages(
                sample, "官方 SYSTEM_PROMPT", lambda s: "官方格式的对话历史"
            )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "官方 SYSTEM_PROMPT"},
                {"role": "user", "content": "官方格式的对话历史"},
            ],
        )

    def test_accepts_system_prompt_detection(self) -> None:
        self.assertTrue(self.wiring._accepts_system_prompt(fusion.build_messages))

        def legacy(sample, retrieved=None):  # 旧版两参签名
            return []

        self.assertFalse(self.wiring._accepts_system_prompt(legacy))


if __name__ == "__main__":
    unittest.main()
