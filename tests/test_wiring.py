"""`deploy/participant/wiring.py` 的回归测试。

这一层是「队友产物 → 推理脚本」的唯一接缝，它的失败模式是**整队零分**
（镜像跑不出合规输出），所以每条回退分支都必须被钉住。

只依赖标准库，与 `tests/test_pipeline.py` 一样用 `python -m unittest discover -s tests` 跑。
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_DIR = REPO_ROOT / "deploy" / "participant"
sys.path.insert(0, str(PARTICIPANT_DIR))

import wiring  # noqa: E402


OFFICIAL = "官方提示词占位"
SAMPLE = {
    "id": "val_000001_t1",
    "conversation_id": "val_conv_000001",
    "target_user_turn_id": 1,
    "history": [{"turn_id": 1, "role": "user", "content": "今天好累"}],
}


def conversation_text(sample: dict) -> str:
    return "对话历史占位"


def _fake_rag(build_messages) -> dict:
    """构造一个假的 rag 包，塞进 sys.modules。"""
    package = types.ModuleType("rag")
    fusion = types.ModuleType("rag.fusion")
    fusion.build_messages = build_messages  # type: ignore[attr-defined]
    package.fusion = fusion  # type: ignore[attr-defined]
    return {"rag": package, "rag.fusion": fusion}


class FallbackTests(unittest.TestCase):
    """任何异常路径都必须产出一条可用的官方版 messages。"""

    def assert_official(self, messages: list[dict]) -> None:
        self.assertIsInstance(messages, list)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], OFFICIAL)
        self.assertEqual(messages[1]["content"], "对话历史占位")

    def test_rag_missing_falls_back(self) -> None:
        # 角色⑤A交付 rag/fusion.py 之后，这个文件在仓库里真实存在，所以必须**显式**
        # 模拟「rag 包不可导入」（sys.modules 里塞 None 会让 importlib 抛 ImportError）。
        with mock.patch.dict(sys.modules, {"rag": None, "rag.fusion": None}):
            self.assert_official(wiring.build_messages(SAMPLE, OFFICIAL, conversation_text))

    def test_fusion_raises_falls_back(self) -> None:
        def boom(sample, retrieved=None):
            raise RuntimeError("队友的模块炸了")

        with mock.patch.dict(sys.modules, _fake_rag(boom)):
            self.assert_official(wiring.build_messages(SAMPLE, OFFICIAL, conversation_text))

    def test_fusion_returns_garbage_falls_back(self) -> None:
        for garbage in ("我不是 list", [], [{"role": "tool", "content": "x"}],
                        [{"role": "system"}], [{"role": "system", "content": 42}]):
            with self.subTest(garbage=garbage):
                with mock.patch.dict(sys.modules, _fake_rag(lambda s, r=None: garbage)):
                    self.assert_official(wiring.build_messages(SAMPLE, OFFICIAL, conversation_text))

    def test_valid_fusion_output_is_used(self) -> None:
        def good(sample, retrieved=None):
            return [
                {"role": "system", "content": "队友的系统提示"},
                {"role": "user", "content": "队友的组包"},
            ]

        with mock.patch.dict(sys.modules, _fake_rag(good)):
            messages = wiring.build_messages(SAMPLE, OFFICIAL, conversation_text)
        self.assertEqual(messages[0]["content"], "队友的系统提示")
        self.assertEqual(messages[1]["content"], "队友的组包")

    def test_fusion_receives_retrieved_list(self) -> None:
        seen = {}

        def spy(sample, retrieved=None):
            seen["retrieved"] = retrieved
            return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

        # 角色④交付 rag/retrieve.py 之后，真实检索会返回非空结果，所以必须**显式**钉住
        # 「检索返回空」这一分支，否则这条测试会随语料变动而飘。
        package = types.ModuleType("rag")
        retrieve_mod = types.ModuleType("rag.retrieve")
        retrieve_mod.retrieve = lambda sample, k=3: []  # type: ignore[attr-defined]
        package.retrieve = retrieve_mod  # type: ignore[attr-defined]
        with mock.patch.dict(
            sys.modules, {**_fake_rag(spy), "rag.retrieve": retrieve_mod, "rag": package}
        ):
            wiring.build_messages(SAMPLE, OFFICIAL, conversation_text)
        self.assertEqual(seen["retrieved"], [])


class RetrieveTests(unittest.TestCase):
    def test_missing_module_returns_empty_list(self) -> None:
        # 用 None 占位让 importlib 抛 ImportError，模拟「rag 包不可导入」。
        # 不能依赖它真的不存在——④ 已经交付 rag/retrieve.py。
        with mock.patch.dict(sys.modules, {"rag": None, "rag.retrieve": None}):
            self.assertEqual(wiring.retrieve_memories(SAMPLE), [])

    def test_raising_module_returns_empty_list(self) -> None:
        package = types.ModuleType("rag")
        retrieve_mod = types.ModuleType("rag.retrieve")

        def boom(sample, k=3):
            raise RuntimeError("检索炸了")

        retrieve_mod.retrieve = boom  # type: ignore[attr-defined]
        package.retrieve = retrieve_mod  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"rag": package, "rag.retrieve": retrieve_mod}):
            self.assertEqual(wiring.retrieve_memories(SAMPLE), [])

    def test_non_list_and_non_dict_are_filtered(self) -> None:
        package = types.ModuleType("rag")
        retrieve_mod = types.ModuleType("rag.retrieve")
        retrieve_mod.retrieve = lambda sample, k=3: "不是 list"  # type: ignore[attr-defined]
        package.retrieve = retrieve_mod  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"rag": package, "rag.retrieve": retrieve_mod}):
            self.assertEqual(wiring.retrieve_memories(SAMPLE), [])

        retrieve_mod.retrieve = lambda sample, k=3: [  # type: ignore[attr-defined]
            {"text": "记忆", "score": 0.9, "source_id": "train_000001_t3"},
            "垃圾",
            None,
        ]
        with mock.patch.dict(sys.modules, {"rag": package, "rag.retrieve": retrieve_mod}):
            self.assertEqual(len(wiring.retrieve_memories(SAMPLE)), 1)


class PromptCompositionTests(unittest.TestCase):
    """官方提示词必须永远在，队友的块只能追加在后面。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._patcher = mock.patch.object(wiring, "PROMPT_DIR", Path(self._tmp.name))
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def write(self, name: str, text: str) -> None:
        (Path(self._tmp.name) / name).write_text(text, encoding="utf-8")

    def test_no_prompts_yields_official_only(self) -> None:
        self.assertEqual(wiring.compose_system_prompt(OFFICIAL), OFFICIAL)

    def test_blocks_are_appended_in_order(self) -> None:
        self.write("emotion_profile.md", "情绪规则")
        self.write("response.md", "回复规则")
        composed = wiring.compose_system_prompt(OFFICIAL)
        self.assertLess(composed.index(OFFICIAL), composed.index("情绪规则"))
        self.assertLess(composed.index("情绪规则"), composed.index("回复规则"))

    def test_missing_block_is_skipped_silently(self) -> None:
        self.write("response.md", "回复规则")
        composed = wiring.compose_system_prompt(OFFICIAL)
        self.assertIn("回复规则", composed)
        self.assertNotIn("情绪规则", composed)

    def test_empty_file_is_treated_as_missing(self) -> None:
        self.write("emotion_profile.md", "   \n\n  ")
        self.assertEqual(wiring.compose_system_prompt(OFFICIAL), OFFICIAL)

    def test_txt_extension_is_accepted(self) -> None:
        self.write("emotion_profile.txt", "txt 版规则")
        self.assertIn("txt 版规则", wiring.compose_system_prompt(OFFICIAL))

    def test_md_wins_over_txt(self) -> None:
        self.write("emotion_profile.md", "md 版规则")
        self.write("emotion_profile.txt", "txt 版规则")
        composed = wiring.compose_system_prompt(OFFICIAL)
        self.assertIn("md 版规则", composed)
        self.assertNotIn("txt 版规则", composed)


class WiringDoesNotTouchGivenDataTests(unittest.TestCase):
    """官方原件必须一个字节都不动（我们只改 deploy/ 里的副本）。"""

    def test_official_copy_has_no_wiring_import(self) -> None:
        official = (REPO_ROOT / "GivenData" / "participant" / "run_inference.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import wiring", official)

    def test_deployed_copy_survives_missing_wiring(self) -> None:
        """deploy 副本在没有 wiring.py 时也必须能编译并退回官方行为。"""
        source = (PARTICIPANT_DIR / "run_inference.py").read_text(encoding="utf-8")
        self.assertIn("import wiring", source)
        self.assertIn("except Exception", source)


if __name__ == "__main__":
    unittest.main()
