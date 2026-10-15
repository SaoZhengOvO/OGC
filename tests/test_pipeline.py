"""本地评测闭环的回归测试（只依赖标准库，用 unittest 跑）。

这份测试锁的是**踩过的坑**，不是覆盖率：

1. ``memory_refs`` 编号是**轮次**不是数组下标（见 CONTEXT.md），解析器必须按轮次取；
2. 训练集里真实存在的 4 条错位样本、2 条真泄漏样本必须被默认丢弃，
   而 39 条"短回应重复"是正常对话现象、**不能**被当成泄漏丢掉；
3. 官方 schema 的 id 正则 ``^test_[0-9]{6}_t[0-9]+$`` 与官方推理样例的
   ``test1_000001_t5`` 互相矛盾，宽松模式必须放过后者。

运行::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data_processor  # noqa: E402
import evaluate  # noqa: E402
import label_space  # noqa: E402
import validate_submission  # noqa: E402


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def make_record(sample_id: str, **overrides) -> dict:
    """一条 schema 合法的提交记录。"""
    record = {
        "id": sample_id,
        "response_text": "听起来今天挺不容易的，愿意多说两句吗？",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    }
    record.update(overrides)
    return record


def empty_target(response: str, emotion: str = "anxiety") -> dict:
    return {
        "response_text": response,
        "emotion_label": emotion,
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    }


class LabelSpaceTest(unittest.TestCase):
    def test_emotion_space_has_17_values(self) -> None:
        # 16 个情绪 + 空串（"宁缺勿滥"：标注不到就不写）
        self.assertEqual(len(label_space.EMOTION_VALUES_WITH_EMPTY), 17)
        self.assertIn("", label_space.EMOTION_VALUES_WITH_EMPTY)
        self.assertIn("anxiety", label_space.EMOTION_VALUES_WITH_EMPTY)

    def test_profile_dimensions_are_three(self) -> None:
        self.assertEqual(
            tuple(label_space.PROFILE_DIMENSIONS),
            ("personality_traits", "interests", "style"),
        )


class ValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_official_prefixed_ids_pass_lenient(self) -> None:
        """官方推理样例用的是 test1_ 前缀，宽松模式必须放过（strict 会拒）。"""
        path = write_jsonl(
            self.dir / "sub.jsonl",
            [make_record("test1_000001_t5"), make_record("test1_000001_t7")],
        )
        self.assertTrue(validate_submission.validate_submission(path).ok)

    def test_local_train_val_ids_pass_lenient_but_fail_strict(self) -> None:
        """本地自测要放行 train_/val_，严格模式仍按官方 schema 拒绝。"""
        path = write_jsonl(self.dir / "sub.jsonl", [make_record("val_000001_t1")])
        self.assertTrue(validate_submission.validate_submission(path).ok)
        strict = validate_submission.validate_submission(path, strict_schema_id=True)
        self.assertFalse(strict.ok)
        self.assertEqual([i.rule for i in strict.issues], ["id.pattern"])

    def test_duplicate_id_is_caught(self) -> None:
        path = write_jsonl(self.dir / "sub.jsonl", [make_record("test_000001_t1")] * 2)
        result = validate_submission.validate_submission(path)
        self.assertIn("id.duplicate", [i.rule for i in result.issues])

    def test_field_typo_is_caught(self) -> None:
        """README 点名过的坑：emotion 写成 emotions 会静默丢分。"""
        record = make_record("test_000001_t1")
        record["emotions"] = record.pop("emotion_label")
        path = write_jsonl(self.dir / "sub.jsonl", [record])
        rules = [i.rule for i in validate_submission.validate_submission(path).issues]
        self.assertIn("field.extra", rules)
        self.assertIn("field.missing", rules)

    def test_enum_and_pattern_violations_are_caught(self) -> None:
        cases = {
            "emotion_label.enum": {"emotion_label": None},
            "user_profile.interests.enum": {
                "user_profile": {
                    "personality_traits": [],
                    "interests": ["not_a_real_tag"],
                    "style": [],
                }
            },
            "memory_refs.pattern": {"memory_refs": ["mem_1"]},
        }
        for rule, override in cases.items():
            with self.subTest(rule=rule):
                path = write_jsonl(
                    self.dir / f"{rule}.jsonl",
                    [make_record("test_000001_t1", **override)],
                )
                result = validate_submission.validate_submission(path)
                self.assertIn(rule, [i.rule for i in result.issues])

    def test_id_coverage_against_input_is_checked(self) -> None:
        """漏条是致命的：输入有 2 条、提交只有 1 条必须报出来。"""
        input_path = write_jsonl(
            self.dir / "input.jsonl",
            [{"id": "test_000001_t1"}, {"id": "test_000001_t3"}],
        )
        sub_path = write_jsonl(self.dir / "sub.jsonl", [make_record("test_000001_t1")])
        result = validate_submission.validate_submission(sub_path, input_path=input_path)
        self.assertFalse(result.ok)

    def test_bad_json_line_reports_line_number(self) -> None:
        path = self.dir / "sub.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(make_record("test_000001_t1"), ensure_ascii=False) + "\n{坏行\n",
            encoding="utf-8",
        )
        result = validate_submission.validate_submission(path)
        json_issues = [i for i in result.issues if i.rule == "json"]
        self.assertEqual(len(json_issues), 1)
        self.assertEqual(json_issues[0].line, 2)


class DataProcessorTest(unittest.TestCase):
    """按真实数据集里那几条坏样本的形状构造样例，确认过滤行为。"""

    USER1 = "最近有点累。"

    def _conversation(self, target_turn: int, assistant_turn: int, target: dict,
                      turns: list[dict] | None = None) -> dict:
        return {
            "conversation_id": "test1_conv_000001",
            "turns": turns if turns is not None else [
                {"turn_id": 1, "role": "user", "text": self.USER1}
            ],
            "prediction_points": [
                {
                    "target_user_turn_id": target_turn,
                    "assistant_response_turn_id": assistant_turn,
                    "target": target,
                }
            ],
        }

    def test_turn_mismatch_dropped_by_default(self) -> None:
        conv = self._conversation(7, 6, empty_target("我在。"))
        samples, quality = data_processor.convert_conversations([conv])
        self.assertEqual(samples, [])
        self.assertEqual(quality.skipped_turn_mismatch, 1)
        self.assertIn("test1_000001_t7", quality.turn_mismatch_ids[0])

    def test_turn_mismatch_kept_when_asked(self) -> None:
        conv = self._conversation(7, 6, empty_target("我在。"))
        samples, _ = data_processor.convert_conversations(
            [conv], drop_turn_mismatch=False
        )
        self.assertEqual(len(samples), 1)

    def test_sample_id_follows_official_naming(self) -> None:
        """``test1_conv_000001`` + 第 5 个用户轮 -> ``test1_000001_t5``。"""
        self.assertEqual(
            data_processor.make_sample_id("test1_conv_000001", 5), "test1_000001_t5"
        )

    def test_short_repeat_is_not_a_leak(self) -> None:
        """'嗯，我在听。' 在陪伴场景本来就该重复，不能当泄漏丢掉。"""
        short = "嗯，我在听。"
        conv = self._conversation(
            3,
            4,
            empty_target(short, emotion="neutral"),
            turns=[
                {"turn_id": 1, "role": "user", "text": self.USER1},
                {"turn_id": 2, "role": "assistant", "text": short},
                {"turn_id": 3, "role": "user", "text": "嗯。"},
            ],
        )
        samples, quality = data_processor.convert_conversations([conv])
        self.assertEqual(len(samples), 1, "短回应重复必须保留")
        self.assertEqual(quality.leaked_samples, 0)
        self.assertEqual(quality.repeat_response_samples, 1)

    def test_long_verbatim_reuse_is_dropped_as_leak(self) -> None:
        long_reply = (
            "我听到你说最近压力很大，我们一起把它拆小一点：先照顾好睡眠，"
            "再挑一件最急的事推进，其他的可以往后放一放，别急着一次做完。"
        )
        conv = self._conversation(
            3,
            4,
            empty_target(long_reply),
            turns=[
                {"turn_id": 1, "role": "user", "text": self.USER1},
                {"turn_id": 2, "role": "assistant", "text": long_reply},
                {"turn_id": 3, "role": "user", "text": "嗯。"},
            ],
        )
        samples, quality = data_processor.convert_conversations([conv])
        self.assertEqual(samples, [], "答案已写在输入里的样本是标注事故，必须丢")
        self.assertEqual(quality.leaked_samples, 1)

        kept, _ = data_processor.convert_conversations([conv], drop_true_leaks=False)
        self.assertEqual(len(kept), 1, "关闭开关后应保留，便于人工复核")

    def test_turn_content_accepts_both_content_and_text(self) -> None:
        """官方 schema 写 content，实际 JSONL 行里写 text，两种都得认。"""
        self.assertEqual(data_processor.turn_content({"content": "甲"}), "甲")
        self.assertEqual(data_processor.turn_content({"text": "乙"}), "乙")
        with self.assertRaises(data_processor.DataContractError):
            data_processor.turn_content({"role": "user"})

    def test_history_is_capped_at_target_turn(self) -> None:
        """上下文必须截止到目标轮（含），后面的话不能泄进输入。"""
        turns = [
            {"turn_id": 1, "role": "user", "text": "甲"},
            {"turn_id": 2, "role": "assistant", "text": "乙"},
            {"turn_id": 3, "role": "user", "text": "丙"},
            {"turn_id": 4, "role": "assistant", "text": "丁"},
        ]
        history = data_processor.build_history(turns, 3)
        self.assertEqual([h["turn_id"] for h in history], [1, 2, 3])
        self.assertNotIn("丁", [h["content"] for h in history])

    def test_missing_conversation_id_raises_with_context(self) -> None:
        with self.assertRaises(data_processor.DataContractError):
            data_processor.convert_conversations([{"prediction_points": []}])

    def test_load_inference_samples_does_not_need_labels(self) -> None:
        """推理输入只有 history，没有 target——这是测试集，不应要求标签。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(
                Path(tmp) / "test.jsonl",
                [
                    {
                        "id": "test1_000001_t5",
                        "conversation_id": "test1_conv_000001",
                        "target_user_turn_id": 5,
                        "history": [
                            {"turn_id": 1, "role": "user", "content": self.USER1}
                        ],
                    }
                ],
            )
            samples = data_processor.load_inference_samples(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["id"], "test1_000001_t5")


class EvaluateTest(unittest.TestCase):
    ID = "test_000001_t1"

    def _pair(self) -> tuple[list[dict], dict[str, dict]]:
        # references 表的值就是 target 本体（load_references 取 sample["target"]），
        # 不是整条样本记录——这里容易写错，写错会静默变成全 0 分。
        #
        # 画像标签必须用官方 ASCII 词表（music / travel_outdoor ...），不能用中文：
        # normalize_profile 会把词表外的取值直接丢掉，于是"乱填"和"空"得分相同。
        target = {
            "response_text": "听起来今天挺不容易的。",
            "emotion_label": "anxiety",
            "user_profile": {
                "personality_traits": ["sensitive"],
                "interests": ["music"],
                "style": ["brief"],
            },
            "memory_refs": ["mem_000002"],
        }
        # 必须深拷贝：prediction 与 reference 若共享同一个嵌套 dict，
        # 测试里改 prediction 会连带改掉 reference，断言就会假通过。
        return [{"id": self.ID, **copy.deepcopy(target)}], {self.ID: target}

    def test_perfect_prediction_scores_one(self) -> None:
        prediction, references = self._pair()
        metrics = evaluate.evaluate_predictions(prediction, references).metrics
        self.assertAlmostEqual(metrics["emotion_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(metrics["profile_accuracy_mean_3dim"], 1.0, places=6)
        self.assertAlmostEqual(metrics["response_quality"]["bleu4"], 1.0, places=6)
        self.assertAlmostEqual(metrics["response_quality"]["rouge_l_f"], 1.0, places=6)

    def test_wrong_emotion_costs_accuracy_not_bleu(self) -> None:
        """三项相互独立，不能一损俱损。"""
        prediction, references = self._pair()
        prediction[0]["emotion_label"] = "joy"
        metrics = evaluate.evaluate_predictions(prediction, references).metrics
        self.assertAlmostEqual(metrics["emotion_accuracy"], 0.0, places=6)
        self.assertAlmostEqual(metrics["response_quality"]["bleu4"], 1.0, places=6)

    def test_empty_emotion_counts_as_failure(self) -> None:
        """"宁缺勿滥"策略的代价：空值在评测里不是"跳过"，是判错。"""
        prediction, references = self._pair()
        prediction[0]["emotion_label"] = ""
        metrics = evaluate.evaluate_predictions(prediction, references).metrics
        self.assertAlmostEqual(metrics["emotion_accuracy"], 0.0, places=6)
        self.assertAlmostEqual(metrics["emotion_empty_rate"], 1.0, places=6)

    def test_missing_prediction_is_warned_and_zeroed(self) -> None:
        _, references = self._pair()
        report = evaluate.evaluate_predictions([], references)
        self.assertEqual(report.metrics["n_comparable"], 0)
        self.assertEqual(report.metrics["n_references"], 1)
        self.assertTrue(report.warnings, "漏条必须产生告警，否则会被静默吞掉")

    def test_profile_scoring_is_set_based(self) -> None:
        """画像按集合比，多报一个标签要扣分（否则乱填能刷分）。"""
        prediction, references = self._pair()
        prediction[0]["user_profile"]["interests"] = ["music", "travel_outdoor"]
        metrics = evaluate.evaluate_predictions(prediction, references).metrics
        detail = metrics["profile_dimension_scores"]["interests"]
        self.assertLess(detail["macro_jaccard"], 1.0)
        self.assertLess(detail["micro_f1"], 1.0)

    def test_out_of_vocabulary_profile_tag_is_dropped_to_empty(self) -> None:
        """词表外的画像取值会被静默丢掉——所以预测必须输出官方 ASCII 标签。

        这条测试是"防静默"的：丢掉之后不仅拿不到分，还会和"空预测"无从区分，
        因此这里把行为固定下来，将来若改成"报错"就说明有意变更。
        """
        prediction, references = self._pair()
        prediction[0]["user_profile"]["interests"] = ["音乐"]
        metrics = evaluate.evaluate_predictions(prediction, references).metrics
        detail = metrics["profile_dimension_scores"]["interests"]
        self.assertEqual(detail["exact_match_acc"], 0.0)
        self.assertEqual(detail["micro_f1"], 0.0)

    def test_memory_refs_reported_but_excluded_by_default(self) -> None:
        """官方口径未公布，memory_refs 默认不计入画像主指标。"""
        prediction, references = self._pair()
        metrics = evaluate.evaluate_predictions(prediction, references).metrics
        self.assertAlmostEqual(metrics["memory_refs_scores"]["micro_f1"], 1.0, places=6)
        self.assertEqual(
            metrics["profile_dimensions_included"],
            ["personality_traits", "interests", "style"],
        )
        included = evaluate.evaluate_predictions(
            prediction, references, include_memory_refs=True
        ).metrics
        self.assertEqual(len(included["profile_dimensions_included"]), 4)

    def test_char_tokenizer_is_default_and_unknown_is_rejected(self) -> None:
        self.assertEqual(evaluate.get_tokenizer("char")("你好"), list("你好"))
        with self.assertRaises(ValueError):
            evaluate.get_tokenizer("no_such_tokenizer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
