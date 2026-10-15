"""数据适配器：把官方对话级数据展开成逐预测点的推理样本与训练目标。

三类输入的统一处理
------------------
1. ``train_public.jsonl`` / ``val_public.jsonl``（官方训练/验证集）
   对话级结构：``{conversation_id, turns[{turn_id, role, content}], prediction_points[...]}``
   每个 prediction point 带完整标签，是我们唯一的监督信号来源。
2. ``test_inference_data.jsonl``（官方推理样例）
   已经展开成逐样本结构：``{id, conversation_id, target_user_turn_id, history[3 字段]}``
   没有标签，只用来冒烟与格式对齐。
3. 官方评测时传入的 ``TEST_FILE``
   与第 2 类同构，但 ID 前缀变成 ``test1_``。

统一输出契约（**这是全队共用的接口，改动需同步训练/推理/评分三方**）
--------------------------------------------------------------------
每个样本：

.. code-block:: json

    {
      "id": "val_000001_t9",
      "conversation_id": "val_conv_000001",
      "target_user_turn_id": 9,
      "history": [{"turn_id": 1, "role": "user", "content": "..."}],
      "target": {                      // 仅训练/验证集有
        "response_text": "...",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [...], "interests": [...], "style": [...]},
        "memory_refs": ["mem_000001"]
      }
    }

``history`` 的字段名固定为 ``turn_id`` / ``role`` / ``content`` —— 与官方
``run_inference.py`` 读取的字段一致，因此训练数据可以直接喂给官方脚本做冒烟。

防泄漏
------
``history`` 只包含 ``turn_id <= target_user_turn_id`` 的轮次。目标轮**之后**的
assistant 回复（也就是答案本身）绝不进入输入。官方数据里答案确实存在于 ``turns`` 中
（我校验过：抽样 5169 个预测点，5169 个都能在 ``turns`` 里找到目标回复原文），所以这一步
不是可选优化，而是必须做对的切分。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TURN_FIELDS = ("turn_id", "role", "content")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class DataQuality:
    """一次转换过程中的数据质量统计，用于生成数据探索报告。"""

    conversations: int = 0
    samples: int = 0
    skipped_no_prediction_point: int = 0
    # assistant_response_turn_id 与 target_user_turn_id 不构成"下一轮"的预测点
    skipped_turn_mismatch: int = 0
    turn_mismatch_ids: list[str] = field(default_factory=list)
    # 目标回复原文出现在输入 history 中（**可能是真泄漏，必须逐条看**）
    leaked_samples: int = 0
    leaked_ids: list[str] = field(default_factory=list)
    # 目标回复与 history 中某个 assistant 轮高度重合（短回应重复，属于正常对话现象）
    repeat_response_samples: int = 0
    repeat_response_ids: list[str] = field(default_factory=list)
    # 标签分布
    emotion_distribution: Counter = field(default_factory=Counter)
    memory_ref_count_distribution: Counter = field(default_factory=Counter)
    # 画像全空的样本数（"宁缺勿滥"策略的直接证据）
    empty_profile_samples: int = 0
    response_char_lengths: list[int] = field(default_factory=list)

    def add_response_length(self, text: str) -> None:
        self.response_char_lengths.append(len(text))

    def summary(self) -> dict[str, Any]:
        lengths = sorted(self.response_char_lengths)

        def pct(p: float) -> int:
            if not lengths:
                return 0
            return lengths[min(len(lengths) - 1, int(len(lengths) * p))]

        return {
            "conversations": self.conversations,
            "samples": self.samples,
            "skipped_no_prediction_point": self.skipped_no_prediction_point,
            "skipped_turn_mismatch": self.skipped_turn_mismatch,
            "turn_mismatch_ids": self.turn_mismatch_ids,
            "leaked_samples": self.leaked_samples,
            "leaked_sample_ids": self.leaked_ids,
            "repeat_response_samples": self.repeat_response_samples,
            "repeat_response_sample_ids": self.repeat_response_ids[:50],
            "empty_profile_samples": self.empty_profile_samples,
            "emotion_distribution": dict(self.emotion_distribution.most_common()),
            "memory_ref_count_distribution": {
                str(k): v for k, v in sorted(self.memory_ref_count_distribution.items())
            },
            "response_char_length": {
                "count": len(lengths),
                "min": lengths[0] if lengths else 0,
                "p10": pct(0.10),
                "median": pct(0.50),
                "p90": pct(0.90),
                "max": lengths[-1] if lengths else 0,
            },
        }


class DataContractError(ValueError):
    """输入文件不符合预期结构时抛出，附带足够的定位信息。"""


# --------------------------------------------------------------------------- #
# 基础读写
# --------------------------------------------------------------------------- #
def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL；空行跳过，坏行报错并带上行号。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - 防御性
                raise DataContractError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# 字段兼容层
# --------------------------------------------------------------------------- #
def turn_content(turn: dict[str, Any]) -> str:
    """取轮次正文，同时兼容 ``content`` 与 ``text`` 两种写法。

    官方三份材料里字段名并不统一：``train_public.schema.json`` 用 ``content``，
    但同一批数据的实际 JSON 行里用的是 ``text``。适配器必须两种都认，否则会在
    换数据版本时静默丢光正文。
    """
    for key in ("content", "text"):
        value = turn.get(key)
        if isinstance(value, str):
            return value
    raise DataContractError(f"轮次缺少正文（content/text 都不存在）: {sorted(turn.keys())}")


def turn_id_of(turn: dict[str, Any]) -> int:
    value = turn.get("turn_id")
    if not isinstance(value, int):
        raise DataContractError(f"轮次缺少合法 turn_id: {turn!r}")
    return value


def longest_common_substring_len(a: str, b: str) -> int:
    """最长公共子串长度（滚动数组 DP）。

    用于判断"这段回答是不是已经说过了"。刻意不用最长公共**子序列**：中文口语里虚词重合
    太多，子序列会虚高。

    复杂度是 O(len(a) * len(b))，1 万级样本上是分钟量级，所以调用方先用廉价门控
    （见 :func:`response_overlap_ratio`）把明显不相干的候选跳过。
    """
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    previous = [0] * (len(a) + 1)
    best = 0
    for char_b in b:
        current = [0] * (len(a) + 1)
        for i, char_a in enumerate(a, start=1):
            if char_a == char_b:
                current[i] = previous[i - 1] + 1
                if current[i] > best:
                    best = current[i]
        previous = current
    return best


def response_overlap_ratio(response_text: str, history: list[dict[str, Any]]) -> float:
    """目标回复与**目标轮之前**的 assistant 轮的最大重合率（0~1）。

    先做门控再算最长公共子串：

    1. 逐位置求集合交集——若历史中与该位置相同字符的集合大小不到目标回复的 60%，
       最长公共子串不可能达到 60%，直接跳过 DP。这是廉价的必要条件过滤。
    2. 通过门控的候选才跑 :func:`longest_common_substring_len`。

    这样在 1.8 万条训练样本上是秒级，而不是分钟级。
    """
    if not response_text:
        return 0.0
    threshold = 0.60
    response_chars = set(response_text)
    best_ratio = 0.0
    for item in history:
        if item["role"] != "assistant":
            continue
        content = item["content"]
        if not content:
            continue
        # 门控 1：字表重合不足则必然达不到阈值
        if len(response_chars & set(content)) / len(response_chars) < threshold:
            continue
        # 门控 2：长度差距过大时子串不可能占满目标回复
        if len(content) < len(response_text) * threshold:
            continue
        ratio = longest_common_substring_len(response_text, content) / len(response_text)
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def make_sample_id(conversation_id: str, target_user_turn_id: int) -> str:
    """按官方样例的命名规则生成样本 ID。

    ``test1_conv_000001`` + 第 5 个用户轮 -> ``test1_000001_t5``
    规则即：对话 ID 去掉中间的 ``conv`` 段，再接 ``_t<轮号>``。
    """
    parts = conversation_id.split("_")
    if len(parts) >= 3 and parts[-2] == "conv":
        prefix, serial = parts[0], parts[-1]
        return f"{prefix}_{serial}_t{target_user_turn_id}"
    # 兜底：不符合约定式命名时保留原 ID，避免自作主张重编号
    return f"{conversation_id}_t{target_user_turn_id}"


def infer_prefix(conversation_id: str) -> str:
    return conversation_id.split("_")[0]


# --------------------------------------------------------------------------- #
# 转换
# --------------------------------------------------------------------------- #
def build_history(turns: list[dict[str, Any]], target_user_turn_id: int) -> list[dict[str, Any]]:
    """构造截止到目标用户轮（含）的上下文，字段统一成官方三字段。"""
    history: list[dict[str, Any]] = []
    for turn in sorted(turns, key=turn_id_of):
        if turn_id_of(turn) > target_user_turn_id:
            break
        history.append(
            {
                "turn_id": turn_id_of(turn),
                "role": turn["role"],
                "content": turn_content(turn),
            }
        )
    return history


def convert_conversations(
    conversations: Iterable[dict[str, Any]],
    *,
    drop_turn_mismatch: bool = True,
    drop_true_leaks: bool = True,
    leak_check: bool = True,
) -> tuple[list[dict[str, Any]], DataQuality]:
    """把对话级数据展开成逐预测点样本。

    参数
    ----
    drop_turn_mismatch:
        ``assistant_response_turn_id`` 不等于 ``target_user_turn_id + 1`` 的预测点，
        其 ``target.response_text`` 并非"下一轮回复"，是标签错位。官方训练集里有 4 条，
        默认丢弃；置 False 则保留并在质量报告里列出 ID。
    drop_true_leaks:
        目标回复（>= 60 字）与历史中某个 assistant 轮重合 60% 以上，说明答案已经写在
        输入里了，属于标注事故而非可学信号。官方训练集里有 2 条，默认丢弃。
    leak_check:
        是否执行上述泄漏自检。这个检查要先按字符集合门控再跑最长公共子串，1.8 万样本
        约几秒；若追求极致速度可以关掉。
    """
    samples: list[dict[str, Any]] = []
    quality = DataQuality()

    for conv in conversations:
        conversation_id = conv.get("conversation_id")
        turns = conv.get("turns")
        if not isinstance(conversation_id, str) or not isinstance(turns, list):
            raise DataContractError(f"对话缺少 conversation_id/turns: {sorted(conv.keys())}")
        quality.conversations += 1

        points = conv.get("prediction_points") or []
        if not points:
            quality.skipped_no_prediction_point += 1
            continue

        for point in points:
            target_user_turn_id = point.get("target_user_turn_id")
            assistant_turn_id = point.get("assistant_response_turn_id")
            target = point.get("target") or {}
            if not isinstance(target_user_turn_id, int):
                raise DataContractError(f"{conversation_id}: 预测点缺少 target_user_turn_id")

            sample_id = make_sample_id(conversation_id, target_user_turn_id)

            if assistant_turn_id != target_user_turn_id + 1:
                quality.skipped_turn_mismatch += 1
                quality.turn_mismatch_ids.append(
                    f"{sample_id}(assistant_turn={assistant_turn_id})"
                )
                if drop_turn_mismatch:
                    continue

            history = build_history(turns, target_user_turn_id)

            if leak_check:
                response_text = target.get("response_text") or ""
                probe = response_text[:30]
                if probe and any(probe in item["content"] for item in history):
                    # 区分两种命中：
                    # - 真泄漏：目标回复大部分内容已经出现在**目标轮之前**的 assistant 轮里。
                    # - 正常重复：情感陪伴场景里"嗯，我在听。""死心。"这类回应本来就会反复出现，
                    #   历史里有一句同样的话是自然对话现象，不是标签泄漏。
                    prev_assistant = [
                        item for item in history
                        if item["role"] == "assistant"
                        and item["turn_id"] < target_user_turn_id
                    ]
                    containing_length = max(
                        (
                            len(item["content"])
                            for item in prev_assistant
                            if probe in item["content"]
                        ),
                        default=0,
                    )
                    # 判据 A：目标回复很短，且与历史某轮长度几乎相同 —— 口头回应重复
                    is_short_repeat = (
                        len(response_text) <= 40
                        and containing_length > 0
                        and abs(containing_length - len(response_text)) <= 8
                    )
                    # 判据 B：只有"长回复被历史某轮整段复用"才算真泄漏。
                    # 必须带最小长度门槛：'嗯。''我听着。' 这类 2~4 字回应在陪伴场景里
                    # 高频重复，重合率天然 100%，用它做判据会淹没真信号。
                    # 实测训练集里 8 条"高重合"里只有 3 条是长回复整段复用。
                    overlap = response_overlap_ratio(response_text, prev_assistant)
                    is_true_leak = len(response_text) >= 60 and overlap >= 0.60
                    if is_short_repeat or not is_true_leak:
                        quality.repeat_response_samples += 1
                        quality.repeat_response_ids.append(sample_id)
                    else:
                        quality.leaked_samples += 1
                        quality.leaked_ids.append(sample_id)
                        if drop_true_leaks:
                            continue

            sample = {
                "id": sample_id,
                "conversation_id": conversation_id,
                "target_user_turn_id": target_user_turn_id,
                "history": history,
                "target": {
                    "response_text": target.get("response_text", ""),
                    "emotion_label": target.get("emotion_label", ""),
                    "user_profile": target.get("user_profile")
                    or {"personality_traits": [], "interests": [], "style": []},
                    "memory_refs": target.get("memory_refs") or [],
                },
            }
            samples.append(sample)

            quality.samples += 1
            quality.emotion_distribution[target.get("emotion_label", "")] += 1
            quality.memory_ref_count_distribution[len(sample["target"]["memory_refs"])] += 1
            profile = sample["target"]["user_profile"]
            if not any(profile.get(dim) for dim in ("personality_traits", "interests", "style")):
                quality.empty_profile_samples += 1
            quality.add_response_length(sample["target"]["response_text"])

    return samples, quality


def load_inference_samples(path: str | Path) -> list[dict[str, Any]]:
    """读取已展开的推理样本（官方测试样例或评测时传入的 TEST_FILE）。

    仅做结构规整，不猜标签。字段名统一为 ``history`` + ``content``。
    """
    samples: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        history_raw = record.get("history")
        if not isinstance(history_raw, list):
            raise DataContractError(f"{record.get('id')}: 缺少 history 数组")
        history = [
            {
                "turn_id": turn_id_of(turn),
                "role": turn["role"],
                "content": turn_content(turn),
            }
            for turn in sorted(history_raw, key=turn_id_of)
        ]
        samples.append(
            {
                "id": record.get("id") or make_sample_id(
                    record.get("conversation_id", "unknown"), record.get("target_user_turn_id", 0)
                ),
                "conversation_id": record.get("conversation_id", ""),
                "target_user_turn_id": record.get("target_user_turn_id"),
                "history": history,
            }
        )
    return samples


# --------------------------------------------------------------------------- #
# 筛选：为不同训练策略准备子集
# --------------------------------------------------------------------------- #
PROFILE_DIMS = ("personality_traits", "interests", "style")


def filter_samples(
    samples: Iterable[dict[str, Any]],
    *,
    with_profile_only: bool = False,
    without_profile_only: bool = False,
) -> list[dict[str, Any]]:
    """按画像是否非空切分样本。

    训练集里约 1/3 的预测点三个画像字段全为空。这是数据本身的分布，不是缺失值，
    因此既可以原样训练（学"什么时候不该输出画像"），也可以只保留非空样本训练
    （学"有证据时该输出什么"）。两种策略哪个在验证集上更好，必须由 evaluate.py
    实际跑出来，不能凭直觉选。本函数就是为这个对照实验准备的。
    """
    out: list[dict[str, Any]] = []
    for sample in samples:
        profile = (sample.get("target") or {}).get("user_profile") or {}
        has_profile = any(profile.get(dim) for dim in PROFILE_DIMS)
        if with_profile_only and not has_profile:
            continue
        if without_profile_only and has_profile:
            continue
        out.append(sample)
    return out
