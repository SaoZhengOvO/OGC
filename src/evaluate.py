"""本地离线评分工具：情绪准确率、画像准确率、回应质量（BLEU-4 / ROUGE-L / BERTScore）、时延。

写在前面：关于"和官方分数有多接近"
-----------------------------------
官方评分脚本没有随材料下发，本文件里的每一项指标定义都是**本地近似**，用于版本之间横向
比较与回归检测，不等于官方分数。需要在报告里如实标注的近似点：

====================================  ==================================================
指标                                  本地假设
====================================  ==================================================
情绪 Acc                              取 16 类单选，预测需与真值完全相同；空串计 0 分（与官方
                                      "空值不得分"一致）
画像 Acc                              三个子维度分别计算"集合精确匹配"，再对维度取平均；
                                     另附逐维度 micro-P/R/F1 与 Jaccard 作为诊断指标
``memory_refs``                       单独报告，默认不计入画像总分。官方口径为"记忆与属性抽取
                                     维度平均准确率"，该维度是否评分、权重多少均未公布
BLEU-4                                 corpus 级 + 标准简短惩罚；分词器可切换，默认纯 Python
                                     中英混合分词（无第三方依赖）
ROUGE-L                               LCS 的 F-measure 按样本平均
BERTScore                             默认关闭（需神经网络编码器）；提供接口供有环境的机器开启
时延                                  直接读官方 ``performance_report.json``，不做二次计时
====================================  ==================================================

设计约束
--------
- **零强制第三方依赖**：BLEU/ROUGE 全部手写，保证任何机器（含离线的评测容器）都能跑。
- 分词器可切换，用于评估"指标对分词方式的敏感度"。
- 所有输出都是纯 Python 基本类型，可直接 ``json.dump``。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from label_space import (
    EMOTIONS,
    PROFILE_DIMENSIONS,
    PROFILE_VALUES,
    normalize_emotion,
    normalize_profile,
)

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# --------------------------------------------------------------------------- #
# 分词器
# --------------------------------------------------------------------------- #
def char_tokenize(text: str) -> list[str]:
    """字符级分词：中文按字、英文数字按词，标点与空白丢弃。

    这是默认分词器。选择它是因为它完全不依赖外部词典，结果可复现，且对中文回复的
    区分度足够做版本间对比。**它与官方分词器可能不同**，所以 BLEU/ROUGE 的绝对值
    只能用于相对比较。
    """
    tokens: list[str] = []
    buffer: list[str] = []
    for ch in text:
        if _CJK.match(ch):
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()
            tokens.append(ch)
        elif ch.isalnum():
            buffer.append(ch)
        else:
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()
    if buffer:
        tokens.append("".join(buffer).lower())
    return tokens


def jieba_tokenize(text: str) -> list[str]:
    """jieba 分词（若环境装了 jieba）。装不上时由 ``get_tokenizer`` 自动退回字符级。"""
    import jieba  # noqa: PLC0415 - 可选依赖，按需导入

    return [tok for tok in jieba.lcut(text.lower()) if tok.strip()]


def get_tokenizer(name: str = "char") -> Callable[[str], list[str]]:
    if name == "char":
        return char_tokenize
    if name == "jieba":
        try:
            return jieba_tokenize
        except ImportError:  # pragma: no cover - 环境相关
            return char_tokenize
    raise ValueError(f"未知分词器: {name}（可选 char / jieba）")


# --------------------------------------------------------------------------- #
# BLEU-4
# --------------------------------------------------------------------------- #
def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


@dataclass
class BleuAccumulator:
    """corpus 级 BLEU-4 累加器（标准做法：先全语料累加裁剪计数，再算一次）。"""

    max_n: int = 4
    clipped: Counter = field(default_factory=lambda: Counter())
    total: Counter = field(default_factory=lambda: Counter())
    candidate_length: int = 0
    reference_length: int = 0

    def add(self, candidate: Sequence[str], reference: Sequence[str]) -> None:
        for n in range(1, self.max_n + 1):
            cand = _ngram_counts(candidate, n)
            ref = _ngram_counts(reference, n)
            for gram, count in cand.items():
                self.clipped[n] += min(count, ref.get(gram, 0))
            self.total[n] += max(0, len(candidate) - n + 1)
        self.candidate_length += len(candidate)
        self.reference_length += len(reference)

    def score(self) -> dict[str, float]:
        precisions: list[float] = []
        for n in range(1, self.max_n + 1):
            if self.total[n] == 0 or self.clipped[n] == 0:
                # 任一级命中为 0 时标准 BLEU 直接整体为 0，这里保留 0 以贴合标准定义
                return {"bleu4": 0.0, "brevity_penalty": 0.0, "precisions": precisions + [0.0]}
            precisions.append(self.clipped[n] / self.total[n])
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / self.max_n)
        if self.candidate_length == 0:
            bp = 0.0
        elif self.candidate_length > self.reference_length:
            bp = 1.0
        else:
            bp = math.exp(1 - self.reference_length / self.candidate_length)
        return {
            "bleu4": geo_mean * bp,
            "brevity_penalty": bp,
            "precisions": precisions,
        }


# --------------------------------------------------------------------------- #
# ROUGE-L
# --------------------------------------------------------------------------- #
def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """最长公共子序列长度（滚动数组实现，空间 O(min(len a, len b))）。"""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


@dataclass
class RougeLAccumulator:
    f_scores: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)

    def add(self, candidate: Sequence[str], reference: Sequence[str]) -> None:
        lcs = lcs_length(candidate, reference)
        if not candidate or not reference:
            self.f_scores.append(0.0)
            self.precision.append(0.0)
            self.recall.append(0.0)
            return
        p = lcs / len(candidate)
        r = lcs / len(reference)
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        self.precision.append(p)
        self.recall.append(r)
        self.f_scores.append(f)

    def score(self) -> dict[str, float]:
        if not self.f_scores:
            return {"rouge_l_f": 0.0, "rouge_l_precision": 0.0, "rouge_l_recall": 0.0}
        n = len(self.f_scores)
        return {
            "rouge_l_f": sum(self.f_scores) / n,
            "rouge_l_precision": sum(self.precision) / n,
            "rouge_l_recall": sum(self.recall) / n,
        }


# --------------------------------------------------------------------------- #
# 集合类指标（画像维度）
# --------------------------------------------------------------------------- #
@dataclass
class SetAccumulator:
    """用于一个画像维度（或 memory_refs）的集合匹配统计。"""

    exact_match: int = 0
    # 官方口径（GivenData/participant/README.md:100-102）：
    #   「画像子项和记忆引用写为空数组。空值作为该项预测失败参与评分，不会获得该项命中分。」
    # 所以预测为空 = 该项判错，即使金标也是空。我们原来的 exact_match 把「空 vs 空」
    # 算成命中，会系统性高估——两个口径都留，报告里并列。
    strict_match: int = 0
    empty_pred: int = 0
    total: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    jaccard_sum: float = 0.0

    def add(self, predicted: Iterable[str], reference: Iterable[str]) -> None:
        pred_set, ref_set = set(predicted), set(reference)
        self.total += 1
        if pred_set == ref_set:
            self.exact_match += 1
        if not pred_set:
            self.empty_pred += 1
        elif pred_set == ref_set:
            self.strict_match += 1
        self.tp += len(pred_set & ref_set)
        self.fp += len(pred_set - ref_set)
        self.fn += len(ref_set - pred_set)
        union = pred_set | ref_set
        self.jaccard_sum += 1.0 if not union else len(pred_set & ref_set) / len(union)

    def score(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        denom = self.total or 1
        return {
            "n": self.total,
            "exact_match_acc": self.exact_match / self.total if self.total else 0.0,
            "exact_match_acc_official": self.strict_match / denom,
            "empty_pred_rate": self.empty_pred / denom,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "macro_jaccard": self.jaccard_sum / self.total if self.total else 0.0,
        }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationReport:
    metrics: dict[str, Any]
    per_sample: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"metrics": self.metrics, "warnings": self.warnings}, ensure_ascii=False, indent=2
        )


def evaluate_predictions(
    predictions: Iterable[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    tokenizer_name: str = "char",
    include_memory_refs: bool = False,
    bertscore_encoder: Callable[[list[str]], list[Any]] | None = None,
) -> EvaluationReport:
    """对照打分。

    参数
    ----
    predictions:
        预测记录，需含 ``id`` 与 ``response_text`` / ``emotion_label`` / ``user_profile``
        / ``memory_refs``。字段缺失按空值处理（与官方"空值计失败"一致），不抛异常。
    references:
        ``{sample_id: {"response_text": ..., "emotion_label": ..., "user_profile": {...}}}``，
        由 :func:`load_references` 从验证集构造。
    include_memory_refs:
        是否把 ``memory_refs`` 计入画像维度平均。官方口径未公布，**默认不计入**，
        仅单独报告该维度的分数，避免用未确认的假设污染主指标。
    bertscore_encoder:
        可选，形如 ``f(list[str]) -> list[vector]`` 的编码器。给出后额外计算 BERTScore
        （余弦相似度 + 贪心匹配的 F1）。默认 None 则跳过，报告中标注为未计算。
    """
    tokenize = get_tokenizer(tokenizer_name)

    emotion_correct = 0
    emotion_confusion: dict[str, Counter] = defaultdict(Counter)
    profile_acc: dict[str, SetAccumulator] = {dim: SetAccumulator() for dim in PROFILE_DIMENSIONS}
    memory_acc = SetAccumulator()
    bleu = BleuAccumulator()
    rouge = RougeLAccumulator()
    per_sample: list[dict[str, Any]] = []

    missing_ids: list[str] = []
    extra_ids: list[str] = []
    empty_response = 0
    empty_emotion = 0
    cand_texts: list[str] = []
    ref_texts: list[str] = []

    seen: set[str] = set()
    for record in predictions:
        sample_id = record.get("id")
        if not isinstance(sample_id, str):
            missing_ids.append(repr(record.get("id")))
            continue
        if sample_id in seen:
            extra_ids.append(sample_id)
            continue
        seen.add(sample_id)

        reference = references.get(sample_id)
        if reference is None:
            extra_ids.append(sample_id)
            continue

        response_text = record.get("response_text")
        response_text = response_text if isinstance(response_text, str) else ""
        if not response_text:
            empty_response += 1

        raw_emotion = record.get("emotion_label")
        emotion = normalize_emotion(raw_emotion)
        if not emotion:
            empty_emotion += 1
        gold_emotion = normalize_emotion(reference.get("emotion_label"))
        if emotion and emotion == gold_emotion:
            emotion_correct += 1
        emotion_confusion[gold_emotion or "<empty>"][emotion or "<empty>"] += 1

        predicted_profile = normalize_profile(record.get("user_profile"))
        gold_profile = normalize_profile(reference.get("user_profile"))
        for dim in PROFILE_DIMENSIONS:
            profile_acc[dim].add(predicted_profile[dim], gold_profile[dim])

        predicted_memory = sorted(
            item for item in (record.get("memory_refs") or []) if isinstance(item, str)
        )
        memory_acc.add(predicted_memory, reference.get("memory_refs") or [])

        candidate_tokens = tokenize(response_text)
        reference_tokens = tokenize(reference.get("response_text", "") or "")
        bleu.add(candidate_tokens, reference_tokens)
        rouge.add(candidate_tokens, reference_tokens)
        cand_texts.append(response_text)
        ref_texts.append(reference.get("response_text", "") or "")

        per_sample.append(
            {
                "id": sample_id,
                "emotion_pred": emotion or "<empty>",
                "emotion_gold": gold_emotion or "<empty>",
                "emotion_hit": bool(emotion and emotion == gold_emotion),
                "profile_pred": predicted_profile,
                "profile_gold": gold_profile,
                "profile_dim_hit": {
                    dim: predicted_profile[dim] == gold_profile[dim] for dim in PROFILE_DIMENSIONS
                },
                "response_chars": len(response_text),
                "reference_chars": len(reference.get("response_text", "") or ""),
            }
        )

    for sample_id in references:
        if sample_id not in seen:
            missing_ids.append(sample_id)

    comparable = len(per_sample)
    warnings: list[str] = []
    if missing_ids:
        warnings.append(
            f"有 {len(missing_ids)} 个验证样本没有对应预测（示例 {missing_ids[:5]}）；"
            "这些样本在情绪/画像/回复三项上均计 0 分。"
        )
    if extra_ids:
        warnings.append(
            f"有 {len(extra_ids)} 条预测的 id 不在验证集里（示例 {extra_ids[:5]}），已忽略。"
        )
    if comparable == 0:
        warnings.append("没有任何 id 能对上，指标全部为 0；请先检查 id 生成规则。")
    if comparable and missing_ids:
        # 把未预测样本按 0 分计入分母，才反映官方口径
        emotion_denominator = comparable + len([i for i in missing_ids if i in references])
    else:
        emotion_denominator = comparable or 1

    profile_dim_scores = {dim: profile_acc[dim].score() for dim in PROFILE_DIMENSIONS}
    # 未预测样本已计入 warnings；主指标分母统一为验证集大小，避免"少交很多条反而分高"
    reference_total = len(references) or 1
    profile_acc_mean = sum(
        profile_acc[dim].exact_match for dim in PROFILE_DIMENSIONS
    ) / (reference_total * len(PROFILE_DIMENSIONS))
    # 官方口径下的同一指标（空预测一律判错），主指标应与它对齐。
    profile_acc_mean_official = sum(
        profile_acc[dim].strict_match for dim in PROFILE_DIMENSIONS
    ) / (reference_total * len(PROFILE_DIMENSIONS))

    dimensions_for_profile_mean = list(PROFILE_DIMENSIONS)
    if include_memory_refs:
        dimensions_for_profile_mean = dimensions_for_profile_mean + ["memory_refs"]

    metrics: dict[str, Any] = {
        "n_predictions": len(seen),
        "n_references": len(references),
        "n_comparable": comparable,
        "emotion_accuracy": emotion_correct / emotion_denominator,
        "emotion_empty_rate": empty_emotion / comparable if comparable else 0.0,
        "emotion_confusion": {gold: dict(row) for gold, row in sorted(emotion_confusion.items())},
        "profile_dimension_scores": profile_dim_scores,
        "profile_accuracy_mean_3dim": profile_acc_mean,
        "profile_accuracy_mean_3dim_official": profile_acc_mean_official,
        "profile_dimensions_included": dimensions_for_profile_mean,
        "memory_refs_scores": memory_acc.score(),
        "response_quality": {
            **bleu.score(),
            **rouge.score(),
            "tokenizer": tokenizer_name,
            "empty_response_rate": empty_response / comparable if comparable else 0.0,
        },
        "bertscore": _bertscore(
            cand_texts, ref_texts, bertscore_encoder
        ),
        "_caveat": (
            "本地近似实现，官方评分脚本未下发；绝对值仅供参考，请用版本间对比。"
            "画像两项口径并存：profile_accuracy_mean_3dim 把「空 vs 空」算命中（旧口径，"
            "会高估），profile_accuracy_mean_3dim_official 按 GivenData/participant/README.md "
            "的「空值作为该项预测失败」判，后者才对齐官方。"
        ),
    }
    return EvaluationReport(metrics=metrics, per_sample=per_sample, warnings=warnings)


def _bertscore(
    candidates: list[str],
    references: list[str],
    encoder: Callable[[list[str]], list[Any]] | None,
) -> dict[str, Any]:
    """BERTScore 的贪心匹配实现；未提供编码器时返回状态说明而非假分数。"""
    if encoder is None:
        return {
            "computed": False,
            "reason": "未提供 bertscore_encoder；该项需要神经网络编码器（离线环境需预置本地模型）。",
        }
    if not candidates:
        return {"computed": False, "reason": "没有可比对样本。"}

    def cosine(a: Any, b: Any) -> float:
        import numpy as np  # noqa: PLC0415

        va, vb = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return 0.0 if denom == 0 else float(va @ vb) / denom

    cand_vectors = encoder(candidates)
    ref_vectors = encoder(references)
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for cand_vec, ref_vec in zip(cand_vectors, ref_vectors):
        # predictables: 候选 token 各自最相似的参考 token 相似度均值
        precision_scores.append(
            sum(max(cosine(c, r) for r in ref_vec) for c in cand_vec) / max(1, len(cand_vec))
        )
        recall_scores.append(
            sum(max(cosine(r, c) for c in cand_vec) for r in ref_vec) / max(1, len(ref_vec))
        )
    p = sum(precision_scores) / len(precision_scores)
    r = sum(recall_scores) / len(recall_scores)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"computed": True, "precision": p, "recall": r, "f1": f}


# --------------------------------------------------------------------------- #
# 数据装载
# --------------------------------------------------------------------------- #
def load_references(path: str | Path) -> dict[str, dict[str, Any]]:
    """从逐样本 JSONL（适配器产物）构造评分所需的真值表。"""
    references: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            target = sample.get("target")
            if not isinstance(target, dict):
                continue
            references[sample["id"]] = target
    return references


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """读取预测 JSONL（官方 ``submission.jsonl`` 或本地产物）。"""
    predictions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc
    return predictions
