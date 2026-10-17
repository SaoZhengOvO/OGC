"""角色④：检索。契约（`docs/roles/CONTRACT.md`、`docs/roles/04-rag-corpus.md`）冻结为：

    def retrieve(sample: dict, k: int = 3) -> list[dict]:
        \"\"\"返回按相关度降序的最多 k 条：[{"text": str, "score": float, "source_id": str}]\"\"\"

## 三条静默丢弃的坑（写在最前面，因为这三点不会报错）

`rag/fusion.py` 的 `_usable_retrieved()`（约 265-300 行）对检索结果做三道过滤，
**不合规的直接丢掉，不抛异常、不打日志**。所以返回值必须同时满足：

1. ``text`` 是非空字符串；
2. ``source_id`` 不等于本样本的 ``conversation_id``，也不等于本样本的 ``id``
   —— 这叫自泄漏，等于把答案喂给自己；
3. ``source_id`` 不以下划线前的 ``val_`` 开头 —— 语料只取 train。

第 2 条有个额外陷阱：fusion 拿 ``source_id`` 去比 ``conversation_id``，但真实数据里
``source_id`` 形如 ``train_conv_000001_t9``、``conversation_id`` 是
``train_conv_000001``，**两者永远不相等**，所以 fusion 那道检查实际上抓不到同对话泄漏。
本模块因此在检索侧自己按 ``conv`` 字段排除同对话（一段对话最多有 8 个预测点，
不排除就会把同一段对话里别的目标轮回复喂进去）。

## 为什么不索引整段对话

单段对话最长 40,262 字符，18,651 段全塞进倒排会撑爆内存，而且 BM25 的长度归一化会
把长文本的分数冲淡。**只索引 query（用户被回复的那一句，中位 78 字）**，
reply 只作为返回内容，不进倒排。

## 依赖

只用标准库：``array`` + ``math`` + ``re`` + ``json``。镜像里没有 sklearn、没有 jieba，
任何第三方依赖都会让离线包变大并且多一个装不上的风险点。
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import threading
from array import array

__all__ = ["retrieve", "corpus_stats", "warm_up"]

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

#: 语料文件。默认仓库内 ``rag/corpus/corpus.jsonl``，可用环境变量覆盖
#: （镜像里 rag/ 被拷到 /root/rag/，默认值自然跟着变，不需要改）。
CORPUS_PATH = os.environ.get("PARTICIPANT_RAG_CORPUS") or os.path.join(
    _HERE, "corpus", "corpus.jsonl"
)
#: 索引缓存。可有可无：写不进去就每次现建，不影响正确性。
INDEX_PATH = os.environ.get("PARTICIPANT_RAG_INDEX") or os.path.join(
    _HERE, "index", "bm25.pkl"
)

#: BM25 参数。中文短查询 + 短文档，k1/b 用常见默认值即可。
_K1 = 1.2
_B = 0.75

# --------------------------------------------------------------------------
# 分词：中文字符 bigram + 英文/数字整词，纯标准库
# --------------------------------------------------------------------------

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """中文取字符 bigram（单字段落退化成一元），英文数字取小写整词。

    不做停用词过滤：中文停用词表对 78 字的短查询伤害大于收益（「我」「不」「没」
    这些在情绪表达里恰恰是信号）。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_WORD.findall(lowered))
    return tokens


# --------------------------------------------------------------------------
# 语料与索引（惰性加载，进程内只建一次）
# --------------------------------------------------------------------------

_lock = threading.Lock()
_records: list[dict] | None = None
_index: dict[str, array] | None = None
_doc_len: array | None = None
_avg_len: float = 0.0
_load_error: str | None = None


def _read_corpus(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item.get("q") or ""
            reply = item.get("r") or ""
            source_id = item.get("id") or ""
            if not query or not reply or not source_id:
                continue
            records.append(
                {
                    "id": source_id,
                    "conv": item.get("conv") or "",
                    "q": query,
                    "r": reply,
                    "emo": item.get("emo") or "",
                }
            )
    return records


def _build_index(records: list[dict]) -> tuple[dict[str, array], array]:
    """倒排表：term -> array('i')，按 (doc_id, tf) 交错排列。

    交错存比「doc_ids 一个数组 + tfs 一个数组」少一次 dict 查表，
    pickle 出来的体积也小一半（1.4M posting ≈ 11 MB）。
    """
    buckets: dict[str, list[int]] = {}
    lengths = array("i")
    for doc_id, record in enumerate(records):
        tokens = _tokens(record["q"])
        lengths.append(len(tokens))
        if not tokens:
            continue
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        for token, freq in term_freq.items():
            bucket = buckets.get(token)
            if bucket is None:
                bucket = buckets[token] = []
            bucket.append(doc_id)
            bucket.append(freq)

    packed: dict[str, array] = {}
    for token, flat in buckets.items():
        packed[token] = array("i", flat)
    return packed, lengths


def _load() -> None:
    """首次调用时把语料和索引装进内存。失败则置 _load_error，retrieve 退化为 []。"""
    global _records, _index, _doc_len, _avg_len, _load_error
    if _records is not None or _load_error is not None:
        return
    with _lock:
        if _records is not None or _load_error is not None:
            return

        # 先试索引缓存：命中就只用反序列化，省掉 15 秒建表。
        if os.path.exists(INDEX_PATH) and os.path.exists(CORPUS_PATH):
            try:
                with open(INDEX_PATH, "rb") as handle:
                    payload = pickle.load(handle)
                if payload.get("version") == 1:
                    _records = _read_corpus(CORPUS_PATH)
                    if payload.get("corpus_size") == len(_records):
                        _index = payload["index"]
                        _doc_len = payload["doc_len"]
                        _avg_len = payload["avg_len"]
                        return
            except Exception:  # noqa: BLE001 - 缓存坏了就重建，不能因此挂掉
                pass

        if not os.path.exists(CORPUS_PATH):
            _load_error = f"语料不存在：{CORPUS_PATH}"
            return
        try:
            _records = _read_corpus(CORPUS_PATH)
            _index, _doc_len = _build_index(_records)
            total = sum(_doc_len) or 1
            _avg_len = total / max(len(_doc_len), 1)
        except Exception as exc:  # noqa: BLE001 - 检索线失败必须降级，不能拖垮推理
            _load_error = f"{type(exc).__name__}: {exc}"
            _records = None
            return

        # 顺手把索引缓存下来，失败无所谓。
        try:
            os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
            with open(INDEX_PATH, "wb") as handle:
                pickle.dump(
                    {
                        "version": 1,
                        "corpus_size": len(_records),
                        "index": _index,
                        "doc_len": _doc_len,
                        "avg_len": _avg_len,
                    },
                    handle,
                    protocol=4,
                )
        except Exception:  # noqa: BLE001
            pass


def warm_up() -> str:
    """预建索引，把十几秒的建表成本挪到启动阶段。返回状态字符串。"""
    _load()
    if _load_error:
        return f"error: {_load_error}"
    return f"ok: {len(_records or []):,} 条语料，{len(_index or {})} 个 term"


def corpus_stats() -> dict:
    """给 wiring.describe() / 自检脚本看的摘要。"""
    _load()
    return {
        "corpus_path": CORPUS_PATH,
        "corpus_exists": os.path.exists(CORPUS_PATH),
        "corpus_size": len(_records or []),
        "terms": len(_index or {}),
        "avg_doc_len": round(_avg_len, 2),
        "load_error": _load_error,
        "has_retrieve": callable(globals().get("retrieve")),
    }


# --------------------------------------------------------------------------
# 查询侧：从 sample 里取「我要回复的那句用户话」
# --------------------------------------------------------------------------


def _target_query(sample: dict) -> str:
    """取出目标用户轮的正文。

    优先按 ``target_user_turn_id`` 精确匹配（官方样本都带这个字段）；
    取不到就退回「history 里最后一条 user 消息」—— 推理链路里这两者等价。
    """
    history = sample.get("history") or []
    target_turn_id = sample.get("target_user_turn_id")
    if target_turn_id is not None:
        for turn in history:
            if not isinstance(turn, dict):
                continue
            if turn.get("turn_id") == target_turn_id and turn.get("role") == "user":
                return str(turn.get("content") or "")
    for turn in reversed(history):
        if isinstance(turn, dict) and turn.get("role") == "user":
            return str(turn.get("content") or "")
    return ""


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def retrieve(sample: dict, k: int = 3) -> list[dict]:
    """按相关度返回最多 k 条相似情境的助手回复。

    Args:
        sample: 官方样本，含 ``history`` / ``target_user_turn_id`` / ``conversation_id``。
        k: 返回条数上限。会被 fusion 的 ``PARTICIPANT_MAX_RETRIEVED`` 再截一次。

    Returns:
        ``[{"text": str, "score": float, "source_id": str}, ...]``，按 score 降序。
        语料缺失或查询为空时返回 ``[]``——**这是合法状态，fusion 会当作没有检索**。
    """
    if not isinstance(sample, dict) or k <= 0:
        return []
    _load()
    if _load_error or not _records or not _index or not _doc_len:
        return []

    query = _target_query(sample)
    if not query.strip():
        return []
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    own_id = str(sample.get("id") or "")
    own_conv = str(sample.get("conversation_id") or "")

    scores: dict[int, float] = {}
    total_docs = len(_records)
    for token in query_tokens:
        bucket = _index.get(token)
        if bucket is None:
            continue
        idf = math.log(1.0 + (total_docs - len(bucket) // 2 + 0.5) / (len(bucket) // 2 + 0.5))
        if idf <= 0.0:
            continue
        for offset in range(0, len(bucket), 2):
            doc_id = bucket[offset]
            freq = bucket[offset + 1]
            length = _doc_len[doc_id] or 0
            denominator = freq + _K1 * (1.0 - _B + _B * length / (_avg_len or 1.0))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (_K1 + 1.0) / denominator

    if not scores:
        return []

    # 排除自泄漏：同一条样本、同一段对话、以及任何 val_ 来源。
    ranked: list[tuple[float, int]] = []
    for doc_id, score in scores.items():
        record = _records[doc_id]
        source_id = record["id"]
        if not source_id or source_id.startswith("val_"):
            continue
        if own_id and source_id == own_id:
            continue
        if own_conv and record["conv"] == own_conv:
            continue
        if not record["r"]:
            continue
        ranked.append((score, doc_id))

    ranked.sort(reverse=True)

    results: list[dict] = []
    for score, doc_id in ranked[:k]:
        results.append(
            {
                "text": _records[doc_id]["r"],
                "score": round(float(score), 6),
                "source_id": _records[doc_id]["id"],
            }
        )
    return results
