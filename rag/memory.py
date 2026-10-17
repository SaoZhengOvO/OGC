"""角色④：记忆抽取。契约（`docs/roles/CONTRACT.md`）冻结为：

    def extract_memories(history: list) -> list[dict]:
        \"\"\"从历史里抽取可长期保留的用户事实：[\"text\": str, \"turn_id\": int, "kind": str}]\"\"\"

## 为什么宁可少抽，不可多抽

`rag/fusion.py` 的 `_extract_memories()` 会**吞掉所有异常**（失败只返回 `[]`），
`render_memories()` 再把结果拼进 system 提示词。所以：

* 抽错一条，等于往提示词里塞一条假事实，模型会当真——伤害比不抽大得多；
* 抽少、抽空，最坏情况只是退化成没有记忆，和现在（④未实现）等价。

因此本模块的策略是**高精度优先**：只认「第一人称 + 持久属性」的短句，
明确排除「今天/刚才/现在」这类瞬时状态。

## 不做什么

* 不做指代消解、不跨轮合并（历史里每轮都是完整的用户发言，合并反而引入错误）；
* 不调模型——离线镜像里没有额外算力预算，规则抽取是唯一稳的选择；
* 不改写原句。原文摘录让模型看不出编造痕迹，改写过的「总结」一旦偏了就全偏。

## 输出上限

最多 5 条。`render_memories()` 不限条数，但历史最多 20 轮，再多就是在挤占
system 提示词的预算（提示词总长度直接影响首 token 延迟）。
"""

from __future__ import annotations

import re

__all__ = ["extract_memories", "memory_stats"]

_MAX_MEMORIES = 5
_MAX_TEXT_CHARS = 80

# --------------------------------------------------------------------------
# 分句
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+|(?<=\.)\s+")

#: 瞬时状态标记。含这些词的句子**不是**持久事实，直接丢。
_TRANSIENT = (
    "今天", "昨天", "前天", "刚才", "刚刚", "现在", "这会儿", "今晚", "明早",
    "早上", "中午", "下午", "晚上", "上周", "这周", "下周", "上个月", "这个月",
    "最近这", "前几", "待会", "等会", "马上",
)

#: 持久属性信号：(正则, kind)。命中任一条才收。
_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 年龄
    (re.compile(r"我(?:今年|才|已经)?\s*\d{1,2}\s*岁"), "profile"),
    (re.compile(r"我(?:今年|才|已经)?\s*[一二三四五六七八九十]{1,3}十[一二三四五六七八九]?\s*岁"), "profile"),
    # 身份 / 职业 / 专业
    (
        re.compile(
            r"我(?:是|在|做|干|当)(?:一个|一名|一位|个)?[^，。！？\s]{0,8}"
            r"(?:学生|老师|教师|教授|医生|护士|程序员|工程师|设计师|会计|律师|销售|"
            r"运营|产品经理|研究生|本科生|大学生|硕士|博士|实习生|自由职业|"
            r"打工人|上班族|创业者|店主|厨师|司机|快递|外卖|保安|公务员)"
        ),
        "profile",
    ),
    (re.compile(r"我(?:的专业|学的|念的|专业)是?[^，。！？\s]{1,12}"), "profile"),
    # 地点
    (
        re.compile(
            r"我(?:住在|家在|老家在|在|搬到了)[^，。！？\s]{0,4}"
            r"(?:北京|上海|广州|深圳|杭州|成都|重庆|武汉|西安|南京|天津|苏州|长沙|"
            r"郑州|青岛|大连|厦门|福州|济南|合肥|昆明|宁波|无锡|佛山|东莞|沈阳|"
            r"哈尔滨|长春|石家庄|太原|南昌|贵阳|南宁|兰州|海口|三亚|拉萨|银川|"
            r"西宁|乌鲁木齐|呼和浩特|香港|澳门|台北|日本|美国|英国|澳洲|"
            r"澳大利亚|加拿大|新加坡|德国|法国|韩国|泰国)"
        ),
        "profile",
    ),
    # 家庭与关系
    (
        re.compile(
            r"我(?:有|家)(?:一个|两个|个)?[^，。！？\s]{0,6}"
            r"(?:妈|爸|父母|老公|老婆|丈夫|妻子|男朋友|女朋友|对象|孩子|儿子|女儿|"
            r"哥哥|姐姐|弟弟|妹妹|爷爷|奶奶|姥姥|姥爷|外公|外婆)"
        ),
        "relationship",
    ),
    (
        re.compile(
            r"(?:我妈|我爸|我父母|我老公|我老婆|我男朋友|我女朋友|我对象|"
            r"我孩子|我儿子|我女儿|我哥|我姐|我弟|我妹)"
        ),
        "relationship",
    ),
    # 长期偏好。**刻意不收「我怕」「我不敢」**：实测在 2354 条 val 上，这两个词
    # 几乎全是当下的担忧（「我怕拒绝会让他们失望」「我怕问多了显得我很闲」），
    # 当成持久偏好写进提示词就是往模型嘴里塞假事实。
    (
        re.compile(
            r"我(?:一直|从小|向来|平时|经常|特别|很|挺|比较|就)?"
            r"(?:喜欢|爱|爱好|迷|热衷|习惯)"
        ),
        "interest",
    ),
    (
        re.compile(
            r"我(?:一直|向来|从小|特别|很|挺|比较|就)?"
            r"(?:讨厌|不喜欢|受不了)"
        ),
        "interest",
    ),
    # 长期状态 / 健康。只认明确的自我诊断，不放宽通配——「我...病」这种模式
    # 会命中「搞得好像我有什么病似的」这种反驳句。
    (
        re.compile(r"我(?:有|患有)(?:社交恐惧|焦虑症|抑郁症|失眠症|强迫症|ADHD|双相)"),
        "health",
    ),
    (
        re.compile(
            r"我(?:一直|常年|从小就)(?:失眠|焦虑|抑郁|社恐|睡不好)"
        ),
        "health",
    ),
    # 明确的自述性格
    (
        re.compile(
            r"我(?:这个人|性格|本来|天生|从小)[^，。！？\s]{0,10}"
            r"(?:内向|外向|敏感|慢热|理性|感性|急躁|温和|孤僻|话少|爱说|社恐|自来熟)"
        ),
        "profile",
    ),
)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("“”\"'，,、；;：:")
    return text


def _iter_user_sentences(history: list) -> list[tuple[int, str]]:
    """把 user 轮拆成句子，保留 (turn_id, sentence)。

    只取 ``role == "user"``：assistant 的回复里出现「我」指的是模型自己，
    抽进来会变成「模型喜欢看电影」这种荒谬记忆。
    """
    collected: list[tuple[int, str]] = []
    for index, turn in enumerate(history or []):
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "user":
            continue
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, int):
            turn_id = index + 1
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        for sentence in _SENTENCE_SPLIT.split(content):
            cleaned = _clean(sentence)
            if 4 <= len(cleaned) <= _MAX_TEXT_CHARS:
                collected.append((turn_id, cleaned))
    return collected


def extract_memories(history: list) -> list[dict]:
    """从对话历史里抽取可长期保留的用户事实。

    实参就是官方样本里的 ``history``（list of ``{turn_id, role, content}``）,
    不需要包一层。返回 ``[{"text": str, "turn_id": int, "kind": str}, ...]``，
    按出现顺序，最多 5 条；抽不到就返回 ``[]``（合法状态）。

    任何异常都不会抛出——fusion 那边直接吞异常，与其静默失败不如自己兜住。
    """
    try:
        return _extract(history)
    except Exception:  # noqa: BLE001 - 记忆抽取不能让整条推理链挂掉
        return []


def _extract(history: list) -> list[dict]:
    if not isinstance(history, list) or not history:
        return []

    results: list[dict] = []
    seen: set[str] = set()
    for turn_id, sentence in _iter_user_sentences(history):
        if any(marker in sentence for marker in _TRANSIENT):
            continue
        kind = None
        for pattern, pattern_kind in _FACT_PATTERNS:
            if pattern.search(sentence):
                kind = pattern_kind
                break
        if kind is None:
            continue
        # 只收第一人称陈述：句子里必须有「我」，且不能是反问/假设。
        if "我" not in sentence:
            continue
        if any(marker in sentence for marker in ("如果", "假如", "要是", "会不会", "是不是")):
            continue
        key = sentence[:40]
        if key in seen:
            continue
        seen.add(key)
        results.append({"text": sentence, "turn_id": turn_id, "kind": kind})
        if len(results) >= _MAX_MEMORIES:
            break
    return results


def memory_stats() -> dict:
    """给 wiring.describe() / 自检脚本看的摘要。"""
    return {
        "patterns": len(_FACT_PATTERNS),
        "max_memories": _MAX_MEMORIES,
        "max_text_chars": _MAX_TEXT_CHARS,
        "has_extract": callable(globals().get("extract_memories")),
    }
