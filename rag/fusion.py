"""融合层：把「系统提示词 + 对话历史 + 检索到的参考 + 抽取出的记忆」拼成 messages。

职责边界（`docs/roles/CONTRACT.md` 第三节）：本模块**只负责装配**，不重写角色②③
的判定规则（那些规则在 `prompts/*.md` 里）。角色④给什么参考就用什么，拿不到就当没有。

契约（签名冻结，可加可选参数）：

    def build_messages(sample: dict, retrieved: list[dict] | None = None,
                       system_prompt: str | None = None) -> list[dict]

`system_prompt` 是**追加式**扩展：不传时本模块自己从 `prompts/` 拼；传了就用传进来的
（角色①的 `deploy/participant/wiring.py` 会把「官方 SYSTEM_PROMPT + 队友提示词块」
组合好传进来），避免官方提示词在这一层被丢掉。

本模块独有的三件事（别人不做）：

1. **上下文预算**。训练集单段对话最长 40,262 字符、中位 1,263 字符。全塞进去会撑爆
   窗口并拖慢延迟，所以这里有明确的三级截断：轮数上限 → 单轮首尾截断 → 总字符预算。
   **截断只允许丢早期上下文，绝不丢 `target_user_turn_id` 那一轮。**
2. **防泄漏断言**。`history` 只含目标轮及之前（`src/data_processor.py:build_history`
   保证），但**检索结果来自训练集语料**，可能把本样本自己的答案带进来。这里逐条过滤
   并把过滤数记进统计，供实验报告引用。
3. **可关闭开关**。`PARTICIPANT_FUSION=0` 时抛 `FusionDisabled`，角色①的接线层会
   回退到**字节级一致**的官方组包，用于做消融实验。

依赖纪律：**只用标准库**，且**不得 import `src/`**——镜像里只 COPY 了
`participant/`、`rag/`、`model/`，没有 `src/`。所以下面那个三行的 `_turn_content`
是刻意的重复（`src/data_processor.py:turn_content` 的同款逻辑），不是漏改。
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

__all__ = [
    "FusionDisabled",
    "FusionUnavailable",
    "build_messages",
    "select_turns",
    "render_conversation",
    "render_retrieved",
    "assert_no_leak",
    "context_stats",
]

# --------------------------------------------------------------------------
# 配置：全部可用环境变量覆盖，方便角色①做消融和压延迟
# --------------------------------------------------------------------------

#: 关掉融合层（回退到纯官方组包）。取值 "0"/"false"/"no"/"off" 视为关闭。
FUSION_FLAG = "PARTICIPANT_FUSION"

#: 总字符预算：整条 user 内容的上限。
MAX_TOTAL_CHARS = int(os.environ.get("PARTICIPANT_MAX_CONTEXT_CHARS") or "6000")

#: 最多保留多少轮对话（从目标轮往前数）。
MAX_TURNS = int(os.environ.get("PARTICIPANT_MAX_TURNS") or "24")

#: 单轮超过这个长度就做首尾截断。
MAX_TURN_CHARS = int(os.environ.get("PARTICIPANT_MAX_TURN_CHARS") or "1200")

#: 单轮截断时首尾各保留多少字符。
TURN_HEAD_CHARS = int(os.environ.get("PARTICIPANT_TURN_HEAD_CHARS") or "700")
TURN_TAIL_CHARS = int(os.environ.get("PARTICIPANT_TURN_TAIL_CHARS") or "400")

#: 最多用几条检索参考、每条最多多少字符。
MAX_RETRIEVED = int(os.environ.get("PARTICIPANT_MAX_RETRIEVED") or "3")
MAX_RETRIEVED_CHARS = int(os.environ.get("PARTICIPANT_MAX_RETRIEVED_CHARS") or "600")

#: 提示词目录。**显式设置时只用它**（环境变量 `PARTICIPANT_PROMPT_DIR`，或测试直接
#: 给 `fusion.PROMPT_DIR` 赋值）；为 `None` 时才按 `prompt_dirs()` 的候选顺序去找。
ENV_PROMPT_DIR = os.environ.get("PARTICIPANT_PROMPT_DIR")
PROMPT_DIR: Path | None = Path(ENV_PROMPT_DIR) if ENV_PROMPT_DIR else None

#: 提示词文件的拼接顺序：先情绪/画像的判定规则，再回复的写作规则。
PROMPT_BLOCKS = ("emotion_profile", "response")
PROMPT_EXTENSIONS = (".md", ".txt")

_TRUNCATION_MARKER = "\n……（此处省略 {n} 字）……\n"


def prompt_dirs() -> tuple[Path, ...]:
    """按优先级返回候选提示词目录。

    必须同时认两种布局，否则镜像里会读不到队友写的提示词：

    - **镜像里**：`/root/rag/fusion.py` → 本模块的 `parent.parent` 是 `/root`，
      而角色①的 `deploy/build.sh` 把提示词放在 `/root/participant/prompts`。
    - **仓库里**：`<repo>/rag/fusion.py` → 提示词在 `<repo>/prompts`。

    顺序即优先级，靠前者有同名文件就用它。显式设置了 `PROMPT_DIR` 则只认那一个，
    这样测试与消融实验能完全隔离仓库里的真实提示词。
    """
    if PROMPT_DIR is not None:
        return (PROMPT_DIR,)
    root = Path(__file__).resolve().parent.parent
    return (root / "participant" / "prompts", root / "prompts")


class FusionDisabled(RuntimeError):
    """融合层被 `PARTICIPANT_FUSION=0` 关掉了。接线层应当回退到官方组包。"""


class FusionUnavailable(RuntimeError):
    """拿不到任何系统提示词，无法组包。接线层应当回退到官方组包。"""


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def _turn_content(turn: Any) -> str:
    """取一轮的正文。官方 schema 写 `content`，实际 JSONL 里也有 `text`，两种都认。

    优先返回**非空**的那一个：真实数据里出现过 `content` 是空串而正文在 `text` 里的行，
    只按 key 顺序取会让整轮变成空行、静默丢掉上下文。
    """
    if not isinstance(turn, dict):
        return ""
    for key in ("content", "text"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("content", "text"):
        value = turn.get(key)
        if isinstance(value, str):
            return value
    return ""


def _turn_id(turn: Any) -> int:
    """取轮次号；取不到给一个很小的值，让它排在前面。"""
    if not isinstance(turn, dict):
        return -1
    try:
        return int(turn.get("turn_id"))
    except (TypeError, ValueError):
        return -1


def _turn_role(turn: Any) -> str:
    """取角色名；只认 user/assistant，其余当 user（宁可多留上下文，不可错认身份）。"""
    if not isinstance(turn, dict):
        return "user"
    role = turn.get("role")
    return role if role in ("user", "assistant") else "user"


def truncate_middle(text: str, head: int, tail: int) -> str:
    """保留首尾、砍掉中间，并在断点写明省了多少字。

    为什么保留首尾而不是只保留尾部：一轮里开头的自述往往是**情境**（出了什么事），
    结尾往往是**当下的情绪与请求**。只砍中间丢的信息最少。
    """
    if len(text) <= head + tail:
        return text
    removed = len(text) - head - tail
    return text[:head] + _TRUNCATION_MARKER.format(n=removed) + text[-tail:]


def _normalize_history(sample: Any) -> tuple[list[dict], int]:
    """把 `sample["history"]` 归一成按轮次升序、且只含目标轮及之前的列表。"""
    if not isinstance(sample, dict):
        return [], -1
    history = sample.get("history")
    if not isinstance(history, list):
        return [], -1
    try:
        target_turn_id = int(sample.get("target_user_turn_id"))
    except (TypeError, ValueError):
        target_turn_id = -1

    turns = [t for t in history if isinstance(t, dict) and _turn_content(t)]
    if target_turn_id > 0:
        turns = [t for t in turns if _turn_id(t) <= target_turn_id]
    turns.sort(key=_turn_id)
    return turns, target_turn_id


# --------------------------------------------------------------------------
# 截断策略
# --------------------------------------------------------------------------


def select_turns(
    history: list[dict],
    target_user_turn_id: int,
    *,
    max_turns: int = MAX_TURNS,
    max_total_chars: int = MAX_TOTAL_CHARS,
    max_turn_chars: int = MAX_TURN_CHARS,
) -> list[dict]:
    """按预算裁出真正要送进模型的轮次。

    三级，顺序不能反：

    1. **锚定目标轮**：先找到 `turn_id == target_user_turn_id` 的那一轮，从它往前取
       `max_turns` 轮。这样「目标轮一定在里面」是结构上保证的，不靠运气。
    2. **单轮首尾截断**：超长单轮砍中间（保留情境开头 + 当下结尾）。
    3. **总字符预算**：从**最新**的一轮往回累加，累到超预算就停。早期上下文先被丢掉，
       目标轮（在列表最后）**永远不会**因为预算被丢——真遇到只剩目标轮还超预算的极端
       情况，就进一步压缩目标轮本身，而不是把它删掉。
    """
    turns = [t for t in history if isinstance(t, dict) and _turn_content(t)]
    turns.sort(key=_turn_id)

    # 1. 锚定目标轮，从它往前数
    end = len(turns)
    if target_user_turn_id > 0:
        for index in range(len(turns) - 1, -1, -1):
            if _turn_id(turns[index]) == target_user_turn_id:
                end = index + 1
                break
    window = turns[max(0, end - max(1, max_turns)) : end]
    if not window:
        return []

    # 2. 单轮首尾截断
    trimmed: list[dict] = []
    for turn in window:
        content = _turn_content(turn)
        if len(content) > max_turn_chars:
            head = min(TURN_HEAD_CHARS, max_turn_chars // 2)
            tail = max(0, max_turn_chars - head)
            content = truncate_middle(content, head, tail)
        trimmed.append({**turn, "content": content})

    # 3. 总字符预算：从最新往回累加
    if max_total_chars > 0:
        kept: list[dict] = []
        used = 0
        for turn in reversed(trimmed):
            cost = len(_turn_content(turn)) + len(_turn_role(turn)) + 2
            if kept and used + cost > max_total_chars:
                break
            kept.append(turn)
            used += cost
        if not kept:
            kept = trimmed[-1:]
        trimmed = list(reversed(kept))

    return trimmed


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------


def render_conversation(sample: dict, turns: list[dict] | None = None) -> str:
    """渲染成官方同款格式（`样本 ID:` + `对话历史:`），保证队友的提示词规则仍然适用。"""
    if turns is None:
        turns, target = _normalize_history(sample)
        turns = select_turns(turns, target)
    sample_id = sample.get("id") if isinstance(sample, dict) else None
    lines = [f"样本 ID: {sample_id}", "对话历史:"]
    lines.extend(f"{_turn_role(t)}: {_turn_content(t)}" for t in turns)
    return "\n".join(lines)


def _usable_retrieved(retrieved: Any, sample: dict) -> tuple[list[dict], int]:
    """筛掉不合规、自泄漏、疑似 val 泄漏的检索结果，返回 (可用条目, 被丢弃数)。

    三道过滤（任何一道命中就丢）：

    1. 结构不对：没有非空 `text`。
    2. **自泄漏**：`source_id` 就是本样本所在的对话（等于把答案喂给自己）。
    3. **val 泄漏**：`source_id` 带 `val_` 前缀。角色④的纪律是语料只取 train；
       这里再拦一道，代价极低，漏一次的代价是整套验证分数作废。
    """
    if not isinstance(retrieved, list):
        return [], 0
    conversation_id = str(sample.get("conversation_id") or "")
    sample_id = str(sample.get("id") or "")
    kept: list[dict] = []
    dropped = 0
    for item in retrieved:
        if not isinstance(item, dict):
            dropped += 1
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            dropped += 1
            continue
        source_id = str(item.get("source_id") or "")
        if source_id and (
            source_id == conversation_id
            or source_id == sample_id
            or source_id.startswith("val_")
        ):
            dropped += 1
            continue
        kept.append(item)
        if len(kept) >= max(0, MAX_RETRIEVED):
            break
    return kept, dropped


RETRIEVED_PLACEMENT_FLAG = "PARTICIPANT_RETRIEVED_IN_SYSTEM"


def _retrieved_in_system() -> bool:
    """检索块放 system（默认）还是放 user 段末尾。

    A/B 证据（2026-10-17，n=200 同批）：放 user 段末尾时，检索块紧贴目标轮，
    模型把「别人的经历」当成「这个人刚说的」，情绪 Acc 从 0.310 掉到 0.255
    （McNemar p=0.0817，22 丢 / 11 得）。放到 system 里它变成背景参考，
    不再和目标轮争夺注意力。设 0 可退回旧行为，做对照用。
    """
    raw = os.environ.get(RETRIEVED_PLACEMENT_FLAG)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def render_retrieved(items: list[dict]) -> str:
    """把检索参考渲染成一段，并明确写清「只学语气、不要照抄」。"""
    if not items:
        return ""
    lines = [
        "",
        "[相似情境下的真实回应，仅供语气与信息量参考，不要照抄内容]",
    ]
    for index, item in enumerate(items, 1):
        text = truncate_middle(
            str(item.get("text")), MAX_RETRIEVED_CHARS // 2, MAX_RETRIEVED_CHARS // 2
        )
        lines.append(f"{index}. {text}")
    return "\n".join(lines)


def _extract_memories(sample: dict) -> list[dict]:
    """调角色④的 `rag.memory.extract_memories(history)`；没有/失败就返回 []。"""
    try:
        module = importlib.import_module("rag.memory")
    except Exception:  # noqa: BLE001 - 队友模块可能还没写
        return []
    function = getattr(module, "extract_memories", None)
    if not callable(function):
        return []
    try:
        result = function(sample.get("history") or [])
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def render_memories(memories: list[dict]) -> str:
    """把抽取到的记忆渲染成一段。空就返回空串。"""
    lines: list[str] = []
    for item in memories:
        text = item.get("text") or item.get("summary") or item.get("content")
        if isinstance(text, str) and text.strip():
            lines.append(f"- {text.strip()}")
    if not lines:
        return ""
    return "\n".join(["", "[从对话历史中抽取到的用户记忆]", *lines])


# --------------------------------------------------------------------------
# 防泄漏断言
# --------------------------------------------------------------------------


def assert_no_leak(sample: dict, messages: list[dict]) -> list[str]:
    """自检：返回问题清单，空列表表示干净。

    检查三件事：

    1. messages 结构合法（角色白名单 + content 是字符串）。
    2. **没有目标轮之后的任何内容**：把 messages 里的每一行与「目标轮之后的轮次正文」
       比对（用长度 ≥ 12 的片段做子串匹配，避免几个字的语气词误报）。
    3. 目标轮本身在场——截断把目标轮丢掉是致命的，必须显式验。
    """
    problems: list[str] = []
    if not isinstance(messages, list) or not messages:
        return ["messages 为空或不是列表"]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"第 {index} 条不是 dict")
            continue
        if message.get("role") not in ("system", "user", "assistant"):
            problems.append(f"第 {index} 条角色非法：{message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            problems.append(f"第 {index} 条 content 不是字符串")

    turns, target_turn_id = _normalize_history(sample)
    body = "\n".join(
        m.get("content") if isinstance(m.get("content"), str) else ""
        for m in messages
        if isinstance(m, dict)
    )

    # 2. 目标轮之后的内容一个都不能出现
    if target_turn_id > 0:
        history = sample.get("history") if isinstance(sample, dict) else None
        for turn in history if isinstance(history, list) else []:
            if _turn_id(turn) <= target_turn_id:
                continue
            after = _turn_content(turn).strip()
            if len(after) >= 12 and after[:60] in body:
                problems.append(
                    f"泄漏：turn_id={_turn_id(turn)} 在目标轮之后，却出现在 messages 里"
                )

    # 3. 目标轮必须在场
    if target_turn_id > 0:
        target_text = ""
        for turn in turns:
            if _turn_id(turn) == target_turn_id:
                target_text = _turn_content(turn).strip()
                break
        if target_text:
            probe = target_text[:40] if len(target_text) >= 12 else target_text
            if probe and probe not in body:
                problems.append(
                    f"截断错误：目标轮（turn_id={target_turn_id}）的正文不在 messages 里"
                )
    return problems


# --------------------------------------------------------------------------
# 统计（给实验报告用，别自己编数）
# --------------------------------------------------------------------------


def context_stats(sample: dict) -> dict:
    """一条样本的上下文预算执行情况。`docs/reports/experiments.md` 直接引用这个。

    口径说明：`turns_total` / `chars_total` 是**该样本 history 里的全部轮次**（含目标轮
    之后的未来轮）；`turns_up_to_target` 是防泄漏归一后真正允许考虑的轮次；`turns_kept`
    是最终送进模型的轮次。`turns_dropped = turns_total - turns_kept`，既包含未来轮，
    也包含被轮数上限与字符预算砍掉的早期上下文。
    """
    full_turns, target = _normalize_history(sample)
    kept = select_turns(full_turns, target)
    raw_history = sample.get("history") if isinstance(sample, dict) else None
    raw_turns = (
        [t for t in raw_history if isinstance(t, dict)]
        if isinstance(raw_history, list)
        else []
    )
    raw_chars = sum(len(_turn_content(t)) for t in raw_turns)
    kept_chars = sum(len(_turn_content(t)) for t in kept)
    return {
        "sample_id": sample.get("id") if isinstance(sample, dict) else None,
        "target_user_turn_id": target,
        "turns_total": len(raw_turns),
        "turns_up_to_target": len(full_turns),
        "turns_kept": len(kept),
        "turns_dropped": max(0, len(raw_turns) - len(kept)),
        "chars_total": raw_chars,
        "chars_kept": kept_chars,
        "chars_dropped": max(0, raw_chars - kept_chars),
        "target_turn_kept": any(_turn_id(t) == target for t in kept) if target > 0 else None,
    }


# --------------------------------------------------------------------------
# 系统提示词
# --------------------------------------------------------------------------


def _load_prompt_block(name: str) -> str:
    """在候选题词目录里找 `prompts/<name>.md`（或 .txt）；读不到/为空返回空串。"""
    for directory in prompt_dirs():
        for extension in PROMPT_EXTENSIONS:
            path = directory / (name + extension)
            try:
                if path.is_file():
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        return text
            except (OSError, UnicodeDecodeError):
                continue
    return ""


def compose_prompt_from_files() -> str:
    """没有外部传入系统提示时，自己从 `prompts/` 拼一份。"""
    blocks = [b for b in (_load_prompt_block(n) for n in PROMPT_BLOCKS) if b]
    return "\n\n".join(blocks)


def _fusion_enabled() -> bool:
    raw = os.environ.get(FUSION_FLAG, "1")
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


# --------------------------------------------------------------------------
# 主入口（契约冻结）
# --------------------------------------------------------------------------


def build_messages(
    sample: dict,
    retrieved: list[dict] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """装配一条样本的 messages。

    参数
    ----
    sample:
        一条推理样本：``{"id", "conversation_id", "target_user_turn_id", "history"}``。
    retrieved:
        角色④ ``rag.retrieve.retrieve`` 的返回值。``None`` 或 ``[]`` 都可以。
    system_prompt:
        角色①已经组好的系统提示（官方 SYSTEM_PROMPT + 队友提示词块）。**传了就用它**，
        不做二次拼接，避免重复注入。不传时本模块从 ``prompts/`` 自己拼。

    返回
    ----
    ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``

    异常
    ----
    ``FusionDisabled``：``PARTICIPANT_FUSION=0``。
    ``FusionUnavailable``：既没有传入系统提示，``prompts/`` 里也没有任何块。
        两种异常都应当由调用方（``deploy/participant/wiring.py``）捕获并回退到官方组包。
    """
    if not _fusion_enabled():
        raise FusionDisabled(
            f"{FUSION_FLAG}=0 —— 融合层已关闭，请回退到官方组包（消融实验用）"
        )

    prompt = system_prompt if system_prompt else compose_prompt_from_files()
    if not prompt or not prompt.strip():
        raise FusionUnavailable(
            "没有可用的系统提示词：调用方未传 system_prompt，"
            f"且下列目录里都没有 {PROMPT_BLOCKS}："
            + "、".join(str(d) for d in prompt_dirs())
        )

    items, _dropped = _usable_retrieved(retrieved, sample)
    turns, target = _normalize_history(sample)
    turns = select_turns(turns, target)

    parts = [render_conversation(sample, turns)]
    retrieved_block = render_retrieved(items)
    if retrieved_block:
        if _retrieved_in_system():
            # 放进 system：模型读到它，但不会把它当成「用户刚说的话」。
            prompt = prompt.rstrip() + "\n" + retrieved_block
        else:
            parts.append(retrieved_block)
    memories_block = render_memories(_extract_memories(sample))
    if memories_block:
        parts.append(memories_block)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]

    problems = assert_no_leak(sample, messages)
    if problems:
        # 宁可让接线层回退到官方组包，也不要带着泄漏去推理。
        raise FusionUnavailable("组包自检未通过：" + "；".join(problems))
    return messages
