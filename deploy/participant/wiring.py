"""参赛副本的接线层：提示词加载 + RAG 模块挂载 + 官方回退。

这一层是**唯一**允许把队友产物接进推理脚本的地方。改动前请先看
`docs/roles/CONTRACT.md`。

三条设计原则（改这一层时不要破坏）：

1. **永不抛异常**。任何一步失败（导入失败、函数不存在、返回类型不对、
   文件缺失）都退回官方内置提示词，保证镜像一定能跑出合规输出。
   镜像跑不起来是全队零分，不能拿它赌队友的进度。
2. **不改官方契约**。返回的 messages 必须是 `[{"role": "system", ...},
   {"role": "user", ...}]`，且**不得**包含目标 assistant 回复或目标轮
   之后的任何内容。
3. **只用标准库**。这一层会被打进离线镜像，不能有第三方依赖。

装配顺序（与 CONTRACT.md 第三节一致）：

    messages = rag.fusion.build_messages(sample, retrieve(sample))
    失败 → 官方 SYSTEM_PROMPT 版 messages

检索失败不算整体失败：拿不到记忆就当没有记忆，仍然走 RAG 组包。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

PARTICIPANT_DIR = Path(__file__).resolve().parent

#: 镜像里 /root/participant 的上一层（= /root），本地布局里 = 仓库根目录。
CONTENT_ROOT = Path(
    os.environ.get("PARTICIPANT_CONTENT_ROOT") or str(PARTICIPANT_DIR.parent)
)

#: 提示词目录，可用环境变量覆盖。
PROMPT_DIR = Path(
    os.environ.get("PARTICIPANT_PROMPT_DIR") or str(PARTICIPANT_DIR / "prompts")
)

#: 会被拼进系统提示的提示词文件（不含扩展名），顺序即拼接顺序。
PROMPT_BLOCKS = ("emotion_profile", "response")

PROMPT_EXTENSIONS = (".md", ".txt")

_VALID_ROLES = {"system", "user", "assistant"}


def _ensure_import_paths() -> list[str]:
    """把 rag 包可能所在的目录加到 sys.path，返回实际加进去的目录。"""
    candidates = [
        CONTENT_ROOT,                   # 镜像: /root -> /root/rag；本地: <repo>
        PARTICIPANT_DIR.parent.parent,  # 本地: deploy/participant 的上两层 = <repo>
        Path("/root"),                  # 镜像里的兜底位置
    ]
    added: list[str] = []
    for candidate in candidates:
        try:
            if candidate.is_dir():
                text = str(candidate)
                if text not in sys.path:
                    sys.path.insert(0, text)
                    added.append(text)
        except OSError:
            continue
    return added


_PATH_READY = False


def _import(name: str) -> Any:
    """导入模块；任何失败都返回 None，绝不抛异常。"""
    global _PATH_READY
    if not _PATH_READY:
        _ensure_import_paths()
        _PATH_READY = True
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001 - 故意的：镜像必须能跑
        return None


def load_prompt_block(name: str) -> str:
    """读 `prompts/<name>.md`（或 .txt）。读不到/为空返回空串。"""
    for extension in PROMPT_EXTENSIONS:
        path = PROMPT_DIR / (name + extension)
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except (OSError, UnicodeDecodeError):
            continue
    return ""


def compose_system_prompt(official_prompt: str) -> str:
    """官方提示词打底，后面追加角色②③写的规则块（有才加）。"""
    blocks = [official_prompt]
    for name in PROMPT_BLOCKS:
        block = load_prompt_block(name)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def retrieve_memories(sample: dict, k: int = 3) -> list[dict]:
    """调角色④的 `rag.retrieve.retrieve`；拿不到就返回 []。"""
    module = _import("rag.retrieve")
    if module is None:
        return []
    function = getattr(module, "retrieve", None)
    if not callable(function):
        return []
    try:
        result = function(sample, k)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def _messages_are_usable(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in _VALID_ROLES:
            return False
        if not isinstance(message.get("content"), str):
            return False
    return True


def default_messages(
    sample: dict, system_prompt: str, conversation_text: Callable[[dict], str]
) -> list[dict]:
    """官方基线版组包。任何情况下都能用。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation_text(sample)},
    ]


def build_messages(
    sample: dict,
    system_prompt: str,
    conversation_text: Callable[[dict], str],
) -> list[dict]:
    """按契约装配一条样本的 messages。**永不抛异常。**"""
    fallback = default_messages(sample, system_prompt, conversation_text)
    module = _import("rag.fusion")
    if module is None:
        return fallback
    function = getattr(module, "build_messages", None)
    if not callable(function):
        return fallback
    retrieved = retrieve_memories(sample)
    try:
        produced = function(sample, retrieved)
    except Exception:  # noqa: BLE001
        return fallback
    if not _messages_are_usable(produced):
        return fallback
    return produced


def describe() -> dict:
    """给运行日志和平台留档用：这一趟到底走了哪条路。"""
    _ensure_import_paths()
    retrieve_module = _import("rag.retrieve")
    fusion_module = _import("rag.fusion")
    return {
        "content_root": str(CONTENT_ROOT),
        "prompt_dir": str(PROMPT_DIR),
        "prompt_blocks_present": [
            name for name in PROMPT_BLOCKS if load_prompt_block(name)
        ],
        "rag_retrieve": "ok" if retrieve_module is not None else "missing",
        "rag_fusion": "ok" if fusion_module is not None else "missing",
    }


def log_wiring() -> None:
    """启动时打印一行，落进运行日志里当证据。"""
    info = describe()
    print(
        "[wiring] content_root={content_root} prompt_dir={prompt_dir} "
        "prompts={prompts} rag.retrieve={retrieve} rag.fusion={fusion}".format(
            content_root=info["content_root"],
            prompt_dir=info["prompt_dir"],
            prompts=",".join(info["prompt_blocks_present"]) or "(none)",
            retrieve=info["rag_retrieve"],
            fusion=info["rag_fusion"],
        ),
        flush=True,
    )
