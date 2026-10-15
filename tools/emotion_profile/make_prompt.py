#!/usr/bin/env python3
"""生成 `prompts/emotion_profile.md`：把判定规则与**脚本文生成**的词表块拼在一起。

为什么要有这个脚本：`CONTRACT.md` 第四节要求提示词里的 16 类情绪与 33 个画像标签
必须用 `tools/gen_label_block.py` 从 `configs/label_space.json`（唯一真相来源）生成，
不许手抄。手抄出错的方式是静默的——词表外的取值会被 `src/label_space.py` 的
`normalize_*` 丢掉，指标上只表现为「那一维全空」，看不出是自己拼错了。

因此本脚本 = 手写的规则模板（`prompt_rules.txt`，里面的 `{{LABEL_BLOCK}}`
是占位符） + `gen_label_block` 现渲染的词表块。

用法::

    python tools/emotion_profile/make_prompt.py            # 覆盖写 prompts/emotion_profile.md
    python tools/emotion_profile/make_prompt.py --check    # 只校验产物是否是最新的（CI 用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
TEMPLATE = TOOL_DIR / "prompt_rules.txt"
TARGET = REPO_ROOT / "prompts" / "emotion_profile.md"
PLACEHOLDER = "{{LABEL_BLOCK}}"

sys.path.insert(0, str(REPO_ROOT / "tools"))
import gen_label_block  # noqa: E402  (需要先修好 sys.path)


def render() -> str:
    """渲染最终提示词全文。"""
    if not TEMPLATE.exists():
        raise SystemExit(f"找不到模板 {TEMPLATE}")
    rules = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in rules:
        raise SystemExit(f"模板里没有占位符 {PLACEHOLDER}，拒绝生成")

    space = gen_label_block.load_label_space()
    label_block = "\n\n".join(
        [
            gen_label_block.render_emotion(list(space["emotions"])),
            gen_label_block.render_profile(space["profile_values"]),
        ]
    )
    text = rules.replace(PLACEHOLDER, label_block)
    # 提示词是纯文本契约产物：收敛多余空行，保证末尾只有一个换行。
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 prompts/emotion_profile.md")
    parser.add_argument("--check", action="store_true", help="只校验产物是否与模板一致，不写文件")
    args = parser.parse_args(argv)

    text = render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} 与模板不一致，请重跑本脚本", file=sys.stderr)
            return 1
        print(f"{TARGET} 是最新的（{len(text)} 字符）")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    emo = len(gen_label_block.load_label_space()["emotions"])
    print(f"已写入 {TARGET}（{len(text)} 字符，含 {emo} 类情绪与三维画像词表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
