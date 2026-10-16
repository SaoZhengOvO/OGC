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

    # 用候选模板渲染到别处（做 A/B 时不让候选版污染已定稿的产物）
    python tools/emotion_profile/make_prompt.py \
        --template tools/emotion_profile/prompt_rules.v4.txt \
        --out /tmp/v4.md
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


def strip_comments(text: str) -> str:
    """剥掉模板里以 `#` 开头的注释行。

    为什么要有这一步：`prompts/emotion_profile.md` 是**原样拼进系统提示**的
    （`deploy/participant/wiring.py:compose_system_prompt`），所以"为什么这么写"
    这类注解不能留在这个文件里，否则会被模型当成规则读。
    注解写在模板 `prompt_rules.txt` 里，由本函数剥离；这样源头留得住理由，产物保持干净。
    """
    kept: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def render(template: Path | None = None) -> str:
    """渲染最终提示词全文。`template` 可指向候选模板（默认用 prompt_rules.txt）。"""
    source = template or TEMPLATE
    if not source.exists():
        raise SystemExit(f"找不到模板 {source}")
    rules = strip_comments(source.read_text(encoding="utf-8"))
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
    rendered = "\n".join(collapsed).strip() + "\n"

    # 护栏：产物里不许再出现注释标记，否则说明剥离逻辑漏了。
    for number, line in enumerate(rendered.splitlines(), 1):
        if line.lstrip().startswith("#"):
            raise SystemExit(f"渲染结果第 {number} 行看起来是没剥干净的注释：{line[:40]!r}")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 prompts/emotion_profile.md")
    parser.add_argument("--check", action="store_true", help="只校验产物是否与模板一致，不写文件")
    parser.add_argument("--template", default=None, help="改用候选模板（默认 prompt_rules.txt）")
    parser.add_argument("--out", default=None, help="改写到别的路径（默认 prompts/emotion_profile.md）")
    args = parser.parse_args(argv)

    if args.check and (args.template or args.out):
        raise SystemExit("--check 只能搭配默认模板与默认产物路径使用")

    template = Path(args.template) if args.template else None
    if template is not None and not template.is_absolute():
        template = REPO_ROOT / template
    target = Path(args.out) if args.out else TARGET
    if not target.is_absolute():
        target = REPO_ROOT / target

    text = render(template)
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} 与模板不一致，请重跑本脚本", file=sys.stderr)
            return 1
        print(f"{TARGET} 是最新的（{len(text)} 字符）")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    emo = len(gen_label_block.load_label_space()["emotions"])
    print(f"已写入 {target}（{len(text)} 字符，含 {emo} 类情绪与三维画像词表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
