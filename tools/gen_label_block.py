"""从官方词表生成可以直接贴进提示词文件的「取值约束」文本块。

角色②（情绪与画像）与角色③（回复生成）都要在提示词里写明合法取值。
手抄 16 类情绪与 33 个画像标签极易出错，而出错的方式是**静默**的：
`src/label_space.py` 的 `normalize_*` 会把词表外的取值丢掉，模型写了错标签
只会表现为「那一维全空」，从指标上根本看不出来是自己拼错了。

因此这份词表只有一处真相来源：`configs/label_space.json`（由
`python src/cli.py build` 从 `src/label_space.py` 导出）。本脚本把它渲染成
纯文本块，供提示词直接引用。

用法::

    python tools/gen_label_block.py                 # 打印全部三个块
    python tools/gen_label_block.py --part emotion  # 只打印情绪块
    python tools/gen_label_block.py --part profile  # 只打印画像块
    python tools/gen_label_block.py --out tools/label_block.txt

注意：输出是**纯文本**，不含 Markdown 标记，可以直接拼进系统提示。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import label_space  # noqa: E402  (需要先修好 sys.path)

LABEL_SPACE_JSON = REPO_ROOT / "configs" / "label_space.json"


def load_label_space() -> dict:
    """优先读 configs/label_space.json（提交进版本库的导出产物），读不到就回退到模块常量。

    两份来源统一成同一个形状：``{"emotions": [...], "profile_values": {dim: [...]}}``。
    """
    if LABEL_SPACE_JSON.exists():
        data = json.loads(LABEL_SPACE_JSON.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("emotions") and data.get("profile_values"):
            return {
                "emotions": list(data["emotions"]),
                "profile_values": {k: list(v) for k, v in data["profile_values"].items()},
            }
    return {
        "emotions": list(label_space.EMOTIONS),
        "profile_values": {dim: list(vals) for dim, vals in label_space.PROFILE_VALUES.items()},
    }


def render_emotion(values: list[str]) -> str:
    """渲染情绪取值块。"""
    lines = [
        "情绪标签 emotion_label 只能取下面 16 个值中的一个，必须恰好选一个，不要自造词：",
        "、".join(values) + "。",
        "写空字符串会被判为该项预测失败，所以即使不确定也必须从上面选一个最接近的。",
    ]
    return "\n".join(lines)


def render_profile(profile: dict[str, list[str]]) -> str:
    """渲染画像取值块。"""
    dim_names = {
        "personality_traits": "人格倾向 personality_traits",
        "interests": "兴趣爱好 interests",
        "style": "表达风格 style",
    }
    lines = [
        "用户画像 user_profile 是一个 object，只含 personality_traits、interests、style 三个数组。",
        "每个数组里的每个取值都必须是下面词表里的英文标签，不能写中文、不能自造、不能重复：",
    ]
    for dim, values in profile.items():
        lines.append(f"{dim_names.get(dim, dim)}：{'、'.join(values)}。")
    lines.append(
        "三个维度互相独立，允许只填其中一维、也允许三维全部留空。"
        "画像按维度的集合精确匹配计分：多写一个错标签会导致整个维度判错，"
        "所以没有充分证据时就留空数组，宁可少写也不要多写。"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--part",
        choices=("emotion", "profile", "all"),
        default="all",
        help="只输出其中一块（默认全部）",
    )
    parser.add_argument("--out", default=None, help="写入文件而不是打印到标准输出")
    args = parser.parse_args()

    data = load_label_space()
    blocks: list[str] = []
    if args.part in ("emotion", "all"):
        blocks.append(render_emotion(list(data["emotions"])))
    if args.part in ("profile", "all"):
        blocks.append(render_profile(data["profile_values"]))
    text = "\n\n".join(blocks) + "\n"

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"已写入 {out_path}（{len(text)} 字符）")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
