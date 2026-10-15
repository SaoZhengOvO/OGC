#!/usr/bin/env python3
"""一键开工：建 venv -> 装依赖 -> 下模型 -> 展开数据 -> 冒烟测试。

给队友用的。跑完这条命令，你就能在验证集上打出第一个分数。
细节说明见 START-HERE.md；这个脚本只是把那五步串起来，每一步都会打印在做什么。

用法
----
    python tools/bootstrap.py                 # 全套（Windows / Linux，N 卡，默认 cu130）
    python tools/bootstrap.py --cpu           # 没有 N 卡，装 CPU 版 torch
    python tools/bootstrap.py --skip-model    # 模型已经拷过来了（从队友那拷目录更快）
    python tools/bootstrap.py --limit 50      # 冒烟测试多跑几条
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV = REPO_ROOT / ".venv"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def clean_env() -> dict:
    """把临时目录显式指到系统 TEMP。

    pip 在有些环境下会把解包临时目录（pip-unpack-*/、tmp*）建在**当前工作目录**，
    也就是仓库根目录。2026-10-15 踩过一次：`git add -A` 把 1.9 GB 的 torch wheel
    扫进了提交。这里是指到系统临时目录，`.gitignore` 里还有一道规则兜底。
    """
    env = os.environ.copy()
    tmp = tempfile.gettempdir()
    for key in ("TMPDIR", "TEMP", "TMP"):
        env[key] = tmp
    return env


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str], *, title: str, fatal: bool = True) -> int:
    print(f"\n=== {title} ===")
    print("$ " + " ".join(str(part) for part in cmd))
    code = subprocess.run([str(part) for part in cmd], cwd=str(REPO_ROOT), env=clean_env()).returncode
    if code != 0:
        message = f"『{title}』失败（退出码 {code}）"
        if fatal:
            raise SystemExit(message + "。修好之后重跑同一条命令会跳过已完成的步骤。")
        print(f"（可跳过）{message}")
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="一键开工：环境 + 模型 + 数据 + 冒烟测试")
    parser.add_argument("--cpu", action="store_true", help="装 CPU 版 torch（没有 N 卡时用）")
    parser.add_argument("--torch-index", default=None, help=f"torch 下载源，默认 {DEFAULT_TORCH_INDEX}")
    parser.add_argument("--skip-model", action="store_true", help="跳过模型下载（已经从队友那拷好了）")
    parser.add_argument("--model-repo", default="Qwen/Qwen2.5-3B-Instruct", help="模型仓库")
    parser.add_argument("--model-dest", default="GivenData/participant/model", help="模型落盘目录")
    parser.add_argument("--mirror", default=None, help="模型下载源，例如 https://hf-mirror.com")
    parser.add_argument("--skip-smoke", action="store_true", help="跳过最后的冒烟测试")
    parser.add_argument("--limit", type=int, default=20, help="冒烟测试跑几条（默认 20）")
    args = parser.parse_args(argv)

    print("=" * 60)
    print(" OGC 一键开工")
    print(f" 解释器: {sys.executable}")
    print(f" 仓库根: {REPO_ROOT}")
    print("=" * 60)

    if sys.version_info < (3, 11):
        raise SystemExit(f"需要 Python 3.11 以上，当前 {sys.version.split()[0]}。")

    # 1) venv
    if venv_python().exists():
        print(f"\n=== 1/5 虚拟环境已存在，跳过：{VENV} ===")
    else:
        run([sys.executable, "-m", "venv", str(VENV)], title="1/5 建虚拟环境")

    py = venv_python()
    run([py, "-m", "pip", "install", "-U", "pip"], title="2/5 升级 pip", fatal=False)

    index = None if args.cpu else (args.torch_index or DEFAULT_TORCH_INDEX)
    torch_cmd = [py, "-m", "pip", "install", "torch"]
    if index:
        torch_cmd += ["--index-url", index]
    run(torch_cmd, title=f"3/5 装 torch{'' if args.cpu else f'（{index}）'}")

    run([py, "-m", "pip", "install", "transformers", "accelerate"], title="3/5 装 transformers / accelerate")

    check = (
        "import torch, transformers;"
        "print('torch', torch.__version__, '| cuda', torch.version.cuda, '| 可用', torch.cuda.is_available());"
        "print('设备', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU');"
        "print('transformers', transformers.__version__)"
    )
    run([py, "-c", check], title="环境自检", fatal=False)

    # 4) 模型
    dest = Path(args.model_dest)
    if args.skip_model:
        print("\n=== 4/5 跳过模型下载 ===")
    elif (dest / "config.json").exists() and any(dest.glob("*.safetensors")):
        print(f"\n=== 4/5 模型已就位，跳过：{dest} ===")
    else:
        fetch_cmd = [py, "tools/fetch_model.py", "--repo", args.model_repo, "--dest", args.model_dest]
        if args.mirror:
            fetch_cmd += ["--mirror", args.mirror]
        run(fetch_cmd, title="4/5 下基座模型（6.2 GB，断了重跑接着下）")

    # 5) 数据 + 冒烟
    run([py, "src/cli.py", "build"], title="5/5 展开官方数据")

    if args.skip_smoke:
        print("\n跳过冒烟测试。自己跑：python tools/eval_loop.py --tag smoke --limit 20")
    elif not (dest / "config.json").exists():
        print(f"\n模型不在 {dest}，跳过冒烟测试。")
    else:
        smoke_cmd = [
            py,
            "tools/eval_loop.py",
            "--tag",
            "smoke",
            "--limit",
            str(args.limit),
            "--model-path",
            str(dest),
        ]
        run(smoke_cmd, title=f"5/5 冒烟测试（{args.limit} 条）", fatal=False)

    print("\n" + "=" * 60)
    print(" 开工完成。接下来：")
    print("   1. 打开 ROLES.md 找你的那条线")
    print("   2. 把 docs/roles/0X-*.md 整段贴给你的编码 AI（工作目录=仓库根）")
    print("   3. 正式迭代用全量：python tools/eval_loop.py --tag 你的实验名")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
