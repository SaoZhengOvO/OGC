#!/usr/bin/env python3
"""把基座模型权重下到本地，供离线推理与打包镜像使用。

只用标准库，不需要 huggingface_hub，也不依赖 git-lfs。支持断点续传：
下载中途断掉，重跑同一条命令会从 .part 文件接着下。

用法
----
    # 默认：Qwen2.5-3B-Instruct -> GivenData/participant/model
    python tools/fetch_model.py

    # 换个模型 / 换个目录
    python tools/fetch_model.py --repo Qwen/Qwen2.5-1.5B-Instruct
    python tools/fetch_model.py --repo Qwen/Qwen2.5-7B-Instruct --dest D:/models/qwen7b

    # 国内网络慢就换镜像
    python tools/fetch_model.py --mirror https://hf-mirror.com

    # 只看现在有什么，不下载
    python tools/fetch_model.py --check

说明
----
* 只下推理必需的部件（config / tokenizer / 权重 / generation_config），跳过 .gitattributes。
* `--ends-with .safetensors` 之类的过滤可以只拉部分文件。
* 模型目录默认被 .gitignore 之外的规则保护：权重**不进 Git**，靠这条命令在每台机器上重建。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DEST = REPO_ROOT / "GivenData" / "participant" / "model"
DEFAULT_ENDPOINT = "https://huggingface.co"

# 推理必需；其余文件（README / .gitattributes）默认跳过
KEEP_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    ".safetensors",
    "LICENSE",
)
OPTIONAL_PREFIXES = (".gitattributes", "README", ".md")


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def request(url: str, *, headers: dict | None = None, timeout: int = 30):
    merged = {"User-Agent": "curl/8.0"}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged)


def list_files(repo: str, endpoint: str) -> list[dict]:
    url = f"{endpoint.rstrip('/')}/api/models/{repo}"
    with urllib.request.urlopen(request(url, timeout=30), timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    siblings = payload.get("siblings") or []
    files = []
    for item in siblings:
        name = item.get("rfilename") or item.get("path")
        if not name:
            continue
        size = item.get("size")
        files.append({"name": name, "size": size})
    return files


def resolve_url(repo: str, name: str, endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/{repo}/resolve/main/{name}"


def remote_size(url: str) -> int | None:
    """HEAD 取文件的权威大小。

    HF 的权重走 LFS：302 到 CDN，`X-Linked-Size` 才是真大小，而 302 自身的
    `Content-Length` 只有一千多字节。urllib 会自动跟到最后一跳，所以这里拿到的
    一定是最终 200 的 `Content-Length`。拿不到就返回 None（调用方降级处理）。
    """
    try:
        with urllib.request.urlopen(request(url, timeout=30), timeout=30) as resp:
            for key in ("Content-Length", "X-Linked-Size"):
                value = resp.headers.get(key)
                if value and str(value).strip().isdigit() and int(value) > 0:
                    return int(value)
    except Exception:  # noqa: BLE001
        return None
    return None


def wanted(name: str) -> bool:
    if name.startswith(OPTIONAL_PREFIXES):
        return False
    return any(name.endswith(pattern) or name == pattern for pattern in KEEP_PATTERNS)


def download_one(repo: str, name: str, dest: Path, endpoint: str, *, retries: int = 6) -> Path:
    url = resolve_url(repo, name, endpoint)
    target = dest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    expected = remote_size(url)
    if expected and part.exists() and part.stat().st_size > expected:
        part.unlink()  # 比目标还大 = 之前续传写脏了，只能重来

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        already = part.stat().st_size if part.exists() else 0
        headers = {}
        mode = "wb"
        if already and (not expected or already < expected):
            headers["Range"] = f"bytes={already}-"
            mode = "ab"
        try:
            with urllib.request.urlopen(request(url, headers=headers, timeout=60), timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if mode == "ab" and status != 206:
                    # 服务器不认 Range，返回的是完整文件：必须从头写，否则会拼出脏文件
                    sys.stdout.write(f"\n    {name}: 服务器不支持断点续传，从头开始\n")
                    mode, already = "wb", 0
                total = expected
                if not total and resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"]) + (already if mode == "ab" else 0)
                start = time.time()
                base = already
                written = already
                with open(part, mode) as handle:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        elapsed = time.time() - start
                        if elapsed > 1.5:
                            speed = (written - base) / elapsed
                            if total:
                                pct = 100 * written / total
                                sys.stdout.write(
                                    f"\r    {name}: {pct:5.1f}%  {human(written)}/{human(total)}  {human(speed)}/s   "
                                )
                            else:
                                sys.stdout.write(f"\r    {name}: {human(written)}  {human(speed)}/s   ")
                            sys.stdout.flush()
                            start = time.time()
                            base = written
            sys.stdout.write("\n")

            # 关键：流断了不一定抛异常，urllib 的 read(n) 只是返回 b''。
            # 所以必须自己核对字节数，否则会把半截文件当成功。
            final = part.stat().st_size if part.exists() else 0
            if expected and final != expected:
                raise OSError(f"只下到 {human(final)}，应为 {human(expected)}（连接被提前掐断）")
            if not expected and written == 0:
                raise OSError("一个字节都没读到")

            part.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sys.stdout.write(f"\n    第 {attempt} 次尝试失败: {type(exc).__name__}: {str(exc)[:120]}\n")
            time.sleep(2 * attempt)

    raise SystemExit(f"下载 {name} 失败（{retries} 次都没下完）：{last_error}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="下载基座模型权重（标准库实现，支持断点续传）")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"模型仓库，默认 {DEFAULT_REPO}")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="模型落盘目录")
    parser.add_argument("--mirror", default=None, help="下载源，例如 https://hf-mirror.com（慢或连不上时用）")
    parser.add_argument("--check", action="store_true", help="只列出目录里已有什么，不下载")
    parser.add_argument("--no-verify", action="store_true", help="配合 --check：跳过与远端大小的核对（离线时用）")
    parser.add_argument("--all", action="store_true", help="连 README/LICENSE 一起下，不留任何文件")
    args = parser.parse_args(argv)

    dest = Path(args.dest).resolve()
    endpoint = args.mirror or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT

    print(f"模型仓库: {args.repo}")
    print(f"落盘目录: {dest}")
    print(f"下载源  : {endpoint}")

    if args.check:
        if not dest.exists():
            print("\n目录不存在——模型还没下过。")
            return 1
        names = sorted(p for p in dest.iterdir() if p.is_file() and not p.name.endswith(".part"))
        if not names:
            print("\n目录是空的——模型还没下过。")
            return 1
        print("\n已有文件:")
        total = 0
        bad: list[str] = []
        for path in names:
            size = path.stat().st_size
            total += size
            note = ""
            if not args.no_verify:
                want = remote_size(resolve_url(args.repo, path.name, endpoint))
                if want is None:
                    note = "  ← 拿不到远端大小，没核对"
                elif want != size:
                    note = f"  ← 不完整！应为 {human(want)}"
                    bad.append(path.name)
            print(f"  {path.name:<34} {human(size)}{note}")
        print(f"  合计 {human(total)}")
        left = sorted(p.name for p in dest.iterdir() if p.name.endswith(".part"))
        if left:
            print(f"  没下完的分片: {', '.join(left)}")
        has_config = (dest / "config.json").exists()
        has_weights = any(p.suffix == ".safetensors" for p in names)
        print(f"\n可用性: config.json={'有' if has_config else '缺'} 权重={'有' if has_weights else '缺'}")
        if bad:
            print(f"\n以下文件不完整，重跑本命令会接着下: {', '.join(bad)}")
            return 1
        return 0 if (has_config and has_weights) else 1

    print("正在取文件清单…")
    try:
        files = list_files(args.repo, endpoint)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"取不到文件清单（{type(exc).__name__}: {str(exc)[:160]}）。试试 --mirror https://hf-mirror.com")

    selected = [item for item in files if args.all or wanted(item["name"])]
    weights = [item for item in selected if item["name"].endswith(".safetensors")]
    if not weights:
        raise SystemExit(f"{args.repo} 里没有 .safetensors 权重，可能是个 adapter 仓库，请确认模型名。")

    known = sum(item["size"] for item in selected if item.get("size"))
    print(f"待下载 {len(selected)} 个文件，约 {human(known) if known else '未知大小'}\n")

    for item in selected:
        name = item["name"]
        target = dest / name
        want = remote_size(resolve_url(args.repo, name, endpoint))
        if target.exists():
            got = target.stat().st_size
            if want is None:
                print(f"  跳过（已存在，远端大小没拿到，没核对）: {name}")
                continue
            if got == want:
                print(f"  跳过（已完整）: {name}")
                continue
            # 之前下了一半就当成功的那种文件：改名成 .part，让下面的续传接着下
            print(f"  {name} 不完整（{human(got)}/{human(want)}），接着下")
            part = target.with_name(target.name + ".part")
            if not part.exists() or part.stat().st_size < got:
                if part.exists():
                    part.unlink()
                target.replace(part)
        print(f"  下载: {name}")
        download_one(args.repo, name, dest, endpoint)

    print("\n完成。校验一遍：")
    return main(["--check", "--dest", str(dest)])


if __name__ == "__main__":
    raise SystemExit(main())
