"""检查 `docker save` 出来的镜像 tar 里到底装了什么。

为什么需要这个工具
------------------
本地是 Windows（没有 WSL2，Docker Desktop 起不来），AMD 那台是 k8s pod
（`no docker` / `no podman` / 无 `/var/run/docker.sock`）。**手边没有一台机器能执行
`docker load`**，而自动评测 70 分全押在这个镜像上。

于是提交前能做的「预演」只剩一件：把 tar 拆开，逐项确认里面确实有该有的东西，
而且文件字节与仓库当前 HEAD 一致。这不能替代真机 `docker load`，但能挡住最丢人的
那类事故——镜像里躺的是三天前的提示词，或者模型权重根本没进层。

用法
----
    python tools/inspect_image_tar.py dist/ogc-participant-OGC.tar.gz
    python tools/inspect_image_tar.py dist/xxx.tar.gz --expect-dir . --json-out report.json

检查项
------
1. 外层是合法 tar.gz，能读到 `manifest.json`。
2. 每一层都能当 tar 读（`docker save` 的经典格式里层是未压缩的 `layer.tar`）。
3. 关键路径存在：
   - `root/participant/run_inference.py`
   - `root/participant/start.sh`
   - `root/participant/configs/inference_config.json`
   - `root/participant/model/config.json` 与至少一个 `*.safetensors`
   - `opt/offline_check.sh`
   - `root/AI技术赛道-数字人综合情感对话模型-*/`（专家评审材料的硬性要求）
4. 某几个文件与本地仓库**逐字节**一致（默认比 `deploy/participant/run_inference.py`
   与 `prompts/*.md`）：这一条才是真正会咬人的。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 镜像里必须有这些路径（前导不带 /，tar 里的成员名就是这个形状）。
REQUIRED_PATHS = (
    "root/participant/run_inference.py",
    "root/participant/start.sh",
    "root/participant/wiring.py",
    "root/participant/configs/inference_config.json",
    "root/participant/model/config.json",
    "opt/offline_check.sh",
)

# 要逐字节比对的文件：镜像内路径 ← 仓库内路径。
BYTE_COMPARE = {
    "root/participant/run_inference.py": "deploy/participant/run_inference.py",
    "root/participant/start.sh": "deploy/participant/start.sh",
    "root/participant/wiring.py": "deploy/participant/wiring.py",
    "root/participant/configs/inference_config.json": "deploy/participant/configs/inference_config.json",
    "opt/offline_check.sh": "deploy/offline_check.sh",
    "root/participant/prompts/response.md": "prompts/response.md",
    "root/participant/prompts/emotion_profile.md": "prompts/emotion_profile.md",
}


def _norm(name: str) -> str:
    return name[2:] if name.startswith("./") else name


_BLOB_LAYER = re.compile(r"^blobs/sha256/[0-9a-f]{64}$")


def _is_layer(name: str, layer_names: set[str] | None = None) -> bool:
    """判断外层 tar 里的这个成员是不是镜像层。

    历史格式（docker save 老版本）用 `<id>/layer.tar` 或 `xxx.tar`，
    新版本改用 OCI 布局：层是 `blobs/sha256/<digest>`，**没有 .tar 后缀**。
    只按后缀判断会把 OCI 镜像判成 0 层 —— 这正是 2026-10-17 第一次拿真镜像
    试跑时踩到的：所有检查项全 FAIL，看着像镜像坏了，其实是工具自己瞎了。

    不能只靠 manifest.json 的 Layers 列表：`docker save` 会把 blobs/ 写在
    manifest.json 前面，流式读到层的时候 manifest 还没出现。所以这里按
    「blobs/sha256/<64 位十六进制>」这种形状先认成候选，真正是不是 tar
    由 `_walk_layer` 打开来判（config blob 是 JSON，打不开就跳过）。
    """
    if layer_names and name in layer_names:
        return True
    if _BLOB_LAYER.match(name):
        return True
    return name.endswith(".tar") or "/layer.tar" in name


def _walk_layer(fileobj, *, collect: dict[str, dict], keep_bytes: set[str]) -> int | None:
    """把一层当 tar 读，记录成员；需要比对的成员把内容抓出来。

    不是 tar（比如 OCI 布局里的 config blob 是 JSON）返回 None，调用方据此跳过。
    """
    count = 0
    try:
        inner = tarfile.open(fileobj=fileobj, mode="r|")
    except tarfile.TarError:
        return None
    for member in inner:
        count += 1
        key = _norm(member.name)
        record = collect.setdefault(
            key,
            {"size": member.size, "layer": True, "sha256": None},
        )
        record["size"] = member.size
        if key in keep_bytes and member.isfile():
            handle = inner.extractfile(member)
            if handle is not None:
                payload = handle.read()
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                record["bytes"] = len(payload)
    return count


def _manifest_layer_names(manifest) -> set[str]:
    """从 manifest.json 里抽出所有层的成员名（含 ./ 前缀的归一化形式）。"""
    names: set[str] = set()
    if not isinstance(manifest, list):
        return names
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        for layer in entry.get("Layers") or []:
            if isinstance(layer, str):
                names.add(_norm(layer))
    return names


def inspect(path: Path, *, expect_dir: Path | None) -> dict:
    result: dict = {
        "tar": str(path),
        "tar_bytes": path.stat().st_size,
        "manifest": None,
        "layers": [],
        "members": {},
        "checks": [],
        "ok": True,
    }

    wanted = set(BYTE_COMPARE) 
    collect: dict[str, dict] = {}
    layer_index = 0
    layer_names: set[str] = set()

    with gzip.open(path, "rb") as gz:
        outer = tarfile.open(fileobj=gz, mode="r|")
        for member in outer:
            name = _norm(member.name)
            if name == "manifest.json":
                payload = outer.extractfile(member).read()
                result["manifest"] = json.loads(payload.decode("utf-8"))
                layer_names = _manifest_layer_names(result["manifest"])
            elif name == "repositories":
                continue
            elif _is_layer(name, layer_names):
                handle = outer.extractfile(member)
                if handle is None:
                    continue
                before = len(collect)
                members_in_layer = _walk_layer(handle, collect=collect, keep_bytes=wanted)
                if members_in_layer is None:
                    # 不是 tar（OCI 布局里的 config blob 就是 JSON），跳过
                    continue
                result["layers"].append(
                    {"index": layer_index, "name": name, "size": member.size,
                     "members": members_in_layer,
                     "new_paths": len(collect) - before}
                )
                layer_index += 1
            else:
                # 配置文件之类的小成员，记个名字就行
                collect.setdefault(name, {"size": member.size, "layer": False, "sha256": None})

    result["members"] = collect

    checks = result["checks"]

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})
        if not ok:
            result["ok"] = False

    add("manifest.json 可解析", bool(result["manifest"]), f"{len(result['manifest'] or [])} 条记录")
    add("至少一层", layer_index > 0, f"{layer_index} 层")

    for required in REQUIRED_PATHS:
        present = required in collect
        add(f"存在 {required}", present, f"{collect.get(required, {}).get('size', 0)} 字节")

    # 模型权重：只要有一个 safetensors 就算过，但要把总大小打出来
    shards = {k: v for k, v in collect.items() if k.endswith(".safetensors")}
    total = sum(v.get("size", 0) for v in shards.values())
    add(
        "模型权重在层里",
        bool(shards) and total > 4 * 1024**3,
        f"{len(shards)} 个 shard，共 {total / 1024**3:.2f} GB（Qwen2.5-3B bf16 应约 5.8 GB）",
    )

    # 专家评审材料目录（官方硬性要求：/root 下、以队名结尾）
    review = [k for k in collect if k.startswith("root/AI技术赛道-数字人综合情感对话模型-")]
    add(
        "专家评审材料目录存在",
        bool(review),
        ", ".join(sorted({k.split("/")[1] for k in review})) or "未找到",
    )

    # 逐字节比对
    if expect_dir is not None:
        for image_path, repo_path in BYTE_COMPARE.items():
            local = expect_dir / repo_path
            record = collect.get(image_path)
            if record is None:
                add(f"比对 {image_path}", False, "镜像里没有这个文件")
                continue
            if not local.exists():
                add(f"比对 {image_path}", False, f"本地没有 {repo_path}，无法比对")
                continue
            local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
            image_sha = record.get("sha256")
            if image_sha is None:
                add(f"比对 {image_path}", False, "没读到内容（可能是白名单没覆盖）")
                continue
            add(
                f"比对 {image_path}",
                local_sha == image_sha,
                f"镜像 {image_sha[:12]} / 本地 {local_sha[:12]}",
            )

    return result


def main(argv=None) -> int:
    # allow_abbrev=False 不能省：argparse 默认会把 `--expect` 当成 `--expect-dir` 的前缀缩写。
    # 2026-10-17 第一次用真镜像验证时就栽在这里——我传了 `--expect /root/.../start.sh`，
    # 它被解释成 `--expect-dir /root/.../start.sh`，于是逐字节比对全部报「本地没有这个文件」，
    # 看着像镜像坏了，其实是我自己的参数被静默改写了。这类静默改写比报错危险得多。
    parser = argparse.ArgumentParser(
        description="检查 docker save 出来的镜像 tar 内容", allow_abbrev=False
    )
    parser.add_argument("tar", help=".tar 或 .tar.gz")
    parser.add_argument("--expect-dir", default=None, help="与哪个目录逐字节比对（默认仓库根）")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="路径",
        help="断言镜像里存在这个路径（可重复），例如 --expect /root/participant/start.sh",
    )
    parser.add_argument("--json-out", "--json", dest="json_out", default=None,
                        help="把完整报告写到这个路径")
    args = parser.parse_args(argv)

    path = Path(args.tar)
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    expect_dir = Path(args.expect_dir).resolve() if args.expect_dir else REPO_ROOT

    print(f"检查 {path}（{path.stat().st_size / 1024**3:.2f} GB）……")
    result = inspect(path, expect_dir=expect_dir)

    for want in args.expect:
        key = _norm(want.lstrip("/")) if want.startswith("/") else _norm(want)
        found = key in result["members"]
        result["checks"].append(
            {
                "check": f"存在 {want}",
                "ok": found,
                "detail": (
                    f"{result['members'][key]['size']} 字节" if found else "镜像里没有"
                ),
            }
        )
        result["ok"] = result["ok"] and found

    print(f"\nmanifest：{result['manifest']}")
    print(f"层数：{len(result['layers'])}")
    for layer in result["layers"]:
        print(f"  [{layer['index']}] {layer['name']}  {layer['size'] / 1024**3:.2f} GB  新增 {layer['new_paths']} 个路径")

    print("\n检查项：")
    for item in result["checks"]:
        mark = "OK  " if item["ok"] else "FAIL"
        print(f"  [{mark}] {item['check']} —— {item['detail']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n完整报告：{args.json_out}")

    if not result["ok"]:
        print("\n有检查项没过。这个 tar 不能交。")
        return 1
    print("\n全部通过。注意：这不等价于 docker load 成功，也不等价于平台能起来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
