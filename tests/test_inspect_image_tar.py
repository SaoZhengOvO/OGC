"""`tools/inspect_image_tar.py` 的测试。

用合成的小 tar 验证「拆层 + 找路径 + 逐字节比对」这套逻辑本身对，不等真镜像。
真镜像 7.65 GB，跑一次几分钟，不适合当单元测试。
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.inspect_image_tar import BYTE_COMPARE, inspect  # noqa: E402


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _image_tar(path: Path, *, layer_files: dict[str, bytes], extra_layer: dict[str, bytes] | None = None) -> Path:
    manifest = [{"Config": "config.json", "RepoTags": ["ogc-participant:latest"], "Layers": ["aaa/layer.tar"]}]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as outer:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            outer.addfile(info, io.BytesIO(payload))

        add("manifest.json", json.dumps(manifest).encode())
        add("config.json", b"{}")
        add("aaa/layer.tar", _tar_bytes(layer_files))
        if extra_layer:
            add("bbb/layer.tar", _tar_bytes(extra_layer))
    path.write_bytes(gzip.compress(buf.getvalue(), 1))
    return path


def _repo_bytes(rel: str) -> bytes:
    return (REPO_ROOT / rel).read_bytes()


def _good_layer() -> dict[str, bytes]:
    layer = {
        "root/participant/run_inference.py": _repo_bytes("deploy/participant/run_inference.py"),
        "root/participant/start.sh": _repo_bytes("deploy/participant/start.sh"),
        "root/participant/wiring.py": _repo_bytes("deploy/participant/wiring.py"),
        "root/participant/configs/inference_config.json": _repo_bytes(
            "deploy/participant/configs/inference_config.json"
        ),
        "opt/offline_check.sh": _repo_bytes("deploy/offline_check.sh"),
        "root/participant/prompts/response.md": _repo_bytes("prompts/response.md"),
        "root/participant/prompts/emotion_profile.md": _repo_bytes("prompts/emotion_profile.md"),
        "root/participant/model/config.json": b'{"model_type":"qwen2"}',
        "root/AI技术赛道-数字人综合情感对话模型-OGC/技术方案.md": b"# plan",
    }
    return layer


def _checks(result: dict) -> dict[str, bool]:
    return {item["check"]: item["ok"] for item in result["checks"]}


def test_good_layer_passes_every_non_weight_check(tmp_path: Path) -> None:
    path = _image_tar(tmp_path / "img.tar.gz", layer_files=_good_layer())

    result = inspect(path, expect_dir=REPO_ROOT)
    checks = _checks(result)

    assert checks["manifest.json 可解析"]
    assert checks["至少一层"]
    assert checks["专家评审材料目录存在"]
    for image_path in BYTE_COMPARE:
        assert checks[f"比对 {image_path}"], image_path

    # 合成 tar 里没有真权重，这一条必须如实报失败，不能假装通过
    assert not checks["模型权重在层里"]
    assert result["ok"] is False


def test_prompt_drift_is_caught(tmp_path: Path) -> None:
    layer = _good_layer()
    layer["root/participant/prompts/response.md"] = b"STALE PROMPT\n"
    path = _image_tar(tmp_path / "img.tar.gz", layer_files=layer)

    result = inspect(path, expect_dir=REPO_ROOT)
    checks = _checks(result)

    assert not checks["比对 root/participant/prompts/response.md"], "提示词漂移必须被抓出来"
    assert checks["比对 root/participant/run_inference.py"], "别的文件不该跟着变红"


def test_missing_required_path_is_caught(tmp_path: Path) -> None:
    layer = _good_layer()
    del layer["root/participant/start.sh"]
    path = _image_tar(tmp_path / "img.tar.gz", layer_files=layer)

    result = inspect(path, expect_dir=REPO_ROOT)
    checks = _checks(result)

    assert not checks["存在 root/participant/start.sh"]
    assert checks["存在 root/participant/run_inference.py"]


def test_files_across_multiple_layers_are_merged(tmp_path: Path) -> None:
    layer = _good_layer()
    tail = {"root/participant/model/model-00001-of-00002.safetensors": b"x" * 1024}
    path = _image_tar(tmp_path / "img.tar.gz", layer_files=layer, extra_layer=tail)

    result = inspect(path, expect_dir=REPO_ROOT)

    assert "root/participant/model/model-00001-of-00002.safetensors" in result["members"]
    assert len(result["layers"]) == 2


def test_byte_compare_records_image_sha(tmp_path: Path) -> None:
    path = _image_tar(tmp_path / "img.tar.gz", layer_files=_good_layer())

    result = inspect(path, expect_dir=REPO_ROOT)

    expected = hashlib.sha256(_repo_bytes("prompts/response.md")).hexdigest()
    assert result["members"]["root/participant/prompts/response.md"]["sha256"] == expected
