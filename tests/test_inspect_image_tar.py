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

from tools.inspect_image_tar import BYTE_COMPARE, inspect, main  # noqa: E402


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


def _oci_image(path: Path, *, layer_files: dict[str, bytes], layer_digest: str) -> Path:
    """按新版 docker save 的 OCI 布局造镜像。

    真镜像的两个特征都在这里复现：层名是 `blobs/sha256/<64 位十六进制>`（**没有 .tar
    后缀**），而且 blobs 写在 manifest.json 前面 —— 流式读的时候，读到层时 manifest
    还没出现。2026-10-17 就是因为这两点，检查一个完好镜像时报了 0 层、所有项全 FAIL。
    """
    layer = _tar_bytes(layer_files)
    config_digest = "b" * 64
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as outer:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            outer.addfile(info, io.BytesIO(payload))

        add(f"blobs/sha256/{layer_digest}", layer)
        add(f"blobs/sha256/{config_digest}", b'{"architecture":"amd64","os":"linux"}')
        add(
            "manifest.json",
            json.dumps(
                [
                    {
                        "Config": f"blobs/sha256/{config_digest}",
                        "RepoTags": ["ogc-participant:latest"],
                        "Layers": [f"blobs/sha256/{layer_digest}"],
                    }
                ]
            ).encode(),
        )
    path.write_bytes(gzip.compress(buf.getvalue()))
    return path


def test_oci_layout_layers_are_detected(tmp_path: Path) -> None:
    """没有 .tar 后缀的层也要认出来。"""
    digest = "a" * 64
    image = _oci_image(tmp_path / "oci.tar.gz", layer_files=_good_layer(), layer_digest=digest)
    result = inspect(image, expect_dir=REPO_ROOT)
    assert [layer["name"] for layer in result["layers"]] == [f"blobs/sha256/{digest}"]
    assert "root/participant/start.sh" in result["members"]


def test_oci_config_blob_is_not_counted_as_a_layer(tmp_path: Path) -> None:
    """config blob 也是 blobs/sha256/<hex>，但它是 JSON 不是 tar，不能算一层。"""
    digest = "a" * 64
    image = _oci_image(tmp_path / "oci.tar.gz", layer_files=_good_layer(), layer_digest=digest)
    result = inspect(image, expect_dir=REPO_ROOT)
    assert len(result["layers"]) == 1


def test_expect_flag_is_honored_and_not_abbreviated(tmp_path: Path, capsys) -> None:
    """`--expect` 不能被 argparse 当成 `--expect-dir` 的前缀缩写。

    这一条是拿真镜像验证时踩出来的：当时传 `--expect /root/.../start.sh`，被解释成
    `--expect-dir /root/.../start.sh`，于是每一行逐字节比对都报「本地没有这个文件」，
    看着像镜像坏了，其实是参数被静默改写。
    """
    digest = "c" * 64
    image = _oci_image(tmp_path / "oci.tar.gz", layer_files=_good_layer(), layer_digest=digest)

    main([str(image), "--expect", "/root/participant/start.sh"])
    assert "存在 /root/participant/start.sh" in capsys.readouterr().out

    assert main([str(image), "--expect", "/root/根本不存在的路径"]) == 1
    assert "/root/根本不存在的路径" in capsys.readouterr().out