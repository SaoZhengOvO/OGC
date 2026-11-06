"""把镜像 tar.gz 传到咪咕「外部平台」的 FileZilla 存储。

实测（2026-11-06）：`183.222.230.93:50001` 是 **vsFTPd 3.0.5 + 显式 TLS**。
- 明文 FTP 会被拒：`530 Non-anonymous sessions must use encryption.`
- SSH/SFTP 连不上（读到的是 FTP banner，不是 SSH banner），别用 paramiko。

所以用 `FTP_TLS`（AUTH TLS + PROT P）。凭据只从环境变量读：
MIGU_FTP_HOST / MIGU_FTP_PORT / MIGU_FTP_USER / MIGU_FTP_PASSWORD。

断点续传：远端已有同名文件时用 REST 从已有字节数接着传。

用法：
    python tools/migu_ftp_upload.py ls [远端目录]
    python tools/migu_ftp_upload.py upload 本地文件 [远端目录]
"""

from __future__ import annotations

import ftplib
import os
import sys
import time


def _connect() -> ftplib.FTP_TLS:
    host = os.environ["MIGU_FTP_HOST"]
    port = int(os.environ.get("MIGU_FTP_PORT", "50001"))
    user = os.environ["MIGU_FTP_USER"]
    password = os.environ["MIGU_FTP_PASSWORD"]

    ftp = ftplib.FTP_TLS(timeout=60)
    ftp.connect(host, port)
    ftp.login(user, password)
    ftp.prot_p()
    ftp.set_pasv(True)
    return ftp


def cmd_ls(remote_dir: str) -> None:
    ftp = _connect()
    for name in ftp.nlst(remote_dir):
        try:
            size = ftp.size(name)
        except ftplib.error_perm:
            size = None
        print(f"{size if size is not None else '?':>14}  {name}")
    ftp.quit()


def cmd_upload(local: str, remote_dir: str) -> None:
    size = os.path.getsize(local)
    name = os.path.basename(local)
    remote = remote_dir.rstrip("/") + "/" + name if remote_dir not in ("", ".") else name

    ftp = _connect()
    try:
        done = ftp.size(remote) or 0
    except ftplib.error_perm:
        done = 0

    if done == size:
        print(f"已完整：{remote}（{size:,} 字节）")
        ftp.quit()
        return
    if done > size:
        print(f"远端更大（{done:,} > {size:,}），删掉重传")
        ftp.delete(remote)
        done = 0

    print(f"续传起点 {done:,} / {size:,}" if done else f"全新上传 {size:,} 字节")

    start = time.time()
    last = start
    last_bytes = done

    def _tick(block: bytes) -> None:
        nonlocal done, last, last_bytes
        done += len(block)
        now = time.time()
        if now - last >= 5:
            rate = (done - last_bytes) / (now - last)
            print(
                f"  {done:,} / {size:,}  {done / size * 100:5.1f}%  {rate / 1e6:.2f} MB/s",
                flush=True,
            )
            last, last_bytes = now, done

    with open(local, "rb") as src:
        if done:
            src.seek(done)
        ftp.storbinary(f"STOR {remote}", src, blocksize=1024 * 1024, callback=_tick, rest=done)

    final = ftp.size(remote)
    print(f"完成 {remote}：远端 {final:,} 字节，用时 {(time.time() - start) / 60:.1f} 分钟")
    ftp.quit()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "ls":
        cmd_ls(argv[2] if len(argv) > 2 else ".")
        return 0
    if argv[1] == "upload":
        if len(argv) < 3:
            print("缺少本地文件路径")
            return 2
        cmd_upload(argv[2], argv[3] if len(argv) > 3 else ".")
        return 0
    print(f"未知动作：{argv[1]}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
