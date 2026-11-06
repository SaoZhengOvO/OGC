# 作品提交记录（2026-11-06）

> 本文件记录**第一次真正提交到咪咕平台**的全过程：走的是哪条路、为什么只能走这条、
> 提交物是什么、平台回执是什么。数字与截图里的字段一一对应。

## 1. 结论先行

| 项 | 值 |
|---|---|
| 提交时间 | 2026-11-06 18:52（UTC+8） |
| 机房 | 西南一区 |
| 作品来源 | **外部平台** |
| 作品文件 | `ogc-participant-OGC.tar.gz`（7,993,816,941 B） |
| 作品状态 | **测评中**（进度 25%） |
| 剩余提交机会 | **5 次**（测评失败不计次，仅成功扣 1 次） |

## 2. 为什么不是「上传大赛镜像」

官方操作手册写的是「上传大赛镜像」一键提交，但那一步在本账号上**服务端报错**：

```
POST /v1/cloud/competition/image/commit
504 Gateway Timeout（前三次，60 s 网关超时）
500 {"errors":[{"code":"system_exception","source":"500000",
                "message":"提交镜像时初始化用户存储空间失败"}]}
```

实例状态确实是「已关机」，系统盘只用了 5.9 GB（远小于免费 30 GB 镜像额度），
所以不是我们的数据问题。报错后前端按钮一直转圈，`我的镜像` 里始终没有条目。

`worksubmit` 页面给了第二条路：把「作品来源」切成 **外部平台**，页面会给出一个
FileZilla 存储地址，手工上传作品镜像后再提交。**这条路是平台自己写的操作手册里
的选项，不是绕过。**

## 3. 存储与上传

| 项 | 值 |
|---|---|
| 地址 | `183.222.230.93:50001` |
| 协议 | **vsFTPd 3.0.5 + 显式 TLS**（`FTP_TLS`，`AUTH TLS` + `PROT P`） |
| 用户 | `user-20261013162303964-6023` |

踩到的坑，按顺序：

1. **不是 SFTP**。用 paramiko 连会得到 `Error reading SSH protocol banner`——
   那个端口回的是 FTP banner。
2. **明文 FTP 被拒**：`530 Non-anonymous sessions must use encryption.`
3. 存储地址**只在西南一区显示**；选西南二区时三个字段是空的（我们的实例在西南二区，
   这也是「上传大赛镜像」失败的旁证——西南二区没有给参赛用户初始化比赛存储）。
4. 传 7.99 GB 时连接会被中途掐断（`ConnectionAbortedError 10053` /
   `ConnectionResetError 10054`），所以脚本必须支持 **REST 断点续传** 并套重试循环。

工具：`tools/migu_ftp_upload.py`（`ls` / `upload`，FTPS + REST 续传 + 5 秒一次进度），
重试循环：`deploy/dist/migu/upload_loop.ps1`。

## 4. 传输实测

| 线路 | 速率 | 7.99 GB 预计 |
|---|---|---|
| OpenClash 全局（国内 IP 也绕美国） | 1.2–1.5 MB/s | ~100 分钟 |
| OpenClash 规则模式（国内直连） | **6.3 MB/s** | ~21 分钟 |

实际：18:20 起传，中途断了两次，续传后 18:44 完成。

## 5. 完整性校验

传完不是看「进度条 100%」就算数——REST 续传一旦偏移算错，文件就是坏的，而平台不会
替你校验。所以做了**定长分块抽样比对**：在 5 个偏移各取 64 KiB，与本地文件同偏移
逐字节比对。

```console
OK   0            65536  6ae9d7497dbe24e8  6ae9d7497dbe24e8
OK   1073741824   65536  d17c21e066aa179e  d17c21e066aa179e
OK   3221225472   65536  482d5d88617fbe9e  482d5d88617fbe9e
OK   5368709120   65536  6e7ca0829204af7d  6e7ca0829204af7d
OK   7993751405   65536  ee752b2b8c9873a5  ee752b2b8c9873a5
全部一致
```

首字节、1/3/5 GB 处、尾块全对，远端大小 `7,993,816,941` 与本地完全相等。

## 6. 提交物本体

`ogc-participant-OGC.tar.gz` 是 GitHub Actions（run `42176133195`，commit `1a98d35`）
构建的 `docker save` 产物，本地双重哈希校验过：

- 外层 zip：`e719411d6a08ee5eeae2b3a12b2d99b6c232e6560e673fee26693c0f75fca0d4`
- 内层 tar.gz：`8596aadb5452b9bafc67ee789ab58a9332edaf83a690f655ee512edd48caf7a9`

`inspect_image_tar.py` 复核：10 层、模型权重 2 shard 共 5.75 GB、
`/root/participant/{start.sh,run_inference.py}` 与 `model/config.json`、
`offline_check.sh`、评审目录 `AI技术赛道-数字人综合情感陪伴对话模型-OGC` 全部在位，
提示词 `emotion_profile.md` sha `4c7cf055`、`response.md` sha `639ce8c4`。

## 7. 待观察

- 测评进度 25% → 完成后的 `作品状态` 与分数（若有）需要回填到本文件。
- 「测评失败不计次」是平台明写的规则，所以第一次失败不等于浪费额度；
  重提只有成功才扣次数。
- 若「上传大赛镜像」之后恢复正常，那条路可以再试一次做交叉验证——
  两条路的**判分口径应该一致**，不一致就要问主办方。
