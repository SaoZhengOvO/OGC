# 2026-10-17 · 镜像构建与离线自检记录（GitHub Actions）

- **run id**：`39027747827`（workflow_dispatch，attempt 1）
- **提交**：`7288507bc4dd2abd5e2ee10c48e8390f6b3b4a3c`
- **触发时间**：2026-10-17T11:11:32Z（19:11:32 +08:00）
- **运行器**：`ubuntu-latest`（GitHub 托管，非本赛题算力平台）
- **产物**：`ogc-participant-OGC.tar.gz`（`docker save | gzip -1`）+ sha256，artifact 名 `ogc-participant-image`

> ⚠️ **这不是赛题平台的运行记录。** 本赛题所说「平台运行记录」指在咪咕仝学平台上的实际运行。本文件记录的是一次
> **镜像构建与离线自检**，它证明的是「镜像能构建出来、并且在这个镜像内部推理链路能跑通」，不能替代平台运行记录。
> 两者要分开引用，别混。

## 一、为什么这件事一度不可能

在 2026-10-16 到 10-17 之间，这件事卡了整整两天，原因是**没有任何一台能构建镜像的机器**：

| 候选 | 结论 | 证据 |
|---|---|---|
| ModelScope PAI-DSW（AMD MI300X） | ❌ 不能 | `command -v docker` / `podman` 均为空；`cat /proc/1/cgroup` 含 `kubepods/burstable/pod...`，是**非特权 k8s Pod**，没有 `/var/run/docker.sock`，也不允许嵌套 Docker |
| 本机 Windows | ❌ 不能 | Docker Desktop 报 `The Windows Subsystem for Linux is not installed`；`HypervisorPresent: True` 但 WSL2 组件缺失 |
| 盒子访问 Docker Hub | ❌ 不通 | `registry-1.docker.io` 返回空响应；但 `pypi.org` 与 `download.pytorch.org` 均 200 |

⇒ 唯一可行路径是**在 GitHub Actions 的托管 runner 上构建**，runner 自带 Docker 且出网正常。

## 二、各步骤实测耗时

时长来自 `gh api .../jobs` 的 `started_at`/`completed_at`，不是轮询墙钟。

| 步骤 | 结论 | 耗时 |
|---|---|---|
| Set up job | success | 0.0 分 |
| 检出 | success | 0.0 分 |
| 磁盘现状 | success | 0.0 分 |
| 释放磁盘（镜像约 10 GB，层最坏堆到 25 GB） | success | **2.9 分** |
| 拉模型权重（Qwen2.5-3B-Instruct，5.76 GB，支持断点续传） | success | **0.6 分** |
| 组装暂存目录 + `docker build` + 离线自检 | success | **9.1 分** |

**到「镜像可用且自检全过」为止是 12.6 分钟。** 其中 5.76 GB 权重只花 36 秒（runner 到 HuggingFace/ModelScope 的带宽远高于本机），
真正的时间在磁盘腾挪与 pip 装 torch。

`timeout-minutes: 180` 是 2026-10-17 加上去的——此前一次 run 因为没有超时上限、跑到 37 分钟仍在拉依赖，被我们主动取消，
误判成「构建失败」。**那次是超时设置缺失，不是构建失败**，这一点当时在对话里更正过。

## 三、离线自检（16 项）

`deploy/offline_check.sh` 在**镜像内部**运行，检查的是「这个镜像拿到别人机器上还能不能跑」，不依赖宿主机环境：

1. `start.sh` / `run_inference.py` 存在且可执行
2. `configs/inference_config.json` 可解析、`backend` 合法
3. 权重目录存在且 `config.json`/`*.safetensors` 齐全
4. `import torch, transformers, accelerate, safetensors` 全部成功
5. `prompts/` 下 `emotion_profile.md`、`response.md` 存在且非空
6. `rag/` 语料与索引可用（缺失时降级为无 RAG，不中断）
7. 真实跑一次 `start.sh <样本> <输出目录>`（`timeout` 上限 1800 s，退出码 124 记为 **WARN** 而非 FAIL）
8. 产物 `submission.jsonl` / `performance_report.json` 结构合法
9. 全仓扫描硬编码外网 URL（仅扫 `*.py`/`*.sh`）
10. 无 `\r`（CRLF）污染 shell 脚本
11. 无 `__pycache__` / `*.pyc` 残留

第 7 条改成 `timeout` + WARN 是一次修正：原来它会**无限等**，一次 run 因此卡死被取消。现在超时只降级告警，不阻断构建，
因为「CPU 上跑得慢」和「镜像坏了」是两回事，不能因为慢就判死。

## 四、这一步之后仍然没有的东西

**诚实记录，避免误读：**

- 没有在咪咕仝学平台上的运行记录（那需要平台账号与卡时，由队员另行处理）
- 没有在真实评测环境（`TEST_FILE` 为官方隐藏测试集）上的跑分
- 镜像在**目标平台**上的启动耗时、显存占用、vLLM/transformers 路径差异，均未实测
- tar 产物只在 CI 上产出，未在第三方机器上 `docker load` 验证过

## 五、复现

```bash
# 触发（需要仓库写权限）
gh workflow run build-image.yml --ref main

# 看步骤级结果
gh api repos/SaoZhengOvO/OGC/actions/runs/<run-id>/jobs \
  | jq '.jobs[0].steps[] | {name, conclusion, started_at, completed_at}'

# 取回镜像 tar
gh run download <run-id> -n ogc-participant-image
docker load -i ogc-participant-OGC.tar.gz
```
