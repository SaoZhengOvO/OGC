# 咪咕仝学平台首次实机运行记录（2026-11-06）

> 这是本项目**第一次在咪咕仝学平台自己的实例里跑通官方 `start.sh`**。此前所有数字
> 都来自 ModelScope PAI-DSW 的 AMD MI300X 机器（见相邻两个目录），那台机器与比赛
> 评测机不同源，只能作为相对比较。本记录里的每一项都能在下方命令与输出里复现。

## 1. 实例与镜像

| 项 | 值 |
|---|---|
| 实例名 | `ogc-duihua-01` |
| 实例 ID | `e1780968-f575-48a2-b237-91baf589f29a` |
| 机房 | 西南二区 |
| GPU | RTX 4090 24G × **1** |
| CPU / 内存 / 系统盘 | 16 核 / 64 GB / 50 GB（免费额度） |
| 计费 | 按量计费 · 抢占 · **1 卡时/小时** |
| 镜像 | 公有镜像 `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8`（28.29 GB，大赛专用） |
| 网络盘 | `/mnt/storage` 200 GB |
| 创建时间 | 2026-11-06 17:25:12 |
| 运行开始 | 2026-11-06 17:30:41 |

镜像选了大赛专属的「vLLM（大赛专用镜像-对话模型）」，理由是 `start.sh` 的两条后端
路径（transformers / vLLM）都需要 Python 3.12 + CUDA 12.8 的运行时，官方镜像自带。

## 2. 环境实测（与官方 README 的对照）

```console
$ /opt/conda/envs/conda_env/bin/python -V
Python 3.12.12

$ /opt/conda/envs/conda_env/bin/python -c "import vllm, torch, transformers; ..."
vllm 0.19.0 | torch 2.10.0+cu128 | transformers 4.57.6
```

- **系统 `python3` 是 3.10.12 的裸环境**，没有 torch。真正的运行时在
  `/opt/conda/envs/conda_env/`。
- `accelerate` 缺失，`pip install accelerate` 后为 1.15.0（本项目用
  `device_map` 之外的标准加载路径，装它是为了保险）。
- `modelscope 1.35.4`、`huggingface_hub 0.36.2` 自带，可以直接拉权重。
- 网络：`pypi.org` 不通；`mirrors.aliyun.com/pypi`（pip 已预配）与 `pypi.tuna` 通；
  `modelscope.cn`、`hf-mirror.com`、`github.com`、`codeload.github.com` 均 200。
- **没有 `git`**，拉代码只能用 `curl` 下 codeload 的 tarball。

## 3. 部署步骤（全部在实例内完成）

```bash
# ① 拉代码（无 git）
curl -sSL -o repo.tar.gz \
  https://codeload.github.com/SaoZhengOvO/OGC/tar.gz/refs/heads/main
mkdir -p /root/work/repo && tar xzf repo.tar.gz -C /root/work/repo --strip-components=1

# ② 拉权重（ModelScope，实测 9–11 MB/s，5.76 GB 约 9 分钟）
/opt/conda/envs/conda_env/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-3B-Instruct", local_dir="/root/participant/model")
PY

# ③ 组装交付目录
cp -a deploy/participant/. /root/participant/      # run_inference.py / wiring.py / start.sh / configs / templates
cp -a prompts /root/participant/prompts
cp -a rag /root/rag                                # corpus.jsonl 被 gitignore，见 ④
chmod +x /root/participant/start.sh

# ④ 重建 RAG 语料（仓库里不入库，避免 17 MB 二进制进 git）
cd /root/work/repo && python rag/corpus/build_corpus.py
#   样本 18,657 条 / 对话 8,333 段 / 正文 5,674,374 字符 / 17.3 MB / 0.5 秒
cp rag/corpus/corpus.jsonl /root/rag/corpus/corpus.jsonl

# ⑤ 评审材料（官方要求放在 /root 下，名字带队名）
python docs/reports/pack_review.py --team OGC \
  --out '/root/AI技术赛道-数字人综合情感陪伴对话模型-OGC'
```

`wiring.py` 的 `CONTENT_ROOT` 默认取 `participant/` 的上一级，所以 `/root/participant`
+ `/root/rag` 这个布局正好命中，不需要设任何环境变量。

## 4. 两次失败与修复

### 4.1 `source activate conda_env` 在非交互 shell 里不存在

首次执行直接挂掉：

```console
[start.sh] 警告：检测到 conda 但 conda_env 激活失败，改用当前 PATH 上的 python3
后端: transformers（直接加载本地模型）
ModuleNotFoundError: No module named 'torch'
RuntimeError: 请先安装 participant/requirements.txt
```

根因：`/opt/conda/bin/conda` 存在，但 `activate` 是交互式 init 脚本才定义的 shell
函数；`bash start.sh` 是非交互的，`source activate` 找不到这个命令，脚本按设计退回
`python3` —— 那是 3.10 裸环境。**这是我们的 bug，不是平台的**：官方原件同样写
`source activate conda_env`，说明它预期在交互式 shell 或已经 init 过的环境里跑。

修复（commit `f401df9`）：改成三级回退 —— 先 `source /opt/conda/etc/profile.d/conda.sh`
再 `conda activate conda_env`；失败就退到 `/opt/conda/envs/conda_env/bin/python`
这个绝对路径；最后才退 `python3`。`PYTHON_BIN` 显式指定时优先。

### 4.2 测试文件带 UTF-8 BOM

```console
ValueError: /root/work/test3.jsonl:1: JSON 解析失败: Unexpected UTF-8 BOM
```

BOM 是本地 PowerShell `Set-Content -Encoding utf8` 写出来的（Windows PowerShell 5.1
默认带 BOM）。重新用 `UTF8Encoding($false)` 写盘后正常。**这条只影响我们自己造的
测试文件**，平台下发的测试集不会带 BOM；`run_inference.py` 仍按官方口径用 `utf-8`
严格解析，没有为此放宽。

## 5. 运行结果

```console
$ bash /root/participant/start.sh /root/work/test3.jsonl /root/work/out3
[wiring] content_root=/root prompt_dir=/root/participant/prompts \
         prompts=emotion_profile,response rag.retrieve=ok rag.fusion=ok
[profile] mode=force priors={"interests": "career_development",
         "personality_traits": "sensitive", "style": "rational"}
[1/3] val_000001_t1: 3561.801 ms
[2/3] val_000001_t3: 2457.660 ms
[3/3] val_000002_t5: 2651.575 ms
[wiring] 降级 0 次：全程走融合层组包
PASS: /root/work/out3/submission.jsonl
PASS: /root/work/out3/performance_report.json
```

- **接线全部命中**：`prompts=emotion_profile,response`、`rag.retrieve=ok`、
  `rag.fusion=ok`、**降级 0 次**（融合层在每一条上都成功接管组包，没有退回官方组包）。
- **延迟**：平均 **2890 ms**、中位 2652、p95 3562、最大 3562（3 条样本）。
- **解析**：`parse_failure_count = 0`。
- 输出三条：

| id | emotion_label | response（截断） |
|---|---|---|
| `val_000001_t1` | `anxiety` | 你现在感觉很慌张，好像所有的时间都在飞快流逝… |
| `val_000001_t3` | `anxiety` | 你现在好像在经历一种「风暴前的宁静」… |
| `val_000002_t5` | `loneliness` | 你现在感觉有点孤单，像是周围的人来人往… |

三条都吐出了合法的 `emotion_label`，没有再出现本地 RTX 5070 上那种「情绪字段塌陷
为空、重试一次仍为空」的情况。

## 6. 与 AMD 机器的对比（同一份 3 条样本）

| 机器 | 平均延迟 | 备注 |
|---|---|---|
| AMD MI300X（ModelScope，191.7 GiB） | 2809 ms | n=2354 全量均值，见 `2026-10-16-sft-lora/` |
| RTX 4090 24G（咪咕，本记录） | 2890 ms | n=3，只作量级参考 |

MI300X 的 2809 ms 是大样本均值，本记录只有 3 条，两者**不能直接比**；能说的只有
「同一量级」，说明 3B 模型在这个平台上不会被算力卡成瓶颈。

## 7. 提交路径（重要：平台没有「上传 tar.gz」入口）

查了控制台与前端接口后确认：`模型镜像 → 我的镜像` 没有任何上传/导入按钮，镜像服务
接口只有 list / delete / `mirrorversion/list`，前端路由表里也没有 `/mirror/upload`。
唯一的大赛镜像是 `POST /v1/cloud/competition/image/commit`，对应实例列表里的
**「上传大赛镜像」** 按钮，提示「只有**关机状态**才能提交镜像」。

所以作品 = **系统盘快照**，不是外部 tar：在实例里把代码、权重、评审材料摆好 →
关机 → 点「上传大赛镜像」→ 在 `worksubmit` 选这个镜像提交。

## 8. 复现清单

- [x] 代码 tarball 拉取成功（26,824,752 B）
- [x] 权重 5.76 GB 落地，两个 shard 尺寸与官方一致
- [x] RAG 语料在实例上重建，18,144,001 B，与本地一致
- [x] 评审材料 01–08 + `MANIFEST.md` 齐全
- [x] `start.sh` 跑通，两份产物 PASS
- [x] 融合层 0 降级
- [x] `parse_failure_count = 0`
