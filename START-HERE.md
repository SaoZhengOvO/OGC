# START HERE —— 队友开工指南

本仓库是「2026 动感地带 AI+ 高校创智计划」AI 技术赛道赛题二 **数字人综合情感陪伴对话模型**的参赛工程，全队 5 人。

这份文件只讲一件事：**从零到这台机器上跑出第一个真实分数**。大约 40 分钟，其中 35 分钟是等下载。

> **懒得自己照着做？** 把 [`docs/roles/ONBOARD.md`](docs/roles/ONBOARD.md) **整段丢给你机器上的 AI**。它会替你问清分工、把你的 GitHub 用户名登记进 `ROLES.md`，再一步一步把下面五步跑完。

---

## 0. 开工前先知道三件事（能省你半天）

1. **我们不训练、不微调。** 路线是「小模型 + 提示词 + RAG」。别去找训练脚本，仓库里没有，也不会有。
2. **没有官方基础镜像可以下。** 客服已确认。组委会说的「基础镜像」指的是咪咕平台的公有镜像（vLLM / Ubuntu / PyTorch 那几个），到平台上选一个自建就行。
3. **唯一的评测入口是我们自己的裁判。** 谁改了什么，都用 `python tools/eval_loop.py` 在同一份验证集上打分，结果自动进 `runs/leaderboard.tsv`。**不要自己另算一套指标**，那样五个人的数没法比。

---

## 1. 你是哪条线

> 原「⑤ RAG·融合与文档」已拆成 **⑤A 融合代码**与 **⑤B 专家评审文档**（2026-10-16）。队长 = ① + ⑤B。

| 线 | 提示词文件 | 主责 | 你能动的目录 |
|---|---|---|---|
| ① 部署与评测 | [`docs/roles/01-deploy-eval.md`](docs/roles/01-deploy-eval.md) | 推理性能 10 分 + 平台运行记录 10 分 | `deploy/`、`docs/platform-log/` |
| ② 提示词·情绪与画像 | [`docs/roles/02-prompt-emotion-profile.md`](docs/roles/02-prompt-emotion-profile.md) | 情感识别 25 分 + 用户画像 15 分 | `prompts/emotion_profile.md`、`tools/emotion_profile/` |
| ③ 提示词·共情回复 | [`docs/roles/03-prompt-response.md`](docs/roles/03-prompt-response.md) | 回应质量 20 分 | `prompts/response.md`、`tools/response/` |
| ④ RAG·语料与检索 | [`docs/roles/04-rag-corpus.md`](docs/roles/04-rag-corpus.md) | 记忆融合 | `rag/corpus/`、`rag/index/`、`rag/retrieve.py`、`rag/memory.py` |
| ⑤A RAG·融合 | [`docs/roles/05a-rag-fusion.md`](docs/roles/05a-rag-fusion.md) | 融合层 | `rag/fusion.py` |
| ⑤B 专家评审文档 | [`docs/roles/05b-reports.md`](docs/roles/05b-reports.md) | 方案完整性 10 分 + 代码规范性 10 分 | `docs/reports/` |

**接口怎么拼**在 [`docs/roles/CONTRACT.md`](docs/roles/CONTRACT.md) 里冻结了，动之前先读。
**别人的目录只读**，要改先在 issue 里说一声。

---

## 2. 五步跑起来

> **懒人版**：Windows / Linux 上跑 `python tools/bootstrap.py`，下面五步它会自动串起来做完。
> 想看清楚每一步在干什么、或者中途出错了要单独重跑，就照下面手动来。

### 第 1 步 · 拉仓库

```bash
git clone https://github.com/SaoZhengOvO/OGC.git
cd OGC
```

### 第 2 步 · 建 Python 环境

需要 Python 3.11 以上（**3.14 也可以用**，torch / transformers / safetensors 都有对应轮子）。

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers accelerate
```

> **显卡是 RTX 50 系（5060 / 5070）的人注意**：必须用 CUDA 12.8 以上的构建，上面这条 `cu130` 就是为此选的。装成旧版 CUDA 会在推理时报 `no kernel image is available for execution on the device`。
> 没有 N 卡也能用，把 `pip install torch` 去掉 `--index-url` 装 CPU 版即可，只是慢。

### 第 3 步 · 下基座模型

```bash
python tools/fetch_model.py
```

约 **6.2 GB**，落到 `GivenData/participant/model/`。断点续传：中途断了重跑同一条命令，会从断点接着下。国内慢就换镜像：

```bash
python tools/fetch_model.py --mirror https://hf-mirror.com
```

看看下好了没：

```bash
python tools/fetch_model.py --check
```

**显存只有 8 GB 的话**，3B 模型（fp16 约 6.2 GB）会很紧，先照上面跑；如果报显存不足，换 1.5B：

```bash
python tools/fetch_model.py --repo Qwen/Qwen2.5-1.5B-Instruct --dest GivenData/participant/model-1.5b
```

### 第 4 步 · 展开官方数据

```bash
python src/cli.py build
```

约一分钟，产出 `data/processed/`（220 MB，不进 Git，随时可重建）。

### 第 5 步 · 冒烟测试，拿到第一个分数

```bash
python tools/eval_loop.py --tag smoke --limit 20
```

它会依次：造验证输入 → 调官方 `run_inference.py` 推理 → 校验提交格式 → 本地打分 → 在 `runs/leaderboard.tsv` 记一行。

**到这里你就通了。** 三条指标（情绪准确率 / 画像准确率 / 回复 BLEU-4 与 ROUGE-L）会直接打在屏幕上。

> `--limit 20` 只是链路冒烟，样本太少指标波动大，**不要**拿它跟全量比。正式迭代跑全量：
> ```bash
> python tools/eval_loop.py --tag 你的实验名
> ```

---

## 3. 把提示词交给你的 AI

这是重点。**你不需要自己记住规则——提示词里都写好了。**

1. 打开你机器上的编码 AI（Claude Code / Cursor / DeepSeek Harness，或者本地部署的任意 agent）。
2. 把它的**工作目录设成这个仓库的根目录**。
3. 打开上面表格里属于你的那个 `docs/roles/0X-*.md`，**整段复制，贴进对话框**。
4. 它会先读 `docs/roles/README.md`（公共背景）、`CONTRACT.md`（接口契约）和你那份提示词，然后开始干活。

提示词里已经写明：先读哪几个文件、你这条线的目标、**只能动哪些目录**、跑什么命令自测、交付什么。

---

## 4. 日常怎么干活

```bash
# 全量打分（每次改完提示词 / RAG 都跑一遍）
python tools/eval_loop.py --tag r2

# 只给已有的提交文件重新打分（不重跑推理）
python tools/eval_loop.py --tag r2 --skip-inference --predictions runs/eval-r2/submission.jsonl

# 看历次实验台账
cat runs/leaderboard.tsv

# 提交前自检
python -m unittest discover -s tests
```

### 交给别人的东西必须过这三关

1. `python -m unittest discover -s tests` 全绿；
2. `python tools/eval_loop.py --tag <你的实验名>` 跑完并留了台账行；
3. 你的改动**没有越界**到别人的目录。

---

## 5. 常见坑

| 现象 | 原因 / 怎么办 |
|---|---|
| `FileNotFoundError: 本地模型目录不存在` | 模型还没下。跑第 3 步。 |
| `no kernel image is available` | torch 的 CUDA 版本太旧。重装 `cu130`。 |
| 权重像是下完了，其实只有一半 | 用 `python tools/fetch_model.py --check` 核对——它会连远端比大小，不完整会退出码 1。这个静默截断的坑已经修掉，但旧版本下过的目录要重新 `--check` 一次。 |
| `OSError: Error no file named model.safetensors, or pytorch_model.bin` | 分片权重缺**目录文件** `model.safetensors.index.json`。只有分片、没有它时 transformers 不认分片，报错却像是"模型没下完"。再跑一次 `python tools/fetch_model.py` 就会补下（`--check` 现在也会报这一项）。 |
| 下载看着"卡住"不动了 | Windows 对**正在写入**的文件只报目录项里的旧大小。别信 `Get-ChildItem`；用 `[System.IO.File]::Open($p,'Open','Read','ReadWrite')` 读 `$fs.Length` 才是真进度。 |
| `CUDA out of memory` | 8 GB 卡跑不动 3B。换 1.5B，或加 `--limit` 先跑小样本。 |
| `找不到 data/processed/...` | 没跑第 4 步 `python src/cli.py build`。 |
| 下载龟速 | 加 `--mirror https://hf-mirror.com`。 |
| `git status` 里冒出 `pip-unpack-*` / `tmp*` / 一个 GB 级的 `.whl` | pip 把解包临时目录建在了仓库根（在根目录跑 `pip install` 就会这样）。**`.gitignore` 已经挡住它们**，正常 `git add -A` 不会带上。千万别 `git add -f`。 |
| `git status` 里冒出 6 GB 文件 | 不该发生——`.gitignore` 已挡住 `*.safetensors`。真出现就直接 `git checkout .gitignore` 后找负责人。 |

---

## 6. 硬约束（踩了直接丢分）

- **完全离线**：提交镜像里不许有任何公网调用。评测机不联网。
- **唯一入口**：`bash /root/participant/start.sh TEST_FILE RESULT_DIR`。别自创入口。
- **镜像 ≤ 50 GB**（平台系统盘 50 GB + 公有镜像已用空间之和），超了直接提交失败。
- **只有 5 次成功测评机会**，失败不占次数但只有最后一次成功结果计分。**平台上机要省着用**（卡时券只有 10 卡时）。
- **不许伪造**：训练日志、跑分、截图一律要真跑出来，跑不出来就写「未跑」。
- **`GivenData/` 是官方材料，只读**。唯一例外是角色①的 `participant/configs/inference_config.json`。
- **验证集（val）绝对不能进 RAG 语料或任何形式的训练**。角色④只读 `train`。
