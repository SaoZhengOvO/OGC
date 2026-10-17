# 复现指南：从裸机到产出提交文件

> 作者：角色⑤B（专家评审文档）。对应官方评分项「专家评审 — 代码规范性（10 分）」。
> 本文的命令分三类，**每类都标了它到底有没有真跑过**：
> ① **本机真跑**（Windows + MSYS `sh`）：`deploy/build.sh --stage-only`、`deploy/offline_check.sh`、单元测试、评测；
> ② **本机跑不了、但在别处真跑过**：`docker build` + `docker run --network none`，跑在 GitHub Actions 的
> `ubuntu-latest` runner 上（`.github/workflows/build-image.yml`）；
> ③ **仍未实测**：在官方平台上的提交与评测（截止前会做一次预演）。
> 本机没有 WSL2（Docker Desktop 起不来），算力机是 k8s pod（`no docker` / `no podman`），
> 所以 **docker 相关一律走 CI，不在本机跑**。

## 零、环境前提

| 项 | 要求 | 本机实测值 |
|---|---|---|
| Python | 3.11 以上（本机 3.14.7） | `C:\Python314\python.exe` |
| GPU | NVIDIA，CUDA 可用 | RTX 5070 Laptop，sm_120（Blackwell） |
| PyTorch 索引 | **cu130**（cp314 有轮子 + 支持 sm_120） | `torch 2.14.1+cu130`，`torch.version.cuda = 13.0` |
| 磁盘 | 权重约 5.8 GB + 展开数据约 221 MB | D 盘剩余 223 GB |

各 CUDA 索引的实测对照（**选错任一项都会装不上或跑不起来**）：

| 索引 | 版本 | cp314 有轮子 | 支持 sm_120 |
|---|---|---|---|
| cu126 | 2.14.1+cu126 | 有 | 未验证 |
| cu128 | 2.11.0+cu128 | 有 | 未验证 |
| cu129 | 2.9.0+cu129 | 有 | 未验证 |
| **cu130** | **2.14.1+cu130** | **有** | **是（实测 `(12, 0)`）** |
| cu131 | 无匹配 | — | — |

## 一、拿到代码与依赖

```bash
git clone https://github.com/SaoZhengOvO/OGC.git
cd OGC
```

### 路线 A：一键（推荐新人）

```bash
python tools/bootstrap.py
```

`tools/bootstrap.py` 会依次建 `.venv`、升级 pip、装 torch（默认 cu130 索引）、装
transformers/accelerate、做一次 import 自检、下模型权重、跑 `src/cli.py build`，
最后跑一个 20 条的小样本评测。可选参数：`--cpu`（不装 CUDA 版 torch）、
`--torch-index <url>`、`--limit N`。

> 脚本会强制把 `TMPDIR`/`TEMP`/`TMP` 指向系统临时目录。**不要去掉这一步**：
> pip 在 Windows 上曾把解包目录建在仓库根目录，`git add -A` 会把 34 MB 的半成品 wheel
> 扫进提交（历史上真的发生过一次，靠 `git commit --amend` + `--force-with-lease` 修掉）。

### 路线 B：手动

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install transformers accelerate
```

自检（必须全部符合预期才继续）：

```powershell
.venv\Scripts\python.exe -c "import torch, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 期望：2.14.1+cu130 13.0 True (12, 0)
```

## 二、下载基座模型

```powershell
.venv\Scripts\python.exe tools\fetch_model.py
```

- 默认仓库 `Qwen/Qwen2.5-3B-Instruct`，落盘到 `GivenData/participant/model/`（已被 .gitignore）。
- 纯标准库实现，**不需要 `huggingface_hub`、不需要 `git-lfs`**；支持断点续传。
- 国内网络慢就加 `--mirror https://hf-mirror.com`。
- **下完必须看这一行**（曾经踩过坑）：

```
可用性: config.json=有 权重=有 分片目录=有
```

三个都必须是「有」。分片权重的 `model.safetensors.index.json` 是目录文件，
**只有分片、没有它时 transformers 会报
`OSError: Error no file named model.safetensors, or pytorch_model.bin`** ——
看起来像「模型没下完」，其实是少了一个 34 KB 的 JSON。

## 三、展开数据与词表

```powershell
python src\cli.py build
```

产物：

| 路径 | 内容 |
|---|---|
| `data/processed/train_samples.jsonl` | 训练样本 18,651 条（含 target 标签） |
| `data/processed/val_samples.jsonl` | 验证样本 2,354 条 |
| `data/processed/*_references.jsonl` | 打分用的参考答案 |
| `configs/label_space.json` | 16 类情绪 + 三维画像词表 |
| `data/reports/{train,val}_quality.json` | 数据质量报告（**这个要提交**） |

`data/processed/` 约 221 MB，已在 `.gitignore` 中；`data/reports/` 只有 3.5 KB，
是「我们查到了什么」的留档证据，**必须提交**。

## 四、跑一次评测（全队共用的唯一裁判）

```powershell
# 冒烟：5 条，验证链路通不通
.venv\Scripts\python.exe tools\eval_loop.py --tag smoke --limit 5 --backend transformers

# 官方裸基线：2,354 条 —— 必须显式指定官方那棵树，默认跑的不是它
.venv\Scripts\python.exe tools\eval_loop.py --tag baseline --backend transformers --inference-script official

# 我们要提交的那棵树：2,354 条
.venv\Scripts\python.exe tools\eval_loop.py --tag wired --backend transformers
```

**默认跑的是哪份推理脚本**：`deploy/participant/run_inference.py` —— **我们要提交的那棵树**，
带接线层，会读 `prompts/` 与 `rag/`。`GivenData/participant/` 那份是官方原件、**没有**接线层，
只在做裸基线对照时用 `--inference-script official` 显式指定。依据是主办方唯一执行的命令
`bash /root/participant/start.sh TEST_FILE RESULT_DIR`（`GivenData/participant/README.md:80-86`，
「确保脚本位置无更改」）——跑的是容器里那份。跑分前工具还会 `probe_wiring()` 问接线层自己走了哪条路，
把答案写进日志，并在台账的 `script` / `prompts` 两列留证。

> **这两个列并非每行都有值**，别把它当成台账能自证出处：只有**真的跑了推理**的行才有；
> `--skip-inference` 重新打分的行写 `(skip)`（产物来自文件，工具无法替它声明出处）；
> `9b54fae` 之前的旧行在列头迁移时一律补 `n/a` —— **已跑完的 E1 基线就是这种**，
> 它的出处记在 `experiments.md`。另外 `runs/` 被 `.gitignore` 覆盖，
> **台账与各次产物都不进仓库**。

`tools/eval_loop.py` 一步做完四件事：

1. 从 `val_samples.jsonl` 只取 `id/conversation_id/target_user_turn_id/history`
   写成 `runs/eval-<tag>/val_input.jsonl`，**并断言输出里不含 `target`**（防把答案喂给模型）；
2. 用环境变量覆盖调官方 `run_inference.py`；
3. 调 `src/cli.py validate` 做提交格式校验；
4. 调 `src/cli.py eval` 打分，并把一行追加进 `runs/leaderboard.tsv`。

> **必须用 `.venv\Scripts\python.exe` 运行它**：脚本用 `sys.executable` 拉起官方推理脚本，
> 用系统 Python 跑会落到没有 torch 的解释器上，报「共享服务没起来、或 Python 版本没有对应轮子」。

常用开关：`--backend openai_compatible --base-url http://<host>:8000/v1 --api-model <name>`
（指向共享推理服务）、`--inference-script {wired|official|<路径>}`（默认 `wired`：我们要提交的那棵树；
`official` 是官方裸基线）、`--skip-inference --predictions runs/xxx.jsonl`（只打分不推理）、
`--tokenizer jieba`、`--include-memory-refs`、`--limit/--offset`。

## 五、自测与规范检查

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # 只装 pytest，镜像不需要
.venv\Scripts\python.exe -m pytest
# 期望：162 passed, 14 subtests passed（数字随开发增长，关键是 passed 且 0 failed）

# 不要用 `python -m unittest discover -s tests` 代替：它只收集到 144 条并报 "OK"，
# 而用 pytest fixture（tmp_path / capsys / monkeypatch）写的 18 条它根本收集不到 ——
# 「测试全绿」在 unittest 下是局部为真。2026-10-17 实测就是这个差值。

.venv\Scripts\python.exe docs\reports\audit_code.py
# 代码规范性巡检：模块 docstring / 类型标注 / 写死路径 / 中文标识符 /
# 裸 except / 重复代码块。只读不写，逐项输出证据（file:line）。
# 完整结果落到 docs/reports/code-review.audit.json；结论见 docs/reports/code-review.md。
```

**改动推理或融合逻辑后必须再跑一遍**，并在群里说明验证步骤。

## 六、打包镜像

**本机跑不了 `docker build`**（Windows 没有 WSL2，Docker Desktop 起不来；算力机是 k8s pod，
`no docker` / `no podman` / 无 `docker.sock`）。所以分两条路：

### 路线 A（推荐）：GitHub Actions 一键构建

```bash
gh workflow run build-image.yml -f team_name=OGC -f push_image=false
gh run list --workflow=build-image.yml --limit 1
gh run view <run-id> --log              # 作业完成后才可取日志
```

`.github/workflows/build-image.yml` 在 `ubuntu-latest` 上依次做：
释放磁盘 → `tools/fetch_model.py` 拉权重（5.76 GB，带断点续传）→ `bash deploy/build.sh`
（组装 + `docker build` + `docker run --network none` 跑 `offline_check.sh`）→ 打印镜像信息。
`push_image=true` 时额外推到 `ghcr.io/<owner>/ogc-participant`。

### 路线 B：任何有 Docker 的 Linux 机器

```bash
python tools/fetch_model.py --check      # 先确认权重完整
bash deploy/build.sh                     # 组装 context + docker build + 断网自检
bash deploy/build.sh --tar               # 额外导出 deploy/dist/ogc-participant.tar
bash deploy/build.sh --stage-only        # 只组装 deploy/dist/context/，不构建
```

`deploy/build.sh` 会：

1. 检查 docker 与权重完整性；
2. 组装 `deploy/dist/context/{participant,rag,model}`（`prompts/` 或 `rag/` 缺失时补空目录并**打警告**）；
3. 权重用 `cp -al` 硬链接接入，不额外占磁盘；
4. `docker build`；
5. **自动跑一次 `docker run --rm --network none <镜像> bash /opt/offline_check.sh`**。

`deploy/offline_check.sh` 六节检查：文件齐不齐 → 权重在不在镜像里 → python/torch 可用性 →
静态扫描外网 URL 与 `pip install` → 提示词与 RAG 接线状态 → **断网真跑 3 条样例并校验产物格式**。
任一 `[FAIL]` 都是退出码 1，**这份镜像不要提交**。

> ⚠️ 第 6 节在纯 CPU runner 上很慢：3B 模型 fp32 逐条生成，3 条样例可能要十几分钟。
> 这是正常现象，**不是卡死**；作业上限 6 小时，足够。

## 七、平台侧

```bash
# 镜像内，官方唯一入口
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

产物：`RESULT_DIR/submission.jsonl` 与 `RESULT_DIR/performance_report.json`。
提交前逐条打勾 `docs/platform-log/preflight-checklist.md`。

## 八、未实测的部分（如实标注）

| 项 | 状态 | 原因 |
|---|---|---|
| `deploy/build.sh --stage-only` | **已真跑** | MSYS `sh.exe`（`portable-git/usr/bin/sh.exe`），29 个文件组装成功、行尾断言通过、`__pycache__` 清干净 |
| `deploy/offline_check.sh` | **已真跑** | 在暂存目录上 `通过 16 项，失败 0 项`（权重硬链接、语料 18,162,658 B、索引 11.7 MB） |
| `deploy/participant/start.sh` | **已在队友机器真跑** | 角色②用 Git Bash 跑通，`[wiring] prompts=emotion_profile,response`、`PASS: submission.jsonl` |
| `docker build` / 镜像体积 | **改在 CI 跑** | 见 §六 路线 A；本机与算力机都没有 docker daemon |
| 在官方平台上提交 | **未执行** | 截止前做一次预演；每队最多 5 次成功测评，失败不占次数 |
