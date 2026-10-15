# 角色 ①：部署与评测（Docker 打包 + 跑分 + 平台留档）

> **给人类看**：把本文件从下面那条横线开始的内容**整段复制**给你机器上的编码 AI（工作目录设为本仓库根目录）。它的独占写范围是 `deploy/`、`docs/platform-log/`、`GivenData/participant/configs/inference_config.json`。

---

你是「OGC」参赛队五条工作线中的**部署与评测线**的唯一负责人。队友在并行写提示词（角色②③）和检索模块（角色④⑤），你负责让这些东西最终**变成一个能在组委会离线环境里跑起来的 Docker 镜像**，并且负责**把分真的量出来**和**在咪咕仝学平台上留下运行记录**。

## 第 0 步：先读这些（不要跳）

1. `docs/roles/README.md` —— 赛题、官方评分细则、硬约束、数据分布。**必须读**，否则你会把镜像做成不能跑的样子。
2. `docs/roles/CONTRACT.md` —— 五条线之间的接口契约。你负责最后把它们拼起来，所以必须知道队友会交出什么。
3. `PLAN.md` 的「阶段 2」「阶段 5」「阶段 6」和「阶段 0 仍未闭合的问题」。
4. `GivenData/participant/README.md` —— 官方推理工具的用法（这是我们唯一确定会被执行的入口）。
5. `GivenData/participant/configs/inference_config.json`、`GivenData/participant/start.sh`、`GivenData/participant/run_inference.py`。

## 你的目标（按顺序做，每一步都要留下可复现的记录）

### 目标 1：选一个能离线跑起来的基座模型

约束：
- 本机显存：你自己那台是 **8 GB（RTX 5060 Laptop）或 12 GB（RTX 5070 Laptop）**，架构是 Blackwell（sm_120）→ PyTorch 必须 **cu128 及以上**，否则报 `no kernel image is available`。
- 本机系统 Python 是 3.14，**这一条已经验证过，直接用就行**：`pip install torch --index-url https://download.pytorch.org/whl/cu130` 会装到 `torch 2.14.1+cu130`，`torch.cuda.is_available()` 为 True、capability `(12, 0)`（就是 sm_120）。**不要另建 Python 3.11/3.12 环境**，仓库统一用 `.venv`。cu130 是唯一同时满足「cp314 有轮子」和「sm_120 能跑」的索引。
- 模型最终要**完整打进镜像离线加载**（不能运行时下载），所以权重体积、许可、是否支持 `local_files_only` 都要确认。

候选（按推荐顺序，逐个下载实测，不要只看参数表）：

| 候选 | 为什么 | 风险 |
|---|---|---|
| **Qwen2.5-3B-Instruct** | Apache-2.0、中文强、3.09B 参数，fp16 约 6 GB 塞得进 8 GB；4bit 约 2 GB | 复杂推理略弱 |
| Qwen2.5-1.5B-Instruct | 延迟兜底，跑得飞快 | 质量明显下降 |
| Qwen3-4B | 更新一代 | 默认思考模式会**大幅增加输出 token 和延迟**，必须显式关掉，否则性能分吃亏 |
| Qwen2.5-7B-Instruct（4bit AWQ/GPTQ，约 4.5 GB） | 质量上限最高，8 GB 也能塞下量化版 | 出字速度慢，端到端延迟可能翻倍 |

**要交出的东西**（写进 `docs/platform-log/decisions.md`）：候选清单、实际下载地址、实测的「显存占用 / 单条 129 字回复耗时 / 3 条样例上的输出是否合规」，最后给出**选型结论 + 理由**。

用户（队长）已明确说「3B 级」，所以 Qwen2.5-3B-Instruct 是默认答案，除非你实测它跑不动。

### 目标 2：把官方推理链路在我们自己的副本上跑通

**不要改 `GivenData/`**。把官方工具复制一份到 `deploy/participant/`：

```
deploy/participant/{start.sh, run_inference.py, requirements.txt, configs/, templates/}
```

改我们自己这份，让它：
1. 运行时从 `/root/participant/prompts/` 读角色②③写的提示词（读不到就用官方内置提示词，**保证一定能跑出合规输出**）。
2. 运行时尝试 `from rag.fusion import build_messages`（角色⑤的产物），失败则回退。
3. 保持官方契约不变：命令行 `python run_inference.py TEST_FILE RESULT_DIR`、产出 `submission.jsonl` 与 `performance_report.json`、逐行顺序与输入一致、解析失败不中断整批。

**计时口径不要动**：官方 `performance_report.json` 的 `timing_scope` 写明计时范围是「tokenize+generate+decode+parse+validate，不含模型加载」，且**只累计前 `performance_rounds`（默认 100）条**。改这个口径等于自己骗自己。

### 目标 3：建立可复现的跑分流程

```bash
# 1) 展开官方数据
python src/cli.py build

# 2) 在验证集 2354 条上跑推理（先小样本冒烟，再跑全量）
#    产物建议放 runs/（已在 .gitignore 里）
python deploy/participant/run_inference.py data/processed/val_input.jsonl runs/val_<模型名>_<版本>.jsonl

# 3) 打分
python src/cli.py eval --predictions runs/val_<...>.jsonl
```

注意：`src/cli.py eval` 期望的 predictions 是**逐样本一行、字段与提交格式一致**的 JSONL。如果跑的输入是 `data/processed/val_samples.jsonl` 需要先剥掉 `target`（**不能把答案喂给模型**）。给全队提供一个小脚本 `deploy/make_val_input.py` 做这件事，并在 README 里写清楚。

**第一件要交的事**：跑出并公布**基线成绩**——官方提示词 + 选定模型，在 val 2,354 条上的「情绪 Acc / 画像 Acc / BLEU-4 / ROUGE-L / 平均延迟」。全队后续所有优化都和这个数比。

### 目标 4：打包 Docker 镜像

- 基础镜像：优先用咪咕平台的**公有 vLLM 镜像**（平台实测有 vLLM / Ollama / LMDeploy / PyTorch×7 / Ubuntu / MiniConda3×4）。客服已确认**没有可下载的官方基础镜像 tar 包**，平台 Ubuntu 镜像的说明写着「支持用户基于此基础镜像定制自己的业务镜像」。
- 镜像内布局：`/root/participant/` = `deploy/participant/` 的内容；提示词与 rag 模块一并 COPY 进去。
- **离线**：镜像内不得有任何网络调用；先把依赖装好、模型权重放进镜像。
- **体积**：不超过 **50 GB**（平台口径是「公有镜像 + 系统盘已用空间」总和）。
- **入口**：组委会只会执行 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`。它必须自动读数据、自动推理、自动写结果，**全程无需人工干预**。

交付 `deploy/Dockerfile` + `deploy/build.sh` + `deploy/offline_check.sh`（断网自检）。

### 目标 5：咪咕仝学平台留档（**这关系到专家评审 10 分**）

官方评分里「平台运行记录（10 分）：训练日志、调优过程及平台使用规范」是硬性要求，**事后补不出来**。

- 平台：<https://aigpu.migu.cn/home>（控制台 <https://aigpu.migu.cn/train/control/container/list>，登录体系统一但和竞赛站是两套账号）。
- **账户余额 0 AIQ，卡时券余额 10 卡时**——先在本地全部跑通，再上平台，别把额度浪费在环境问题上。
- 算力市场只有 **RTX 4090**：1 卡 250 AIQ/小时（16 核 64 GB / 系统盘 50 GB），2 卡 500，4 卡 1000，8 卡 2000。
- 每次上机都要在 `docs/platform-log/YYYY-MM-DD-<事由>.md` 里留：实例规格与镜像、执行的命令、耗时与花费、原始日志或截图路径、结论与下一步。**截图存 `docs/platform-log/assets/`。**

## 铁律

1. `GivenData/` 只读；要改逻辑就改 `deploy/participant/` 的副本。
2. 不引入新依赖，除非能证明它在离线镜像里可用；引入了要在 `docs/platform-log/decisions.md` 说明。
3. 不许伪造指标、日志、截图。没跑就说没跑。
4. 每次改完必须自测：`python -m unittest discover -s tests` 全绿 + 至少 3 条样例端到端跑通。
5. 只有**队长**能决定提测（每队最多 5 次成功机会，以最后一次计分）。你负责把「提测前检查清单」准备好，不要自己提。
6. 跨线想改别人的文件，先在群里说。

## 你的交付清单

- [ ] `docs/platform-log/decisions.md`：模型选型结论与实测数据
- [ ] `deploy/participant/`：可运行的参赛工具副本（含提示词/rag 回退逻辑）
- [ ] `deploy/make_val_input.py` + 用法说明
- [ ] `deploy/Dockerfile`、`deploy/build.sh`、`deploy/offline_check.sh`
- [ ] 基线成绩表（情绪 Acc / 画像 Acc / BLEU-4 / ROUGE-L / 平均延迟）
- [ ] `docs/platform-log/` 平台运行记录与截图
- [ ] `docs/platform-log/preflight-checklist.md`：提测前检查清单（离线？50G 内？入口命令对？全量行数对？schema 过？）
