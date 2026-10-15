# 角色分工与提示词（ROLES）

> **第一次来先看 [`START-HERE.md`](START-HERE.md)** —— 从零到跑出第一个分数的五步命令、常见坑、硬约束。
> **还不知道自己该挑哪条线？看 [`docs/roles/CHOOSE.md`](docs/roles/CHOOSE.md)** —— 每条线的活儿、需要的底子、难度风险，按强项对号入座。

本仓库是「2026 动感地带 AI+ 高校创智计划」AI 技术赛道赛题二**数字人综合情感陪伴对话模型**的参赛工程。

我们五个人各自认领一条线，**每条线配一份提示词**。用法：把自己那台机器上的编码 AI（Claude Code / Cursor / DeepSeek Harness / 本地部署的任意 agent）的工作目录设为本仓库根目录，把对应提示词**整段贴进去**，它就知道该干什么、该动哪些文件、怎么自测。

## 五条线

| 线 | 提示词 | 主责评分项 | 独占写文件范围 |
|---|---|---|---|
| ① 部署与评测 | [`docs/roles/01-deploy-eval.md`](docs/roles/01-deploy-eval.md) | 推理性能 10 分 + 平台运行记录 10 分 | `deploy/`、`docs/platform-log/` |
| ② 提示词·情绪与画像 | [`docs/roles/02-prompt-emotion-profile.md`](docs/roles/02-prompt-emotion-profile.md) | 情感识别 25 分 + 用户画像 15 分 | `prompts/emotion_profile.md`、`tools/emotion_profile/` |
| ③ 提示词·共情回复 | [`docs/roles/03-prompt-response.md`](docs/roles/03-prompt-response.md) | 回应质量 20 分 | `prompts/response.md`、`tools/response/` |
| ④ RAG·语料与检索 | [`docs/roles/04-rag-corpus.md`](docs/roles/04-rag-corpus.md) | 记忆融合（赛题核心能力） | `rag/corpus/`、`rag/index/`、`rag/retrieve.py`、`rag/memory.py`、`tools/rag/` |
| ⑤ RAG·融合与文档 | [`docs/roles/05-rag-fusion-docs.md`](docs/roles/05-rag-fusion-docs.md) | 方案完整性 10 分 + 代码规范性 10 分 | `rag/fusion.py`、`docs/reports/` |

**接口契约见 [`docs/roles/CONTRACT.md`](docs/roles/CONTRACT.md)** —— 五个人的产出的拼接方式（`rag/retrieve.py`、`rag/fusion.py` 的函数签名、提示词文件格式、评测口径）都在那里冻结了。

> 提示词文件里都写了「先读什么、动哪些文件、跑什么命令自测」。跨线只读别人目录，**不要改别人的文件**；要改就先在群里说。

## 技术路线（已定，2026-10-15）

**不微调，走「提示词 + RAG + 离线小模型」。**

- 队友结论：两天时间不够微调；量化/权重不是我们能在本机搞定的事；先选一个模型、直接上。
- 因此 `PLAN.md` 里阶段 3（SFT 多任务训练）**不再执行**，改为提示词工程 + 检索增强。
- 基座模型选型见 `PLAN.md` 阶段 2；本机 8–12 GB 显存决定了模型规模上限。

## 公共背景

读 [`docs/roles/README.md`](docs/roles/README.md)。每份提示词开头也复制了最关键的那几条硬约束，单独贴给 AI 也不会跑偏。

## 协作规约

1. **只在 GitHub issue 里讨论跨线事项**，用 `gh` CLI（见 [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md)）。
2. **`GivenData/` 是官方材料，只读**，任何情况下不修改、不删除、不覆盖（唯一的例外是角色①的 `participant/configs/inference_config.json`，那是我们自己的参赛配置）。
3. **提交前必须跑**：`python -m unittest discover -s tests`、`python src/cli.py eval --predictions <你的产物>`。
4. **不许伪造数据**：训练日志、跑分、截图、指标一律要真跑出来；跑不出来就写「未跑」。
