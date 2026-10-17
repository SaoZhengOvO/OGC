# tools/ —— 手工工具台账

这里的脚本**不在主流程里自动跑**（主流程是 `tools/bootstrap.py` + `tools/eval_loop.py`），
它们是评测、对照、救火时手动调用的。没有这份台账的时候，读者只能靠文件名猜，猜错就
会重复踩已经踩过的坑——所以每一节都写清「什么时候用」和「为什么会有这个工具」。

跑之前的通用前提：`.venv` 已建好（`tools/bootstrap.py`），仓库根有 `data/processed/`。

---

## 评测主链路

### `eval_loop.py`

一键闭环：造验证输入 → 跑推理 → 格式校验 → 本地打分 → 写实验台账。

```bash
python tools/eval_loop.py --tag my-run --limit 300
python tools/eval_loop.py --tag arm-B --limit 300 --prompt-dir /path/to/other/prompts
python tools/eval_loop.py --tag rescore --skip-inference --predictions runs/xxx/submission.jsonl
```

产物固定在 `runs/eval-<tag>/`：`val_input.jsonl`、`submission.jsonl`、`score.json`、
以及追加到 `runs/leaderboard.csv` 的一行。

**它现在会做的事**（都是被真实事故逼出来的，不是提前设计）：

- 格式校验退出码非零就中止，不再带着非法 submission 进打分。要绕过必须显式
  `--skip-validate`。
- 打分前按 id 去重（`dedupe_submission`），保留最后一次并留 `.dedupe.bak`。
  2026-10-16 两个推理进程并发写同一个输出目录，2545 行只剩 2354 个唯一 id、
  164 个 id 内容冲突，情绪分 0.4146 → 0.1619，而 schema 校验一条都没报。
- `--limit` / `--offset` 时会把真值一起收窄成 `val_references_subset.jsonl`，
  否则分母永远是完整的 2354 条，分数会被严重低估。

### `score_submission.py`

对**已经存在**的一份 `submission.jsonl` 单独打分，不跑推理、不改文件。

```bash
python tools/score_submission.py --submission runs/xxx/submission.jsonl --references data/processed/val_references.jsonl
```

和 `eval_loop` 内置的打分是同一套口径（`src/evaluate.py`），区别是它把
「去重 / 行数 / 唯一 id / 冲突数」单独打出来。拿到别人给的 submission、或者怀疑
分数不对劲想复查时用这个。

---

## 补算（不重跑模型）

这一类工具存在的理由：`submission.jsonl` 里已经存了预测文本和画像，**指标变化
不需要重新推理**。全量 2354 跑一次要一个多小时，补算只要几秒到几分钟。

### `apply_profile_floor.py`

给已有 submission 补上画像。两种模式：

```bash
# floor（默认）：只填空，不覆盖模型自己给的标签
python tools/apply_profile_floor.py --submission runs/x/submission.jsonl
# force：三维一律写成先验常量，无视模型输出
python tools/apply_profile_floor.py --submission runs/x/submission.jsonl --mode force
```

⚠️ 它读写的键是 `user_profile`（跟 `deploy/participant/run_inference.py` 写盘的键
一致）。曾经这里写成 `profile`，于是补算往每行塞了个评分器不认识的键，
`score.json` 前后逐字段相同，一度被误判成「兜底无效」。`tests/test_profile_floor_sync.py`
现在会同时断言两边的键名和常量字面量，防止再漂。

### `bertscore_rescore.py`

给已有 submission 补算 BERTScore（官方口径之一，回应质量 20 分里的一项）。

```bash
HF_HUB_DISABLE_XET=1 python tools/bertscore_rescore.py --submission ... --encoder bert-base-chinese
```

`HF_HUB_DISABLE_XET=1` 不是可选的：hf-mirror 的 xet 协议会对 `bert-base-chinese`
返回 404，关掉 xet 走传统下载 11 秒就完事。

⚠️ 默认编码器不是官方那个（官方未公布具体模型），所以这个分数是**代理值**，
只用来在臂与臂之间比相对高低，不能当成官方分写进结论。

### `adjudicate_arms.py`

把「该发哪一版提示词」从叙事变成算术。读若干 `runs/eval-<tag>/score.json`，
在四种加权假设下各算一次综合分；四种都指向同一臂才输出「结论稳健」。

```bash
python tools/adjudicate_arms.py runs/eval-clean-A runs/eval-clean-B
```

⚠️ 它不处理 `--help`，把 `--help` 当路径传会 `FileNotFoundError`。

---

## 数据与模型准备

| 脚本 | 用途 |
|---|---|
| `bootstrap.py` | 一键开工：venv → 依赖 → 下模型 → 展开数据 → 冒烟测试 |
| `fetch_model.py` | 单独下基座权重（`deploy/build.sh` 也调它） |
| `gen_label_block.py` | 从官方词表生成可贴进提示词的「取值约束」文本块 |
| `make_sft_data.py` | 已处理样本 → SFT 训练数据（chat 格式） |
| `train_sft.py` | LoRA 微调，让模型直接吐评测要的那一个 JSON object |
| `assign_role.py` | 把队员认领的分工写进 `ROLES.md` 的「当前负责人」列 |

### `sft_train.py` 与 `train_sft.py` 的关系

`train_sft.py` 是**当前在用的**那个（`docs/platform-log/modelscope-amd/` 里的 MI300X
训练记录就是它跑出来的）。`sft_train.py` 是更早的一版结构化多任务实现，功能被
`train_sft.py` 覆盖，全仓无调用方。保留是因为它的数据组织方式不同（多任务头而
不是单 JSON），要做多任务变体时可以接着改；**不要**把它当成主训练脚本。

---

## 已知没有被任何代码调用的接口

审计（2026-10-17，65 文件 / 12,842 行）查出下面几个函数只有定义、没有调用方。
留着是因为它们是模块的对外接口（别人 import 时可能会用），但读者容易误以为
有自动化在跑：

- `rag/retrieve.py` 的 `warm_up()`、`corpus_stats()`
- `rag/memory.py` 的 `memory_stats()`

线上路径实际走的是 `deploy/participant/wiring.py` 里的封装，不是这些函数。
