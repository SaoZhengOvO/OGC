# 实验设计与结果

> 作者：角色⑤B（专家评审文档）。对应官方评分项「专家评审 — 方案完整性（10 分）」。
> **写作纪律**：每个数字都必须能指回具体产物文件；没跑的实验写「未执行」并说明原因。
> 基线以下的所有对比都是**相对本地评测器**的，官方未公布评分公式，本地数字只是代理指标。

## 〇、测量口径（先读这一节，否则会读错数字）

| 项 | 口径 | 出处 |
|---|---|---|
| 情绪 Acc | 单标签准确率；**空值计错**，不是跳过 | `src/evaluate.py` |
| 画像 Acc | 三维（personality_traits / interests / style）**集合精确匹配**准确率的算术平均 | 同上 |
| 多写一个错标签 | 该维**直接归零**（不是扣分） | 同上 |
| BLEU-4 / ROUGE-L | char 分词（中英混合，中文按字、英文按词），零依赖 | 同上 |
| BERTScore | **未接入**：需要神经网络编码器，离线环境需预置本地模型 | 评估器返回 `{"computed": false}`，不输出假分数 |
| memory_refs | 默认**不计入**画像主指标（官方口径待确认），可用 `--include-memory-refs` 单看 | `tools/eval_loop.py` |

### ⚠️ 三个会读错数的地方

1. **部分样本实验（`--limit N`）的分母是全量 2,354**。
   缺预测的样本按 0 分计入，所以 `--limit 5` 跑出来的「情绪 Acc 0.0013」不是模型崩了，
   而是 `0.6 × 5/2354` 的稀释结果。**看部分实验请只看相对变化，或直接比较 `n_comparable`
   相同的那几行**。这一点已作为待办提给角色①（评测归属方），建议加一列「可比对口径」。
2. **本地评测器不是官方评分器**。官方未公布各指标权重与延迟满分阈值，
   本文件的数字只能用于「同一份数据、同一套代码下 A 方案是否比 B 方案好」的判断。
3. **E0 / E1 量的是哪一棵树**（2026-10-16 补记，`9b54fae`）。
   `tools/eval_loop.py` 在 `9b54fae` 之前的默认推理脚本是 `GivenData/participant/run_inference.py`
   —— 那是**官方原件，没有接线层**，读不到队友的 `prompts/` 与 `rag/`。所以 E0 与 E1 量到的都是
   「官方提示词 + 3B 模型」，**不是**我们要提交的那棵树。主办方唯一执行的命令是
   `bash /root/participant/start.sh TEST_FILE RESULT_DIR`（`GivenData/participant/README.md:80-86`），
   跑的是容器里那份，即 `deploy/participant/`。

   这对 E0 / E1 本身不算错——我们本来就需要一条裸基线——**错的是台账上看不出这件事**：
   一行数字分不出「官方裸基线」还是「带队友提示词与 RAG 的那棵树」。`9b54fae` 的修法是：

   - `--inference-script {wired|official|<路径>}`，默认 `wired`；要裸基线就显式写 `official`
   - 台账加 `script` / `prompts` 两列，每行数字从此自带「量的是哪棵树」的证据
   - 开跑前先问接线层自己「这一趟走了哪条路」，把答案写进日志（`probe_wiring()`）

   两个连带的坑一并处理了：`deploy/participant/model/` 不存在（5.9 GB 不复制第二份），而两份
   `inference_config.json` 都写 `"./model"` 且以**工具目录**为基准 —— 换脚本而不同时指定模型，
   模型就会找不到；`deploy/participant/prompts/` 本地同样不存在（镜像里由 `deploy/build.sh` 现拷），
   不显式指回仓库根的 `prompts/` 就会**静默**退化成官方提示词，量到的数不对应提交的那棵树。

## 一、实验环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop，12,231 MiB，sm_120（Blackwell） |
| PyTorch | 2.14.1+cu130（`torch.version.cuda = 13.0`，`cuda.is_available() = True`） |
| transformers / accelerate | 5.19.0 / 1.15.0 |
| 基座模型 | `Qwen/Qwen2.5-3B-Instruct`，本地目录 `GivenData/participant/model/`，5.8 GB，2 分片 |
| 生成参数 | `do_sample=false`，`max_new_tokens=1200`（`configs/inference_config.json`） |
| 数据 | train 18,651 条 / val 2,354 条（清洗后口径，见 `data/reports/`） |

## 二、已执行的实验

### E-1 管道自证：拿真值当完美提交

**目的**：证明打分器没有写反、校验器不是「永远通过」。

```bash
.venv/Scripts/python.exe tools/eval_loop.py --tag selftest-perfect \
    --skip-inference --predictions runs/_selftest_perfect.jsonl
```

**结果**（`runs/leaderboard.tsv` 第 1 行，2026-10-15 22:36:38，n=2354）：

| 指标 | 值 |
|---|---|
| 情绪 Acc | 1.0000 |
| 画像 Acc（三维均值） | 1.0000（三维各 1.0000） |
| BLEU-4 | 1.0000 |
| ROUGE-L | 1.0000 |
| memory_refs micro-F1 | 1.0000 |

**对照**：注入 5 类错误的 `runs/_selftest_broken.jsonl`（2355 条，含 1 条重复 id）
被校验器**逐条抓出 7 处问题**（字段名拼错、画像枚举越界、memory_refs 正则不符、
id 格式错、id 重复）。管道可信。

### E0 链路冒烟：真模型跑通全流程

**目的**：验证「加载权重 → 生成 → 解析 → 校验 → 打分」整条链在不依赖任何队友产物时能跑通。

```bash
.venv/Scripts/python.exe tools/eval_loop.py --tag smoke --limit 5 --backend transformers
```

**量的树**：官方原件 `GivenData/participant/`（**无**接线层，读不到 `prompts/`）——见 §〇 第 3 条。

**结果**（2026-10-15 23:28:49，`runs/eval-smoke/`）：

| 指标 | 值 |
|---|---|
| 端到端单条延迟 | 平均 **2,932 ms**（中位 2,792 / P95 3,462 / 最小 2,621 / 最大 3,462） |
| 解析失败数 | **0**（`parse_failure_count = 0`，5 条全部产出合法 JSON） |
| 提交格式校验 | 通过（`空情绪 0 / 空回复 0 / 画像三维全空 4`） |
| 模型加载 | 434 个权重张量，约 6 秒（**不计入**官方计时口径） |

**结论**：
1. 官方基线的「官方提示词 + 3B 模型」在本机可以产出**结构完全合规**的输出，解析兜底一次都没用上。
2. 单条约 2.9 秒 —— 这是**性能分的现实起点**。官方口径只统计前 100 条的
   `tokenize+generate+decode+parse+validate`，模型加载不计入。
3. 5 条里 4 条画像三维全空。这与训练集分布一致（36.6% 的预测点三维全空），
   但也提示模型倾向于「少写」；这正是角色②要解决的问题。

### E1 官方基线（全量）

```bash
.venv/Scripts/python.exe tools/eval_loop.py --tag baseline --backend transformers
```

**量的树**：官方原件 `GivenData/participant/`（**无**接线层）——见 §〇 第 3 条。这是一条**裸基线**，
不是带队友提示词与 RAG 的那棵树；后者要等 ②③④ 交付、并用默认的 `--inference-script wired` 另跑一次。

**状态：运行中**（2026-10-15 启动）。预计约 2 小时（2,354 × 2.9 s）。
结果将写入 `runs/leaderboard.tsv` 的 `baseline` 行与 `runs/eval-baseline/score.json`。
**本文档在本行数字落盘后必须更新**，不得留空占位。

## 三、未执行的实验（如实列出，含原因）

| 实验 | 状态 | 原因 |
|---|---|---|
| RAG 检索对 val 指标的影响 | **未执行** | 依赖角色④的 `rag/retrieve.py` 与语料产物，尚未交付 |
| 提示词 v1 / v2 的 A/B | **未执行** | 依赖角色②③的 `prompts/*.md`，尚未交付 |
| `memory_refs` 三种策略对照（固定 `[]` / 抽历史记忆编号 / 随机合法 id） | **未执行** | 依赖 `rag/memory.py`（角色④）；**已知 train 的 26,883 个唯一 id 与 val 的 3,301 个交集为 0**，随机或记忆编号路线预期在 val 上必然为 0，用数据即可先验否定 |
| 上下文预算消融（关掉截断 / 关掉融合） | **未执行** | 开关已就绪（`PARTICIPANT_FUSION=0`、`PARTICIPANT_MAX_CONTEXT_CHARS`），等 E1 有基线后跑 |
| 4bit 量化对延迟与指标的影响 | **未执行** | 需要额外依赖，且全队铁律「不引入新依赖」，需先在全队讨论 |
| 7B 模型的对照 | **未执行** | 工期与 8 GB 显存约束下不现实（12 GB 机器上 4bit 约 4.5 GB 可跑，但延迟翻倍） |

## 四、负结果（试过但没采用的方案）

> 这一节是加分项：把死路记下来，专家评审能看到我们真的做了实验设计。

1. **「目标回复前 30 字出现在历史里」作为泄漏判据** —— 误报 41 条。
   情感陪伴场景里 `嗯，我在听。`/`死心。` 这类 2~4 字口头回应**本就高频重复**，
   重合率天然 100%。最终判据改成双门槛：短重复要求
   `len ≤ 40 且长度差 ≤ 8`，真泄漏要求 `len ≥ 60 且重合率 ≥ 0.60`，
   结果从 41 条收敛到 **2 条真泄漏**（`train_002907_t15`、`train_005955_t15`）。
2. **「训练集记忆编号能被模型学会」** —— 不成立。train 与 val 的 memory id
   **零重叠**（26,883 vs 3,301），靠记住编号预测 `memory_refs` 在验证集上必然为 0。
3. **全量跑 O(n·m) 最长公共子串** —— 18,653 条上超过 120 秒。
   改为先过两道廉价门控（字符集合交集比例、长度倍数），再跑 DP。
