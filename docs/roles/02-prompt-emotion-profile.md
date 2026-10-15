# 角色 ②：提示词·情绪与画像

> **给人类看**：把本文件从下面那条横线开始的内容**整段复制**给你机器上的编码 AI（工作目录设为本仓库根目录）。独占写范围：`prompts/emotion_profile.md`、`tools/emotion_profile/`。

---

你是「OGC」参赛队工作线中的**情绪与画像线**负责人。你的产物是一份**纯文本提示词** `prompts/emotion_profile.md`，它会被拼进系统提示，决定模型输出的两个字段：`emotion_label`（16 类单选）与 `user_profile`（三维标签数组）。

**这两个字段对应自动评测的 40 分**（情感识别准确性 25 分 + 用户画像推断 15 分），是全部评分项里最大的一块。你的工作就是用提示词尽可能把这块吃掉。

## 第 0 步：先读这些（不要跳）

1. `docs/roles/README.md` —— 赛题、评分细则、硬约束、字段取值约束。**必须读**。
2. `docs/roles/CONTRACT.md` —— 你必须遵守的接口契约（提示词格式、评测口径、基线定义）。
3. `configs/label_space.json` 与 `src/label_space.py` —— **官方词表的唯一真相**。不要手抄词表，用脚本从它生成。
4. `src/evaluate.py` —— 看清楚你的两个字段**到底怎么算分**（这决定了你的策略）。
5. `data/reports/train_quality.json`、`data/reports/val_quality.json` —— 类别分布与画像空值率。
6. `GivenData/participant/run_inference.py` —— 官方内置提示词长什么样，以及输出 JSON 是怎么被解析与兜底的。

## 关键：先搞懂算分方式，再动手写提示词

打开 `src/evaluate.py` 确认两件事，这会直接决定你的策略：

1. **情绪 = Accuracy**。单标签，只有一个答案对。所以要么最大概率猜对，要么别猜。训练集分布极度倾斜：`anxiety` 占 34.0%，`relaxed` 15.1%，而 `care` 只有 0.4%、`surprise` 只有 0.3%。**一个无脑输出 anxiety 的模型也有约 0.34 的情绪分**——你的及格线是这个数，你的目标是把小类也捞回来。
2. **画像 = 三个维度各自「集合精确匹配」准确率的平均**（见 `evaluate.py` 的 `SetAccumulator`）。这意味着：
   - 多写一个错标签 → **整个维度判错**，比不写还亏。
   - 训练集里 **36.6% 的预测点三维全空**（`data/reports/train_quality.json` 的 `empty_profile_samples`）。空数组是常见且正确的答案。
   - **所以策略必须是「有强证据才写」，而不是「尽量多写」。** 这一条是本项目最容易踩的坑。

还有一个容易搞错的事实：**画像标签是逐预测点的，不是对话级恒定的**。前 4000 段对话里有 1299 段的画像在对话内发生了变化，其中 978 段是**非单调**的（先加了标签后面又去掉）。所以不能「第一轮推断一次就缓存到底」，必须每一轮都重新判定。

## 你的任务

### 任务 1：写出 `prompts/emotion_profile.md`

**格式要求**（来自 `CONTRACT.md`）：纯文本，不要 Markdown 标题/表格/代码围栏，用中文短句写规则。它会原样拼进系统提示，模型会照着它的风格输出。

必须包含：
- 输出字段的说明：`emotion_label` 从 16 类里选**恰好一个**；`user_profile` 只含 `personality_traits` / `interests` / `style` 三个数组。
- 16 类情绪的**判定规则**（这是核心，见下面的方法论）。
- 三维画像的**判定规则 + 证据门槛**：什么才算「充分证据」；没有证据时输出空数组；三维互相独立，可以只填其中一维。
- 词表本身（**用脚本从 `configs/label_space.json` 生成**后贴进去，别手打，别写成中文）：

  ```bash
  python tools/gen_label_block.py --part profile    # 打印画像取值块
  python tools/gen_label_block.py --part emotion    # 打印情绪取值块
  ```

  手抄词表出错的方式是**静默的**：`src/label_space.py` 的 `normalize_*` 会把词表外取值丢掉，模型写了错标签只会表现为「那一维全空」，从指标上看不出是自己拼错了。

### 任务 2：方法论（不是让你照抄，是让你有起点）

情绪判定的可操作思路：
- 先找**明说的情绪词**（「我好焦虑」「开心」「难过」）→ 直接映射到最近的那一类。
- 没有明说时，看**情境**：时间压力/考试/绩点 → anxiety；被否定/被冷落 → sadness 或 loneliness；深夜独自一人谈心事 → loneliness 或 helplessness；已发生且无法改变 → sadness；未来不确定 → anxiety。**anxiety / sadness / loneliness / helplessness 这四个的区别是本任务最容易混淆的地方，值得专门写规则。**
- `mixed` 只用于**同时存在两个相反极性且都明确**的情况（如又开心又害怕）。不要把它当垃圾桶。
- `neutral` 只用于**真的没有情绪信息**的陈述（如问事实性问题）。不要因为读不出来就丢 neutral——那会退化成无脑 anxiety。
- 小类（`care` `surprise` `pride` `disgust` `joy`）在训练集里样本极少，提示词里要专门给它们各自一条清晰的触发条件，否则模型永远不会输出它们。
- 判定依据是**用户最后一条消息**（在明确的情境下可结合上一轮），不是整个对话的平均情绪。

画像判定的可操作思路：
- `personality_traits` 反映**稳定的**倾向，一次情绪波动不算证据。要能引用到具体表述（如「我总是提前做计划」→ `high_conscientiousness`）。
- `interests` 要有**明确提到或反复涉及**的兴趣点。
- `style` 反映用户**说话方式**（句子长短 → `brief` / `detailed`；用不用表情 → `emoji_user`；理性 vs 感性表达 → `rational` / `high_emotional_expression`）。style 比前两维更容易从样本本身看出来，通常是最好拿分的一维。
- **允许并鼓励输出空数组**。写清楚「没有证据就留空」。

### 任务 3：建立迭代闭环并交数据

你要能**只改提示词、其它都不变**地反复跑分。流程：

```bash
# 一次性准备
python src/cli.py build                 # 展开官方数据 → data/processed/
# 造不含答案的验证输入（若 deploy/make_val_input.py 还没就绪，自己写一个放 tools/emotion_profile/）
python tools/emotion_profile/make_val_input.py

# 每一轮
python deploy/participant/run_inference.py data/processed/val_input.jsonl runs/val_<你的版本>.jsonl
python src/cli.py eval --predictions runs/val_<你的版本>.jsonl
```

再自己写一个 `tools/emotion_profile/report.py`，把 `eval` 的 `--per-sample-out` 结果加工成：
- 情绪**混淆矩阵**（16×16）与**每类 P/R/F1**，重点看 anxiety/sadness/loneliness/helplessness 之间怎么混的；
- 画像**逐维度**准确率、以及「该空却填了 / 该填却空了」两类错误各占多少。

这些数字是你写规则的**唯一依据**。不要凭感觉调提示词。

**基线**：先复现角色①公布的基线成绩（官方提示词 + 选定模型），把你每一轮的改动和它对比，写进 `tools/emotion_profile/LOG.md`：改了什么规则 → 哪个指标动了多少 → 是否保留。一次只改一处。

## 铁律

1. `GivenData/` 只读。
2. 不引入新依赖（只用标准库）。
3. 不伪造指标。每一轮都要有真实的产物文件路径。
4. `prompts/emotion_profile.md` 是**契约产物**，只改这一个文件就能切换你的方案；不要把逻辑散在别处。
5. 不要为了让 val 好看而写「针对特定样本」的规则（比如把某几个 id 硬编码）。官方测试集是隐藏的，过拟合等于零分。
6. 跨线想改别人的文件，先在群里说。

## 你的交付清单

- [ ] `prompts/emotion_profile.md`：可用的提示词（纯文本、含词表、含规则）
- [ ] `tools/emotion_profile/make_val_input.py`、`report.py`：造输入 + 出混淆矩阵/逐维度报告
- [ ] `tools/emotion_profile/LOG.md`：每轮改动的实验记录（改了什么 → 指标变化 → 是否保留）
- [ ] 一份结论：情绪 Acc 相对基线提升了多少、画像三维各提升了多少、还差在哪里
