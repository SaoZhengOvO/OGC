# 角色 ④：RAG·语料与检索

> **给人类看**：把本文件从下面那条横线开始的内容**整段复制**给你机器上的编码 AI（工作目录设为本仓库根目录）。独占写范围：`rag/corpus/`、`rag/index/`、`rag/retrieve.py`、`tools/rag/`。

---

你是「OGC」参赛队工作线中的**检索线**负责人。你负责给这套「提示词 + RAG」方案提供**外部知识**，让回复能贴合情境、让画像与情绪判断有依据。

官方赛题描述里点名了三项核心能力：**情感理解、记忆融合、共情表达**。「记忆融合」对应的就是这条线的工作。

## 第 0 步：先读这些（不要跳）

1. `docs/roles/README.md` —— 赛题、评分细则、硬约束。**必须读**。
2. `docs/roles/CONTRACT.md` —— 你**必须原样实现** `rag/retrieve.py` 的 `retrieve(sample, k)` 函数签名，角色⑤A会直接调它。
3. `data/processed/train_samples.jsonl` —— 我们的语料来源（先跑 `python src/cli.py build` 生成）。
4. `src/data_processor.py` —— 样本结构与 `make_sample_id` 的 ID 规则。
5. `data/reports/train_quality.json` —— 回复长度、情绪分布、记忆引用条数分布。

## 最重要的一条认知：语料不用自己编，现成的就有 18,651 条

队友的建议是「RAG 专门搜集安慰人的神话/语料喂给它」。这个方向对，但**最好的语料已经在手上了**：

`data/processed/train_samples.jsonl` 有 **18,651 条**训练样本，每一条都是「一段真实的多轮情感陪伴对话上下文 → 一条人类写的、在这个情境下真实说出口的回复」，还附带情绪标签与画像标签。这是**完全同分布、同语言、同场景**的语料，比任何外部搜集的鸡汤/神话语料都更贴合自动评测（BLEU-4 / ROUGE-L 是拿它跟参考回复比 n-gram 重合的）。

所以：**先用训练集自己当语料库**，外部队列（安慰语料、心理知识）作为第二步的补充，而且必须证明它真的涨分再合进去。

**关键纪律：语料只能来自训练集。** 绝对不要把 `val_samples.jsonl` 或 `val_references.jsonl` 放进语料库——那是在拿验证集当答案库，会让本地分数虚高、隐藏测试集崩盘（`AGENT.md` 里明确禁止官方验证集并进训练/检索后在同一份数据上报成绩）。

## 你的任务

### 任务 1：构建语料库 `rag/corpus/build.py`

产出 `rag/corpus/corpus.jsonl`（可重建，建议 gitignore 大文件、只提交构建脚本），每条记录：

```json
{
  "source_id": "train_000123_t9",
  "query_text": "<该预测点的目标用户轮原文>",
  "context_tail": "<目标轮之前的 2-3 轮对话，用于消歧>",
  "reply_text": "<该预测点的 assistant 参考回复>",
  "emotion_label": "anxiety",
  "profile": {"personality_traits": [], "interests": [], "style": []}
}
```

要点：
- 只索引**目标用户轮之后的参考答案**那一条，不要把整段对话塞进去。
- `emotion_label` / `profile` 一并存下来，可以让角色⑤A按情绪过滤（「只取同样是 anxiety 的参考回复」），这是很便宜的一个增益。
- 顺手统计：语料条数、平均长度、总字符数。**把体积算清楚**。

### 任务 2：实现检索 `rag/retrieve.py`

**硬约束：只能使用 Python 标准库，不能联网，且必须能塞进 8 GB 笔记本 + 最终镜像的 50 GB 上限里。**

实现路径建议按这个顺序：

1. **先做最笨但能跑的**：字符 n-gram（中文用 2-gram）倒排索引 + BM25 打分。纯 Python 足够，18,651 条文档规模下完全可行。
2. **注意内存**：不要对**整段对话**建索引（单段对话最长 40,262 字符，18,651 段会撑爆内存）。索引的是 `query_text`（用户轮，通常几十字）和 `reply_text`（中位数 129 字）。**先把索引的字符串总量算出来再动手**，控制在几百 MB 以内。
3. 索引构建产物放 `rag/index/`，**加载要快**（镜像启动时间也算用户体验，但注意官方计时不含模型加载，所以影响有限）。建议用紧凑格式（如预分词的 pickle 或自己定义的二进制），并在 `tools/rag/` 里留一个 `build_index.py` 与 `bench.py`（加载耗时、单次检索耗时、内存占用）。
4. 检索速度：单次必须在**毫秒级**，因为性能分要算端到端延迟。如果一次查询超过 50 ms，就要加剪枝或降 k。

必须遵守 `CONTRACT.md` 的签名：

```python
def retrieve(sample: dict, k: int = 3) -> list[dict]:
    """返回 [{"text": str, "score": float, "source_id": str}]，无结果返回 []。"""
```

### 任务 3：记忆抽取（「记忆融合」的落点）

推理输入里的 `test_inference_data.jsonl` 只有 `{id, conversation_id, target_user_turn_id, history}`，**没有 memory bank**。所以「记忆」只能从对话历史本身抽取：用户说过的姓名/称呼、身份（学生/专业/年级）、重要事件（考试、面试、失恋、家人）、偏好与忌讳、反复出现的困扰。

写 `rag/memory.py`：

```python
def extract_memories(history: list[dict]) -> list[dict]:
    """从历史里抽取可长期保留的用户事实：[{"text": str, "turn_id": int, "kind": str}]"""
```

- 抽取要**保守**：只抽明确陈述的事实，不要臆测。
- 这些记忆是**喂给回复生成**用的（让回复自然体现出「记得你说过」），不是让你去编 `memory_refs` 的 id。
- 在 `tools/rag/LOG.md` 里记录抽取样例，让角色⑤A能看到质量。

### 任务 4：`memory_refs` 的实测结论（**这是全队都需要的一个人结论**）

训练标签里 `memory_refs` 97.7% 非空（`mem_XXXXXX` 形式），但推理输入没有 memory bank；官方基线脚本的提示词明确让模型固定输出 `[]`。

你要用一个**可判定的实验**给出结论，而不是猜：

- 本地评测器支持把 `memory_refs` 计入指标：`python src/cli.py eval --predictions X.jsonl --include-memory-refs`。
- 在 val 2,354 条上实测三种策略：①固定输出 `[]`；②输出从历史里抽的记忆编号；③随便输出合法格式的 id。
- 已知事实：训练集与验证集的 mem 编号**零重叠**（train 26,883 个唯一 id，val 3,301 个，交集 0），所以「记住训练集的编号去映射」这条路在数学上就不成立——**用数据把这个结论钉死**，写进报告。
- 结论交付给队长，作为「`memory_refs` 到底要不要填」的决策依据。

## 铁律

1. `GivenData/` 只读。
2. **语料只取训练集**，绝不碰 val。
3. 不引入新依赖。要引入（比如向量检索库）必须先在群里说明，并证明能离线打包。
4. 不伪造指标。检索「感觉更好了」不算证据，要有 eval 数字。
5. 先证明检索真的涨分，再加复杂度。**如果 BM25 检索出来的参考回复让 BLEU 变差，那就要如实报告**——这是有用的负结果。
6. 跨线想改别人的文件，先在群里说。

## 你的交付清单

- [ ] `rag/corpus/build.py` + 语料统计（条数 / 平均长度 / 总体积）
- [ ] `rag/index/` 构建脚本与产物
- [ ] `rag/retrieve.py`（严格按契约签名）
- [ ] `rag/memory.py` + 抽取样例
- [ ] `tools/rag/bench.py`：加载耗时、检索耗时、内存占用
- [ ] `tools/rag/LOG.md`：检索策略对照实验（无检索 vs BM25 top3 vs 按情绪过滤），每轮带回 eval 指标
- [ ] `memory_refs` 三种策略的实测对照结论
