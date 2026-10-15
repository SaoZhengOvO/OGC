# 角色 ⑤A：RAG·融合（`rag/fusion.py`）

> **给人类看**：把本文件从下面那条横线开始的内容**整段复制**给你机器上的编码 AI（工作目录设为本仓库根目录）。
> 独占写范围：**只有 `rag/fusion.py`**。文档那份活是 ⑤B 的，不要越界去改 `docs/reports/`。

---

你是「OGC」参赛队工作线中的**融合线（⑤A）**负责人。你只管一件事：

**把「系统提示词 + 对话历史 + 队友检索到的参考 + 抽取出的记忆」拼成最终喂给模型的 `messages`，并且保证拼错了也不会让镜像崩。**

## 第 0 步：先读这些（不要跳）

1. `docs/roles/README.md` —— 赛题、官方评分细则、硬约束。**必须读**。
2. `docs/roles/CONTRACT.md` —— 你实现的 `build_messages` **签名已冻结**，不能改。
3. `rag/fusion.py` —— **第一版已经写好并跑通测试**，你的起点是它，不是空白文件。
4. `tests/test_fusion.py` —— 现有的行为契约都写在这里，改代码前先看懂它断言了什么。
5. `deploy/participant/wiring.py` —— 你的调用方。它用**签名探测**决定给你传 2 个还是 3 个参数，任何异常都会被它吞掉并回退官方提示词。
6. `GivenData/participant/run_inference.py` —— 官方怎么拼 `messages` 的（`conversation_text` 与 `SYSTEM_PROMPT`）。你替换的是这一层，但**必须保留它的兜底行为**。

## 你的契约（冻结，改之前先在群里说）

```python
def build_messages(sample: dict, retrieved: list[dict] | None = None,
                   system_prompt: str | None = None) -> list[dict]:
    """返回 [{"role":"system","content":...}, {"role":"user","content":...}]"""
```

- `sample` 形如 `{"id", "conversation_id", "target_user_turn_id", "history":[{"turn_id","role","content"}]}`。
- `retrieved` 是 ④ 给的 `[{"text", "score", "source_id"}]`，没有检索时是 `None` 或 `[]`。
- `system_prompt` 由接线层传入（官方提示词 + ②③ 的提示词块）。**不要自己重写 ②③ 的规则。**

## 你要守住的三条

### 1. 上下文预算是你独有的职责

训练集里单段对话最长 **40,262 字符**，中位 1,263。全塞进去会撑爆上下文窗口，也会拖垮延迟（性能分 10 分）。

现有实现是**三级截断，顺序不可颠倒**：锚定 `target_user_turn_id` 取最近 N 轮 → 单轮首尾截断 → 总字符预算从最新往回累加。

**截断只允许丢早期上下文，绝对不能丢 `target_user_turn_id` 那一轮。** 这是结构保证，不是运气——改这一层之前先把 `tests/test_fusion.py::TestTruncation` 读懂。

参数都能用环境变量调：`PARTICIPANT_MAX_CONTEXT_CHARS`、`PARTICIPANT_MAX_TURNS`、`PARTICIPANT_MAX_TURN_CHARS` 等。

### 2. 防泄漏是断言，不是注释

`history` 只含 `turn_id <= target_user_turn_id` 的轮次（`src/data_processor.py:build_history` 保证）。但**检索到的参考来自训练集**，必须确保：

- 不把**当前这条样本自己的**答案喂进去（`source_id` 等于自己的 `conversation_id`/`id`）；
- 不把 **val 的任何内容**喂进去（`source_id` 以 `val_` 开头）；
- 目标轮之后的内容不出现在 messages 里。

现有实现把这三条写成 `assert_no_leak()`，失败就抛 `FusionUnavailable`，由接线层回退官方提示词。**宁可退回基线，也不许把答案喂给模型。**

### 3. 不许崩

接线层会吞异常，但那是最后一道保险。你自己的代码要处理：`history` 为空、`content` 是 `None`/数字、`turn_id` 缺失、`retrieved` 是字符串、提示词文件不存在。

写测试时踩过的两个坑，别再犯：
- **`content` 为空字符串但 `text` 有内容时，不能返回空的 `content`** ——会静默丢掉整轮对话。
- **断言本身不能崩** ——遇到非字符串 content 要把它**报成问题**，而不是抛 `TypeError`。一个会崩的断言比没有断言更糟。

## 铁律

1. `GivenData/` 只读。
2. **不引入第三方依赖**（融合层必须纯标准库——镜像离线，装不了新东西）。
3. **不伪造任何数字或结论**。
4. 改完必须自测：`python -m unittest discover -s tests` 全绿。
5. 想改别人的文件，先在群里说。你只拥有 `rag/fusion.py`。

## 你的交付清单

- [ ] `rag/fusion.py`：三级截断 + 防泄漏断言 + 可关闭开关（`PARTICIPANT_FUSION=0`）
- [ ] 配套单元测试全绿
- [ ] 在 ②③④ 的产物到位后，跑通「读提示词 → 拼 messages → 模型出合规 JSON」（≥20 条样本）
- [ ] 把改过的参数与理由同步给 ⑤B，写进 `docs/reports/experiments.md`

## 现在缺什么（2026-10-16 状态）

融合层已就位且 74 条测试全绿，但它现在是**空转**的：`prompts/emotion_profile.md`、`prompts/response.md`、`rag/retrieve.py`、`rag/memory.py` **都还不存在**，所以走的全是「官方提示词 + 截断后的上下文」这条回退路径。

**②③④ 谁先交东西，融合层立刻就能吃进去，不需要改代码。** 你的价值是：等他们的产物一到位，你能马上把接缝调对，而不是等到最后一天才发现拼不上。
