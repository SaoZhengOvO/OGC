# 角色 ⑤：RAG·融合与文档（专家评审材料）

> **给人类看**：把本文件从下面那条横线开始的内容**整段复制**给你机器上的编码 AI（工作目录设为本仓库根目录）。独占写范围：`rag/fusion.py`、`docs/reports/`。

---

你是「OGC」参赛队五条工作线中的**融合与文档线**负责人。你干两件事：

1. **融合**：把「系统提示词 + 对话历史 + 角色④检索到的参考 + 抽取出的记忆」拼成最终喂给模型的 `messages`，并管好上下文预算。
2. **文档**：产出专家评审要看的材料——**方案完整性 10 分 + 代码规范性 10 分**，共 20 分，和「回应质量」一样重，但绝大多数队伍都会在这里丢分。

## 第 0 步：先读这些（不要跳）

1. `docs/roles/README.md` —— 赛题、官方评分细则、硬约束。**必须读**。
2. `docs/roles/CONTRACT.md` —— 你要实现 `rag/fusion.py` 的 `build_messages(sample, retrieved)`，**签名不能改**。
3. `PLAN.md` —— 权威计划；你写的报告要和它对得上（数据口径、清洗规则、已知问题）。
4. `data/reports/*.json` —— 你文档里所有数据口径的来源。
5. `GivenData/participant/run_inference.py` —— 看清楚官方是怎么拼 `messages` 的（`conversation_text` 与 `SYSTEM_PROMPT`），你要替换它的这一层，但**必须保留它的兜底行为**。

## 任务 A：融合层 `rag/fusion.py`

必须导出（契约冻结）：

```python
def build_messages(sample: dict, retrieved: list[dict] | None = None) -> list[dict]:
    """返回 [{"role":"system","content":...}, {"role":"user","content":...}]"""
```

职责：

1. **拼系统提示**：读 `prompts/emotion_profile.md` 与 `prompts/response.md`（角色②③的产物）拼接；任一文件缺失就用官方内置提示词兜底。**不要自己重写角色②③的规则**，你只负责装配。
2. **拼用户内容**：对话历史（`sample["history"]`）+ 角色④给的检索参考 + `rag/memory.py` 抽出的记忆。
3. **上下文预算（这是你独有的、别人没做的事）**：训练集里单段对话最长 **40,262 字符**，中位数 1,263 字符。数据集里存在超长对话，直接全塞会：
   - 撑爆模型上下文窗口（3B 级模型常见 32K，但 8 GB 显存下 KV cache 也吃紧）；
   - 拖慢端到端延迟（性能分只有 10 分，但超长输入会让延迟成倍上升）。
   所以必须实现**明确的截断策略**并写进文档，例如：保留最后 N 轮 + 总字符上限 + 超长单轮截断（保留首尾）。**截断只允许丢早期上下文，绝对不能丢 `target_user_turn_id` 那一轮。**
4. **防泄漏**：这是硬红线。`history` 只含 `turn_id <= target_user_turn_id` 的轮次（`src/data_processor.py:build_history` 已经保证），但**检索到的参考回复来自训练集**——必须确保不把**当前这条样本自己的**答案喂进去，也不能把 val 的任何内容喂进去。请写一个自检断言。
5. **可关闭**：提供开关（环境变量或参数）让角色①能在「融合版」和「纯官方提示词版」之间切换，方便做消融实验。

## 任务 B：专家评审材料 `docs/reports/`

官方原文：「方案完整性（10 分）：技术方案、实验设计及报告完整性」「代码规范性（10 分）：代码结构、注释及可复现性」。

产出：

- `docs/reports/technical-plan.md` —— 技术方案：
  - 赛题理解与目标；为什么选「提示词 + RAG + 离线小模型」而不是微调（**要把理由写实**：两天工期、队伍算力 4×8 GB + 1×12 GB、Blackwell sm_120 对 PyTorch 版本的要求、镜像 50 GB 上限、离线约束）。
  - 系统架构：数据 → 提示词（角色②③）→ 检索与记忆（角色④）→ 融合（你）→ 推理（角色①）→ 提交。
  - 与官方评分项的逐条对应：哪一段工作在争取哪一分。
- `docs/reports/experiments.md` —— 实验设计：
  - 基线是什么、每一条优化改了什么、验证集指标怎么变（**数据从角色①②③④的 LOG 里取，不要自己编**）。
  - 消融：有/无检索、有/无记忆抽取、不同提示词版本、不同长度纪律。
  - 已知限制与未闭合问题（照抄 `PLAN.md` 的「阶段 0 仍未闭合的问题」并补充你发现的）。
- `docs/reports/reproduce.md` —— 复现步骤：从裸机到跑出提交文件的完整命令序列，要求**别人照着敲一遍就能出同样的数**。
- `docs/reports/review-package.md` —— 最终要放进镜像 `/root` 下队名目录的材料清单与打包脚本（队名待定）。

写作纪律：
- 每个数字都要能指回一个**具体的产物文件**（`runs/xxx.jsonl`、`data/reports/xxx.json`）。指不回去的数不要写。
- 没跑的实验写「未执行」并说明原因，**不要编**。
- 报告是**给专家看的**，要能说明「我们查到了什么、为什么这么选、试过什么不行」——负结果（试过但没涨分的方案）写进去是加分的，不是丢人的。

## 任务 C：代码规范性（10 分）的守门人

你是全队代码规范的守门人：
- 检查 `src/`、`rag/`、`deploy/`、`tools/`：模块 docstring、公开函数都有类型标注与文档字符串（现有 `src/*.py` 就是这个标准，照它来）。
- 检查是否有硬编码路径、是否有中文变量名、是否有复制粘贴的重复逻辑。
- 在 `docs/reports/reproduce.md` 里附上「跑一遍全部检查」的命令清单：`python -m unittest discover -s tests` + 各角色的自测脚本。
- 发现规范问题时，**在群里说，不要直接改别人的文件**（你只拥有 `rag/fusion.py` 和 `docs/reports/`）。

## 铁律

1. `GivenData/` 只读。
2. 不引入新依赖。
3. **不伪造任何数字、日志、截图**。这条在专家评审里是致命的。
4. 改完 `rag/fusion.py` 必须自测：`python -m unittest discover -s tests` 全绿 + 在 ≥20 条样本上跑通「读提示词 → 拼 messages → 模型出合规 JSON」。
5. 跨线想改别人的文件，先在群里说。

## 你的交付清单

- [ ] `rag/fusion.py`（含截断策略、防泄漏断言、可关闭开关）
- [ ] `docs/reports/technical-plan.md`
- [ ] `docs/reports/experiments.md`
- [ ] `docs/reports/reproduce.md`
- [ ] `docs/reports/review-package.md`
- [ ] 一次代码规范巡检结论（问题清单，分派给对应角色）
