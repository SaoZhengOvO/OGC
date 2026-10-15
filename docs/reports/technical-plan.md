# 技术方案：数字人综合情感陪伴对话模型

> 作者：角色⑤（RAG·融合与文档）。对应官方评分项「专家评审 — 方案完整性（10 分）」。
> 写作纪律：本文每一个数字都必须能指回仓库里的具体产物文件；没有实测的地方写「未执行」并说明原因，不编造。

## 一、赛题理解

官方对本题的表述（`docs/roles/README.md`「赛题与官方描述」一节引用了原文）是围绕
**情感理解、记忆融合、共情表达** 三项核心能力做模型训练与优化。把它们翻译成可交付的工程目标：

| 官方能力 | 落到本赛题的数据字段 | 判决方式 |
|---|---|---|
| 情感理解 | `emotion_label`（16 类闭集） | 单标签分类准确率 |
| 记忆融合 | `memory_refs`（`mem_XXXXXX`） | 官方未公布是否计分 |
| 共情表达 | `response_text` | BLEU-4 / ROUGE-L / BERTScore |
| 用户画像 | `user_profile` 三维标签集合 | 维度平均准确率 |

推理输入只有 `history`（截至目标轮为止的对话），要求模型一次前向输出**一个 JSON object**，
同时给齐四个字段。这是一个「一次输出、四路评分」的多任务问题。

## 二、硬约束（决定技术路线的前提）

全部来自官方材料，逐条在 `docs/roles/README.md` 与 `PLAN.md` 有出处：

1. **完全离线**：镜像内不得调用公网 API，权重与推理引擎必须全量集成。
2. **唯一入口**：官方只运行 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`
   （`GivenData/participant/README.md`）。脚本必须自动读数据、自动推理、按格式落盘、无需人工干预。
3. **镜像 ≤ 50 GB**：咪咕平台「公有镜像与系统盘已使用空间总和不超过 50G」。
   注意平台 1 卡规格的系统盘**正好是 50 GB**。
4. **5 次测评成功机会**，以最后一次成功提交计分，运行失败不占用次数。
5. **官方不提供可下载的基础镜像 tar 包**（客服确认），基础环境用平台公有镜像自建。
6. 产物固定为 `RESULT_DIR/submission.jsonl` + `RESULT_DIR/performance_report.json`；
   性能分只统计**前 100 条**的端到端耗时（`performance_rounds = 100`）。

## 三、技术路线：为什么是「提示词 + RAG + 离线小模型」，而不是微调

这是本方案最需要解释的一个决定。结论是**放弃微调**，理由按权重排序：

| 约束 | 实测/事实 | 对微调的影响 |
|---|---|---|
| 工期 | 初赛窗口只有数天，且队伍同时在做数据适配、打包、跑通链路 | 微调需要的「数据→训练→评测→迭代」闭环跑不满一轮 |
| 本地算力 | 1×RTX 5070 Laptop **12 GB** + 4×RTX 5060 Laptop **8 GB**，无高速互联，跨机 DDP 不可行 | 全参微调不可能；QLoRA 只能单机小步迭代 |
| 平台算力预算 | 咪咕仝学平台**只有 RTX 4090**，1 卡 250 AIQ/小时；账户余额 **0 AIQ**，卡时券 **10 卡时** | 10 卡时≈10 小时单卡，不够跑完一轮像样的 SFT |
| 权重体积 | 镜像 ≤ 50 GB | 训练产物（LoRA 适配器）体积不是问题，问题是训练本身 |
| 依赖栈 | 本机 GPU 是 Blackwell **sm_120**，需要 cu130 的 PyTorch | 训练侧依赖（peft / bitsandbytes / flash-attn）在该组合下的可用性未验证，风险高 |
| 评分口径 | 回应质量靠 BLEU-4 / ROUGE-L / BERTScore，情感与画像靠分类准确率 | 这两类都是**可以由强提示词 + 约束输出**显著改善的任务，微调不是唯一路径 |

同时，**有两条结构性观察支持这条路**：

1. **70 分里的 40 分是分类/集合预测**（情感识别 25 + 画像 15）。它们不需要大模型生成能力，
   需要的是「稳定的结构化输出」——提示词里写死枚举、写死证据门槛，就能直接改善。
2. **官方只校验提交文件的格式**，不要求四个字段来自同一个模型。因此架构上允许
   「融合层统一装配 messages → 单模型一次前向 → 解析并兜底」这条最简单、也最稳的链路。

原 `PLAN.md` 阶段 3 的 SFT 方案已标注为「**备选（不执行）**」，不是删除——如果后续工期允许，
它是唯一能进一步压低延迟、提升风格一致性的手段。

## 四、系统架构

```
TEST_FILE (jsonl: id / conversation_id / target_user_turn_id / history)
        │
        ▼
start.sh  ──►  run_inference.py  ──►  wiring.py  ──►  rag/fusion.py
（官方入口）      （官方脚本副本）      （接线层）      （融合层）
                                        │                │
                                        │                ├─ prompts/emotion_profile.md  ← 角色②
                                        │                ├─ prompts/response.md          ← 角色③
                                        │                ├─ rag/retrieve.py              ← 角色④
                                        │                └─ rag/memory.py                ← 角色④
                                        │
                                        ▼
                              messages = [system, user]
                                        │
                                        ▼
                       Qwen2.5-3B-Instruct（本地目录，离线加载）
                                        │
                                        ▼
                   解析 JSON → 字段归一化 → submission.jsonl
                                        │
                                        └─► performance_report.json
```

### 4.1 各层职责与边界

- **`deploy/participant/`（角色①）**：官方工具的**副本**。`GivenData/` 保持只读，所有改动落在副本上。
- **`deploy/participant/wiring.py`（角色①）**：接线层。唯一职责是把「官方脚本」与
  「队友的融合层」缝在一起，**永不抛异常**——任何一步失败都退回官方提示词与官方组包，
  保证镜像在任何情况下都能跑出合规输出。这是「5 次机会 + 10 卡时」下的必要设计。
- **`rag/fusion.py`（角色⑤）**：装配 `messages`。负责提示词拼接、检索结果过滤、
  上下文预算与截断、防泄漏断言、可关闭开关。
- **`prompts/*.md`（角色②③）**：纯文本提示词块，不许在融合层里重写。
- **`rag/`（角色④）**：纯标准库检索 + 记忆抽取。语料只取训练集。

### 4.2 接口契约

冻结在 `docs/roles/CONTRACT.md`，三个签名不得单方面改动：

```python
# rag/retrieve.py（角色④）
def retrieve(sample: dict, k: int = 3) -> list[dict]:
    """返回 [{"text": str, "score": float, "source_id": str}]，无结果返回 []。"""

# rag/fusion.py（角色⑤）
def build_messages(sample: dict, retrieved: list[dict] | None = None,
                   system_prompt: str | None = None) -> list[dict]:
    """给不得时抛 FusionUnavailable，由接线层回退官方组包。"""

# rag/memory.py（角色④）
def extract_memories(history: list[dict]) -> list[dict]: ...
```

### 4.3 上下文预算与防泄漏（本方案的技术核心之一）

训练集单段对话**最长 40,262 字符**（中位 1,263），远超 3B 模型的舒适上下文。
`rag/fusion.py:select_turns()` 做三级裁剪，**顺序不可颠倒**：

1. **锚定目标轮**：先定位 `turn_id == target_user_turn_id`，从它往前取最多 24 轮。
   「目标轮一定在里面」由结构保证，不靠运气。
2. **单轮首尾截断**：超长单轮保留开头（情境）+ 结尾（当下情绪），砍中间并写明省略字数。
3. **总字符预算**：从最新往回累加，超 6,000 字符就停。早期上下文先丢，**目标轮在列表末尾，
   永远不会因预算被删**。

防泄漏是**断言**而不是注释（`assert_no_leak()`）：校验 messages 结构、检查目标轮之后的
任何一轮正文都没有出现在 messages 里、并确认目标轮本身在场。断言失败即抛
`FusionUnavailable`，由接线层回退——**宁可退回官方基线，也不许把答案喂给模型**。

检索结果另有过滤：丢掉结构不合规的、丢掉 `source_id` 等于本样本自身或其对话 ID 的、
丢掉 `source_id` 以 `val_` 开头的（验证集内容一律不得进入推理上下文）。

## 五、与官方评分项的逐条对应

| 官方评分项 | 分值 | 本方案由谁承担 | 具体手段 |
|---|---|---|---|
| 情感识别准确性 | 25 | 角色② | 16 类判定规则写进 `prompts/emotion_profile.md`；先拿「无脑输出最大类」的免费基线做下界 |
| 回应质量（BLEU-4 / ROUGE-L / BERTScore） | 20 | 角色③ | 控制回复长度贴近训练集分布（中位 129 字）；只学检索到的语气、不抄内容 |
| 用户画像推断 | 15 | 角色② | 三维**集合精确匹配**，多写错标签会让整维归零 → 策略是「有强证据才写」 |
| 推理性能 | 10 | 角色① | 3B 级模型 + 明确计时口径；权重打进镜像避免加载抖动 |
| 方案完整性 | 10 | 角色⑤ | 本文件 + `experiments.md` |
| 代码规范性 | 10 | 角色⑤（守门） | 统一 docstring/类型标注规范，`reproduce.md` 附全量检查命令 |
| 平台运行记录 | 10 | 角色① | `docs/platform-log/YYYY-MM-DD-<事由>.md` + 截图，**从第一天开始记**，事后补不出来 |

## 六、已知限制与风险

1. **官方未公布各指标的满分标准**：延迟多少毫秒拿满、三项回复指标如何加权，均无官方口径。
   本地评测器只能作为**相对改进的代理指标**。
2. **`memory_refs` 是否计分未知**：训练标签 97.7% 非空，但推理输入没有 memory bank；
   官方基线 `run_inference.py` 的 SYSTEM_PROMPT 明确指示固定输出 `[]`。我们按 `[]` 处理，
   同时用 `--include-memory-refs` 在验证集上给出「三种策略」的对照实验（见 `experiments.md`）。
3. **官方 schema 的 ID 正则与实际样例矛盾**：`submission_schema.json` 要求
   `^test_[0-9]{6}_t[0-9]+$`，官方样例却是 `test1_000001_t5`。策略是**沿用输入 ID、不重编号**，
   并保留 `--strict-schema-id` 开关。
4. **3B 级模型的能力上限**：结构化输出与长上下文遵循能力弱于 7B，靠提示词约束与解析兜底弥补。
5. **本地无 bash、无 docker**：`deploy/*.sh` 与镜像构建**从未在本机执行过**，仅做过逐行检查。
   这一条已在 `docs/platform-log/preflight-checklist.md` 的「本地已知限制」中如实标注。
