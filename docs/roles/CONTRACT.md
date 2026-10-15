# 接口契约（各线之间的接缝，改之前必须先在全队说）

路线已定：**提示词 + RAG + 离线小模型**。为了让五个人能并行开发、最后由角色①的 Dockerfile 拼起来，下面这些路径与函数签名**冻结**。谁要改，先在群里说，改完全队同步。

## 一、目录所有权

```
prompts/emotion_profile.md     角色② 写    情绪 + 画像的判定规则（纯文本）
prompts/response.md            角色③ 写    回复生成的写作规则（纯文本）
rag/corpus/                    角色④ 写    语料构建脚本与语料产物
rag/index/                     角色④ 写    检索索引（构建产物，可重建）
rag/retrieve.py                角色④ 写    检索接口
rag/fusion.py                  角色⑤A 写   把「画像 + 检索结果 + 对话」拼成最终 messages
deploy/                        角色① 写    Dockerfile、participant 工具副本、离线打包脚本
deploy/participant/            角色① 写    /root/participant 的内容（官方工具的我们自己的副本）
docs/platform-log/             角色① 写    平台运行留档
docs/reports/                  角色⑤B 写   技术方案 / 实验报告 / 专家评审材料
```

> **2026-10-16**：原「⑤ RAG·融合与文档」拆成 **⑤A（融合代码，只写 `rag/fusion.py`）** 与 **⑤B（专家评审文档，只写 `docs/reports/`）**。两条线不再由同一人承担，其余归属不变。

**`GivenData/` 永远只读。** 我们要改官方推理脚本时，改的是 `deploy/participant/` 里的**副本**，不是 `GivenData/participant/` 原件。

## 二、检索接口（角色④ → 角色⑤A）

`rag/retrieve.py` 必须导出：

```python
def retrieve(sample: dict, k: int = 3) -> list[dict]:
    """sample 为一条推理样本：{"id", "conversation_id", "target_user_turn_id", "history": [{"turn_id","role","content"}]}
    返回按相关度降序的最多 k 条：[{"text": str, "score": float, "source_id": str}]
    无结果返回 []。不允许联网、不允许引入第三方依赖。"""
```

## 三、组包接口（角色⑤A → 角色①）

`rag/fusion.py` 必须导出：

```python
def build_messages(sample: dict, retrieved: list[dict] | None = None) -> list[dict]:
    """返回直接喂给 chat template 的 messages：
    [{"role": "system", "content": <由 prompts/*.md 拼成的系统提示>},
     {"role": "user",   "content": <对话历史 + 检索到的记忆>}]
    必须保证：不把目标 assistant 回复或目标轮之后的任何内容放进 messages。"""
```

角色①的推理脚本按这个顺序加载，任何一步失败就退回官方内置提示词（**保证镜像永远能跑出合规输出**）：

```
try: from rag.fusion import build_messages; messages = build_messages(sample, retrieved)
except Exception: messages = <官方 SYSTEM_PROMPT 版>
```

## 四、提示词文件格式

`prompts/*.md` 是**纯文本**，会被原样拼进系统提示，因此：

- 不要写 Markdown 标题、表格、代码围栏（模型会照抄格式）。
- 用中文短句写规则。
- 必须包含输出字段的取值约束（16 类情绪、三维画像的 ASCII 词表）。
- 其中**画像与情绪词表不要手抄**：用 `python tools/gen_label_block.py`（读 `configs/label_space.json`，也就是 `src/label_space.py` 导出的唯一真相）生成纯文本块后贴进提示词。手抄出错是静默的——词表外的取值会被 `normalize_*` 丢掉，只表现为「那一维全空」。

## 五、评测口径（各线共用同一把尺子）

```bash
python src/cli.py eval --predictions runs/<你的产物>.jsonl
```

- 情绪：Accuracy（全类别）
- 画像：三维各自「集合精确匹配」准确率的均值
- 回复：BLEU-4、ROUGE-L（char 分词）；BERTScore 本地尚未接入
- 汇报时必须写明：本轮产物路径、跑了多少条、对比的基线是什么

**基线（务必先复现出来再谈优化）**：官方 `run_inference.py` 的原始提示词 + 选定基座模型，在 val 2,354 条上的成绩。任何改动都要和这条基线比。

## 六、资产共享（谁需要传什么）

**结论：除了模型权重，什么都不用传；而模型权重也不用互相传。**

| 资产 | 体积 | 怎么获得 | 要不要互相传 |
|---|---|---|---|
| 训练集 / 验证集（`GivenData/训练-验证-数据集/`） | 66.4 MB + 5.8 MB | `git clone` 就有 | 不用 |
| 展开后的样本（`data/processed/`，10 个文件） | 220.9 MB | `python src/cli.py build` 重建 | 不用（已 gitignore） |
| 语料与索引（`rag/corpus/`、`rag/index/`） | 待 ④ 测 | ④ 的 `build.py` 从**训练集**重建 | 不用（已 gitignore） |
| 提示词（`prompts/*.md`） | 几 KB | `git pull` | 不用，走 git |
| 代码（`rag/*.py`、`src/`、`tools/`） | 几 KB | `git pull` | 不用，走 git |
| 模型权重（Qwen2.5-3B-Instruct） | **5.9 GB** | `python tools/fetch_model.py` | 想省流量可以，但没必要 |

**为什么不要互相传语料/索引**：语料只能来自训练集，而训练集人人都有；`build.py` 是确定性的，同一个 commit + 同一份数据，谁跑出来的索引都一样。互相传文件反而制造「我用的是 v2、你用的是 v1」这种**静默不一致**——分数对不上却查不出原因。**版本一致性靠 git commit 号，不靠传文件。**

**如果实在想省那 5.9 GB**：把模型放到一台有公网的机器上起 vLLM / Ollama，别人用 `openai_compatible` 后端指过去：

```bash
export EVALUATION_BACKEND=openai_compatible
export EVALUATION_BASE_URL=http://<那台机器>:8000/v1
export EVALUATION_API_MODEL=<模型名>
python tools/eval_loop.py --tag <实验名>
```

官方脚本原生支持这条路（`run_inference.py` 已内置这些环境变量），**不需要改一行代码**。

**但有一个陷阱**：走共享服务跑出来的**延迟数字没有意义**（官方性能分是 4090 上单条端到端计时，走网络会含往返与排队），**准确率类指标不受影响**（情绪 Acc / 画像 Acc / BLEU-4 / ROUGE-L 都是纯文本比对，与算力在哪无关）。所以：

- ②③④ 用共享服务迭代**准确率**——完全可以，还能省下各自下模型的 5.9 GB；
- 做**性能验证**——必须用本机 GPU，或者直接上咪咕平台的 4090。
