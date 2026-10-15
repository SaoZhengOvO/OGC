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
