# 动感地带 AI+ 高校创智计划 · AI 技术赛道
## 赛题二：数字人综合情感陪伴对话模型

> **本 README 已于 2026-10-15 按实际技术路线重写。**
>
> 早先版本推荐的路线（Qwen2.5-7B + LoRA/SFT + DPO + vLLM 微调）**已被全队否决**，
> 第三节与第五节已按真实情况改写，第七节的命令已替换为真正能跑的那一套。
> 评审时如果对照 git 历史看到旧内容，以本文件为准；修订留痕见 [`PLAN.md`](PLAN.md)。
>
> 想直接看结果的专家，请走这条线：
> [`docs/reports/technical-plan.md`](docs/reports/technical-plan.md)（方案）
> → [`docs/reports/experiments.md`](docs/reports/experiments.md)（实验与数据）
> → [`docs/reports/code-review.md`](docs/reports/code-review.md)（代码规范性自审）
> → [`docs/reports/reproduce.md`](docs/reports/reproduce.md)（复现）。

---

## 目录

- [一、赛题与提交形式](#一赛题与提交形式)
- [二、评分标准](#二评分标准)
- [三、我们实际的技术方案](#三我们实际的技术方案)
- [四、数据工程：我们在训练集里实测到的](#四数据工程我们在训练集里实测到的)
- [五、仓库实际结构](#五仓库实际结构)
- [六、怎么跑（可复现命令）](#六怎么跑可复现命令)
- [七、硬约束与踩过的坑](#七硬约束与踩过的坑)
- [八、分工](#八分工)
- [九、尚未完成的部分（如实标注）](#九尚未完成的部分如实标注)

---

## 一、赛题与提交形式

赛题聚焦数字人情感陪伴与智能对话，要求模型具备**情感识别、用户画像推断、记忆融合、共情表达**
四类能力，并在多轮对话中保持自然度与连贯性。

- **提交形式**：Docker 镜像，平台侧离线自动化测评。
- **环境特性**：**完全禁网**。不能调用任何公网 API，模型权重与推理引擎必须全部固化在镜像内。
- **评测机会**：初赛阶段最多 **5 次**成功评测；成绩以**最后一次成功提交**计入排名；
  容器报错或运行失败**不占用**次数（这条决定了我们的工程策略：宁可退化成基线，也绝不能让容器崩掉）。

## 二、评分标准

总分 **100 分** = 自动评测 **70** + 内部技术专家评审 **30**。

```
总分 100 分
├── 一、自动评测 (70分) ── [初赛排名的决定性指标]
│   ├── 1. 情感识别准确性 (25分) ── Acc（多标签分类）
│   ├── 2. 回应质量 (20分) ─────── BLEU-4 + ROUGE-L + BERTScore
│   ├── 3. 用户画像推断 (15分) ─── 三个维度的平均 Acc
│   └── 4. 推理性能 (10分) ─────── 端到端平均推理延迟
│
└── 二、内部技术专家评审 (30分) ── [拉开档次的护城河]
    ├── 1. 方案完整性 (10分) ───── 技术方案、共情理论支撑、实验消融完整性
    ├── 2. 代码规范性 (10分) ───── 工程结构、注释、类型注解、一键复现
    └── 3. 平台运行记录 (10分) ─── 【硬性要求】咪咕仝学平台的实例运行与调优留档
```

权重已与官方细则逐条核对。**注意第三项是硬指标**：没有平台运行记录，这 10 分直接归零，
与模型效果无关——所以角色①从第一天起就在平台留痕，而不是等模型调好再补。

## 三、我们实际的技术方案

### 3.1 路线：不微调，走「提示词 + RAG + 离线小模型」

**这是被约束逼出来的，不是偏好。**三条理由，每条都有实测依据：

1. **工期不够。** 从立项到初赛截止只有两天，SFT + DPO + 量化调参不可能在两天内收敛。
2. **本机算力不匹配。** 队伍算力是 4×RTX 5060 Laptop 8GB + 1×RTX 5070 Laptop 12GB，
   训练与量化的结果不可控；而平台卡时券只有 **10 卡时**（算力市场只有 RTX 4090 一种卡，
   按 250 AIQ/小时计），这点卡时用来跑训练会比用来跑评测亏得多。
3. **收益结构不划算。** 微调主要影响「回应质量」20 分；而情感识别 25 分与画像 15 分
   本质是**受限分类问题**——把标签空间约束进提示词、做好归一化，比微调更能稳定拿分。

### 3.2 架构

```
官方测试集 (test_inference_data)
        │
        ▼
  deploy/participant/run_inference.py     ← 官方入口（原件副本，仅 +22/−2 行）
        │
        ├─ wiring.compose_system_prompt(official_prompt)
        │     └── prompts/emotion_profile.md   (角色②)
        │         prompts/response.md          (角色③)
        │
        ├─ wiring.build_messages(sample, system_prompt, conversation_text)
        │     └── rag.fusion.build_messages
        │           ├── rag.retrieve.retrieve(sample, k)   (角色④)
        │           └── 上下文预算裁剪 + 防泄漏断言
        │
        ▼
  Qwen2.5-3B-Instruct（本地权重，5.9 GB，2 个分片）
        │
        ▼
  submission.jsonl  →  src/validate_submission.py 校验 →  src/evaluate.py 本地打分
```

**关键设计：每一个接缝都允许失败，失败即退回官方基线。**

```
try: from rag.fusion import build_messages; messages = build_messages(sample, retrieved)
except Exception: messages = [官方原始的两条 messages]
```

`deploy/participant/wiring.py` 是整层容错的落点：提示词读不到、检索模块缺失、融合函数抛异常，
任何一步失败都退回官方提示词与官方组包。**删掉 `wiring.py` 就等于完全按官方基线跑。**
在一个「崩溃不计次数但也不出分」的赛制里，这个设计比任何单点优化都值钱。

### 3.3 基座选型：Qwen2.5-3B-Instruct

| 候选 | 结论 |
|---|---|
| Qwen2.5-7B-Instruct | 效果更好，但本机下载与推理更吃力，平台单卡显存也紧张 |
| **Qwen2.5-3B-Instruct** | **选定**：约 5.9 GB，单卡 4090 上延迟可控；中文与结构化输出够用 |
| 更小模型（1.5B / 0.5B） | 延迟更优，但结构化输出稳定性明显下降，会拖垮 25 分的情感识别 |

选型不是拍脑袋：情感识别与画像推断是**闭集分类**，靠提示词约束标签空间就能拿到大部分分；
真正需要模型能力的是回应质量 20 分，而 3B 在这项上的边际损失小于它带来的延迟收益。

### 3.4 与「微调路线」的对照（为什么放弃）

| 维度 | 微调路线（原计划） | 实际路线 |
|---|---|---|
| 交付周期 | 需 5–8 天 | 2 天可完成 |
| 算力需求 | 需 GPU 训练，10 卡时远远不够 | 单卡推理即可 |
| 对回应质量的提升 | 显著 | 有限，靠提示词 + 检索补 |
| 对情感/画像得分的提升 | 小（本质是分类） | 大（标签空间约束 + 归一化） |
| 崩溃风险 | 权重与推理引擎耦合，调试面大 | 每层可退化到官方基线 |
| 平台运行记录 | 有训练日志 | 只有推理日志，**此项偏弱** |

最后一行是我们这条路线的真实代价，不掩饰：平台运行记录 10 分原本可以靠训练日志撑起来，
现在只能靠推理实例的运行留档。

## 四、数据工程：我们在训练集里实测到的

数据来源：`data/reports/dataset_summary.json`、`train_quality.json`、`val_quality.json`
（这三个文件**已提交**，是本节的数字来源，可逐项复算）。

| 项目 | 训练集 | 验证集 |
|---|---:|---:|
| 原始对话 | 8,338 | 900 |
| 展开后样本（每个 assistant 轮一条） | **18,651** | **2,354** |
| 丢弃：无预测点 | 5 | 0 |
| 丢弃：轮次错位 | 4 | 0 |
| 丢弃：真值泄漏 | 2 | 0 |
| 三画像维度全空的样本 | 6,834（36.6%） | 875（37.2%） |
| 回复长度 中位数 / P10 / P90 / 最大 | 129 / 39 / 517 / 1809 字 | — |

### 四个直接影响策略的发现

**1. 情感标签严重偏斜。** 16 类中 `anxiety` 占 6,339 条（**34.0%**），
而 `surprise` 只有 64 条、`care` 79 条、`disgust` 125 条。
→ 意味着**只猜 `anxiety` 一类就能拿到约 34% 的 Acc**，而追求小众类的召回会伤总分。
提示词策略据此定：先保住高频类的判定稳定性，再对小类做规则兜底。

**2. 画像有 36.6% 是空集。** 三个维度全空的样本占训练集三分之一以上。
→ 如果模型习惯性地「填满」三个维度，这 6,834 条会全部判错。
评测口径是**精确匹配的集合**，多填一个值就是错，所以「该空就空」必须写进提示词规则。

**3. 记忆引用几乎总是非空。** `memory_refs` 分布：0 条仅 437 个样本（2.3%），
众数是 2 条，最多 9 条。→ 检索是**必答题**而不是加分项，跳过检索会直接丢掉「记忆融合」这项核心能力。

**4. 训练集与验证集的记忆 id 交集为 0。**
→ 记忆是**全局池 + 对话内局部编号**，不能靠 id 直接跨集匹配。
所以检索必须按内容相似度做，不能按 id 查表。

## 五、仓库实际结构

```
├── GivenData/                    官方材料，只读，任何情况下不修改
│   ├── 训练-验证-数据集/          官方 train/val JSONL
│   └── participant/              官方推理工具原件 + 基座模型（model/ 5.9 GB，已 gitignore）
├── src/                          公共设施：数据适配 / 评分 / 校验
│   ├── cli.py                    统一入口：build | eval | validate
│   ├── data_processor.py         展开官方数据为样本，产出数据质量报告
│   ├── evaluate.py               本地评分：Acc（情绪/画像）、BLEU-4、ROUGE-L、BERTScore、延迟
│   ├── label_space.py            词表唯一真相（情绪 16 类 / 画像三维度取值）
│   └── validate_submission.py    按官方 schema 校验提交文件
├── rag/                          检索增强
│   ├── fusion.py                 ⑤A：把「画像 + 检索结果 + 对话」拼成最终 messages
│   └── retrieve.py               ④：检索接口（contract 冻结签名）— **尚未创建**
├── prompts/                      ②③：纯文本判定规则与写作规则 — **尚未创建**
├── rag/corpus/  rag/index/       ④：语料与索引 — **尚未创建**
├── deploy/                       ①：镜像与部署
│   ├── participant/              官方 /root/participant 的我们自己的副本
│   │   ├── run_inference.py      官方原件 +22/−2 行（接 wiring）
│   │   ├── wiring.py             我们新增的接线层（唯一容错落点）
│   │   ├── start.sh              官方原件 +17/−3 行（conda 容错、单卡 TP）
│   │   └── configs/              我们的参赛推理配置
│   └── build.sh                  组装 context + docker build + 断网自检
├── tools/                        开发工具
│   ├── bootstrap.py              一键建环境（Python 3.14 + torch cu130）
│   ├── fetch_model.py            模型下载（断点续传、镜像源、完整性校验）
│   ├── eval_loop.py              评测循环：跑分 → 记 leaderboard
│   ├── gen_label_block.py        从词表生成纯文本块，防止手抄出错
│   └── assign_role.py            分工登记
├── tests/                        98 个单元测试
├── data/processed/               展开后的样本（220 MB，可重建，已 gitignore）
├── data/reports/                 数据质量报告（体积小，是查证证据，**已提交**）
├── docs/
│   ├── roles/                    各线提示词与接口契约
│   ├── reports/                  专家评审材料（方案 / 实验 / 代码评审 / 复现）
│   └── platform-log/             平台运行留档
├── runs/                         跑分产物与 leaderboard（已 gitignore）
└── PLAN.md / ROLES.md / START-HERE.md   权威计划 / 分工 / 上手指引
```

> **关于上图的准确说明**：标了「尚未创建」的目录确实还不存在——那是角色②③④的产物，
> 目前仍在开发中。本仓库**不放计划中的目录占位**，所以树里没有它们。
>
> 另外，早先版本 README 里的 `models/`、`configs/model_config.yaml`、`src/predictor.py`、
> 根目录 `Dockerfile`、`run_all.sh` **也都不存在**——那是微调路线时期的设想，从未落地。

## 六、怎么跑（可复现命令）

**完整版见 [`docs/reports/reproduce.md`](docs/reports/reproduce.md)。** 最短路径：

```powershell
# 1. 建环境（Python 3.14 + torch 2.14.1+cu130，适配 sm_120）
python tools/bootstrap.py

# 2. 下模型（5.9 GB，支持断点续传与镜像源）
.venv\Scripts\python.exe tools\fetch_model.py
.venv\Scripts\python.exe tools\fetch_model.py --check     # 完整性校验

# 3. 展开数据与词表
.venv\Scripts\python.exe src\cli.py build

# 4. 跑分：冒烟 5 条，全量 2354 条
.venv\Scripts\python.exe src\cli.py eval --predictions runs\<你的产物>.jsonl
.venv\Scripts\python.exe tools\eval_loop.py --tag <实验名>

# 5. 自测与规范检查
.venv\Scripts\python.exe -m unittest discover -s tests    # 期望 OK
.venv\Scripts\python.exe docs\reports\audit_code.py        # 代码规范巡检

# 6. 校验提交文件
.venv\Scripts\python.exe src\cli.py validate --submission runs\<你的产物>.jsonl
```

打包镜像（Linux / WSL）：

```bash
bash deploy/build.sh              # 组装 context + docker build + 断网自检
bash deploy/build.sh --tar        # 额外导出 deploy/dist/ogc-participant.tar
```

## 七、硬约束与踩过的坑

### 硬约束

1. **绝对禁网。** 依赖必须预装进镜像，权重必须固化；代码里不允许出现
   `huggingface_hub.download` 或任何 `requests.post`。
2. **最多 5 次成功评测，只认最后一次。** 崩溃不计次数——所以「不崩」优先于「更好」。
3. **输出字段名错一个就是 0 分**（例如 `emotion` 写成 `emotions`）。
   入口有 `validate_submission.py` 按官方 schema 校验。
4. **平台运行记录 10 分是硬指标**，与模型效果无关，必须从第一天开始留档。
5. **`GivenData/` 只读。** 我们要改官方推理脚本时，改的是 `deploy/participant/` 里的副本。

### 实际踩到的坑（都已修复，留作记录）

| 坑 | 表现 | 处置 |
|---|---|---|
| 官方 `start.sh` 的 `source activate conda_env` | 脚本首行是 `set -euo pipefail`，**没有 conda 的镜像会直接终止，等于零分** | 改为「有 conda 就进，没有就用 PATH 上的 python3」 |
| 官方默认双卡张量并行 | 平台算力市场**只有 RTX 4090 一种卡**，按 1 卡申请时 `--tensor-parallel-size 2` 起不来 | 默认改成 1，可用环境变量覆盖 |
| `fetch_model.py` 漏下分片目录文件 | 两个分片字节数都对、`--check` 也通过，**但推理立刻报 `OSError: ... model.safetensors.index.json`** | 补上下载分片索引文件 |
| 部分样本跑分口径 | 用 `--limit 20` 冒烟时，分母仍是完整 2354 条，指标永远显示约 0.001 | 真值同步收窄（`8a61e72`） |
| pip 把解包临时目录建在工作目录 | 一次 `git add -A` 扫进 1.9 GB 的 torch wheel | `.gitignore` 挡住 + `bootstrap.py` 把 TEMP 指到系统临时目录 |

## 八、分工

六条线并行（两个人兼线），详见 [`ROLES.md`](ROLES.md) 与 [`docs/roles/CONTRACT.md`](docs/roles/CONTRACT.md)。

| 线 | 主责评分项 | 独占写文件范围 |
|---|---|---|
| ① 部署与评测 | 推理性能 10 + 平台运行记录 10 | `deploy/`、`docs/platform-log/` |
| ② 提示词·情绪与画像 | 情感识别 25 + 用户画像 15 | `prompts/emotion_profile.md`、`tools/emotion_profile/` |
| ③ 提示词·共情回复 | 回应质量 20 | `prompts/response.md`、`tools/response/` |
| ④ RAG·语料与检索 | 记忆融合（核心能力） | `rag/corpus/`、`rag/index/`、`rag/retrieve.py`、`rag/memory.py` |
| ⑤A RAG·融合 | ②③的分要经它才落到模型上 | `rag/fusion.py` |
| ⑤B 专家评审文档 | 方案完整性 10 + 代码规范性 10 | `docs/reports/` |

**`src/`、`tools/`、`tests/` 是公共设施，不归任何一条线独占**——这一点已在
[`docs/reports/code-review.md`](docs/reports/code-review.md) 中作为问题 F1 记下，
因为无主目录的代码质量确实低于有主目录。

**提交前必须跑**：`python -m unittest discover -s tests` 与 `python src/cli.py eval --predictions <产物>`。

## 九、尚未完成的部分（如实标注）

按「不许伪造数据」的规约，以下内容**没有做，或者做了但未验证**，不写进任何得分陈述：

- **全量基线分数：正在跑。** 2354 条验证集的官方裸提示词基线尚未跑完，
  `docs/reports/experiments.md` 的 E1 行为空。
- **②③④ 的产物尚未交付**：`prompts/*.md` 与 `rag/retrieve.py` 还在开发。
  融合层（⑤A）已经写好并通过测试，但**在队友产物落地前，它的分值是 0**——
  不是它坏了，是它没有输入。
- **平台运行记录正在积累**，尚未整理成评审材料。
- **未做消融实验**：受工期限制，目前只能给出基线单点，没有「有 RAG vs 无 RAG」的对照。
- **官方口径有三个待确认问题**（已记录，未获答复）：
  1. 官方 ID 正则与实测 ID 格式存在矛盾；
  2. `memory_refs` 是否计入评测指标；
  3. 回应质量三项（BLEU-4 / ROUGE-L / BERTScore）的具体权重。
- **`GivenData/participant/model/` 完整性与推理延迟的真实数字**取决于运行机器，
  本机测得的值不能直接当作平台成绩。

---

**许可与致谢**：本仓库为参赛用途。`GivenData/` 版权归赛事主办方，我们只读不改。
