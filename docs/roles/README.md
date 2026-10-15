# 公共背景（各线都先读这个）

这份文件是给**在本地跑编码 AI 的人**看的：把对应角色的提示词贴给 AI 之前，先让它读完本文件（提示词里已经写了这一步）。

---

## 一、我们在参加什么

「2026 动感地带 AI+ 高校创智计划」AI 技术赛道，赛题二**数字人综合情感陪伴对话模型**。
官方赛题描述：面向数字人情感陪伴与智能对话场景，围绕**情感理解、记忆融合、共情表达**等核心能力开展模型训练与优化，提升模型在多场景、多轮对话中的自然度、连贯性和情感表现。

**交付形式**：提交一个 Docker 镜像。组委会在**离线环境**里用隐藏测试集跑我们的镜像，自动评分。全程不得联网、不得调用公网 API、模型权重与推理引擎必须全部打进镜像。

## 二、官方评分细则（原文，已核对）

来源：<https://ai.migudm.cn/interactive/match/home> →「评审标准 → AI技术赛道 → 02 数字人综合情感陪伴对话模型」

> **一、自动评测 70 分**
> 情感识别准确性（25 分）：情感分类准确率 Acc。
> 回应质量（20 分）：BLEU-4、ROUGE-L、BERTScore 综合得分。
> 用户画像推断（15 分）：画像维度平均准确率。
> 推理性能（10 分）：平均端到端推理延迟。
>
> **二、内部技术专家评审 30 分**
> 方案完整性（10 分）：技术方案、实验设计及报告完整性。
> 代码规范性（10 分）：代码结构、注释及可复现性。
> 平台运行记录（10 分）：训练日志、调优过程及平台使用规范。

**注意：30 分的专家评审占三成，且「平台运行记录」是硬性要求。** 所以哪怕不训练，也必须在咪咕仝学平台上真实跑过、并把过程留档，否则这 10 分直接丢。

## 三、硬约束（违反任意一条，作品直接失败）

| 约束 | 说明 |
|---|---|
| 离线运行 | 镜像里不许下载模型、不许装包、不许请求公网 |
| 唯一入口 | 组委会只会执行 `bash /root/participant/start.sh TEST_FILE RESULT_DIR` |
| 镜像上限 | **50 GB**（咪咕平台是「公有镜像 + 系统盘已用空间」总和；外部平台是镜像文件本身） |
| 测评机会 | 每队最多 **5 次成功**测评；运行失败不占次数，成功后以**最后一次**成绩计分 |
| 输出格式 | `RESULT_DIR/submission.jsonl`（一行一个 JSON）+ `RESULT_DIR/performance_report.json` |
| 性能统计 | 只统计**前 100 条**的端到端耗时（`performance_rounds=100`） |

## 四、提交字段（每个字段都影响分数）

```json
{
  "id": "test1_000001_t5",
  "response_text": "非空中文回复",
  "emotion_label": "anxiety",
  "user_profile": {
    "personality_traits": ["sensitive"],
    "interests": ["career_development"],
    "style": ["rational"]
  },
  "memory_refs": ["mem_000123"]
}
```

- `emotion_label` 只能取 16 类之一：joy / gratitude / relaxed / care / pride / neutral / surprise / mixed / sadness / loneliness / anxiety / anger / fear / disgust / shame / helplessness。**写空字符串等于该项判错。**
- `user_profile` 三维取值是**固定 ASCII 词表**（不是中文），写词表外的值会被判空。完整词表见 `configs/label_space.json`。
- `memory_refs` 格式为 `^mem_[0-9]{6}$`。**推理输入里没有 memory bank**，官方基线脚本的提示词让模型固定输出 `[]`；该字段是否计分官方未答复。
- 数据分布很偏：训练集里 `anxiety` 占 34%，`care` 只有 0.4%、`surprise` 只有 0.3%。**模型若一律输出 anxiety，情感分也能拿到约 0.34**——所以小类能不能救，是拉开差距的地方。

## 五、仓库地图

```
GivenData/                     官方材料，只读
  participant/                 官方推理工具（入口 start.sh + run_inference.py）
  训练-验证-数据集/              train 8338 段对话 / 18657 预测点；val 900 段 / 2354
  test_inference_data.jsonl    3 条推理样例（只够冒烟）
src/                           本地评测闭环（我们自己写的）
  data_processor.py            官方数据 → 逐轮样本
  evaluate.py                  BLEU-4 / ROUGE-L / 情绪 Acc / 画像集合匹配
  validate_submission.py       提交格式校验
  label_space.py               官方词表
  cli.py                       统一入口
data/processed/                build 产物（已 gitignore，可随时重生）
data/reports/                  数据质量报告（已提交，是留档证据）
tests/                         26 条回归测试
PLAN.md                        权威计划
README.md                      队伍方案书（评分权重与官方一致）
ROLES.md                       角色分工索引
docs/roles/                    五份角色提示词
```

## 六、三条命令（每天都用）

```bash
python src/cli.py build                      # 展开官方数据 → data/processed/，写质量报告
python src/cli.py eval --predictions X.jsonl # 在验证集上打分
python src/cli.py validate --submission X.jsonl   # 校验提交格式
python -m unittest discover -s tests         # 26 条回归测试
```

只用 Python 标准库，**不要引入新依赖**（离线镜像里装不了，而且本机没有 pip 源的保证）。

## 七、铁律

1. **`GivenData/` 只读。** 它是官方发的材料，改了就说不清了。
2. **不引入新依赖。** 现有代码只用标准库。要加依赖必须先在全队说清楚，并确认能离线打包。
3. **不伪造。** 指标、日志、截图必须真跑出来；没跑就写「未跑」。这条同时也保护专家评审的 30 分。
4. **不动别人的目录。** 见 `ROLES.md` 的独占范围表。
5. **改完必须自测**：`python -m unittest discover -s tests` 全绿，且你的产物能过 `validate`。
6. **不要自作主张换技术路线。** 路线已经定了（提示词 + RAG + 离线小模型），要改先在群里讨论。

## 八、还没定的事（别自己猜）

| 问题 | 现状 |
|---|---|
| 官方 ID 正则自相矛盾 | schema 要求 `^test_[0-9]{6}_t[0-9]+$`，官方样例却是 `test1_000001_t5`。策略：**沿用输入给的 ID，不重编号** |
| `memory_refs` 是否计分 | 官方未答复。标签里 97.7% 非空，但推理输入没有 memory bank |
| 延迟多少毫秒得满分 | 官方未公布换算方式 |
| 三项回复指标如何加权 | 官方未公布 |
| 数字人赛道基础镜像 | 细则写了要下载，客服说没有；实际只能用平台公有镜像（vLLM / Ubuntu 等） |

## 九、咪咕仝学平台实测（2026-10-15）

- 控制台：<https://aigpu.migu.cn/train/control/container/list>（需登录，和竞赛站 <https://ai.migudm.cn> 是两套账号）
- **公有镜像**：vLLM、Ollama、LMDeploy、PyTorch（7 个版本）、MiniConda3（4 个版本）、Ubuntu、DeepSeek、MiniMax、PaddlePaddle、TensorFlow、ComfyUI、Stable Diffusion WebUI、Qwen-Image。
  其中 **Ubuntu 镜像**说明里明确写了「支持用户基于此基础镜像定制自己的业务镜像」——这就是客服所说「没有可下载的基础镜像 tar 包」时的替代方案。
- **算力市场只有 RTX 4090 一种卡**：1 卡 = 250 AIQ/小时（16 核 64GB、系统盘 50GB）；2 卡 = 500 AIQ/小时；4 卡 = 1000；8 卡 = 2000。机房为西南一区 / 西南二区。
- **账户余额 0 AIQ，卡时券余额 10 卡时**——平台上的每一分钟都很贵，先用本地跑通再上平台。
