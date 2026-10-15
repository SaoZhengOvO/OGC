# 角色② 情绪与画像 · 实验记录（LOG）

> 纪律：一次只改一处；每行都要有真实的产物路径；跑不出来就写「未跑」。

## 口径（全队共用，不许自己另算）

```bash
python src/cli.py build                                    # 一次性：展开官方数据
python tools/emotion_profile/make_val_input.py             # 造不含答案的验证输入（2354 条）
python deploy/participant/run_inference.py <输入> <结果目录>   # 注意用 deploy 的副本，不是 GivenData 官方原件
python src/cli.py eval --predictions <产物>/submission.jsonl --per-sample-out <产物>/samples.jsonl
python tools/emotion_profile/report.py --per-sample <产物>/samples.jsonl
python tools/emotion_profile/badcase_pack.py --per-sample <产物>/samples.jsonl --out <产物>/badcase.md
```

- 验证集：**2354 条**（900 段对话）。`val` 绝不进语料、绝不进训练。
- 提示词产物：`prompts/emotion_profile.md`，由 `tools/emotion_profile/make_prompt.py`
  从 `tools/emotion_profile/prompt_rules.txt` + `tools/gen_label_block.py` 生成
  （词表不许手抄，手抄出错是静默的）。
- 评测口径：情绪 = Accuracy；画像 = 三维各自「集合精确匹配」准确率的均值。

## 接线方式（2026-10-15 角色①⑤交付后确认）

角色①的 `deploy/participant/wiring.py` 决定了我的文件怎么被吃进去：

- 提示词目录 = `PARTICIPANT_PROMPT_DIR`，默认 `deploy/participant/prompts/`；
  镜像里由 `deploy/Dockerfile` 设成 `/root/participant/prompts`，
  内容由 `deploy/build.sh:107-113` 从**仓库根 `prompts/`** 拷进去。→ 我写仓库根就对了。
- 拼接顺序：**官方 SYSTEM_PROMPT 打底，我的块追加在后面**（`compose_system_prompt`）。
  所以我的规则是"补充/收紧"，而不是替换；冲突时我的后置块权重更大。
- **本地自测陷阱**：直接跑 `deploy/participant/run_inference.py` 而不设
  `PARTICIPANT_PROMPT_DIR`，它找的是 `deploy/participant/prompts/`（本地不存在）
  → **静默回退到官方提示词**，你会以为改了提示词却没变化。务必显式设置。

---

## 第 0 轮 · 不依赖模型的下限基线（已跑）

产物：`runs/floor_anxiety.jsonl`（2354 条，桩预测）、`runs/floor_anxiety.samples.jsonl`、
`runs/floor_anxiety.report.json`

桩预测内容：情绪一律 `anxiety`；画像三维一律空数组；回复固定「嗯，我在听。」。

| 指标 | 值 | 说明 |
|---|---|---|
| 情绪 Acc | **0.3190** | = 751/2354，就是「无脑猜 anxiety」的免费分 |
| 画像 Acc（三维均值） | **0.6283** | **画像全空就有 0.6283**，这是画像的真正及格线（不是 0.372） |
| └ personality_traits | 0.6185 | 金标为空的 1456/2354 |
| └ interests | 0.7489 | 金标为空的 1763/2354 |
| └ style | 0.5174 | 金标为空的 1218/2354 |
| 回复 BLEU-4 / ROUGE-L | 0.0000 / 0.0296 | 归角色③，这里只作对照 |

### 由这一轮推出的两条可执行结论

1. **画像的策略不是「多填」，而是「只在有把握时填」。**
   对一个维度：金标为空（personality 1456 / interests 1763 / style 1218）却填了 → 判错 **-1**；
   金标非空（898 / 591 / 1136）且完全一致 → **+1**。
   → **盈亏平衡点是「填了的条目里超过 50% 完全正确」**。低于它比全空更差。
   三个维度余额不同：`style` 有 1136 条「该填却空」(48.3%)，最值得动手；
   `interests` 只有 591 条 (25.1%)，乱填最容易吃掉收益。
2. **情绪必须救大类。** val 里 relaxed 409 + mixed 177 + helplessness 142 + sadness 131 +
   fear 124 = **983 条（41.8%）**被"全猜 anxiety"丢掉；
   care(11)/surprise(9)/joy(20)/disgust(21) 四类合计仅 61 条（2.6%）。
   给小类写触发条件是对的，但为它们牺牲大类不划算。

---

## 环境与吞吐（2026-10-15 实测，本机）

- 本机 GPU：**RTX 5060 Laptop 8GB**，compute capability **(12,0)**（sm_120）。
  `torch 2.9.1+cu130` + `transformers 5.19.0`，CUDA 13.0，GPU 算子实测通过。
- 基座模型：**Qwen2.5-3B-Instruct**，fp16 共 5.8GB，8GB 卡塞得下（`device_map=auto`）。
- 冒烟（官方 3 条样例，空提示词走官方提示词）：**PASS**。
  `runs/smoke_base/submission.jsonl` + `performance_report.json`（parse_failure_count=0）。
  接线状态日志：`prompts=(none) rag.retrieve=missing rag.fusion=ok`。
- **单条端到端延迟 7.87 秒**（3 条实测：9.84 / 9.43 / 4.34 秒）。
  → 全量 2354 条 ≈ **5.1 小时**，不可能每轮都跑。

### 迭代约定（重要）

- **迭代对比**用固定分层子集 `runs/iter/val_input.jsonl`：**300 条**，
  按金标情绪每类最多 20 条（小类全取），seed=20261015。单轮约 **39 分钟**。
  清单见 `runs/iter/subset_manifest.json`。
- **子集的总体 Acc 不能和全量 val 直接比**（分层过采样了稀有类）。
  它只用于**同一子集上不同版本的相对对比**。
- **headline 数字**（要写进报告、要跟队友比的）必须用全量 2354 条重建。

---

## 第 1 轮 · 接线后基线（空提示词）— 进行中

- 命令：`deploy/participant/run_inference.py runs/iter/val_input.jsonl runs/iter_base`，
  `PARTICIPANT_PROMPT_DIR` 指向空目录 → 只有官方提示词。
- **待补**：跑完后填 情绪 Acc / 画像 Acc / BLEU-4 / ROUGE-L / 平均延迟。
- 这一轮的意义：把「接线层 + 融合层截断」的影响先量出来，
  后面换我的提示词就是干净的单一变量对比。

---

## 发现的两个 bug（都是别人的文件，已上报，不要自己改）

1. **`tools/fetch_model.py` 漏下分片索引**：`KEEP_PATTERNS` 只含 `.safetensors`，
   不含 `model.safetensors.index.json`。默认模型 Qwen2.5-3B-Instruct 是**分片**权重，
   缺索引直接报
   `OSError: Error no file named model.safetensors, or pytorch_model.bin, found in directory ...`
   → 冒烟测试第一步就挂，全队都会踩。我手工补下了该文件（35,581 字节）。
   修法：`KEEP_PATTERNS` 加 `model.safetensors.index.json`，并把 `--check` 一并校验它。
2. **`tools/eval_loop.py` 指向官方原件**：`INFERENCE_SCRIPT = GivenData/participant/run_inference.py`，
   而官方原件**没有接线层**，所以它永远只量官方基线，
   量不到角色②③的提示词、也量不到角色④的检索。
   → 「五个人用同一把尺子」这条目前是断的。修法：加 `--inference-script` 开关，
   默认指向 `deploy/participant/run_inference.py`。
