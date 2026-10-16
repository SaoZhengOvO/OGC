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

> **2026-10-16 更新**：两个都已由角色①修掉。`eval_loop.py` 现在有
> `--inference-script wired|official`、`--prompt-dir`、`--model-path`，
> 台账还加了 `script` / `prompts` 两列做证据。`fetch_model.py` 的分片索引问题也已修。

---

# 第二轮：AMD MI300X 上的 v1 → v3（2026-10-16）

## 环境换到 MI300X

`ssh -p 30817 root@frp-rug.com`，代码在 `/mnt/workspace/OGC`，模型已就位。
`torch 2.12.0+git6bbd260`（HIP 7.2.53211）· 显存 191.7 GB · 单条 1.4–2.6 秒 ·
**一轮 smoke(n=20) 约 3.5 分钟**。

## 实验协议（控制变量）

`prompts/response.md` 是角色③在 22:50 交付的，而基线是 22:05 跑的。
为了让「只改了②的提示词」这个结论成立，**v2/v3 实验都把提示词目录隔离**：

```bash
python3 tools/eval_loop.py --tag <tag> --limit 20 \
    --inference-script wired --prompt-dir /mnt/workspace/OGC/runs/p02-only
```

## 逐版结果（n=20，val 前 20 条）

| 指标 | official(裸) | v1 | v2 | **v3** |
|---|---|---|---|---|
| 情绪 Acc | 0.5000 | 0.4500 | 0.4500 | **0.6000** |
| 画像 Acc（三维均值） | 0.4500 | 0.3167 | 0.2833 | **0.5000** |
| └ personality_traits | 0.6000 | 0.2500 | 0.2000 | **0.6000** |
| └ interests | 0.5500 | 0.4000 | 0.5500 | **0.6000** |
| └ style | 0.2000 | 0.3000 | 0.1000 | **0.3000** |
| personality 精确率 | 0.0000 | 0.1667 | — | **0.5000** |
| personality micro-F1 | 0.0000 | 0.2424 | — | **0.3077** |

**v3 三项验收门全部达成**（情绪 >0.50、画像 >0.45、style ≥0.30）。
相对裸基线：**情绪 +0.10、画像 +0.05**。

## 诊断（有数据支撑）

### 一、官方那棵树的画像几乎不填

`official/score.json` 三个维度都满足 `micro_f1 == 0` 且 `exact_match_acc == macro_jaccard`。
由 `src/evaluate.py:213-234`：`macro_jaccard` 在 `pred ∪ ref = ∅` 时记 1.0，
所以预测为空时 Jaccard 恒等于 EMA。
**官方画像 Acc 0.45 = 金标为空的比例，不是模型能力。**

> 因此 `COMPUTE.md` §6 第 3 条「标签词表对不上、改格式就能拿分」是**误诊**：
> 官方那棵树根本没有标签可对，micro-F1 归零是因为 `tp=0`。

这 20 条的金标为空率：personality 13/20、interests 13/20、style 4/20。
→ **personality / interests 的最优策略是「留空」，style 必须填。**

### 二、v1 为什么把 personality_traits 打成 0.25

v1 在画像段落里写了「一句话 → 标签」的示例映射，模型把它当成了填写许可：

- 输出 `['introverted','sensitive']` 这个**完全相同的组合出现 6 次**，
  精确对应 v1 那句 `"我不太敢跟陌生人说话" → introverted 或 sensitive`
- personality 共输出 24 个标签、命中 4 个（**精确率 16.7%**，远低于 50% 盈亏线）
- `该空却填` 7 条 → 把官方靠留空拿到的 12 条打掉 7 条

### 三、v1 为什么把情绪打成 0.45：anxiety 过度吸引

| v1 误判 | 次数 |
|---|---|
| `anger → anxiety` | 3 |
| `helplessness → anxiety` | 2 |
| `fear → anxiety` | 1 |

v1 第一步的情绪词示例「我好焦虑／开心／难受／烦死了／有点慌」里
**3 个指向负面、1 个指向焦虑、0 个指向 anger/fear/helplessness**；
加上第二步 anxiety 那行写了 6 个触发词（其他类只有一句）→ 模型学会「默认焦虑」。
但用户原话里其实**明说了**：「更烦躁」「想摔键盘」「更无力了」「只觉得害怕」。

### 四、v2 的两个新失败（改过头的代价）

- personality：删掉正面示例后，模型把「情绪克制」反向映射成
  **`emotionally_stable`（输出 7 次）**，而该标签在训练集只占非空的 1.0%
  —— 换了套错标签，`该空却填` 反而升到 9 条。
- style：**`brief` 输出了 17 次**（20 条里 17 条），只命中 1 条。
  根因是把 `colloquial/formal` 删掉后，模型退回到列表里最「客观」的长度标签当默认值。

**教训：只要提示词里给了「整句 → 标签」的正面示例，模型就会照搬。
填充分数低的维度应该给负面示例（把"不要填"的情形列出来），而不是正面示例。**

## v3 是怎么修的

1. **personality / interests 全部改成「默认输出 []」+ 只列负面情形**，去掉一切正面示例；
   并点名禁止把「情绪克制」当成 `emotionally_stable`。
2. **style 换成可判定的措辞信号**，按优先级排：
   直接向助手提要求/下指令/划边界 → `direct`；
   给因果、做对比 → `rational`；
   夸张比喻、密集感叹 → `high_emotional_expression`；
   ≤15 字 → `brief`；三段以上 → `detailed`。
   并明确「不要用 colloquial/formal/indirect/humorous 当默认值」。
   这一改把 `direct` 从 2 条提到 5 条（金标 direct 有 9 条），style 精确率从 0.22 提到 0.50。
3. **情绪**：明确「anxiety 和 relaxed 都不是默认答案」，并补两条排除规则
   「只是礼貌回应/下指令/陈述事实 → neutral」「对助手讽刺或拒绝 → anger」。
   情绪从 0.45 提到 0.60（v1 那 6 条 `→ anxiety` 全修好，其中 2 条还多了 2 个正确）。

## 待办

- [ ] **全量 2354 条确认**（`--tag v3-full`，已 `setsid nohup` 挂在服务器后台；
      注意 box 上同时有别人的 `mi300x-v4-full` 与 `train_sft`，GPU 共享会变慢）
- [ ] 全量数字出来后更新本表，并把 `prompts/emotion_profile.md` 定稿
- [ ] `prompts/response.md` 已由角色③交付；若要联合评测，
      去掉 `--prompt-dir` 隔离参数即可（但那样结论不再是「只改了②」）

## 复现命令（box 上）

```bash
cd /mnt/workspace/OGC
# 裸基线
python3 tools/eval_loop.py --tag smoke-official --limit 20 --inference-script official
# 当前提示词的接线版（隔离目录，只含 emotion_profile）
python3 tools/eval_loop.py --tag v3-smoke --limit 20 --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-only
# 全量
python3 tools/eval_loop.py --tag v3-full --inference-script wired \
    --prompt-dir /mnt/workspace/OGC/runs/p02-only
```
