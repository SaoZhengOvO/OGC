# 上机记录 · 2026-10-16 · LoRA SFT 微调基线

## 一、基本信息

| 项 | 值 |
|---|---|
| 平台 | ☑ 魔搭 ModelScope DSW「方式三 AMD GPU 环境」（免费共享池） |
| 日期 / 起止时间 | 2026-10-16 23:05:11 → 2026-10-17 02:27:40（**3 小时 18 分 29 秒**，`elapsed_s = 11873.5`） |
| 操作人 | 队长（①） |
| 算力规格 | MI300X ×1，显存 191.7 GiB；23 vCPU / 200 GB RAM；`/mnt/workspace` 为 NFS 持久盘（1.0 PB） |
| 使用镜像 | 平台预装 `ubuntu22.04-rocm7.2.3-pytorch2.12.0`，`torch 2.12.0+git6bbd260`、`hip 7.2.53211`、Python 3.12.13 |
| 代码版本（commit） | 训练脚本 `tools/train_sft.py`；容器 HEAD 与本地 main 存在差异（容器工作区保留队友未提交改动，故训练时未做 `git pull`），数据与脚本 SHA 以本节末尾记录为准 |
| 本次目标（一句话） | 用 LoRA 在 18,249 条 SFT 样本上微调 3B 基座，看 2 epoch 能否在 4 小时内跑完并且 loss 收敛 |

## 二、本次要回答的问题

**只写一个。**

> **LoRA 微调在 MI300X 单卡上跑 2 epoch / 2,282 步需要多久、loss 收敛到什么水平、显存会不会成为瓶颈？**

**验收标准**（跑出什么算回答完了）：

> `runs/sft-v1/train.log.jsonl` 里出现 `{"event": "done"}`，且 `runs/sft-v1/final/adapter_model.safetensors` 落盘；总时长 < 4 小时（平台 1 小时无操作会自动关机，长任务必须一次跑完）。

## 三、执行命令

```bash
cd /mnt/workspace/OGC

# 1) 生成 SFT 数据（复用 deploy/participant 的官方提示词渲染，保证训练与推理同任务）
python3 tools/make_sft_data.py
#   -> data/processed/sft_train.jsonl 116,249,098 B / 18,249 条 / 8,139 段对话
#   -> data/processed/sft_dev.jsonl     2,664,869 B /    402 条

# 2) 正式训练（setsid nohup 挂后台，断 SSH 不影响）
setsid nohup python3 tools/train_sft.py \
  --out runs/sft-v1 \
  --max-length 2048 \
  --batch-size 16 \
  --grad-accum 1 \
  --epochs 2.0 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-targets q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --log-every 20 --eval-every 200 --save-every 300 \
  > /mnt/workspace/OGC/runs/sft-v1/train.out 2>&1 < /dev/null &

# 3) 观测（不打断训练）
tail -f runs/sft-v1/train.log.jsonl
```

**两次启动的原因（真实记录，不是笔误）**：第一次用 `--batch-size 48` 在 23:05:11 启动，第一步就在 cross-entropy 处 OOM（`Tried to allocate 13.91 GiB`，191.69 GiB 总量不够一次连乘），23:09:47 改用 `--batch-size 16` 重启。日志里因此有 **2 条 `start` 事件**。

## 四、结果

| 指标 | 数值 | 样本量 n | 备注 |
|---|---|---|---|
| 总步数 | **2,282** | 18,249 条 × 2 epoch ÷ 16 | 与 `steps_target` 一致 |
| 总耗时 | **11,873.5 s = 3 h 17 m 53 s** | — | 02:27:40 结束 |
| 稳态吞吐 | **3,556 – 3,626 token/s** | 全程 116 个采样点 | 方差极小，无掉速 |
| 单步耗时 | ≈ **5.2 s/step** | — | 约 62 步/分钟 |
| 可训练参数 | **29,933,568 / 3,115,872,256 = 0.96 %** | — | LoRA r=16 |
| 峰值显存 | ≈ 15.7 GB（早前 20 步 smoke 实测） | — | 191.7 GiB 里用了 8 % |
| 起始 loss | **2.3308**（step 1） | — | 第一次启动那次是 2.3559 |
| 末段 loss | **1.3519**（step 2280） | — | 见下方曲线 |
| dev_loss | **1.3749** | 402 条 dev | `{"event":"eval","step":2282,"final":true}` |
| 产出 | `runs/sft-v1/final/adapter_model.safetensors` **119,801,528 B**（114 MB） | — | 另有 `adapter_config.json`、`README.md` |

**loss 曲线（每 200 步采样，全部来自 `train.log.jsonl`）**

| step | loss | token/s | 时间 |
|---|---|---|---|
| 1 | 2.3308 | 1702 | 23:09:55 |
| 200 | 1.6848 | 3556 | 23:27:32 |
| 400 | 1.5673 | 3579 | 23:44:47 |
| 600 | 1.4661 | 3626 | 00:01:14 |
| 800 | 1.4830 | 3595 | 00:19:05 |
| 1000 | 1.4093 | 3580 | 00:36:26 |
| 1200 | 1.3018 | 3601 | 00:53:46 |
| 1400 | 1.3204 | 3593 | 01:10:31 |
| 1600 | 1.3174 | 3591 | 01:27:44 |
| 1800 | **1.2508** | 3602 | 01:44:36 |
| 2000 | 1.3610 | 3596 | 02:02:03 |
| 2200 | 1.3342 | 3592 | 02:19:36 |
| 2280 | 1.3519 | 3599 | 02:26:45 |

产物路径：

- `runs/sft-v1/train.log.jsonl`（134 行，含 2×start / 116×step / 8×eval / 7×save / 1×done）
- `runs/sft-v1/final/`（`adapter_model.safetensors` 119.8 MB + `adapter_config.json` 1,179 B + `README.md` 5,246 B）
- 中间存档 `runs/sft-v1/step-{300,600,900,1200,1500,1800,2100}/` 全部落在 NAS 上

## 五、资源消耗

| 项 | 数值 |
|---|---|
| 实际运行时长 | 3 h 18 m（`elapsed_s = 11873.5`） |
| 资源消耗 | 魔搭免费额度：**≈ 3.3 小时**（1 张 MI300X，192 GB） |
| 结束后余额/剩余券 | 见返回列表 `RemainingQuota`（秒），本次之后约 85 小时 |
| 是否已关机 | ☑ 是 ☑ 否 —— **本记录结束时未关机**（后续还要跑评测）；实例在 06:04–06:12 之间被平台按「1 小时无操作自动关闭」规则关闭 |

## 六、截图清单

| # | 内容 | 文件 | 已存 |
|---|---|---|---|
| 1 | 实例创建/启动页（卡型、镜像、时间） | `screenshots/01-instance.png` | ☐ |
| 2 | 终端跑起来（命令、时间、进度） | `screenshots/02-running.png` | ☐ |
| 3 | 跑完的结果 | `screenshots/03-result.png` | ☐ |
| 4 | 资源消耗页（卡时/AIQ，或 CPU 时长） | `screenshots/04-quota.png` | ☐ |
| 5 | 关机确认 | `screenshots/05-shutdown.png` | ☐ |

**说明**：本次训练全程通过 SSH 远程驱动，没有在浏览器里截图。可替代的机器可读证据已在第四节给全（完整 `train.log.jsonl` 134 行 + 适配器文件大小 + 时间戳），截图项见第八节。

## 七、结论

**这次回答了问题吗？**

- [x] 是 → 结论：**能跑，而且很快。** 2,282 步 / 2 epoch / 18,249 条样本在 3 小时 18 分内跑完，稳态 3,600 token/s，显存只用了 192 GB 里的 16 GB，loss 从 2.33 降到 1.35（末段 400 步在 1.25–1.36 之间抖动，基本平了）。**限制不在算力，在显存的一次性分配**——batch 48 会 OOM，batch 16 稳定。
- [ ] 否 → 卡在哪：____，下次需要：____

**对最终提交的影响**：这次结果会改变哪个决定？

> 1. **证明了 MI300X 单卡足以在 3 小时内完成一轮 LoRA SFT**，所以「是否微调」不再受算力约束，只受时间（截止 10-20 23:59）和收益约束。
> 2. **确认 batch 16 是安全值**，写进 `runs/GPU-RULES.md`：不要在大 batch 上调，OOM 发生时的报错是 `Tried to allocate 13.91 GiB`，看起来像显存不够，实际是 cross-entropy 的临时张量峰值。
> 3. 产物只有 114 MB，**可以一起打进镜像**（镜像预算 10.2 GB / 上限 50 GB，余量充足）。

## 八、未完成项

| 项 | 原因 | 补救方式 |
|---|---|---|
| 六节截图未存 | 全程 SSH 驱动，没有在浏览器里留图；事后截图补不出来 | 下一次上机（评测/构建镜像）时按模板补齐并保存到 `screenshots/` |
| 适配器效果未评测 | 训练结束时平台即将关机，先把产物落 NAS 保存 | 在 2026-10-17 的全量评测里对比「有/无适配器」的情绪 Acc |

## 九、下一步

> LoRA 适配器（114 MB）在 n=2,354 全量上的情绪 Acc 与画像 Acc，是否比纯提示词 v3 更高？如果高出的幅度小于 1 个百分点，就不值得在镜像里加 114 MB 和 LoRA 加载时间。
