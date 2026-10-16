# 算力变更通告：一台 AMD MI300X（192 GB）

> 2026-10-16 落地。**这份文档推翻了 `START-HERE.md` 里的若干前提，先读这一份。**

---

## 一、现在有什么

| 项 | 实测值 |
|---|---|
| 卡 | **AMD Instinct MI300X 级**（`gfx942`）· **192 GB HBM3** · ECC 已开（`sramecc+`） |
| 实测算力 | bf16 **199.3 TFLOPS** / fp16 199.9 / bf16@n=16384 **204.5 TFLOPS** |
| CPU / 内存 | 23 核 / 200 GB |
| 系统 | Ubuntu 22.04.5 LTS |
| ROCm | **7.2.3** |
| Python | 3.12.13 |
| torch | **2.12.0+git6bbd260**（HIP 7.2.53211，`torch.cuda.is_available() == True`） |
| 显存（torch 视角） | **191.7 GB** |
| transformers / accelerate | 5.15.1 / 1.14.0 |
| pip 源 | 已配 `mirrors.cloud.aliyuncs.com`（阿里云内网，不用自己设） |
| 共享盘 | NAS 1 PB，`/mnt/workspace` 与 `/mnt/data` 是**同一块** |

**结论：一台 MI300X 的算力约等于 8 张 V100-16G。** 3B 级别的模型在这里是「跑得动」的量级，不是「勉强跑」——**微调、全量评测、多方案并行都成为可能**，这是之前按 CPU/小卡做规划时不敢想的前提。

---

## 二、怎么连进去（一条命令）

```
ssh -p 30817 root@frp-rug.com
```
密码：`OGC-amd-2026`

网关走 SakuraFrp 隧道（会员节点 `济南三线PLUS1`，端口固定，不会变）。网络不理想时可按运营商换域名，指向同一个隧道：

- 电信 `dx.frp-rug.com:30817`
- 移动 `yd.frp-rug.com:30817`
- 联通 `lt.frp-rug.com:30817`

### Windows 上想少打字

在 `C:\Users\<你>\.ssh\config` 里加：

```
Host amd
    HostName frp-rug.com
    Port 30817
    User root
    StrictHostKeyChecking accept-new
```

之后 `ssh amd` 即可。

---

## 三、代码和模型都在哪

```
/mnt/workspace/OGC/                        ← 仓库（已 clone）
/mnt/workspace/OGC/GivenData/participant/model/    ← 5.8 G，两片权重齐全
```

**模型已经下好了**（`model-00001/00002-of-00002.safetensors` + `model.safetensors.index.json` + tokenizer），不要重复下。

---

## 四、两个坑，踩了要重来

### 坑 1：绝对不要跑 `bootstrap.py` 的装 torch 那一步

```bash
# ❌ 千万别跑
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

**这是 ROCm 机器。** CUDA 版 torch 会把 ROCm 版顶掉，GPU 直接失效。`tools/bootstrap.py` 的 `--torch-index` 默认值恰好就是 `cu130`——**不要用它装环境**，环境已经全好了，一个 pip 都不用装。

### 坑 2：单次最长 10 小时，会被自动关机

实例 `AutoShutdown: true`，闲置即关。**被关机后隧道地址不变**，但实例里所有进程都没了。重新开机后先跑一次：

```bash
bash /mnt/workspace/.team-ssh/frpc-keeper.sh &
```

（这个脚本在共享盘上，跟着 NAS 持久化，不会被实例回收。）

---

## 五、跑一次评测要多久

| 阶段 | 耗时 |
|---|---|
| 权重加载（434 分片） | **2 分 38 秒** |
| 单条生成 | 1.4 – 2.6 秒 |
| `cli.py build`（展开官方数据） | 6.25 秒 |
| smoke 评测（20 条，含加载） | 约 3 分 30 秒 |

**全量 2354 条估算：约 1.5 – 2 小时**（含加载）。比之前按 CPU 规划的「跑一次一夜」强太多。

---

## 六、基线数字（2026-10-16 实测，n=20 smoke）

```
                    official(裸)   wired(接线)    delta
  情绪 Acc           0.5000        0.4500       -0.050
  画像 Acc(三维均值)  0.4500        0.3167       -0.133
    personality_traits 0.6000       0.2500       -0.350
    interests          0.5500       0.4000       -0.150
    style              0.2000       0.3000       +0.100
  回复 BLEU-4         0.0016        0.0001       -0.0015
  ROUGE-L             0.1093        0.0815       -0.028
```

- `official` = `GivenData/participant/run_inference.py`，官方原件，**无接线层**
- `wired` = `deploy/participant/run_inference.py`，接线层 = `提示词 emotion_profile` + `rag.fusion=ok` + **`rag.retrieve=missing`**
- 台账：`/mnt/workspace/OGC/runs/leaderboard.tsv`

### 从这张表能读出的四件事

1. **提示词 v1 目前是负收益**（4 项里 3 项下降，`personality_traits` 掉 0.35）。n=20 波动大，但方向一致，**不是噪声能解释的**。
2. **`style` 是唯一涨的维度**（0.20 → 0.30），说明提示词在「风格」方向是对的，在「性格/兴趣」方向写反了。
3. **`micro-F1` 全 0 而 `Jaccard` 有 0.55–0.60** —— 预测的标签和真值**措辞对不上**，集合精确匹配挂零、宽松匹配过半。**这是白捡的分。**
4. **`rag.retrieve=missing`** —— 检索层没接上，融合层在空转。

---

## 七、复现命令

```bash
ssh amd
cd /mnt/workspace/OGC

# 1) 展开官方数据（6 秒）
python3 src/cli.py build

# 2) 跑基线（裸官方原件）
python3 tools/eval_loop.py --tag smoke-official --limit 20 --inference-script official

# 3) 跑接线版（你的提示词/RAG 生效）
python3 tools/eval_loop.py --tag smoke-wired --limit 20 --inference-script wired

# 4) 看台账
cat runs/leaderboard.tsv
```

`--limit` 去掉就是全量（约 1.5–2 小时）。

---

## 八、谁该关心这份文档的哪一段

| 线 | 该看的 |
|---|---|
| ① 部署与评测 | 全篇。§6 的四个发现是你的产出 |
| ② 提示词·情绪与画像 | **§6 第 1、2 条** —— 你的 v1 在跌分，`style` 是唯一正确方向 |
| ③ 数据与标签 | **§6 第 3 条** —— 标签词表对不上，白捡的分 |
| ④⑤A RAG | **§6 第 4 条** —— `rag.retrieve=missing` |
| ⑤B 报告 | §1 硬件表 + §6 基线表可直接引用 |
