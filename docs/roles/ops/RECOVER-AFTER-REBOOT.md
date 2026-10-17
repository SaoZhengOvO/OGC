# 换 pod 后的一键恢复（唯一需要人做的动作）

> **为什么需要人**：容器换了新 pod，容器里的一切（`.pth` 钩子、profile.d、sshd 二进制、authorized_keys）全部重置。
> 只有 NAS `/mnt/workspace` 活下来。所以每次换 pod 都必须有人手动触发一次自举。

## 人只做这一步（30 秒）

1. 打开 `https://www.modelscope.cn/my/mynotebook`
2. 找到「**方式三 GPU 环境 NEW**」那张卡，点 **查看 Notebook**（不是「启动」，实例已经在跑了）
3. 如果跳到阿里云登录页 → **扫码登录一次**
4. 进去后在网页终端里粘这一行，回车：

```bash
bash /mnt/workspace/.team-ssh/bootstrap.sh
```

看到 `SSHD-STARTED` / `FRPC-STARTED` / `BOOTSTRAP-DONE` 就成功了。

**然后什么都不用做** —— SSH、隧道、root 密码、公钥会全部恢复，我这边接着跑。

## 为什么不能全自动（诚实说明）

| 我原以为 | 实际 |
|---|---|
| `.pth` 钩子放进 site-packages 就能活过重启 | **只在"容器还在、只是进程被杀"时有效** |
| 自愈实验 PASS = 重启不用人 | 实验只证明了"**钩子存在时**能触发"，**没证明钩子本身能活过换 pod** |

第一次自愈实验（2026-10-16 23:13）的做法是：删掉 sshd/authorized_keys/frpc/密码，保留 site-packages，然后起一个 python → 钩子触发 → 全部恢复。**那一步 PASS 是真的，但它证明的是"钩子能跑"，不是"钩子能活"。**

换 pod 后 `.pth` 不见了，所以钩子根本不存在，bootstrap 没被调用。

**结论：这台免费实例没有"换 pod 后无人值守"的办法**，除非把自举写进镜像本身（`commit_to_image` 是否覆盖这一点尚未验证，且当前 pod 的网络/镜像状态就够我们折腾的，不建议在剩 3.5 天时去赌）。

## 恢复后要立刻做的事

```bash
# 1) 确认 SFT 存档还在
ls /mnt/workspace/OGC/runs/sft-v1/

# 2) 从最新存档续训
cd /mnt/workspace/OGC
setsid nohup python3 tools/train_sft.py --epochs 2 --max-length 2048 \
  --batch-size 16 --grad-accum 1 --log-every 20 --eval-every 300 --save-every 300 \
  --resume runs/sft-v1/step-1500 --out runs/sft-v2 > /tmp/sft-v2.log 2>&1 < /dev/null &

# 3) 把本地补丁同步回机器（新 pod 的仓库可能落后）
git -C /mnt/workspace/OGC pull --rebase
```

## 记录

- 2026-10-16 22:15 起的 pod：`dsw-19vrgn0g4ntgjvy7iy`，2026-10-17 01:30 前后被平台关机
- 2026-10-17 12:52 起的 pod：**`dsw-oqlnz3djz9oou9obzr`**（gateway `dsw-2266641`），Status Running，SSH 隧道未自动恢复
- 该 pod 额度剩余 **316531 秒 ≈ 87.9 小时**
