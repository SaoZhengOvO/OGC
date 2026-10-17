#!/bin/bash
# OGC 团队 SSH 自举脚本 —— 实例重启后跑一次即可恢复 SSH 隧道。
# 触发方式（三重，正常情况不用手动跑）：
#   1) /etc/profile.d/99-ogc-bootstrap.sh —— 有人打开 DSW 网页终端就跑
#   2) /usr/local/lib/python3.12/dist-packages/ogc_boot.pth —— 有 python 进程起来就跑
#   3) 手动：bash /mnt/workspace/.team-ssh/bootstrap.sh
# 密钥凭据（frpc 的隧道 ID）写死在这里；这是队内共享盘，不要往外发。
BIN=/mnt/workspace/.team-ssh/bin
DEBDIR=/mnt/workspace/.team-ssh/debs
FRPC_KEY=nukti4316xdcshm7:29551019
ROOT_PW=OGC-amd-2026

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh

# ---- 1) sshd 二进制：先吃 NAS 上的 deb 缓存（离线可用），再退回 apt ----
if [ ! -x /usr/sbin/sshd ]; then
  if ls $DEBDIR/*.deb >/dev/null 2>&1; then
    dpkg -i $DEBDIR/*.deb >/dev/null 2>&1
  fi
fi
if [ ! -x /usr/sbin/sshd ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq openssh-server >/dev/null 2>&1
fi
[ -x /usr/sbin/sshd ] || { echo 'SSHD-INSTALL-FAILED'; exit 1; }

# ---- 2) 配置：PermitRootLogin 必须是 yes ----
# 注意：prohibit-password / without-password 都会**禁止 root 用密码登录**，
# 队友看到的就是「密码明明对却 Permission denied」。这里刻意用 yes。
# 配合 MaxAuthTries 3 控制爆破面。
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
grep -q '^MaxAuthTries' /etc/ssh/sshd_config || echo 'MaxAuthTries 3' >> /etc/ssh/sshd_config
[ -f /etc/ssh/ssh_host_ed25519_key ] || ssh-keygen -A >/dev/null 2>&1

# ---- 3) root 密码与公钥（容器重建后两者都会丢）----
echo "root:${ROOT_PW}" | chpasswd 2>/dev/null || true
cat /mnt/workspace/.team-ssh/keys.pub > /root/.ssh/authorized_keys 2>/dev/null
chmod 600 /root/.ssh/authorized_keys

# ---- 4) 起 sshd：必须用 /usr/sbin/sshd ----
# 坑（2026-10-17 踩到）：DSW 自带 /etc/dsw/runtime/bin/sshd -p 22 -D 也匹配 'sshd -p 22'，
# 旧守卫误判「已在跑」直接跳过，端口被 DSW 的占着；而它不读 /etc/ssh/sshd_config，
# PermitRootLogin yes 永远不生效 —— 队长的私钥能进、队友的密码进不去。
pkill -f '/etc/dsw/runtime/bin/sshd' 2>/dev/null && sleep 2
if ! pgrep -f '/usr/sbin/sshd -p 22' >/dev/null 2>&1; then
  setsid nohup /usr/sbin/sshd -p 22 -D -o PermitRootLogin=yes -o PasswordAuthentication=yes >/tmp/sshd22.log 2>&1 </dev/null &
  sleep 2
  pgrep -f '/usr/sbin/sshd -p 22' >/dev/null 2>&1 && echo 'sshd-started' || echo 'sshd-MAYBE-FAILED'
fi

# ---- 5) 起 frpc 隧道（地址固定 frp-rug.com:30817，队友不用改命令）----
if ! pgrep -f 'frpc -f' >/dev/null 2>&1; then
  [ -x /usr/local/bin/frpc ] || { cp -f $BIN/frpc /usr/local/bin/frpc && chmod +x /usr/local/bin/frpc; }
  if [ -x /usr/local/bin/frpc ]; then
    setsid nohup /usr/local/bin/frpc -f $FRPC_KEY --disable_log_color >/tmp/frpc.log 2>&1 </dev/null &
    sleep 3
    pgrep -f 'frpc -f' >/dev/null 2>&1 && echo 'frpc-started' || echo 'frpc-MAYBE-FAILED'
  else
    echo 'FRPC-BINARY-MISSING'
  fi
fi

echo 'BOOTSTRAP-DONE'
