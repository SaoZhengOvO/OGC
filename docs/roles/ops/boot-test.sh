#!/usr/bin/env bash
# 模拟一次「实例重启」：把运行时状态全清掉，然后只看 .pth 钩子能不能自己把 SSH 拉回来。
# 这是对「重启后不用人工介入」这个承诺的判决性实验。
# 用法：bash _boot-test.sh   然后断开，60 秒后重新 ssh 看能不能进。
set -u

LOG=/tmp/boot-test.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== 模拟重启开始 $(date '+%F %T') ==="

# 1) 清掉一切「重启后会消失」的东西
pkill -f "sshd -p 22" 2>/dev/null || true
pkill -f "/usr/local/bin/frpc" 2>/dev/null || true
rm -f /usr/sbin/sshd /root/.ssh/authorized_keys /tmp/.ogc-boot /tmp/bootstrap-boot.log
passwd -l root >/dev/null 2>&1 || true

echo "已清空：sshd 二进制 / authorized_keys / 触发标记 / root 密码已锁"
printf '  sshd 二进制: %s\n' "$(ls /usr/sbin/sshd 2>&1 | head -1)"
printf '  root 密码位: %s\n' "$(passwd -S root 2>&1)"
printf '  隧道进程数: %s\n' "$(pgrep -c -f '/usr/local/bin/frpc' || echo 0)"

sleep 3

# 2) 只靠 .pth：新起一个 python 进程，等价于「实例开机后 DSW 自己拉起 python」
echo "=== 新起一个 python 进程（模拟开机时 DSW 拉起 python）==="
python3 -c 'import sys; print("  python 起来了:", sys.version.split()[0])'
echo "  标记文件已生成? $([ -f /tmp/.ogc-boot ] && echo yes || echo no)"

# 3) 等钩子里那条后台 bootstrap 跑完
sleep 55

# 4) 判决
echo "=== 55 秒后 ==="
printf '  sshd 二进制:   %s\n' "$(ls -l /usr/sbin/sshd 2>&1 | head -1)"
printf '  root 密码位:   %s\n' "$(passwd -S root 2>&1)"
printf '  authorized_keys: %s\n' "$(ls -l /root/.ssh/authorized_keys 2>&1 | head -1)"
printf '  sshd 监听:     %s\n' "$(pgrep -f 'sshd -p 22' | head -1 || echo 无)"
printf '  frpc 进程:     %s\n' "$(pgrep -f '/usr/local/bin/frpc' | head -1 || echo 无)"
echo "  --- bootstrap 自启日志 ---"
tail -12 /tmp/bootstrap-boot.log 2>/dev/null || echo "  （没有 bootstrap-boot.log，钩子没触发）"

echo "=== 判决 ==="
if pgrep -f 'sshd -p 22' >/dev/null && pgrep -f '/usr/local/bin/frpc' >/dev/null && [ -f /root/.ssh/authorized_keys ]; then
  echo "PASS —— 全自动恢复成功，人不用碰"
else
  echo "FAIL —— 有环节没起来，逐项看上面"
fi
