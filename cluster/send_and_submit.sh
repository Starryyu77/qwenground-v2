#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash cluster/send_and_submit.sh <QWEN_API_KEY>
# 说明：
# - 将本地项目（排除 .venv）同步到服务器 ~/qwenground/
# - 然后在服务器上提交 Slurm 任务（批处理 AirSim），传入 QWEN_API_KEY
# - 会提示输入服务器密码（建议改用 SSH 密钥免密登录）

API_KEY_ARG=${1:-}
API_KEY="${API_KEY_ARG:-${QWEN_API_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "[ERROR] 缺少 QWEN_API_KEY。" >&2
  echo "用法：bash cluster/send_and_submit.sh sk-xxxx 或先执行 export QWEN_API_KEY=sk-xxxx" >&2
  exit 1
fi

SERVER_IP="172.21.89.130"
SERVER_USER="tianyu016"
REMOTE_DIR="/home/${SERVER_USER}/qwenground"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # 仓库根目录

SSH_OPTS="${SSH_OPTS:-}"

# 选择认证方式：优先使用 sshpass（若提供 SSH_PASS 且已安装），否则走普通 ssh
RSYNC_SSH="ssh $SSH_OPTS"
SSH_CMD="ssh $SSH_OPTS"
if [ -n "${SSH_PASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
  RSYNC_SSH="sshpass -p "$SSH_PASS" ssh $SSH_OPTS -o StrictHostKeyChecking=no"
  SSH_CMD="sshpass -p "$SSH_PASS" ssh $SSH_OPTS -o StrictHostKeyChecking=no"
fi

# 1) 同步到服务器（排除本地虚拟环境，保持目录结构一致）
rsync -avz --delete --exclude '.venv' --exclude '.DS_Store' -e "$RSYNC_SSH" \
  "$LOCAL_DIR/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/"

# 2) 服务器上提交任务（传入 API KEY）
$SSH_CMD "${SERVER_USER}@${SERVER_IP}" \
  "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --export=ALL,QWEN_API_KEY=${API_KEY} cluster/submit.sbatch"

echo "已提交任务。可用以下命令查看队列："
echo "  ssh ${SERVER_USER}@${SERVER_IP} 'squeue -u ${SERVER_USER}'"
echo "查看日志："
echo "  ssh ${SERVER_USER}@${SERVER_IP} 'tail -f ${REMOTE_DIR}/logs/<jobid>.out'"