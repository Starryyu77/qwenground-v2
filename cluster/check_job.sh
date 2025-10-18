#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash cluster/check_job.sh [<jobid>]
# 说明：
# - 不带参数：列出当前用户的队列（squeue -u）
# - 带 jobid：在服务器上 tail -f logs/<jobid>.out 以查看运行日志

SERVER_IP="172.21.89.130"
SERVER_USER="tianyu016"
REMOTE_DIR="/projects/_hdd/SeeGround/qwenground"

SSH_OPTS="${SSH_OPTS:- -o StrictHostKeyChecking=no -o LogLevel=ERROR}"
SSH_CMD="ssh $SSH_OPTS"
if [ -n "${SSH_PASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
  SSH_CMD="sshpass -p \"$SSH_PASS\" ssh $SSH_OPTS -o StrictHostKeyChecking=no"
fi

JOBID=${1:-}
if [ -z "$JOBID" ]; then
  $SSH_CMD "${SERVER_USER}@${SERVER_IP}" "squeue -u ${SERVER_USER}"
else
  $SSH_CMD "${SERVER_USER}@${SERVER_IP}" "tail -f ${REMOTE_DIR}/logs/${JOBID}.out"
fi