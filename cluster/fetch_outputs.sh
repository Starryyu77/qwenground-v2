#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash cluster/fetch_outputs.sh <remote_output_subdir> [<local_dest_dir>]
# 示例：
#   bash cluster/fetch_outputs.sh outputs/airsim_front_0 \
#       "/Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/qwenground/outputs/"
# 说明：
# - 从服务器 /projects/_hdd/SeeGround/qwenground/<remote_output_subdir> 抓取结果到本地 <local_dest_dir>
# - 支持 sshpass（如设置 SSH_PASS 且已安装），否则使用普通 scp。

REMOTE_OUTPUT_SUBDIR=${1:-}
LOCAL_DEST_DIR=${2:-}
if [ -z "$REMOTE_OUTPUT_SUBDIR" ]; then
  echo "[ERROR] 缺少参数：remote_output_subdir，例如 outputs/airsim_front_0" >&2
  exit 1
fi
if [ -z "$LOCAL_DEST_DIR" ]; then
  echo "[WARN] 未提供本地目标目录，将默认保存到 ./outputs/" >&2
  LOCAL_DEST_DIR="$(pwd)/outputs"
fi

SERVER_IP="${SERVER_IP:-172.21.89.130}"
SERVER_USER="${SERVER_USER:-tianyu016}"
REMOTE_DIR="/projects/_hdd/SeeGround/qwenground/${REMOTE_OUTPUT_SUBDIR}"

mkdir -p "$LOCAL_DEST_DIR"

SSH_OPTS="${SSH_OPTS:-"-o StrictHostKeyChecking=no -o LogLevel=ERROR"}"
SCP_CMD="scp $SSH_OPTS -C -r"
if [ -n "${SSH_PASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
  SCP_CMD="sshpass -p "$SSH_PASS" scp $SSH_OPTS -o StrictHostKeyChecking=no -o LogLevel=ERROR -C -r"
fi

$SCP_CMD "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}" "$LOCAL_DEST_DIR/"

echo "已拉取：${REMOTE_DIR} -> ${LOCAL_DEST_DIR}/"