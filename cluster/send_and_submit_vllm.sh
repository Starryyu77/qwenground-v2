#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash cluster/send_and_submit_vllm.sh [SERVER_IP] [SERVER_USER]
# 说明：
# - 将本地项目（排除 .venv、outputs、大型数据）同步到服务器 /projects/_hdd/SeeGround/qwenground/
# - 在服务器上提交 vLLM Slurm 任务（批处理 AirSim），无需云端 API Key
# - 可通过环境变量 VLLM_MODEL/VLLM_BASE_URL/VLLM_API_KEY 进行覆盖

SERVER_IP_ARG=${1:-"172.21.89.130"}
SERVER_USER_ARG=${2:-"tianyu016"}

SERVER_IP="$SERVER_IP_ARG"
SERVER_USER="$SERVER_USER_ARG"
REMOTE_DIR="/projects/_hdd/SeeGround/qwenground"  # 避免 home 配额，使用 /projects
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # 仓库根目录
# 加强 SSH 选项，避免登录横幅/TTY 干扰 rsync 协议
SSH_OPTS="${SSH_OPTS:- -o StrictHostKeyChecking=no -o LogLevel=ERROR -T}"

# 选择认证方式：优先使用 sshpass（若提供 SSH_PASS 且已安装），否则走普通 ssh
RSYNC_SSH="ssh $SSH_OPTS"
SSH_CMD="ssh $SSH_OPTS"
if [ -n "${SSH_PASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
  RSYNC_SSH="sshpass -p "$SSH_PASS" ssh $SSH_OPTS"
  SSH_CMD="sshpass -p "$SSH_PASS" ssh $SSH_OPTS"
fi

# 0) 预创建远程目录，避免 rsync 到 home 导致配额错误
$SSH_CMD "${SERVER_USER}@${SERVER_IP}" "mkdir -p ${REMOTE_DIR} ${REMOTE_DIR}/logs ${REMOTE_DIR}/cluster ${REMOTE_DIR}/scripts ${REMOTE_DIR}/src"

# 1) 分模块同步，避免一次性同步大量数据造成协议中断或配额问题
#    - 保留 --delete 仅用于代码目录，确保远端与本地一致
#    - 排除 .venv、outputs、airsim_camera_seg/sweeps 和 samples 等大体量数据
rsync -avz --delete --exclude '.venv' --exclude '.DS_Store' -e "$RSYNC_SSH" \
  "$LOCAL_DIR/cluster/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/cluster/"
rsync -avz --delete --exclude '.venv' --exclude '.DS_Store' -e "$RSYNC_SSH" \
  "$LOCAL_DIR/scripts/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/scripts/"
rsync -avz --delete --exclude '.venv' --exclude '.DS_Store' -e "$RSYNC_SSH" \
  "$LOCAL_DIR/src/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/src/"
# 根目录零散文件
rsync -avz --exclude '.venv' --exclude '.DS_Store' --exclude 'outputs/' --exclude 'airsim_camera_seg/sweeps/' --exclude 'airsim_camera_seg/samples/' -e "$RSYNC_SSH" \
  "$LOCAL_DIR/README.md" "$LOCAL_DIR/requirements.txt" \
  "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/"
# 可选：仅同步必要的 airsim 元数据（不包含 sweeps 大数据），如需请取消注释
# rsync -avz --exclude '.DS_Store' --exclude 'sweeps/' --exclude 'samples/' -e "$RSYNC_SSH" \
#   "$LOCAL_DIR/airsim_camera_seg/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/airsim_camera_seg/"

# 2) 服务器上提交 vLLM 任务
$SSH_CMD "${SERVER_USER}@${SERVER_IP}" \
  "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --export=ALL ${REMOTE_DIR}/cluster/submit_vllm.sbatch"

echo "已提交 vLLM 任务。查看队列："
echo "  ssh ${SERVER_USER}@${SERVER_IP} 'squeue -u ${SERVER_USER}'"
echo "查看日志："
echo "  ssh ${SERVER_USER}@${SERVER_IP} 'tail -f ${REMOTE_DIR}/logs/<jobid>.out'"