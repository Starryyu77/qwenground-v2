#!/bin/bash
#SBATCH -J qwenground-vllm
#SBATCH -p cluster02
#SBATCH --gres=gpu:v100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%j.out

set -euo pipefail

# Use the dedicated Conda env under /projects to avoid home quota
ENV_PREFIX="/projects/_hdd/SeeGround/envs/qwenground"

# Initialize Conda from cluster module and activate env
if [ -f "/cluster/apps/software/Miniconda3/25.5.1-0/etc/profile.d/conda.sh" ]; then
  source "/cluster/apps/software/Miniconda3/25.5.1-0/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX" || { echo "[ERROR] Failed to activate $ENV_PREFIX" >&2; exit 1; }
else
  echo "[ERROR] Miniconda3 (25.5.1-0) not found on cluster." >&2
  exit 1
fi

# Avoid home quota writes
export TMPDIR="/projects/_hdd/SeeGround/tmp"
export PIP_CACHE_DIR="/projects/_hdd/SeeGround/tmp/pip-cache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" logs
# HF/vLLM 缓存目录，避免写入 home 和 /tmp
export HF_HOME="/projects/_hdd/SeeGround/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export VLLM_WORKDIR="/projects/_hdd/SeeGround/cache/vllm"
# 将 HOME 与 XDG_CACHE_HOME 指向 /projects，避免 home 配额写入
export HOME="/projects/_hdd/SeeGround/home/${USER:-tianyu016}"
export XDG_CACHE_HOME="/projects/_hdd/SeeGround/cache/xdg"
# vLLM 注意力后端兼容 V100（sm70）：使用 PyTorch SDPA 作为后端，避免 FA2/FA3/TRITON_MLA 不兼容
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
unset VLLM_FLASH_ATTN_VERSION  # disabled for V100; avoid forcing FA version
# （可选）关闭 vLLM 使用统计写入，减少磁盘写入风险
export VLLM_DISABLE_USAGE_STATS="1"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$VLLM_WORKDIR" "$HOME" "$XDG_CACHE_HOME"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# 运行时信息打印，便于日志诊断
echo "[INFO] Node: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python - <<'PY'
import torch, vllm, sys
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA toolkit:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("vLLM:", vllm.__version__)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

# Remove user-level pip installs; dependencies must be preinstalled in the activated Conda env
# python3 -m pip install --user -U pip
# python3 -m pip install --user -r requirements.txt
# python3 -m pip install --user -U vllm transformers accelerate tiktoken

# 启动 vLLM OpenAI 兼容服务器
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
# 选择可用端口：若外部已指定且可用则沿用；否则自动选一个空闲端口
if [ -n "${VLLM_PORT:-}" ]; then
  IS_BUSY=$(python -c 'import socket,sys;s=socket.socket();res=s.connect_ex(("127.0.0.1",int(sys.argv[1])));s.close();print("busy" if res==0 else "free")' "$VLLM_PORT")
  if [ "$IS_BUSY" = "busy" ]; then
    echo "[WARN] 指定端口 $VLLM_PORT 已被占用，自动选择可用端口。"
    VLLM_PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  fi
else
  VLLM_PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
fi
echo "[INFO] 使用 vLLM host=${VLLM_HOST} port=${VLLM_PORT}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
MODEL_NAME=${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}

python -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype float16 \
  --download-dir "$HF_HUB_CACHE" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --served-model-name "${MODEL_NAME}" \
  &
VLLM_PID=$!

# 等待服务启动完成（查询 /v1/models）
RETRY=0
MAX_RETRY=120
SLEEP=5
until curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  RETRY=$((RETRY+1))
  if [ $RETRY -ge $MAX_RETRY ]; then
    echo "[ERROR] vLLM 服务器启动失败或超时（/v1/models 不可用）。" >&2
    kill $VLLM_PID || true
    exit 1
  fi
  echo "等待 vLLM 服务启动(${RETRY}/${MAX_RETRY})..."
  sleep $SLEEP
  # 若 vLLM 进程已提前退出，立即失败，避免无效等待
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM 进程已退出，请查看上方引擎日志。" >&2
    exit 1
  fi
 done

# 进一步验证 /v1/chat/completions 路由可用（使用最小文本消息）
CHAT_RETRY=0
CHAT_MAX_RETRY=60
CHAT_SLEEP=5
while true; do
  if curl -fsS -X POST "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"ping\"}]}], \"temperature\": 0.0, \"max_tokens\": 1}" \
    >/dev/null 2>&1; then
    echo "vLLM /v1/chat/completions 路由可用。"
    break
  fi
  CHAT_RETRY=$((CHAT_RETRY+1))
  if [ $CHAT_RETRY -ge $CHAT_MAX_RETRY ]; then
    echo "[ERROR] vLLM /v1/chat/completions 路由不可用或超时。" >&2
    kill $VLLM_PID || true
    exit 1
  fi
  echo "等待 chat 路由可用(${CHAT_RETRY}/${CHAT_MAX_RETRY})..."
  sleep $CHAT_SLEEP
  # 若 vLLM 进程已提前退出，立即失败，避免无效等待
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM 进程已退出，请查看上方引擎日志。" >&2
    exit 1
  fi
 done

# 运行批处理（使用 vLLM）
export VLLM_BASE_URL="${VLLM_BASE_URL}"
python -u scripts/run_batch_airsim.py cluster/tasks_airsim_vllm.json
STATUS=$?

# 结束 vLLM 服务
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true

exit $STATUS