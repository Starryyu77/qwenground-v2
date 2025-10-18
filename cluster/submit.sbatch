#!/bin/bash
#SBATCH -J qwenground
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o logs/%j.out

# ==== 根据学院环境调整以下部分（Conda/模块/路径）====
# 典型方式：启用 conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate qwenground || conda create -y -n qwenground python=3.10 && conda activate qwenground
fi

# 安装依赖（首次运行）
python -m pip install -U pip
python -m pip install -r requirements.txt

# 如果使用云端 Qwen API，需要环境变量 QWEN_API_KEY
if [ -z "$QWEN_API_KEY" ]; then
  echo "[ERROR] QWEN_API_KEY 未设置。请使用 sbatch --export=ALL,QWEN_API_KEY=sk-... 传入。" >&2
  exit 1
fi

# 进入项目并执行 AirSim 批处理任务（API 模式）
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
python scripts/run_batch_airsim.py cluster/tasks_airsim.json