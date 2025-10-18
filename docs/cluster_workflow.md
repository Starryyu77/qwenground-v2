# Qwenground 集群环境与推理工作流指南

本指南涵盖：在集群上通过 vLLM 运行 Qwen2-VL-7B-Instruct 完成 AirSim 2D 定位任务，并在本地 Mac 下载与验证结果的完整流程。内容与项目脚本保持一致，适配 V100 GPU 环境。

## 1. 目标与架构
- 服务器（Slurm + V100）上启动 vLLM（Qwen2-VL-7B-Instruct）服务。
- 使用脚本批量运行 AirSim 2D 定位任务（scripts/run_batch_airsim.py）。
- 通过 cluster/fetch_outputs.sh 将输出拉取到本地 Mac。

## 2. 前置条件
- 服务器账户与基础权限，能使用 Slurm 队列。
- 可选：本地设置 SSH_PASS 环境变量，以免交互执行 ssh/scp。
- 建议的 Conda 环境路径：/projects/_hdd/SeeGround/envs/qwenground。
- 依赖安装参考 requirements.txt；vLLM 在 V100 上的关键环境：
  - TORCH_SDPA=1（启用 SDPA 后端）
  - 取消设置 VLLM_FLASH_ATTN_VERSION（避免不兼容的 Flash-Attn）

## 3. 服务器环境准备（一次性）
```bash
# 登录服务器后：
module load anaconda  # 如集群有模块系统
conda create -y -p /projects/_hdd/SeeGround/envs/qwenground python=3.10
conda activate /projects/_hdd/SeeGround/envs/qwenground

# 安装依赖
pip install -r requirements.txt
pip install vllm

# 关键环境变量（建议在 sbatch 或启动脚本中设置）
export TORCH_SDPA=1
unset VLLM_FLASH_ATTN_VERSION
```

## 4. 代码与数据同步（本地发起）
- 推荐使用 cluster/send_and_submit_vllm.sh 同步并提交作业。
- 若需免交互，先在本地导出 SSH_PASS：
```bash
export SSH_PASS='你的服务器密码'
./cluster/send_and_submit_vllm.sh
```
- 如需单独同步数据集，可使用 rsync：
```bash
rsync -av --progress /path/to/local/dataset \
  <user>@<server>:/projects/_hdd/SeeGround/qwenground/datasets/
```

## 5. Slurm 作业与 vLLM 服务
- submit_vllm.sbatch 已适配 V100：请求 GPU 资源、在作业中设置 TORCH_SDPA=1 并禁用 Flash-Attn 版本。
- 作业会动态选择可用端口并启动 vLLM，日志位于服务器的 /projects/_hdd/SeeGround/qwenground/logs/<jobid>.out。
- 查看队列与日志（本地自动或手动）：
```bash
# 本地脚本（需 SSH_PASS 可免交互）
./cluster/check_job.sh queue
./cluster/check_job.sh tail <jobid>

# 服务器手动：
squeue -u $USER
tail -f /projects/_hdd/SeeGround/qwenground/logs/<jobid>.out
```

## 6. 批量任务配置（cluster/tasks_airsim_vllm.json）
- 关键字段：
  - dataset_root：数据集根目录（服务器路径）
  - channel：图像通道名（例如 "camera_seg"）
  - prompt：指令提示词
  - use_vllm：true（启用 vLLM 推理）
  - vllm_model："Qwen/Qwen2-VL-7B-Instruct"
  - output_dir：输出保存目录（服务器路径）
- 示例（单任务）：
```json
[
  {
    "dataset_root": "/projects/_hdd/SeeGround/qwenground/airsim_camera_seg/v1.0-mini",
    "channel": "camera_seg",
    "prompt": "请输出目标的2D边界框，格式为JSON {\"bbox\": [x1, y1, x2, y2]}。",
    "use_vllm": true,
    "vllm_model": "Qwen/Qwen2-VL-7B-Instruct",
    "output_dir": "/projects/_hdd/SeeGround/qwenground/outputs/airsim_vllm"
  }
]
```

## 7. 运行推理
- 本地一键提交：
```bash
./cluster/send_and_submit_vllm.sh
```
- 服务器单样本调试：
```bash
conda activate /projects/_hdd/SeeGround/envs/qwenground
export VLLM_BASE_URL=http://127.0.0.1:<port>
python scripts/run_airsim_camera_seg.py \
  --dataset-root /projects/_hdd/SeeGround/qwenground/airsim_camera_seg/v1.0-mini \
  --output-dir /projects/_hdd/SeeGround/qwenground/outputs/airsim_debug \
  --use-vllm --vllm-base-url "$VLLM_BASE_URL"
```
- 批量运行（由 sbatch 启动后，run_batch_airsim.py 根据 tasks_airsim_vllm.json 自动处理）：
```bash
export VLLM_BASE_URL=http://127.0.0.1:<port>
python scripts/run_batch_airsim.py --tasks cluster/tasks_airsim_vllm.json
```

## 8. 下载结果到本地
- 使用 cluster/fetch_outputs.sh（从本地 Mac 运行）：
```bash
# 交互式（输入服务器密码）
./cluster/fetch_outputs.sh airsim_vllm

# 非交互（需 SSH_PASS）
export SSH_PASS='你的服务器密码'
./cluster/fetch_outputs.sh airsim_vllm ./outputs/airsim_vllm
```
- 直接 scp 示例：
```bash
scp -r <user>@<server>:/projects/_hdd/SeeGround/qwenground/outputs/airsim_vllm \
  ./outputs/
```

## 9. 本地验证
```bash
ls -lah ./outputs/airsim_vllm
open ./outputs/airsim_vllm/*.jpg  # 预览图像
cat ./outputs/airsim_vllm/*.json | head -n 50
```

## 10. 常见问题排查
- vLLM 启动失败（EngineCore failed to start）：确认 TORCH_SDPA=1，取消 VLLM_FLASH_ATTN_VERSION，端口未被占用。
- 连接错误（Connection error）：确认 VLLM_BASE_URL 正确且服务已启动；防火墙与节点访问范围允许。
- 权限问题：确保输出目录存在且当前用户有写权限。
- 显存不足：减少 batch size 或并发请求。

## 11. 作业生命周期
```bash
# 查看队列
squeue -u $USER
# 取消作业
scancel <jobid>
# 查看日志
tail -f /projects/_hdd/SeeGround/qwenground/logs/<jobid>.out
```

## 12. 关键命令速查
```bash
# 提交与同步
./cluster/send_and_submit_vllm.sh
# 查看队列或日志
./cluster/check_job.sh queue
./cluster/check_job.sh tail <jobid>
# 下载输出
./cluster/fetch_outputs.sh <remote_subdir> [local_dir]
```