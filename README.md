# QwenGround: 3D 视觉定位系统（Qwen-VL + 点云）

目标：输入自然语言（如“最左边的汽车”）、图片和 3D 点云，系统输出目标的 3D 包围盒，并生成标注好的 2D 图像与 3D 场景结果。

核心流程（端到端）：
- 2D 定位：使用 Qwen-VL（GGUF，llama-cpp-python + Metal）根据图像与文本输出目标 2D 边界框（bbox）。
- 3D 提升：利用相机参数将 2D 框反向投影为 3D 视锥，用视锥裁剪点云得到目标点云簇。
- 结果输出：计算该点云簇的 3D 包围盒（AABB/OBB），保存几何、JSON 与可视化图片。

目录说明：
- `src/locator/vision.py`：Qwen-VL 推理（llama-cpp-python），返回结构化 JSON bbox。
- `src/locator/geometry.py`：相机投影、视锥裁剪点云、3D 包围盒计算。
- `src/locator/io.py`：数据读写、可视化辅助（2D bbox 叠加、点云读写）。
- `scripts/generate_sample.py`：生成示例图片/点云/相机参数与 bbox 提示。
- `scripts/run_single.py`：单样本推理（图像+文本+点云+相机）完整输出。
- `scripts/run_batch.py`：读取 `cluster/tasks.json` 做批量处理。
- `cluster/submit.sbatch`：Slurm 提交脚本（示例，需按学院集群环境调整）。
- `cluster/tasks.json`：批处理任务清单（示例）。

快速开始（Mac M4 + Metal）：
1) 进入项目并创建虚拟环境
```
cd "~/Library/Mobile Documents/com~apple~CloudDocs/qwenground"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) 下载 Qwen-VL GGUF 模型（含 mmproj）
- 推荐：Qwen2-VL-Instruct GGUF（7B 或 2B），需同时下载主模型 `*.gguf` 与 `mmproj*.gguf` 文件。
- 将路径配置为环境变量：
```
export QWEN_MODEL_PATH=/absolute/path/to/qwen2-vl-instruct-q4_k_m.gguf
export QWEN_MMPROJ_PATH=/absolute/path/to/mmproj-qwen2-vl-instruct-f16.gguf
```
- 若 `llama-cpp-python` 未启用 Metal，可尝试：
```
pip install --force-reinstall --no-binary :all: llama-cpp-python
# 或使用 CMAKE_ARGS="-DLLAMA_METAL=ON" 重新编译
```

3) 运行示例数据（本地单样本）
```
python scripts/generate_sample.py
python scripts/run_single.py \
  --image sample_data/image.png \
  --pointcloud sample_data/pointcloud.pcd \
  --camera sample_data/camera.yaml \
  --prompt "最左边的汽车" \
  --output_dir outputs
# 如果尚未完成模型下载，可用示例 bbox 提示跳过 2D 检测：
python scripts/run_single.py \
  --image sample_data/image.png \
  --pointcloud sample_data/pointcloud.pcd \
  --camera sample_data/camera.yaml \
  --prompt "最左边的汽车" \
  --bbox_json sample_data/bbox_hint.json \
  --output_dir outputs
```
输出：
- `outputs/annotated.png`：叠加 2D bbox 的图像。
- `outputs/bbox3d.json`：3D 包围盒（中心、尺寸、旋转/朝向等）。
- `outputs/bbox3d.ply`：3D 包围盒几何（Open3D 可视化）。

批处理（学院 GPU 集群）：
- 将本项目与数据集上传到学院服务器（示例）：
```
# 本地 -> 集群
scp -r qwenground tianyu016@172.21.89.130:~/qwenground
```
- 在集群环境准备 Conda/依赖（按学院指南）
- 修改 `cluster/submit.sbatch` 以适配分区、GPU、环境激活等
- 提交任务：
```
sbatch cluster/submit.sbatch
```
- 任务脚本会执行：
```
python scripts/run_batch.py cluster/tasks.json
```
学院 GPU 使用手册参考：[NTUEEECluster/docs][0]

注意与建议：
- 模型：确保主模型 GGUF 与 mmproj 文件匹配版本（Qwen2-VL-Instruct）。
- 上下文长度：`n_ctx` 建议 4k~8k；Metal 加速时 `n_gpu_layers>0`。
- 结构化输出：我们强制 Qwen 返回 JSON，代码包含鲁棒解析与兜底策略。
- 相机坐标系：示例以相机坐标为世界坐标（R=I、t=0）；实际数据请填写真实外参。
- 远近裁剪：`near`、`far` 可在相机配置文件里调整，避免噪点。
- 安全：不要把集群账号密码写入脚本或仓库，登录使用交互式输入或密钥。

[0]: https://github.com/NTUEEECluster/docs.git