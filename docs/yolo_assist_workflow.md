# YOLO 助手（Plan B）端到端流程指南

本指南汇总了在“AirSim 摄像机分割”场景下使用 YOLO 助手（Plan B）完成目标定位的完整流程，包括批处理（旧方法）、单图测试、参数配置、日志与输出拉取、常见问题与排错。该方案保持“fail fast”（无兜底）的原则：若未找到足够有效的车辆候选，则任务立即失败，以便快速迭代参数和样本。

更新时间：2025-10-18

---

## 0. 前置条件
- 代码已更新到最新（包含以下改动）：
  - scripts/run_airsim_camera_seg.py 新增 `--image_path` 参数，支持显式传入图片路径进行单图测试。
  - 支持通过环境变量调整 YOLO 阈值与设备选择：
    - `YOLO_CONF`（默认 0.10）、`YOLO_IOU`（默认 0.60）
    - `YOLO_MIN_W`、`YOLO_MIN_H`、`YOLO_MIN_AREA_RATIO`（用于过滤过小候选框）
    - `YOLO_DEVICE`（如 `cuda:0` 或 `cpu`）
    - `YOLO_MODEL`（如指定 `yolov8s.pt` 或默认优先使用预下载的 `yolov8n.pt`）
  - cluster/submit_vllm.sbatch 默认导出了一组 YOLO 配置（可被用户环境变量覆盖）。
- 服务器环境（示例路径）：
  - Conda 环境：`/projects/_hdd/SeeGround/envs/qwenground`
  - 数据集根目录：`/projects/_hdd/SeeGround/qwenground/airsim_camera_seg`
  - 任务与脚本：`/projects/_hdd/SeeGround/qwenground/cluster` 与 `scripts`
  - 已安装/可安装 `ultralytics` 包（YOLOv8）
  - Slurm 可用（`sbatch` 提交）

---

## 1. 批处理（旧方法）快速开始
该方法通过 `cluster/tasks_airsim_vllm.json` 批量运行任务，使用 `sbatch` 提交。

1) 拉代码并准备环境
- `git pull`
- `source /cluster/apps/software/Miniconda3/25.5.1-0/etc/profile.d/conda.sh`
- `conda activate /projects/_hdd/SeeGround/envs/qwenground`
- `pip install -U ultralytics`

2) 设置更宽松的 YOLO 参数（可选，提高召回）
- `export YOLO_CONF=0.10`
- `export YOLO_IOU=0.60`
- `export YOLO_MIN_W=12`
- `export YOLO_MIN_H=12`
- `export YOLO_MIN_AREA_RATIO=0.001`
- `export YOLO_DEVICE=cuda:0`
- 可选更强模型：`export YOLO_MODEL=/projects/_hdd/SeeGround/models/yolo/yolov8s.pt`（若无则使用 `yolov8n.pt`）

3) 编辑批任务（示例）
- 打开并编辑 `/projects/_hdd/SeeGround/qwenground/cluster/tasks_airsim_vllm.json`，在 YOLO 助手任务中指定合适的通道与索引：
```
{
  "dataset_root": "/projects/_hdd/SeeGround/qwenground/airsim_camera_seg",
  "channel": "CAM_LEFT_id_1",      // 示例：左侧相机
  "index": 0,                       // 建议先用 0（更近更大）；严格对某张图需先查该通道的 key_frame 的 index
  "prompt": "画面中最右侧的车辆。",   // 可简化为单一目标
  "output_dir": "outputs/airsim_left_rightmost_yolo_1",
  "use_vllm": true,
  "yolo_assist": true
}
```

4) 提交作业
- `sbatch /projects/_hdd/SeeGround/qwenground/cluster/submit_vllm.sbatch`

5) 监控日志
- 查看 `logs/<jobid>.out` 或使用 `bash cluster/check_job.sh <job_id>`
- 成功：日志中会出现“输出完成：outputs/airsim_left_rightmost_yolo_1”之类提示
- 失败：若出现“YOLO 未发现足够有效的车类候选，任务失败。”，请继续放宽阈值、切换更近样本（index=0）、或更换通道（如 `CAM_FRONT_id_0`）

6) 拉取输出到本地
- 推荐用引号包裹本地路径（注意 iCloud 路径名为 `com~apple~CloudDocs` 而非空格）：
  - `bash cluster/fetch_outputs.sh outputs/airsim_left_rightmost_yolo_1 "/Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/qwenground/outputs/"`
- 或不传第二个参数，默认拷贝到当前目录的 `./outputs`：
  - `bash cluster/fetch_outputs.sh outputs/airsim_left_rightmost_yolo_1`
- 成功后检查：
  - `outputs/airsim_left_rightmost_yolo_1/bbox2d.json`（应包含 `"yolo_assist": true` 与 `"yolo_candidates"`）
  - `outputs/airsim_left_rightmost_yolo_1/annotated.png`

7) 切换统一输入目录（供下游任务使用）
- `bash cluster/switch_inputs.sh outputs/airsim_left_rightmost_yolo_1`

---

## 2. 单图测试（本地或服务器均可）
用于快速验证指定图片上的 YOLO 助手效果。

- 命令示例（服务器）：
```
python scripts/run_airsim_camera_seg.py \
  --image_path /projects/_hdd/SeeGround/qwenground/airsim_camera_seg/sweeps/CAMERA_LEFT_id_1/1624109527058040832.png \
  --use_vllm true \
  --yolo_assist true \
  --output_dir outputs/single_test_yolo
```
- 在执行前可同样导出 YOLO 环境变量以放宽阈值（见上文 1) - 2)）。
- 若 `sample_data.json` 可匹配到该图片路径，脚本会尽量补全样本元数据（token、intrinsic）；否则仍以显式图片路径进行处理。

---

## 3. 如何为特定图片查找 index（批处理场景）
若需在批任务中严格对齐到某张图片的 `index`，可以在服务器上运行一个辅助脚本：

```python
import json, os
root = "/projects/_hdd/SeeGround/qwenground/airsim_camera_seg/v1.0-mini"
channel = "CAM_LEFT_id_1"
img_name = "1624109527058040832.png"
with open(os.path.join(root, "sample_data.json"), "r") as f:
    sd = json.load(f)
target = None
for it in sd["data"]:
    fn = os.path.basename(it.get("filename", ""))
    if it.get("channel") == channel and fn == img_name:
        target = it
        break
print({
  "index": target.get("index") if target else None,
  "is_key_frame": target.get("is_key_frame") if target else None,
  "token": target.get("token") if target else None,
})
```
将脚本输出贴给我们，我们可以据此在 `tasks_airsim_vllm.json` 中设置对应的 `index`。

---

## 4. 参数与默认值说明（可通过环境变量调整）
- `YOLO_CONF`: 检测置信度阈值（默认 0.10，降低能提高召回但会引入更多噪声）
- `YOLO_IOU`: NMS 的 IoU 阈值（默认 0.60）
- `YOLO_MIN_W` / `YOLO_MIN_H`: 最小候选框宽高（默认脚本内设置，提交脚本也会导出可覆盖的默认值）
- `YOLO_MIN_AREA_RATIO`: 最小面积比例（相对整图面积，过滤极小框）
- `YOLO_DEVICE`: 推理设备，如 `cuda:0` 或 `cpu`
- `YOLO_MODEL`: YOLOv8 模型权重路径（如 `yolov8s.pt` 提升召回；无则用 `yolov8n.pt`）
- 提交脚本 `cluster/submit_vllm.sbatch` 会导出默认 YOLO 配置（如 `YOLO_CONF=0.10`、`YOLO_IOU=0.60` 等），你也可以在提交前手动导出覆盖这些默认值。

---

## 5. 常见问题与排错
- 任务失败：`RuntimeError: YOLO 未发现足够有效的车类候选`
  - 放宽阈值（降低 `YOLO_CONF`、降低最小宽高/面积比例）
  - 切换更近的样本（`index=0` 往往更大更清晰），或换通道为 `CAM_FRONT_id_0`
  - 使用更强模型（`yolov8s.pt`）
  - 简化提示词为单一目标（例如“画面中最右侧的车辆。”）
- 本地路径包含空格（iCloud）导致 `Permission denied`
  - 路径名应为 `com~apple~CloudDocs`（非空格）
  - 使用引号包裹整条路径，或用反斜杠转义空格
  - 可创建无空格软链接以简化：
    - `ln -s "/Users/<you>/Library/Mobile Documents/com~apple~CloudDocs/qwenground" "$HOME/qwenground"`
- 服务器源目录不存在（`scp: No such file or directory`）
  - 先在服务器上用 `ls -al /projects/_hdd/SeeGround/qwenground/outputs/` 确认输出目录确已生成
  - 若不存在，说明批任务未成功或 `output_dir` 名称不一致
- 原则提醒：方案保持“fail fast”，未引入兜底逻辑；请通过参数与样本选择提高成功率。

---

## 6. 参考文件与路径
- 批任务文件：`cluster/tasks_airsim_vllm.json`
- 提交脚本：`cluster/submit_vllm.sbatch`
- 单图脚本：`scripts/run_airsim_camera_seg.py`（支持 `--image_path`）
- 输出与验证：`outputs/<your_output_dir>/bbox2d.json` 与 `outputs/<your_output_dir>/annotated.png`
- 下游输入切换：`cluster/switch_inputs.sh outputs/<your_output_dir>`

如需我们协助直接修改 `tasks_airsim_vllm.json`、准备 `sbatch` 命令或根据日志进一步微调参数，请将相关信息（日志或 `bbox2d.json`）发给我们。