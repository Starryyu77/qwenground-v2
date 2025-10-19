# 多模态查询框架（RGB + 语义分割 + 3D框 + BEV）设计与使用指南

目标：在保留现有 YOLO 助手（Plan B）与文档的基础上，新增一个可融合图像、语义标签、3D边界框以及 BEV 语义的查询与推理框架，以提高查询精度与鲁棒性。

核心原则：
- 保留现有代码与文档，不破坏原有流程；新增为并行可选路径。
- 模块化设计，语义、检测、融合、查询各自独立，便于替换与扩展。
- 可渐进式增强：先实现“RGB+语义分割+YOLO”融合，后续再接入3D框与BEV。

---

## 目录结构建议
- `src/fusion/semantics.py`：语义掩码加载、类别映射、重叠比例计算。
- `src/fusion/fusion.py`：YOLO检测 + 语义融合，输出带语义信息的候选框。
- `scripts/run_multimodal_query.py`：单次运行入口，支持 `--image_path` 或 `channel/index`，输出融合结果与可视化。
- （可选）`src/fusion/queries.py`：将融合后的数据支持一些常见查询（如“最右侧车辆”、“距离最近车辆”等）。

---

## 数据与语义标签对齐
- RGB图像：位于 `airsim_camera_seg/sweeps/<CHANNEL>/<FILENAME>.png`（示例）
- 语义分割标签：与 RGB 图像一一对应（路径需提供或通过规则推断）。
- 类别定义：参考 `airsim_camera_seg/v1.0-mini/category.json`，包含如 `roadway`、`building`、`vehicle`、`others`，用于掩码类别ID映射。
- 初期实现：支持手动或约定路径传入语义标签（`--sem_label_path`），若无法推断则退化为仅 YOLO。

---

## YOLO 与语义融合逻辑（第一阶段）
1. 运行 YOLO 检测（ultralytics），取车辆相关类别（如 car/truck/bus 等）。
2. 对每个候选框，计算其在语义掩码中属于 `vehicle` 类的像素占比（overlap ratio）。
3. 根据阈值（如 `SEM_MIN_VEHICLE_OVERLAP=0.15`）过滤与重打分（例如置信度融合：`score' = yolo_conf * (0.5 + 0.5 * vehicle_overlap)`）。
4. 输出 `fused.json`（包含 bbox、类别、原始 YOLO 置信度、vehicle_overlap、是否通过语义过滤），并生成叠加可视化图 `annotated_mm.png`。

---

## 参数与环境变量（可扩展）
- YOLO相关（复用现有配置）：`YOLO_CONF`、`YOLO_IOU`、`YOLO_DEVICE`、`YOLO_MODEL`、小框过滤阈值等。
- 语义融合阈值：`SEM_MIN_VEHICLE_OVERLAP`（默认 0.15，可通过环境变量覆盖）。
- 输出目录：`--output_dir`（默认 `outputs/mmq_<timestamp>`）。

---

## 使用示例（单图）
```
python scripts/run_multimodal_query.py \
  --image_path /projects/_hdd/SeeGround/qwenground/airsim_camera_seg/sweeps/CAMERA_LEFT_id_1/1624109527058040832.png \
  --sem_label_path /projects/_hdd/SeeGround/qwenground/airsim_camera_seg/semantics/CAMERA_LEFT_id_1/1624109527058040832.png \
  --output_dir outputs/multimodal_left_1624109527
```
- 若未提供 `--sem_label_path` 且无法根据规则推断，将仅输出 YOLO 检测结果（语义字段为空）。

---

## 渐进式扩展（第二阶段）
- 3D边界框：读取 `v1.0-mini/sample_annotation.json` 与相关3D框数据，关联到图像坐标（需要内外参与投影），对 query 提供物理尺度与遮挡状态约束。
- BEV语义：引入栅格化鸟瞰语义（0.25m分辨率），对“道路内车辆”等查询进行空间过滤与排序。
- Query DSL：在 `src/fusion/queries.py` 定义可组合的查询算子（位置、类别、语义、距离/朝向等），上层可由 vLLM 将自然语言解析为结构化查询。

---

## 与现有流程的协同
- 保留 `scripts/run_airsim_camera_seg.py` 与 `cluster/submit_vllm.sbatch` 的用法；新脚本单独运行，不影响现有流程。
- 日志与输出格式对齐：`outputs/<dir>/fused.json` 与 `annotated_mm.png`；可选地在 `switch_inputs.sh` 中支持新的输出目录类型。

---

如需将该框架与批处理（tasks_airsim_vllm.json）联动，我们后续可以补充批运行脚本与任务模版，并在现有 `submit_vllm.sbatch` 中加入条件分支。当前先以单图入口验证与固化接口。