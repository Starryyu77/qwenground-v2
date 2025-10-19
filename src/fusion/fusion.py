import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .semantics import compute_bbox_overlap_ratio


VEHICLE_NAMES = {"vehicle", "car", "truck", "bus"}


def fuse_yolo_with_semantics(
    image_path: str,
    yolo_results: List[Dict[str, Any]],
    semantic_mask: Optional[np.ndarray],
    vehicle_class_id: Optional[int],
    min_vehicle_overlap: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    将 YOLO 检测与语义掩码融合：
    - 若提供 semantic_mask 与 vehicle_class_id，则计算 bbox 在 vehicle 类上的重叠占比，并按阈值筛选。
    - 若未提供，则直接返回 YOLO 结果（附加标记：semantics_used=false）。
    yolo_results 条目示例：{"bbox": [x1,y1,x2,y2], "cls_name": "car", "conf": 0.56}
    输出在每条中附加：vehicle_overlap、passed_semantic_filter、semantics_used。
    同时绘制 annotated_mm.png（红色未通过，绿色通过）。
    """
    fused = []
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    for det in yolo_results:
        bbox = det.get('bbox')
        cls = str(det.get('cls_name', '')).lower()
        conf = float(det.get('conf', 0.0))
        record = {
            'bbox': bbox,
            'cls_name': cls,
            'conf': conf,
        }
        if semantic_mask is not None and vehicle_class_id is not None:
            overlap = compute_bbox_overlap_ratio(semantic_mask, tuple(bbox), vehicle_class_id)
            record['vehicle_overlap'] = overlap
            record['semantics_used'] = True
            record['passed_semantic_filter'] = (overlap >= min_vehicle_overlap)
            # 可选：融合置信度
            record['fused_score'] = conf * (0.5 + 0.5 * overlap)
            color = 'green' if record['passed_semantic_filter'] else 'red'
        else:
            record['vehicle_overlap'] = None
            record['semantics_used'] = False
            record['passed_semantic_filter'] = True  # 未使用语义则不筛掉
            record['fused_score'] = conf
            color = 'yellow'
        # 绘制
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        fused.append(record)

    return fused, img


def filter_vehicle_detections(yolo_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从 YOLO 原始检测中过滤出车辆相关类别（car/truck/bus/vehicle）。
    统一输出结构：bbox, cls_name, conf。
    """
    filtered = []
    for det in yolo_raw:
        cls = str(det.get('cls_name', '')).lower()
        if cls in VEHICLE_NAMES:
            filtered.append({
                'bbox': det['bbox'],
                'cls_name': cls,
                'conf': float(det.get('conf', 0.0))
            })
    return filtered