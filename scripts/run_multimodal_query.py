#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from src.fusion.semantics import load_category_map, load_semantic_mask, derive_semantic_path_from_image
from src.fusion.fusion import filter_vehicle_detections, fuse_yolo_with_semantics


def run_yolo(image_path: str) -> List[Dict[str, Any]]:
    """
    运行YOLOv8推理，返回list[ {bbox:[x1,y1,x2,y2], cls_name:str, conf:float} ]
    使用环境变量配置：YOLO_MODEL、YOLO_CONF、YOLO_IOU、YOLO_DEVICE
    """
    if YOLO is None:
        raise RuntimeError("ultralytics 未安装或导入失败，请先 pip install ultralytics")

    model_path = os.environ.get('YOLO_MODEL', 'yolov8n.pt')
    conf = float(os.environ.get('YOLO_CONF', 0.10))
    iou = float(os.environ.get('YOLO_IOU', 0.60))
    device = os.environ.get('YOLO_DEVICE', 'cpu')

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, iou=iou, device=device)

    out = []
    for r in results:
        if not hasattr(r, 'boxes'):
            continue
        for b in r.boxes:
            # xyxy tensor
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            # class id to name
            cls_id = int(b.cls[0].item()) if hasattr(b, 'cls') else -1
            cls_name = str(r.names.get(cls_id, str(cls_id)))
            conf = float(b.conf[0].item()) if hasattr(b, 'conf') else 0.0
            out.append({'bbox': [x1, y1, x2, y2], 'cls_name': cls_name, 'conf': conf})
    return out


def main():
    parser = argparse.ArgumentParser("Multimodal Query Runner (RGB + Semantics + YOLO)")
    parser.add_argument('--image_path', type=str, required=True, help='显式图像路径')
    parser.add_argument('--sem_label_path', type=str, default=None, help='语义掩码路径（可选）')
    parser.add_argument('--category_json', type=str, default='airsim_camera_seg/v1.0-mini/category.json', help='类别定义JSON路径')
    parser.add_argument('--semantics_root', type=str, default=None, help='用于推断语义标签路径的根目录（可选）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--sem_min_vehicle_overlap', type=float, default=float(os.environ.get('SEM_MIN_VEHICLE_OVERLAP', 0.15)), help='语义过滤阈值')
    args = parser.parse_args()

    image_path = args.image_path
    sem_label_path = args.sem_label_path
    output_dir = args.output_dir or f"outputs/mmq_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)

    # 1) YOLO 推理
    yolo_raw = run_yolo(image_path)
    yolo_vehicle = filter_vehicle_detections(yolo_raw)

    # 2) 语义掩码加载与类别映射
    id_to_name, name_to_id = load_category_map(args.category_json) if os.path.exists(args.category_json) else ({}, {})
    vehicle_id = name_to_id.get('vehicle') if name_to_id else None

    semantic_mask = None
    sem_path_used = None
    if sem_label_path and os.path.exists(sem_label_path):
        semantic_mask = load_semantic_mask(sem_label_path)
        sem_path_used = sem_label_path
    else:
        cand = derive_semantic_path_from_image(image_path, args.semantics_root)
        if cand and os.path.exists(cand):
            semantic_mask = load_semantic_mask(cand)
            sem_path_used = cand

    # 3) YOLO + 语义融合
    fused, annotated_img = fuse_yolo_with_semantics(
        image_path=image_path,
        yolo_results=yolo_vehicle,
        semantic_mask=semantic_mask,
        vehicle_class_id=vehicle_id,
        min_vehicle_overlap=args.sem_min_vehicle_overlap,
    )

    # 4) 输出保存
    fused_json_path = os.path.join(output_dir, 'fused.json')
    with open(fused_json_path, 'w') as f:
        json.dump({
            'image_path': image_path,
            'sem_label_path': sem_path_used,
            'category_json': args.category_json,
            'sem_min_vehicle_overlap': args.sem_min_vehicle_overlap,
            'yolo_conf': float(os.environ.get('YOLO_CONF', 0.10)),
            'yolo_iou': float(os.environ.get('YOLO_IOU', 0.60)),
            'yolo_device': os.environ.get('YOLO_DEVICE', 'cpu'),
            'yolo_model': os.environ.get('YOLO_MODEL', 'yolov8n.pt'),
            'detections': fused
        }, f, ensure_ascii=False, indent=2)

    annotated_path = os.path.join(output_dir, 'annotated_mm.png')
    annotated_img.save(annotated_path)

    print(f"完成：输出目录 {output_dir}\n- {fused_json_path}\n- {annotated_path}")


if __name__ == '__main__':
    main()