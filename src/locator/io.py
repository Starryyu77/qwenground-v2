import json
import os
from typing import Dict, Any, Tuple

import numpy as np
# 使 open3d 成为可选依赖：仅在需要点云相关功能时再报错
try:
    import open3d as o3d
except Exception:
    o3d = None
from PIL import Image, ImageDraw
import yaml


def read_camera_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_bbox_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def overlay_bbox_on_image(img_path: str, bbox: Tuple[int, int, int, int], out_path: str, color=(255, 0, 0)) -> None:
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def load_point_cloud(path: str) -> np.ndarray:
    if o3d is None:
        raise RuntimeError("open3d 未安装或不可用，无法读取点云。请安装 open3d，或在不需要点云时避免调用该函数。")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    return pts


def save_point_cloud(path: str, points: np.ndarray) -> None:
    if o3d is None:
        raise RuntimeError("open3d 未安装或不可用，无法保存点云。请安装 open3d，或在不需要点云时避免调用该函数。")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)