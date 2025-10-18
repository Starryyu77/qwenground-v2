import os
import json
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
import yaml
import open3d as o3d

BASE = os.path.dirname(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(BASE, "sample_data")


def make_image(path: str) -> Tuple[int, int]:
    w, h = 640, 480
    img = Image.new("RGB", (w, h), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    # 三个“汽车”矩形（左、中、右）
    left = (60, 180, 220, 320)
    mid = (240, 180, 400, 320)
    right = (420, 180, 580, 320)
    for rect, col in zip([left, mid, right], [(255, 0, 0), (0, 180, 255), (0, 200, 60)]):
        draw.rectangle(rect, outline=col, width=4)
    img.save(path)
    # 返回左车 bbox 作为 hint
    return w, h


def make_camera_yaml(path: str, width: int, height: int) -> None:
    cam = {
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": width / 2.0, "cy": height / 2.0},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 0.0],
        },
        "width": width,
        "height": height,
        "near": 0.5,
        "far": 50.0,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cam, f, allow_unicode=True)


def make_pointcloud(path: str) -> None:
    # 构造“左车”点簇（与左矩形投影一致）
    # 反投影估算：x像素 60..220 -> X ~ [-1.3, -0.5]（Z≈3, fx≈600）
    # y像素 180..320 -> Y ~ [-0.3, 0.4]
    Z_obj = 3.0
    Xs = np.random.uniform(-1.3, -0.5, size=5000)
    Ys = np.random.uniform(-0.3, 0.4, size=5000)
    Zs = np.random.normal(Z_obj, 0.1, size=5000)
    car_left = np.stack([Xs, Ys, Zs], axis=1)

    # 背景点
    bg_X = np.random.uniform(-2.0, 2.0, size=4000)
    bg_Y = np.random.uniform(-1.0, 1.0, size=4000)
    bg_Z = np.random.uniform(2.0, 8.0, size=4000)
    background = np.stack([bg_X, bg_Y, bg_Z], axis=1)

    pts = np.concatenate([car_left, background], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(path, pcd)


def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    img_path = os.path.join(SAMPLE_DIR, "image.png")
    cam_path = os.path.join(SAMPLE_DIR, "camera.yaml")
    pcd_path = os.path.join(SAMPLE_DIR, "pointcloud.pcd")
    bbox_hint_path = os.path.join(SAMPLE_DIR, "bbox_hint.json")

    w, h = make_image(img_path)
    make_camera_yaml(cam_path, w, h)
    make_pointcloud(pcd_path)

    # 保存左车 bbox 作为提示（便于跳过 2D 检测）
    bbox_hint = {"bbox": [60, 180, 220, 320], "label": "left-car"}
    with open(bbox_hint_path, "w", encoding="utf-8") as f:
        json.dump(bbox_hint, f, ensure_ascii=False, indent=2)

    print(f"生成完成：\n- {img_path}\n- {cam_path}\n- {pcd_path}\n- {bbox_hint_path}")


if __name__ == "__main__":
    main()