import os
import sys
import json
import argparse
from typing import Optional, Tuple

import numpy as np

from locator.vision import QwenVL2DDetector
from locator.geometry import Camera, frustum_crop_points, compute_bounding_box, bbox_geom_to_o3d
from locator.io import read_camera_yaml, overlay_bbox_on_image, load_point_cloud
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def make_camera(cam_cfg: dict) -> Camera:
    K = cam_cfg["intrinsics"]
    E = cam_cfg["extrinsics"]
    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])
    near = float(cam_cfg.get("near", 0.1))
    far = float(cam_cfg.get("far", 100.0))
    R = np.array(E["R"], dtype=float)
    t = np.array(E["t"], dtype=float)
    return Camera(
        fx=float(K["fx"]), fy=float(K["fy"]), cx=float(K["cx"]), cy=float(K["cy"]),
        width=width, height=height, R=R, t=t, near=near, far=far
    )


def run(
    image: str,
    pointcloud: str,
    camera_yaml: str,
    prompt: str,
    output_dir: str,
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = 1,
    bbox_json: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    cam_cfg = read_camera_yaml(camera_yaml)
    cam = make_camera(cam_cfg)
    pts = load_point_cloud(pointcloud)

    if bbox_json and os.path.exists(bbox_json):
        with open(bbox_json, 'r', encoding='utf-8') as f:
            det = json.load(f)
    else:
        # 从环境变量读取默认模型路径
        model_path = model_path or os.environ.get("QWEN_MODEL_PATH")
        mmproj_path = mmproj_path or os.environ.get("QWEN_MMPROJ_PATH")
        if not model_path:
            raise RuntimeError("未提供 QWEN 模型路径。请通过 --model_path 或设置环境变量 QWEN_MODEL_PATH。")
        detector = QwenVL2DDetector(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        det = detector.detect(image, prompt)

    bbox = det["bbox"]
    print(f"2D bbox: {bbox}, label: {det.get('label')}")

    # 叠加 2D bbox 可视化
    overlay_bbox_on_image(image, tuple(bbox), os.path.join(output_dir, "annotated.png"))

    # 视锥裁剪点云
    pts_crop = frustum_crop_points(pts, cam, tuple(bbox))
    print(f"裁剪点数: {pts_crop.shape[0]} / {pts.shape[0]}")

    # 计算 3D 包围盒并保存
    bbox3d = compute_bounding_box(pts_crop, oriented=True)
    with open(os.path.join(output_dir, "bbox3d.json"), 'w', encoding='utf-8') as f:
        json.dump(bbox3d, f, ensure_ascii=False, indent=2)

    # 保存 3D 线框
    ls = bbox_geom_to_o3d(bbox3d)
    if ls is not None:
        o3d.io.write_line_set(os.path.join(output_dir, "bbox3d.ply"), ls)

    print(f"输出完成：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单样本 3D 视觉定位（图像+文本+点云+相机）")
    parser.add_argument("--image", required=True)
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="outputs")

    parser.add_argument("--model_path", default=None)
    parser.add_argument("--mmproj_path", default=None)
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_gpu_layers", type=int, default=1)
    parser.add_argument("--bbox_json", default=None, help="若提供则跳过 Qwen-VL 推理")

    args = parser.parse_args()
    run(
        image=args.image,
        pointcloud=args.pointcloud,
        camera_yaml=args.camera,
        prompt=args.prompt,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mmproj_path=args.mmproj_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        bbox_json=args.bbox_json,
    )