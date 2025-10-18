import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

# 加入 src 到路径，便于导入 locator 包
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from locator.vision import QwenVL2DDetector
from locator.geometry import Camera, frustum_crop_points
from locator.io import overlay_bbox_on_image, load_point_cloud


def load_camera_intrinsic(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    K = data.get('cam_K')
    if K is None or len(K) != 9:
        raise ValueError(f"相机内参 cam_K 格式不正确: {path}")
    return {
        'fx': float(K[0]),
        'fy': float(K[4]),
        'cx': float(K[2]),
        'cy': float(K[5]),
        'cam_D': data.get('cam_D', None),
    }


def load_lidar_to_camera(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    R = np.array(data['rotation'], dtype=float)  # 3x3
    t = np.array(data['translation'], dtype=float).reshape(3)  # 3x1 -> (3,)
    return {'R': R, 't': t}


def select_bbox_from_label(label_path: str, label_type: str = 'Car', strategy: str = 'leftmost') -> Optional[Dict[str, Any]]:
    with open(label_path, 'r', encoding='utf-8') as f:
        items: List[Dict[str, Any]] = json.load(f)
    candidates = [it for it in items if it.get('type') == label_type and '2d_box' in it]
    if not candidates:
        return None
    if strategy == 'leftmost':
        candidates.sort(key=lambda it: it['2d_box']['xmin'])
    elif strategy == 'largest':
        candidates.sort(key=lambda it: (it['2d_box']['xmax'] - it['2d_box']['xmin']) * (it['2d_box']['ymax'] - it['2d_box']['ymin']), reverse=True)
    # 默认取排序后第一个
    box = candidates[0]['2d_box']
    bbox = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
    return {'bbox': bbox, 'label': label_type}


def main():
    parser = argparse.ArgumentParser(description='使用 example-cooperative-vehicle-infrastructure 数据集进行单样本测试（vehicle/infrastructure）')
    parser.add_argument('--side', default='example-cooperative-vehicle-infrastructure/vehicle-side', help='数据子目录：vehicle-side 或 infrastructure-side')
    parser.add_argument('--id', required=True, help='样本ID，如 015344')
    parser.add_argument('--prompt', default='最左边的汽车')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--bbox_from_label', action='store_true', help='从 label/camera/{id}.json 选择 bbox，跳过 Qwen 推理')
    parser.add_argument('--label_type', default='Car')
    parser.add_argument('--label_strategy', default='leftmost', choices=['leftmost', 'largest'])

    parser.add_argument('--model_path', default=None)
    parser.add_argument('--mmproj_path', default=None)
    parser.add_argument('--n_ctx', type=int, default=4096)
    parser.add_argument('--n_gpu_layers', type=int, default=1)

    args = parser.parse_args()

    side_dir = args.side.rstrip('/')
    sid = args.id

    img_path = os.path.join(side_dir, 'image', f'{sid}.jpg')
    pcd_path = os.path.join(side_dir, 'velodyne', f'{sid}.pcd')
    K_path = os.path.join(side_dir, 'calib', 'camera_intrinsic', f'{sid}.json')
    L2C_path = os.path.join(side_dir, 'calib', 'lidar_to_camera', f'{sid}.json')
    label_cam_path = os.path.join(side_dir, 'label', 'camera', f'{sid}.json')

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join('outputs', f'{os.path.basename(side_dir)}_{sid}')
    os.makedirs(out_dir, exist_ok=True)

    # 读图以获得尺寸
    from PIL import Image
    im = Image.open(img_path).convert('RGB')
    width, height = im.size

    # 构造 Camera（将 LiDAR坐标转换到相机坐标）
    ki = load_camera_intrinsic(K_path)
    l2c = load_lidar_to_camera(L2C_path)
    cam = Camera(
        fx=ki['fx'], fy=ki['fy'], cx=ki['cx'], cy=ki['cy'],
        width=width, height=height, R=l2c['R'], t=l2c['t'], near=0.3, far=200.0
    )

    # 点云（LiDAR坐标）
    pts = load_point_cloud(pcd_path)

    # 2D bbox: 来自标签或 Qwen
    if args.bbox_from_label and os.path.exists(label_cam_path):
        det = select_bbox_from_label(label_cam_path, args.label_type, args.label_strategy)
        if det is None:
            print(f'标签未找到 {args.label_type}，改用 Qwen 推理')
    else:
        det = None

    if det is None:
        model_path = args.model_path or os.environ.get('QWEN_MODEL_PATH')
        mmproj_path = args.mmproj_path or os.environ.get('QWEN_MMPROJ_PATH')
        if not model_path:
            raise RuntimeError('未提供 QWEN 模型路径。请通过 --model_path 或设置环境变量 QWEN_MODEL_PATH。若仅测试3D流程，请加 --bbox_from_label 使用标签。')
        detector = QwenVL2DDetector(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )
        det = detector.detect(img_path, args.prompt)

    bbox = det['bbox']
    print(f'2D bbox: {bbox}, label: {det.get("label")}')

    # 2D叠加
    overlay_bbox_on_image(img_path, tuple(bbox), os.path.join(out_dir, 'annotated.png'))

    # 视锥裁剪（LiDAR->Camera 投影）
    pts_crop = frustum_crop_points(pts, cam, tuple(bbox))
    print(f'裁剪点数: {pts_crop.shape[0]} / {pts.shape[0]}')

    # 3D包围盒（NumPy AABB）与保存
    if pts_crop.shape[0] == 0:
        bbox3d = {"empty": True}
    else:
        minv = pts_crop.min(axis=0)
        maxv = pts_crop.max(axis=0)
        center = ((minv + maxv) / 2.0).tolist()
        extent = (maxv - minv).tolist()
        bbox3d = {
            "empty": False,
            "type": "AABB",
            "center": center,
            "extent": extent,
            "R": np.eye(3).tolist(),
        }
    with open(os.path.join(out_dir, 'bbox3d.json'), 'w', encoding='utf-8') as f:
        json.dump(bbox3d, f, ensure_ascii=False, indent=2)

    print(f'输出完成：{out_dir}')


if __name__ == '__main__':
    main()