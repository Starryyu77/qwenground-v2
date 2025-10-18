from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import open3d as o3d


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3
    near: float = 0.1
    far: float = 100.0

    def world_to_camera(self, P_world: np.ndarray) -> np.ndarray:
        """R,t 为相机外参，将世界坐标转换到相机坐标系。P_world: (N,3)"""
        return (self.R @ P_world.T).T + self.t.reshape(1, 3)

    def project(self, P_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将相机坐标投影到像素坐标。返回 (xy, z)，xy: (N,2), z: (N,)"""
        X = P_cam[:, 0]
        Y = P_cam[:, 1]
        Z = P_cam[:, 2]
        x = self.fx * (X / Z) + self.cx
        y = self.fy * (Y / Z) + self.cy
        xy = np.stack([x, y], axis=1)
        return xy, Z


def frustum_crop_points(
    pts_world: np.ndarray,
    cam: Camera,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """根据 2D bbox 在相机视锥中裁剪点云，返回目标点集 (M,3)。"""
    P_cam = cam.world_to_camera(pts_world)
    xy, Z = cam.project(P_cam)

    x1, y1, x2, y2 = bbox
    in_bbox = (
        (xy[:, 0] >= x1)
        & (xy[:, 0] <= x2)
        & (xy[:, 1] >= y1)
        & (xy[:, 1] <= y2)
    )
    in_depth = (Z >= cam.near) & (Z <= cam.far)
    mask = in_bbox & in_depth
    return pts_world[mask]


def compute_bounding_box(points: np.ndarray, oriented: bool = True) -> Dict[str, Any]:
    """计算点集的 3D 包围盒，返回几何信息字典。"""
    if points.shape[0] == 0:
        return {"empty": True}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if oriented:
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        extent = obb.extent
        R = obb.R
        return {
            "empty": False,
            "type": "OBB",
            "center": center.tolist(),
            "extent": extent.tolist(),
            "R": R.tolist(),
        }
    else:
        aabb = pcd.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        center = (min_bound + max_bound) / 2.0
        extent = (max_bound - min_bound)
        return {
            "empty": False,
            "type": "AABB",
            "center": center.tolist(),
            "extent": extent.tolist(),
            "R": np.eye(3).tolist(),
        }


def bbox_geom_to_o3d(bbox_info: Dict[str, Any]) -> Optional[o3d.geometry.LineSet]:
    """将 bbox 信息转换为 Open3D 线框几何（用于保存/可视化）。"""
    if bbox_info.get("empty", False):
        return None
    center = np.array(bbox_info["center"])  # (3,)
    extent = np.array(bbox_info["extent"])  # (3,) -> dx,dy,dz
    R = np.array(bbox_info["R"])          # (3,3)

    # 8 角点（在自身坐标系），再用 R 与 center 平移
    ex = extent / 2.0
    corners_local = np.array([
        [-ex[0], -ex[1], -ex[2]],
        [ ex[0], -ex[1], -ex[2]],
        [ ex[0],  ex[1], -ex[2]],
        [-ex[0],  ex[1], -ex[2]],
        [-ex[0], -ex[1],  ex[2]],
        [ ex[0], -ex[1],  ex[2]],
        [ ex[0],  ex[1],  ex[2]],
        [-ex[0],  ex[1],  ex[2]],
    ])
    corners_world = (R @ corners_local.T).T + center.reshape(1, 3)

    # 12 条边索引
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners_world),
        lines=o3d.utility.Vector2iVector(edges),
    )
    return ls