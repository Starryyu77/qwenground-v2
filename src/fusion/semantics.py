import os
import json
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image


def load_category_map(category_json_path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    加载类别映射：返回 (id_to_name, name_to_id)
    期望 category.json 为数组，每个元素至少包含 {"id": int, "name": str}
    """
    with open(category_json_path, 'r') as f:
        data = json.load(f)
    id_to_name = {}
    name_to_id = {}
    for it in data:
        cid = int(it.get('id'))
        name = str(it.get('name')).lower()
        id_to_name[cid] = name
        name_to_id[name] = cid
    return id_to_name, name_to_id


def load_semantic_mask(mask_path: str) -> np.ndarray:
    """
    加载语义掩码，返回 HxW 的整型数组（每个像素为类别ID）。
    支持常见的灰度/索引色PNG。
    """
    img = Image.open(mask_path)
    # 转为灰度或保持原始，但最终需要整数类别ID
    if img.mode not in ('L', 'I', 'P'):
        img = img.convert('L')
    arr = np.array(img)
    # 若是索引色(P)，arr 为索引号；若是灰度，直接视为类别ID
    return arr.astype(np.int32)


def compute_bbox_overlap_ratio(mask: np.ndarray, bbox: Tuple[int, int, int, int], target_class_id: int) -> float:
    """
    计算 bbox 区域内，属于 target_class_id 的像素占比（overlap ratio）。
    bbox: (x1, y1, x2, y2) in pixels
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    total = region.size
    if total == 0:
        return 0.0
    hit = np.count_nonzero(region == target_class_id)
    return float(hit) / float(total)


def derive_semantic_path_from_image(image_path: str, semantics_root: Optional[str] = None) -> Optional[str]:
    """
    根据约定尝试从图像路径推导语义掩码路径。
    - 若提供 semantics_root，则在该目录下按通道子目录与文件名匹配。
    - 否则尝试将路径中的 `sweeps` 替换为 `semantics`。
    返回可能的路径；若无法推导则返回 None。
    """
    if semantics_root:
        # 语义根目录下使用与图像相同的子路径结构（CHANNEL/filename.png）
        # 从图像路径中提取 CHANNEL 与 filename
        parts = image_path.split('/')
        try:
            idx = parts.index('sweeps')
            channel = parts[idx + 1]
            filename = parts[-1]
            cand = os.path.join(semantics_root, channel, filename)
            return cand
        except Exception:
            # 如果不含 sweeps，尝试直接匹配文件名
            filename = os.path.basename(image_path)
            cand = os.path.join(semantics_root, filename)
            return cand
    # 无 semantics_root，尝试规则替换
    if '/sweeps/' in image_path:
        return image_path.replace('/sweeps/', '/semantics/')
    return None