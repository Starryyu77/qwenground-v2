from typing import List, Dict, Any, Optional, Tuple

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None


class YOLOCandidateDetector:
    """
    轻量车类候选生成器，基于 ultralytics YOLOv8。
    仅生成车类(car)候选框，并返回结构化候选列表。
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.45, device: Optional[str] = None):
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics。请在服务器环境中执行: pip install ultralytics")
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect_cars(self, image_path: str) -> List[Dict[str, Any]]:
        """
        运行 YOLOv8，在图像上检测车辆候选（COCO: car 类索引=2）。
        返回结构: [{"id": int, "bbox": [x1,y1,x2,y2], "conf": float, "label": "car"}, ...]
        id 为 1..N 递增，便于后续 LLM 选择。
        """
        # 运行推理
        results = self.model.predict(image_path, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        candidates: List[Dict[str, Any]] = []
        if not results:
            return candidates
        r0 = results[0]
        if r0.boxes is None:
            return candidates
        # 遍历框
        idx = 1
        for b in r0.boxes:
            # b.cls: 类别索引；b.conf: 置信度；b.xyxy: 张量 [x1,y1,x2,y2]
            try:
                cls = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
                if cls != 2:  # COCO car 类别索引=2
                    continue
                conf = float(b.conf.item()) if hasattr(b.conf, 'item') else float(b.conf)
                xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy, 'tolist') else list(b.xyxy)
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                candidates.append({
                    "id": idx,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "label": "car",
                })
                idx += 1
            except Exception:
                # 跳过异常框
                continue
        return candidates