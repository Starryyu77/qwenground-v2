import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

from PIL import Image

# 允许直接运行脚本时导入 src
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from locator.vision import QwenVL2DDetector, QwenAPI2DDetector, VLLMOpenAI2DDetector
from locator.io import overlay_bbox_on_image


META_DIRNAME = "v1.0-mini"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_channel_for_filename(channel: str) -> str:
    """
    sensor.json 中相机 channel 形如 CAM_FRONT_id_0；
    sample_data.json 的文件路径前缀为 sweeps/CAMERA_FRONT_id_0/xxx.png。
    这里做一个简单映射：CAM_ -> CAMERA_。
    """
    if channel.startswith("CAM_"):
        return channel.replace("CAM_", "CAMERA_")
    return channel


def collect_camera_samples(dataset_root: str, channel_substr: Optional[str] = None, key_frame_only: bool = True) -> List[Dict[str, Any]]:
    meta_dir = os.path.join(dataset_root, META_DIRNAME)
    sd_path = os.path.join(meta_dir, "sample_data.json")
    sample_data = load_json(sd_path)

    results: List[Dict[str, Any]] = []
    for item in sample_data:
        if item.get("fileformat") != ".png":
            continue
        fname = item.get("filename", "")
        if channel_substr and channel_substr not in fname:
            continue
        if key_frame_only and not item.get("is_key_frame", False):
            continue
        results.append({
            "token": item["token"],
            "filename": os.path.join(dataset_root, fname),
            "width": item.get("width"),
            "height": item.get("height"),
            "calibrated_sensor_token": item.get("calibrated_sensor_token"),
            "timestamp": item.get("timestamp"),
        })
    return results


def load_camera_intrinsic(dataset_root: str) -> Dict[str, List[List[float]]]:
    meta_dir = os.path.join(dataset_root, META_DIRNAME)
    calib_path = os.path.join(meta_dir, "calibrated_sensor.json")
    calib = load_json(calib_path)
    mapping: Dict[str, List[List[float]]] = {}
    for it in calib:
        token = it["token"]
        K = it.get("camera_intrinsic")
        if K is not None:
            mapping[token] = K
    return mapping


def run(dataset_root: str,
        channel: str,
        index: int,
        prompt: str,
        output_dir: str,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 1,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_model: str = "qwen2-vl-7b-instruct",
        use_vllm: bool = False,
        vllm_base_url: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        vllm_model: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
    os.makedirs(output_dir, exist_ok=True)

    channel_substr = normalize_channel_for_filename(channel)
    samples = collect_camera_samples(dataset_root, channel_substr=channel_substr, key_frame_only=True)
    if not samples:
        raise RuntimeError(f"未找到相机样本，channel='{channel_substr}'，请检查数据集路径与通道名称。")

    if index < 0 or index >= len(samples):
        raise IndexError(f"index 超出范围：{index}，可选范围 0~{len(samples)-1}")

    sample = samples[index]
    image_path = sample["filename"]

    # 读图，确认尺寸
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 构造更严格的提示，加入图像尺寸与坐标约束，避免仅框局部
    final_prompt = (
        f"{prompt}\n"
        f"图像尺寸：宽 {w} 像素，高 {h} 像素。"
        f"请输出该目标的完整 2D 边界框，bbox=[x1,y1,x2,y2]；坐标为整数，满足 0<=x1<x2<= {w-1}，0<=y1<y2<= {h-1}。"
        "边界框需要尽量覆盖整个车辆外轮廓，不要只框局部或边角。"
        "若不存在该目标，请仅输出 JSON：{{\"bbox\": [0,0,0,0], \"label\": \"none\"}}。"
        "不要输出除 JSON 以外的任何字符。"
    )

    # 加载相机内参（用于记录/调试）
    K_map = load_camera_intrinsic(dataset_root)
    K = K_map.get(sample["calibrated_sensor_token"])  # 3x3

    # Qwen-VL 推理（API / vLLM / 本地 GGUF）
    if use_vllm:
        base_url = vllm_base_url or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1"
        detector = VLLMOpenAI2DDetector(base_url=base_url, api_key=vllm_api_key, model=vllm_model)
    elif use_api:
        detector = QwenAPI2DDetector(api_key=api_key, model=api_model)
    else:
        model_path = model_path or os.environ.get("QWEN_MODEL_PATH")
        mmproj_path = mmproj_path or os.environ.get("QWEN_MMPROJ_PATH")
        if not model_path:
            raise RuntimeError("未提供 QWEN 模型路径。请通过 --model_path 或设置环境变量 QWEN_MODEL_PATH。若使用云端 API 或 vLLM，请分别加入 --use_api 或 --use_vllm。")
        detector = QwenVL2DDetector(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    det = detector.detect(image_path, final_prompt)
    bbox = det["bbox"]
    # 边界裁剪，确保坐标合法
    def _clamp_bbox(b: Any, w: int, h: int) -> List[int]:
        x1, y1, x2, y2 = map(int, b)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]
    bbox = _clamp_bbox(bbox, w, h)

    # 小框重试逻辑：若面积过小或宽高过小，强化提示重试一次
    def _area(b: List[int]) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    min_area = int(w * h * 0.003)  # 至少覆盖约 0.3% 图像面积
    if _area(bbox) < min_area or (bbox[2] - bbox[0]) < 16 or (bbox[3] - bbox[1]) < 16:
        retry_prompt = (
            f"{prompt}\n"
            f"图像尺寸：宽 {w} 像素，高 {h} 像素。"
            f"请输出该目标（例如最右侧的车辆）的完整 2D 边界框，bbox=[x1,y1,x2,y2]；坐标为整数，满足 0<=x1<x2<= {w-1}，0<=y1<y2<= {h-1}。"
            "务必覆盖完整车身，不要只框局部或边角；框的宽和高至少分别不小于 30 像素。"
            "若不存在该目标，请仅输出 JSON：{\"bbox\": [0,0,0,0], \"label\": \"none\"}。"
            "不要输出除 JSON 以外的任何字符。"
        )
        det_retry = detector.detect(image_path, retry_prompt)
        bbox_retry = _clamp_bbox(det_retry.get("bbox", bbox), w, h)
        # 若重试框面积更大，则采用重试结果
        if _area(bbox_retry) > _area(bbox):
            bbox = bbox_retry
            det = det_retry
            print("检测到初次框过小，已重试并采用更合理的框。")

    print(f"2D bbox: {bbox}, label: {det.get('label')}")

    # 保存可视化与 JSON
    overlay_bbox_on_image(image_path, tuple(bbox), os.path.join(output_dir, "annotated.png"))
    result = {
        "sample_token": sample["token"],
        "filename": image_path,
        "timestamp": sample["timestamp"],
        "width": w,
        "height": h,
        "bbox": bbox,
        "label": det.get("label"),
        "camera_intrinsic": K,
        "channel": channel_substr,
        "use_api": use_api,
        "api_model": api_model if use_api else None,
        "use_vllm": use_vllm,
        "vllm_model": vllm_model if use_vllm else None,
        "vllm_base_url": (vllm_base_url or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1") if use_vllm else None,
    }
    with open(os.path.join(output_dir, "bbox2d.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"输出完成：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在 airsim_camera_seg v1.0-mini 上运行 Qwen-VL 2D 定位")
    parser.add_argument("--dataset_root", required=True, help="数据集根目录，如 .../airsim_camera_seg")
    parser.add_argument("--channel", default="CAM_FRONT_id_0", help="相机通道（sensor.json 中的名称，例如 CAM_FRONT_id_0）")
    parser.add_argument("--index", type=int, default=0, help="选取样本索引（按时间排序）")
    parser.add_argument("--prompt", required=True, help="自然语言定位提示，如 '最左边的汽车'")
    parser.add_argument("--output_dir", default="outputs/airsim")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--mmproj_path", default=None)
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_gpu_layers", type=int, default=1)
    parser.add_argument("--use_api", action="store_true", help="使用 Qwen DashScope API 而非本地 GGUF")
    parser.add_argument("--api_key", default=None, help="Qwen API Key（也可用环境变量 QWEN_API_KEY）")
    parser.add_argument("--api_model", default="qwen2-vl-7b-instruct", help="DashScope 模型名")
    # 新增：vLLM 相关参数
    parser.add_argument("--use_vllm", action="store_true", help="使用 vLLM OpenAI 兼容服务器作为后端")
    parser.add_argument("--vllm_base_url", default=None, help="vLLM 服务地址，例如 http://127.0.0.1:8000/v1，也可环境变量 VLLM_BASE_URL")
    parser.add_argument("--vllm_api_key", default=None, help="vLLM API Key（通常可留空，或用环境变量 VLLM_API_KEY）")
    parser.add_argument("--vllm_model", default="Qwen/Qwen2-VL-7B-Instruct", help="vLLM 服务器加载的模型名（需与服务端一致）")

    args = parser.parse_args()
    run(
        dataset_root=args.dataset_root,
        channel=args.channel,
        index=args.index,
        prompt=args.prompt,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mmproj_path=args.mmproj_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        use_api=args.use_api,
        api_key=args.api_key,
        api_model=args.api_model,
        use_vllm=args.use_vllm,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_model=args.vllm_model,
    )