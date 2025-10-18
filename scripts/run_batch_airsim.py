import os
import json
import argparse
from typing import Any, Dict

from run_airsim_camera_seg import run as run_airsim


def main(tasks_path: str):
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    assert isinstance(tasks, list), "tasks.json 须为任务列表"

    for i, t in enumerate(tasks):
        print(f"==== 任务 {i+1}/{len(tasks)} ====")
        run_airsim(
            dataset_root=t["dataset_root"],
            channel=t.get("channel", "CAM_FRONT_id_0"),
            index=int(t.get("index", 0)),
            prompt=t.get("prompt", "最左边的汽车"),
            output_dir=t.get("output_dir", os.path.join("outputs", f"airsim_task_{i+1}")),
            model_path=t.get("model_path"),
            mmproj_path=t.get("mmproj_path"),
            n_ctx=int(t.get("n_ctx", 4096)),
            n_gpu_layers=int(t.get("n_gpu_layers", 1)),
            use_api=bool(t.get("use_api", False)),
            api_key=t.get("api_key"),
            api_model=t.get("api_model", "qwen2-vl-7b-instruct"),
            use_vllm=bool(t.get("use_vllm", False)),
            vllm_base_url=t.get("vllm_base_url"),
            vllm_api_key=t.get("vllm_api_key"),
            vllm_model=t.get("vllm_model", "Qwen/Qwen2-VL-7B-Instruct"),
            list_mode=bool(t.get("list_mode", False)),
            yolo_assist=bool(t.get("yolo_assist", False)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirSim 批处理任务执行器（支持 Qwen API、vLLM 或 GGUF）")
    parser.add_argument("tasks", help="tasks_airsim.json 路径")
    args = parser.parse_args()
    main(args.tasks)