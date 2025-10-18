import os
import json
import argparse
from typing import Dict, Any

from run_single import run  # 直接复用单样本逻辑


def main(tasks_path: str):
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    assert isinstance(tasks, list), "tasks.json 须为任务列表"

    for i, t in enumerate(tasks):
        print(f"==== 任务 {i+1}/{len(tasks)} ====")
        run(
            image=t["image_path"],
            pointcloud=t["pointcloud_path"],
            camera_yaml=t["camera_path"],
            prompt=t.get("prompt", "目标"),
            output_dir=t.get("output_dir", os.path.join("outputs", f"task_{i+1}")),
            model_path=t.get("model_path"),
            mmproj_path=t.get("mmproj_path"),
            n_ctx=int(t.get("n_ctx", 4096)),
            n_gpu_layers=int(t.get("n_gpu_layers", 1)),
            bbox_json=t.get("bbox_json"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批处理任务执行器")
    parser.add_argument("tasks", help="tasks.json 路径")
    args = parser.parse_args()
    main(args.tasks)