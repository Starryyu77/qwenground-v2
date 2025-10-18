import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
from rich import print

import base64

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

# 新增：可选使用 Qwen 官方 DashScope API（云端推理）
try:
    from dashscope import MultiModalConversation
except Exception:
    MultiModalConversation = None


class QwenVL2DDetector:
    """
    Qwen-VL (GGUF) 2D 检测器：输入图像与文本，输出 2D bbox。
    依赖 llama-cpp-python + mmproj；支持 Metal 加速（macOS）。
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 1,
        verbose: bool = False,
    ) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python 未安装，请先 pip install llama-cpp-python"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        llm_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if mmproj_path is not None:
            if not os.path.exists(mmproj_path):
                raise FileNotFoundError(f"mmproj 文件不存在: {mmproj_path}")
            llm_kwargs["mmproj"] = mmproj_path
        self.llm = Llama(**llm_kwargs)

    @staticmethod
    def _force_json_bbox(text: str) -> Optional[Dict[str, Any]]:
        """从模型输出文本中提取 JSON bbox（鲁棒解析）。"""
        text = text.strip()
        # 优先解析 ```json ... ``` 或 ``` ... ``` 代码块中的 JSON
        import re
        m_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if m_fence:
            inner = m_fence.group(1)
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict) and "bbox" in obj:
                    return obj
            except Exception:
                pass
        # 直接 JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "bbox" in obj:
                return obj
        except Exception:
            pass
        # 简单模式：查找类似 [x1, y1, x2, y2]
        m = re.search(r"\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]", text)
        if m:
            nums = [int(x) for x in m.group(1).split(',')]
            return {"bbox": nums, "label": "object"}
        return None

    def detect(self, image_path: str, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        # 构造严格的系统与用户提示，强制 JSON 输出
        system_prompt = (
            "你是一个图像定位助手。根据用户描述，在给定图像中只返回一个 JSON 对象，"
            "包含像素边界框 bbox=[x1,y1,x2,y2] 和 label 字段。不要输出多余文本。"
        )
        user_prompt = (
            f"图像中定位：{prompt}\n"
            "请仅输出：{\"bbox\": [x1, y1, x2, y2], \"label\": \"...\"}"
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    # llama-cpp 支持 images 参数，这里也附带 image_url 作为兜底
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{os.path.abspath(image_path)}"},
                    },
                ],
            },
        ]

        try:
            result = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                images=[image_path],  # 优先使用直接传图（llama-cpp 新版支持）
            )
            content = result["choices"][0]["message"]["content"]
            parsed = self._force_json_bbox(content)
            if parsed is None:
                raise ValueError(f"未解析到 JSON bbox: {content}")
            return parsed
        except Exception as e:
            print(f"[yellow]Qwen-VL 推理失败，使用兜底策略：{e}[/yellow]")
            # 兜底：返回整幅图像的 bbox
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            return {"bbox": [0, 0, w - 1, h - 1], "label": "fallback"}


# 新增：DashScope 云端 Qwen-VL 推理（无需本地 GGUF/llama-cpp）
class QwenAPI2DDetector:
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen2-vl-7b-instruct") -> None:
        if MultiModalConversation is None:
            raise RuntimeError("dashscope 未安装。请先 pip install dashscope")
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise RuntimeError("未提供 QWEN_API_KEY（--api_key 或环境变量 QWEN_API_KEY）")
        self.model = model

    def detect(self, image_path: str, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        system_prompt = (
            "你是一个图像定位助手。根据用户描述，在给定图像中只返回一个 JSON 对象，"
            "包含像素边界框 bbox=[x1,y1,x2,y2] 和 label 字段。不要输出多余文本。"
        )
        user_prompt = (
            f"图像中定位：{prompt}\n"
            "请仅输出：{\"bbox\": [x1, y1, x2, y2], \"label\": \"...\"}"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_path)}"}}
            ]}
        ]
        try:
            resp = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                result_format="message",
                max_tokens=max_tokens,
                temperature=0.0,
            )
            # 调试输出，观察 DashScope 返回结构
            print(f"[blue]DashScope resp: {resp}[/blue]")
            # 兼容多种返回结构：优先 output_text；否则解析 output.choices[0].message.content[*].text
            content: Optional[str] = None
            try:
                content = resp.output_text  # 某些实现下缺失会抛异常
            except Exception:
                content = None
            if not content:
                try:
                    out = resp.output
                except Exception:
                    out = None
                if isinstance(out, dict):
                    choices = out.get("choices") or []
                    if choices:
                        msg = choices[0].get("message", {})
                        cont = msg.get("content", [])
                        for it in cont:
                            if isinstance(it, dict) and (it.get("type") == "text" or "text" in it):
                                content = it.get("text") or it["text"]
                                break
            if not content:
                content = str(resp)
            parsed = QwenVL2DDetector._force_json_bbox(content)
            if parsed is None:
                raise ValueError(f"未解析到 JSON bbox: {content}")
            return parsed
        except Exception as e:
            print(f"[yellow]Qwen API 推理失败，使用兜底策略：{e}[/yellow]")
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            return {"bbox": [0, 0, w - 1, h - 1], "label": "fallback"}

# 新增：vLLM OpenAI 兼容接口（服务器端）
class VLLMOpenAI2DDetector:
    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 model: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
        try:
            from openai import OpenAI
        except Exception:
            raise RuntimeError("openai 客户端未安装。请先 pip install openai")
        if not base_url:
            raise RuntimeError("未提供 vLLM base_url（例如 http://127.0.0.1:8000/v1）")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or "EMPTY"
        self.model = model
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def detect(self, image_path: str, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        system_prompt = (
            "你是一个图像定位助手。根据用户描述，在给定图像中只返回一个 JSON 对象，"
            "包含像素边界框 bbox=[x1,y1,x2,y2] 和 label 字段。不要输出多余文本。"
        )
        user_prompt = (
            f"图像中定位：{prompt}\n"
            "请仅输出：{\"bbox\": [x1, y1, x2, y2], \"label\": \"...\"}"
        )
        data_url = self._image_to_data_url(image_path)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            # vLLM OpenAI 接口通常返回字符串 content
            content = resp.choices[0].message.content if resp and resp.choices else None
            if not content:
                content = str(resp)
            parsed = QwenVL2DDetector._force_json_bbox(content)
            if parsed is None:
                raise ValueError(f"未解析到 JSON bbox: {content}")
            return parsed
        except Exception as e:
            print(f"[yellow]vLLM OpenAI 推理失败，使用兜底策略：{e}[/yellow]")
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            return {"bbox": [0, 0, w - 1, h - 1], "label": "fallback"}