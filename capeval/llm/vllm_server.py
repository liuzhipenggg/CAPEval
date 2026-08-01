"""Remote vLLM OpenAI-compatible server backend."""
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from capeval.llm.common import OptionalDependencyError, _openai

class AMDvLLMServerClient:
    base_url: str
    model: str
    headers: Dict[str, str]


def AMD_vllm_server_client(
    base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
) -> AMDvLLMServerClient:
    """Create a lightweight client for a vLLM server exposing OpenAI-compatible APIs.

    base_url should look like: http://host:port (no trailing slash is required)
    """
    sanitized = (base_url or "").rstrip("/")
    default_headers: Dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)
    return AMDvLLMServerClient(base_url=sanitized, model=model, headers=default_headers)


def _encode_image_b64_local(path: str, default_mime: str = "image/jpeg") -> str:
    mime = mimetypes.guess_type(path)[0] or default_mime
    with open(path, "rb") as f:
        data = f.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def AMD_vllm_server_multimodal_call(
    client: AMDvLLMServerClient,
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    system: str = "You are a helpful assistant.",
    n: int = 1,
    return_all: bool = False,
) -> Union[List[str], List[List[str]]]:
    """Call a running vLLM server (OpenAI-compatible) with image(s) + text.

    Each item is a dict with keys similar to AMD_vllm_multimodal_call:
      - 'text' | 'question' | 'prompt'
      - 'image_paths' | 'images' | 'image' (strings or list of strings)
    """
    try:
        import requests  # lazy import to keep dependency optional
    except Exception:
        raise OptionalDependencyError("requests is not installed. pip install 'requests'")

    batch: List[Dict[str, Any]] = items if isinstance(items, list) else [items]

    all_outputs: List[List[str]] = []
    url = f"{client.base_url}/v1/chat/completions"

    for it in batch:
        if not isinstance(it, dict):
            raise TypeError(f"Each item must be a dict, got {type(it)}")

        # Extract text
        text = None
        for key in ("text", "question", "prompt"):
            if key in it and it[key] is not None:
                text = str(it[key])
                break
        if text is None:
            raise ValueError("Each item must contain 'text', 'question', or 'prompt'")

        # Extract image paths
        image_paths: List[str] = []
        for key in ("image_paths", "images", "image"):
            if key in it and it[key] is not None:
                img_val = it[key]
                if isinstance(img_val, str):
                    image_paths = [img_val]
                elif isinstance(img_val, (list, tuple)):
                    image_paths = [str(p) for p in img_val]
                break
        if not image_paths:
            raise ValueError("Each item must contain 'image_paths', 'images', or 'image'")

        # Build content list: images first, then text, in a single user message
        content_items: List[Dict[str, Any]] = []
        for p in image_paths:
            try:
                url_b64 = _encode_image_b64_local(p)
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": url_b64},
                })
            except Exception:
                # Skip unreadable image; keep going if at least one image remains
                continue
        if not content_items:
            raise ValueError("None of the provided image paths could be read/encoded")
        content_items.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content_items},
        ]

        payload: Dict[str, Any] = {
            "model": client.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "n": int(max(1, n)),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if presence_penalty is not None:
            payload["presence_penalty"] = float(presence_penalty)

        resp = requests.post(url, headers=client.headers, json=payload)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"vLLM server returned non-JSON response: HTTP {resp.status_code}")

        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
            raise RuntimeError(f"vLLM server error {resp.status_code}: {err_msg or data}")

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            all_outputs.append([""] * int(max(1, n)))
            continue

        texts = []
        for i in range(min(len(choices), int(max(1, n)))):
            msg = choices[i].get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            texts.append(str(content).strip())

        while len(texts) < int(max(1, n)):
            texts.append("")
        all_outputs.append(texts)

    if n == 1 and not return_all:
        return [outs[0] if outs else "" for outs in all_outputs]
    return all_outputs

# -------------------- Qwen-VL via Transformers (optional) -----------------
