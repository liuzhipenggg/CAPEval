"""Transformers caption dispatcher (routes to Qwen-VL / InternVL / GLM-4V)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from capeval.llm.internvl_glm import (
    AMDGlm4VTransformersClient,
    AMDInternVLTransformersClient,
    AMD_glm4v_transformers_call,
    AMD_glm4v_transformers_call_batch,
    AMD_glm4v_transformers_client,
    AMD_internvl_transformers_call,
    AMD_internvl_transformers_client,
)
from capeval.llm.qwenvl import (
    AMDQwenVLClient,
    AMD_qwenvl_call,
    AMD_qwenvl_call_batch,
    AMD_qwenvl_client,
)

@dataclass
class AMDTransformersCaptionClient:
    """Unified local HF client for capeval/caption.py (--backend transformers)."""

    kind: str
    qwenvl: Optional[AMDQwenVLClient] = None
    internvl: Optional[AMDInternVLTransformersClient] = None
    glm4v: Optional[AMDGlm4VTransformersClient] = None


def AMD_transformers_caption_client(
    model_id: str,
    *,
    device: Optional[str] = None,
) -> AMDTransformersCaptionClient:
    m = (model_id or "").lower()
    if "internvl" in m:
        return AMDTransformersCaptionClient(
            kind="internvl", internvl=AMD_internvl_transformers_client(model_id, device=device)
        )
    if "glm" in m and (
        "4.6" in m
        or "4_6" in m
        or "zai-org" in m
        or "glm-4.6" in m
    ):
        return AMDTransformersCaptionClient(
            kind="glm4v", glm4v=AMD_glm4v_transformers_client(model_id, device=device)
        )
    return AMDTransformersCaptionClient(
        kind="qwenvl", qwenvl=AMD_qwenvl_client(model_id, device=device)
    )


def AMD_transformers_caption_call(
    client: AMDTransformersCaptionClient,
    image_paths: List[str],
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_image_edge: int = 0,
) -> str:
    if client.kind == "internvl" and client.internvl:
        return AMD_internvl_transformers_call(
            client.internvl,
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    if client.kind == "glm4v" and client.glm4v:
        return AMD_glm4v_transformers_call(
            client.glm4v,
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    if client.qwenvl:
        return AMD_qwenvl_call(
            client.qwenvl,
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_image_edge=max_image_edge,
        )
    raise RuntimeError("Transformers caption client not initialized")


def AMD_transformers_caption_call_batch(
    client: AMDTransformersCaptionClient,
    image_paths: List[str],
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_image_edge: int = 0,
) -> List[str]:
    """One caption per path; InternLM/GLM fall back to sequential single calls."""
    if client.kind == "internvl" and client.internvl:
        return [
            AMD_internvl_transformers_call(
                client.internvl,
                [p],
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for p in image_paths
        ]
    if client.kind == "glm4v" and client.glm4v:
        return AMD_glm4v_transformers_call_batch(
            client.glm4v,
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_image_edge=max_image_edge,
        )
    if client.qwenvl:
        return AMD_qwenvl_call_batch(
            client.qwenvl,
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_image_edge=max_image_edge,
        )
    raise RuntimeError("Transformers caption client not initialized")


__all__ = [
    "AMD_openai_client",
    "AMD_openai_call",
    "AMD_gemini_client",
    "AMD_gemini_call",
    "AMD_claude_client",
    "AMD_claude_call",
    "AMD_vllm_chat_client",
    "AMD_vllm_text_chat_client",  # Backward compatibility alias
    "AMD_vllm_text_chat_call",
    "AMD_vllm_multimodal_call",
    "AMD_vllm_server_client",
    "AMD_vllm_server_multimodal_call",
    "AMD_qwenvl_client",
    "AMD_qwenvl_call",
    "AMD_internvl_transformers_client",
    "AMD_internvl_transformers_call",
    "AMD_glm4v_transformers_client",
    "AMD_glm4v_transformers_call",
    "AMD_glm4v_transformers_call_batch",
    "AMDTransformersCaptionClient",
    "AMD_transformers_caption_client",
    "AMD_transformers_caption_call",
    "AMD_transformers_caption_call_batch",
    "AMD_qwenvl_call_batch",
]
