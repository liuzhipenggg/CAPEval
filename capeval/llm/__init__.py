"""LLM backends for CAPEval. All public ``AMD_*`` names are re-exported here."""
from __future__ import annotations

from capeval.llm.cloud import (
    AMD_claude_call,
    AMD_claude_client,
    AMD_gemini_call,
    AMD_gemini_client,
    AMD_openai_call,
    AMD_openai_client,
)
from capeval.llm.common import (
    OptionalDependencyError,
    _downscale_pil_max_edge,
    _env_capeval,
    _open_pil_rgb,
    current_user,
)
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
from capeval.llm.transformers_vl import (
    AMDTransformersCaptionClient,
    AMD_transformers_caption_call,
    AMD_transformers_caption_call_batch,
    AMD_transformers_caption_client,
)
from capeval.llm.vllm_local import (
    AMDvLLMClient,
    AMD_vllm_chat_client,
    AMD_vllm_multimodal_call,
    AMD_vllm_text_chat_call,
)
from capeval.llm.vllm_server import (
    AMDvLLMServerClient,
    AMD_vllm_server_client,
    AMD_vllm_server_multimodal_call,
)

__all__ = [
    "AMDGlm4VTransformersClient",
    "AMDInternVLTransformersClient",
    "AMDQwenVLClient",
    "AMDTransformersCaptionClient",
    "AMDvLLMClient",
    "AMDvLLMServerClient",
    "AMD_claude_call",
    "AMD_claude_client",
    "AMD_gemini_call",
    "AMD_gemini_client",
    "AMD_glm4v_transformers_call",
    "AMD_glm4v_transformers_call_batch",
    "AMD_glm4v_transformers_client",
    "AMD_internvl_transformers_call",
    "AMD_internvl_transformers_client",
    "AMD_openai_call",
    "AMD_openai_client",
    "AMD_qwenvl_call",
    "AMD_qwenvl_call_batch",
    "AMD_qwenvl_client",
    "AMD_transformers_caption_call",
    "AMD_transformers_caption_call_batch",
    "AMD_transformers_caption_client",
    "AMD_vllm_chat_client",
    "AMD_vllm_multimodal_call",
    "AMD_vllm_server_client",
    "AMD_vllm_server_multimodal_call",
    "AMD_vllm_text_chat_call",
    "OptionalDependencyError",
    "_downscale_pil_max_edge",
    "_env_capeval",
    "_open_pil_rgb",
    "current_user",
]
