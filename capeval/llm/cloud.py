"""Cloud LLM backends (OpenAI / Gemini / Claude)."""
from __future__ import annotations

import mimetypes
import os
from typing import Any, Iterable, List, Optional, Union

from capeval.llm.common import (
    OptionalDependencyError,
    _BadRequestError,
    _anthropic,
    _env_capeval,
    _genai,
    _openai,
    current_user,
    GenerateContentConfig,
    HttpOptions,
    Part,
)


def _amd_gateway_base() -> str:
    """Corporate LLM gateway base URL (must be set via env; no hardcoded host)."""
    base = (
        os.environ.get("AMD_LLM_BASE_URL")
        or os.environ.get("CAPEVAL_AMD_LLM_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if not base:
        raise OptionalDependencyError(
            "Set AMD_LLM_BASE_URL to your corporate LLM gateway base URL "
            "before using AMD cloud backends."
        )
    return base


def AMD_openai_client(model_id: str, amd: bool = False) -> Any:
    """Create a standard OpenAI client using OPENAI_API_KEY.

    The model_id argument is accepted for compatibility with callers but is not
    needed to construct the client.
    """
    if _openai is None:
        raise OptionalDependencyError(
            "openai is not installed. pip install 'openai>=1.0.0'"
        )
    if amd:
        sub_key = os.environ.get("AMD_SUBSCRIPTION_KEY", "").strip()
        if not sub_key:
            raise OptionalDependencyError(
                "AMD_SUBSCRIPTION_KEY environment variable is not set. "
                "Set it before using AMD OpenAI gateway."
            )
        user = current_user()
        url = _amd_gateway_base()
        client = _openai.AzureOpenAI(
            api_key="dummy",
            api_version="2024-12-01-preview",
            base_url=url,
            default_headers={
                "Ocp-Apim-Subscription-Key": sub_key,
                "user": user,
            },
        )
        # keep your existing deployment path pattern
        client.base_url = f"{url}/openai/deployments/{model_id}"
        return client
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        # Allow overriding base URL for proxies/self-hosted compatible servers
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return _openai.OpenAI(api_key=api_key, base_url=base_url)
        return _openai.OpenAI(api_key=api_key)


def AMD_openai_call(
    client: Any,  # do not reference openai types here
    model_id: str,
    messages: Union[str, List[dict]],
    **kwargs: Any,
) -> Any:
    """
    Make a chat completion call. Accepts either a string or a list of message dicts.
    Extra kwargs pass-through (temperature, stream, max_completion_tokens, reasoning_effort, etc.)
    """
    if _openai is None:
        raise OptionalDependencyError(
            "openai is not installed. pip install 'openai>=1.0.0'"
        )

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        return client.chat.completions.create(model=model_id, messages=messages, **kwargs)
    except _BadRequestError as e:
        msg = str(e)
        # Retry without reasoning_effort if backend doesn't accept it
        if "Unrecognized request argument" in msg and ("reasoning_effort" in msg):
            kwargs.pop("reasoning_effort", None)
            return client.chat.completions.create(model=model_id, messages=messages, **kwargs)
        raise

# -------------------- Gemini (optional) --------------------
def AMD_gemini_client() -> Any:
    """Create Gemini (Google GenAI) client via AMD gateway."""
    if _genai is None:
        raise OptionalDependencyError(
            "google-genai is not installed. pip install 'google-genai>=1.0.0'"
        )
    sub_key = (
        os.environ.get("AMD_VERTEX_SUBSCRIPTION_KEY")
        or _env_capeval("AMD_SUBSCRIPTION_KEY", "")
    ).strip()
    if not sub_key or sub_key == "YOUR_SUBSCRIPTION_KEY":
        raise OptionalDependencyError(
            "Set AMD_VERTEX_SUBSCRIPTION_KEY (or CAPEVAL_AMD_SUBSCRIPTION_KEY) for Gemini gateway access."
        )
    vertex_base = (
        os.environ.get("AMD_VERTEX_BASE_URL")
        or os.environ.get("CAPEVAL_AMD_VERTEX_BASE_URL")
        or f"{_amd_gateway_base()}/VertexGen"
    ).strip().rstrip("/")
    client = _genai.Client(
        vertexai=True,
        api_key="dummy",
        http_options=HttpOptions(
            base_url=vertex_base,
            api_version="v1",
            headers={
                "Ocp-Apim-Subscription-Key": sub_key,
            },
        ),
    )
    return client

def AMD_gemini_call(
    client: Any,
    model_id: str,
    messages: str,                 # 只接受字符串
    *,
    image_paths: Iterable[str],    # 只接受本地图片路径（可多张）
    default_mime: str = "image/jpeg",
    **kwargs: Any,                 # 直接传入 GenerateContentConfig（不做过滤）
) -> Any:
    if _genai is None:
        raise OptionalDependencyError(
            "google-genai is not installed. pip install 'google-genai>=1.0.0'"
        )
    # 1) 文本（已保证是字符串）
    text = messages

    # 2) contents：图片在前、文本在后
    contents: List[Any] = []
    for p in image_paths:
        mime = mimetypes.guess_type(p)[0] or default_mime
        with open(p, "rb") as f:
            contents.append(Part.from_bytes(data=f.read(), mime_type=mime))
    contents.append(text)

    # 3) 直接构造 config（kwargs 不合法时让其抛错）
    config = GenerateContentConfig(**kwargs) if kwargs else None

    # 4) 调用
    return client.models.generate_content(
        model=model_id,
        contents=contents if len(contents) > 1 else contents[0],
        config=config,
    )

# -------------------- Claude/Anthropic (optional) --------------------
def AMD_claude_client() -> Any:
    """Create Anthropic (Claude) client via AMD gateway."""
    if _anthropic is None:
        raise OptionalDependencyError(
            "anthropic is not installed. pip install 'anthropic>=0.18.0'"
        )

    sub_key = os.environ.get("AMD_SUBSCRIPTION_KEY", "").strip()
    if not sub_key:
        raise OptionalDependencyError(
            "AMD_SUBSCRIPTION_KEY environment variable is not set. "
            "Set it before using AMD Claude gateway."
        )
    claude_base = (
        os.environ.get("AMD_ANTHROPIC_BASE_URL")
        or os.environ.get("CAPEVAL_AMD_ANTHROPIC_BASE_URL")
        or f"{_amd_gateway_base()}/Anthropic"
    ).strip().rstrip("/")
    client = _anthropic.Anthropic(
        base_url=claude_base,
        api_key="dummy",
        default_headers={
            "Ocp-Apim-Subscription-Key": sub_key,
            "anthropic-version": "2023-10-16"
        },
        timeout=600,
    )
    return client


def AMD_claude_call(
    client: Any,
    model_id: str,
    messages: Union[str, List[dict]],
    **kwargs: Any,
) -> Any:
    """
    Make a Claude API call. Accepts either a string or a list of message dicts.
    Extra kwargs pass-through (max_tokens, temperature, top_p, etc.)
    
    Args:
        client: Anthropic client from AMD_claude_client()
        model_id: Claude model name (e.g., "claude-3-5-sonnet-20241022")
        messages: Either a string or list of message dicts (Claude format)
        **kwargs: Additional parameters (max_tokens, temperature, top_p, etc.)
    
    Returns:
        Anthropic API response object
    
    Example:
        client = AMD_claude_client()
        response = AMD_claude_call(
            client,
            model_id="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": [...]}],
            max_tokens=1024,
            temperature=0.6,
            top_p=0.95
        )
        text = response.content[0].text
    """
    if _anthropic is None:
        raise OptionalDependencyError(
            "anthropic is not installed. pip install 'anthropic>=0.18.0'"
        )
    
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    return client.messages.create(
        model=model_id,
        messages=messages,
        **kwargs
    )

# -------------------- vLLM (optional) ----------------------


