"""Shared LLM helpers and optional dependency imports."""
from __future__ import annotations

import os
import re
import getpass
from pathlib import Path
import mimetypes
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Union, Optional, Any, Dict, Iterable

from capeval.util.env import env_capeval as _env_capeval_raw

# ---------- optional deps (lazy) ----------
_openai = None
_openai_import_error: BaseException | None = None
try:
    import openai as _openai  # Azure OpenAI SDK
    from openai import BadRequestError as _BadRequestError
except Exception as e:  # not installed or import error
    _openai = None
    _openai_import_error = e

    class _BadRequestError(Exception):  # fallback to avoid NameError at runtime
        pass

_genai = None
_genai_import_error: BaseException | None = None
try:
    # Google GenAI SDK（新版）：pip install google-genai
    from google import genai as _genai
    from google.genai.types import HttpOptions, GenerateContentConfig, Part
except Exception as e:
    _genai = None
    _genai_import_error = e
    HttpOptions = None
    GenerateContentConfig = None
    Part = None

_transformers = None
_transformers_import_error: BaseException | None = None
try:
    import transformers as _transformers  # for AutoTokenizer & chat template
except Exception as e:
    _transformers = None
    _transformers_import_error = e

_vllm = None
_vllm_import_error: BaseException | None = None
try:
    import vllm as _vllm  # for LLM, SamplingParams
except Exception as e:
    _vllm = None
    _vllm_import_error = e

_anthropic = None
_anthropic_import_error: BaseException | None = None
try:
    import anthropic as _anthropic  # for Claude models
except Exception as e:
    _anthropic = None
    _anthropic_import_error = e


class OptionalDependencyError(ImportError):
    pass


def require_optional(
    name: str,
    module: Any,
    err: BaseException | None,
    *,
    install_hint: str,
) -> Any:
    """Raise a clear error if an optional import failed (missing *or* version clash)."""
    if module is not None:
        return module
    detail = f" Import failed: {type(err).__name__}: {err}" if err is not None else ""
    clash = ""
    if name == "transformers" and err is not None and "huggingface" in str(err).lower():
        clash = (
            " This is often a transformers 4.x / huggingface-hub>=1 clash; "
            "keep huggingface-hub>=0.34,<1 (see requirements.txt)."
        )
    raise OptionalDependencyError(
        f"{name} is unavailable.{detail}{clash} {install_hint}"
    ) from err


def _env_capeval(name: str, default: str = "") -> str:
    """Read CAPEVAL_<name>, falling back to legacy CAPTIONQA_<name>."""
    val = _env_capeval_raw(name, None)
    if val is None:
        return default
    return val


def current_user() -> str:
    for k in ("AMD_API_USER", "API_USER", "SLURM_JOB_USER",
              "SUDO_USER", "LOGNAME", "USER", "USERNAME"):
        v = os.getenv(k)
        if v:
            return v
    try:
        return getpass.getuser()
    except Exception:
        try:
            import pwd
            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return "unknown"


# -------------------- OpenAI (optional) --------------------
def _open_pil_rgb(path: str) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _downscale_pil_max_edge(img: Any, max_edge: int) -> Any:
    """If max_edge > 0, resize so max(W,H) <= max_edge (faster vision encoder, fewer tokens)."""
    if max_edge <= 0:
        return img
    from PIL import Image

    w, h = img.size
    m = max(w, h)
    if m <= max_edge:
        return img
    scale = max_edge / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR)


def _item_has_predecoded_pils(it: Dict[str, Any]) -> bool:
    p = it.get("pil_images")
    if p is None:
        return False
    try:
        from PIL import Image as _PIL_Image
    except ImportError:
        return False
    if isinstance(p, _PIL_Image.Image):
        return True
    return isinstance(p, (list, tuple)) and len(p) > 0


def _load_pil_images_for_mm_item(it: Dict[str, Any]) -> List[Any]:
    """Decode images for one multimodal item (paths and/or pre-loaded PIL)."""
    pil = it.get("pil_images")
    if pil is not None:
        from PIL import Image as _PIL_Image

        if isinstance(pil, _PIL_Image.Image):
            return [pil]
        return list(pil)
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
        raise ValueError("Each item must contain image_paths/images or pil_images")
    out: List[Any] = []
    for img_path in image_paths:
        try:
            out.append(_open_pil_rgb(img_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}") from e
    return out


