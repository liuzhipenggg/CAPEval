"""Filesystem-safe model id helpers (single implementation)."""
from __future__ import annotations

import os


def model_safe(model_id: str) -> str:
    """Slash/space → underscore (caption output basenames, etc.)."""
    return str(model_id).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def safe_model_subdir(model_id: str) -> str:
    """Sanitize model_id for use as a single path segment under an output dir."""
    name = model_safe(model_id).replace(os.sep, "_").replace("..", "_").strip("._")
    if not name or name in (".", ".."):
        raise ValueError(f"invalid model_id for directory: {model_id!r}")
    if os.path.isabs(name):
        raise ValueError(f"invalid model_id for directory: {model_id!r}")
    return name
