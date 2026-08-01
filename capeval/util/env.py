"""Unified CAPEVAL_* / legacy CAPTIONQA_* environment lookup."""
from __future__ import annotations

import os
from typing import Optional


def env_capeval(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read ``CAPEVAL_<name>``, falling back to legacy ``CAPTIONQA_<name>``.

    ``name`` may be a bare suffix (``FOO`` → ``CAPEVAL_FOO``) or a full
    ``CAPEVAL_*`` / ``CAPTIONQA_*`` key.
    """
    if name.startswith("CAPEVAL_") or name.startswith("CAPTIONQA_"):
        key = name
    else:
        key = f"CAPEVAL_{name}"
    if key.startswith("CAPEVAL_"):
        legacy = "CAPTIONQA_" + key[len("CAPEVAL_") :]
        if key in os.environ:
            return os.environ.get(key)
        if legacy in os.environ:
            return os.environ.get(legacy)
        return default
    return os.environ.get(key, default)
