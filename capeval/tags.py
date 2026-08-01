"""Helpers for multi-label checklist ``Tags`` fields."""

from __future__ import annotations

import re
from typing import List, Optional


def split_raw_tag_field(tags: Optional[str]) -> List[str]:
    """Split a Tags string into atomic tokens (comma / semicolon / Chinese comma)."""
    if not tags or not str(tags).strip():
        return []
    s = str(tags).strip()
    parts = re.split(r"[,，;；|]+", s)
    return [p.strip() for p in parts if p.strip()]
