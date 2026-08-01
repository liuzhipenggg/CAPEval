"""Caption result store: resume/shard JSON and JSONL writes."""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Tuple

from capeval.util.caption_store import (
    append_caption_jsonl as _util_append_jsonl,
    load_caption_store as _util_load_store,
    write_caption_json as _util_write_json,
)


def _parse_caption_shard(spec: str) -> Tuple[int, int]:
    """Parse 'K/N' for parallel caption workers (0 <= K < N)."""
    spec = (spec or "").strip()
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError("--caption-shard must be K/N, e.g. 0/2")
    k, n = int(parts[0].strip()), int(parts[1].strip())
    if n < 1 or k < 0 or k >= n:
        raise ValueError(f"invalid caption shard {spec!r} (need 0 <= K < N)")
    return k, n


def _backup_output_path(output_path: str, args) -> Optional[str]:
    backup_root = getattr(args, "backup_dir", None)
    if not backup_root:
        return None
    try:
        out_abs = os.path.abspath(output_path)
        out_root = os.path.abspath(getattr(args, "output_dir", "") or "")
        if out_root and os.path.commonpath([out_abs, out_root]) == out_root:
            rel = os.path.relpath(out_abs, out_root)
            return os.path.join(os.path.abspath(backup_root), rel)
    except Exception:
        pass
    return os.path.join(os.path.abspath(backup_root), os.path.basename(output_path))


def _caption_store_fmt(args: Any) -> str:
    fmt = (getattr(args, "caption_results_format", None) or "json").lower()
    return fmt if fmt in ("json", "jsonl") else "json"


def _load_caption_store(path: str, fmt: str) -> Dict[str, str]:
    return _util_load_store(path, fmt)


def _append_caption_jsonl(output_path: str, delta: Dict[str, str], args: Any) -> None:
    if not delta:
        return
    _util_append_jsonl(output_path, delta)
    backup_path = _backup_output_path(output_path, args)
    if backup_path:
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(output_path, backup_path)


def _write_results(output_path: str, results: Dict[str, str], args: Any) -> None:
    """Write full JSON snapshot (used when caption_results_format=json)."""
    _util_write_json(output_path, results)
    backup_path = _backup_output_path(output_path, args)
    if backup_path:
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(output_path, backup_path)

