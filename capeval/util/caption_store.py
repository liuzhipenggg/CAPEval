"""Caption map JSON / JSONL load & write; merged-file basename filter."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]


def is_merged_basename(name: str) -> bool:
    """False for parallel shard outputs, e.g. ``*.shard0of8.json``."""
    return ".shard" not in name.lower()


def load_caption_dict(path: PathLike) -> Dict[str, str]:
    """Load a caption map from JSON object or JSONL ``{k,c}`` / ``{image_key,caption}`` rows."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        out: Dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                k = obj.get("k") if "k" in obj else obj.get("image_key")
                c = obj.get("c") if "c" in obj else obj.get("caption")
                if k is not None and c is not None:
                    out[str(k)] = str(c)
        return out
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def write_caption_json(path: PathLike, results: Dict[str, Any], *, indent: int = 2) -> None:
    """Write a pretty-printed JSON caption dict (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=indent)


def append_caption_jsonl(path: PathLike, delta: Dict[str, str]) -> None:
    """Append ``{k, c}`` rows for each entry in ``delta``."""
    if not delta:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for k, v in delta.items():
            f.write(json.dumps({"k": k, "c": v}, ensure_ascii=False) + "\n")


def load_caption_store(path: str, fmt: str) -> Dict[str, str]:
    """Load by explicit format (``json`` / ``jsonl``); used by caption engine resume."""
    fmt = (fmt or "json").lower()
    if fmt == "jsonl" or (fmt == "json" and path.lower().endswith(".jsonl")):
        return load_caption_dict(path)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}
