#!/usr/bin/env python3
"""Merge caption shard files from parallel --caption-shard runs into one JSON file.

Each shard may be a JSON object (path/key -> caption) or capeval caption JSONL
(``{"k":..., "c":...}`` per line). Output is always a single pretty-printed JSON dict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from capeval.util.caption_store import load_caption_dict, write_caption_json


def _load_caption_dict(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        return load_caption_dict(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected top-level JSON object, got {type(data)}")
    return {str(k): v for k, v in data.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "shards",
        nargs="+",
        type=Path,
        help="Paths to *.shardKofN.json (or any caption JSON dicts); later files override keys on collision.",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="Merged JSON path")
    args = p.parse_args()

    merged: Dict[str, Any] = {}
    for path in args.shards:
        data = _load_caption_dict(path)
        overlap = set(merged) & set(data)
        if overlap:
            print(f"warn: {path}: {len(overlap)} keys already present; shard values win")
        merged.update(data)

    write_caption_json(args.output, merged)
    print(f"Wrote {len(merged)} entries -> {args.output}")


if __name__ == "__main__":
    main()
