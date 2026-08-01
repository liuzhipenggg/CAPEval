"""Prepare step: join captions with checklist into prepared.jsonl."""
from __future__ import annotations

import argparse
import os
from typing import Dict

from capeval.score.io import (
    append_jsonl,
    build_checklist_items_with_index,
    default_eval_output_dir,
    image_id_from_row,
    load_captions_by_image_id,
    load_jsonl,
    resolve_caption_paths,
)

def _domain_from_img_path(img_path: str) -> str:
    """Super-category prefix from filename (SO / PA / TI / DK)."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    prefix = []
    for ch in stem:
        if ch.isalpha():
            prefix.append(ch)
        else:
            break
    return "".join(prefix).upper() if prefix else "unknown"


def cmd_prepare(args: argparse.Namespace) -> None:
    if not getattr(args, "output_dir", None):
        args.output_dir = default_eval_output_dir()
        print(f"[prepare] default --output-dir -> {args.output_dir}")
    args.caption_paths = resolve_caption_paths(args)
    stats = {"fallback_img_path": 0}
    gt_rows = load_jsonl(args.gt_jsonl)
    gt_by_id = {image_id_from_row(r, stats): r for r in gt_rows}
    gt_by_id.pop("", None)
    gt_by_img_path: Dict[str, dict] = {}
    for r in gt_rows:
        p = str(r.get("img_path") or "").strip()
        if p:
            gt_by_img_path[p] = r

    cl_rows = load_jsonl(args.checklist_jsonl, checklist_rows_only=True)
    cl_by_id = {image_id_from_row(r, stats): r for r in cl_rows}
    cl_by_id.pop("", None)
    cl_by_img_path: Dict[str, dict] = {}
    for r in cl_rows:
        p = str(r.get("img_path") or "").strip()
        if p:
            cl_by_img_path[p] = r

    out_path = os.path.join(args.output_dir, "prepared.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8"):
        pass

    n_units = 0
    n_skipped = 0
    from capeval.util.paths import model_id_from_caption_path

    for cap_path in args.caption_paths:
        model_id = model_id_from_caption_path(cap_path)
        if getattr(args, "model_name", None) and len(args.caption_paths) == 1:
            model_id = args.model_name
        cap_stats = {"fallback_img_path": 0}
        caps = load_captions_by_image_id(cap_path, cap_stats)
        stats["fallback_img_path"] += cap_stats.get("fallback_img_path", 0)

        for cap_key, caption in caps.items():
            # Prefer img_path (SO001.jpg) as join key — merged JSON captions are keyed by filename.
            cl_row = cl_by_img_path.get(cap_key) or cl_by_id.get(cap_key)
            if cl_row is None:
                n_skipped += 1
                continue
            canonical_id = image_id_from_row(cl_row, stats)
            if not canonical_id:
                n_skipped += 1
                continue
            img_path = str(cl_row.get("img_path") or "").strip()
            # Prefer img_path join: a few GT/checklist rows share path but disagree on `id`.
            gt_row = gt_by_img_path.get(img_path) or gt_by_id.get(canonical_id, {})
            if not img_path:
                img_path = str(gt_row.get("img_path") or "").strip()
            if not img_path:
                n_skipped += 1
                continue
            abs_img = os.path.join(args.image_root, img_path)
            domain = _domain_from_img_path(img_path)
            checklist_items = build_checklist_items_with_index(cl_row)
            if not checklist_items:
                n_skipped += 1
                continue
            unit = {
                # Stable key for eval artifacts: same as checklist/GT `img_path` (e.g. SO001.jpg).
                "image_id": img_path,
                "image_path": img_path,
                "absolute_image_path": abs_img,
                "domain": domain,
                "model_id": model_id,
                "caption": caption,
                "checklist_items": checklist_items,
            }
            append_jsonl(out_path, unit)
            n_units += 1

    if stats.get("fallback_img_path"):
        print(f"[prepare] rows using img_path as id (no id field): {stats['fallback_img_path']}")
    if n_skipped:
        print(f"[prepare] skipped {n_skipped} unmatched / empty caption keys")
    if n_units == 0:
        raise SystemExit(
            "[prepare] wrote 0 units — no caption keys matched checklist img_path/id. "
            "Check caption JSON keys (e.g. SO001.jpg) and --checklist-jsonl / --caption-paths."
        )
    print(f"[prepare] wrote {n_units} units -> {out_path}")
