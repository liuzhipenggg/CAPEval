"""Score I/O: JSONL, caption maps, merged-file discovery."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

_CAPEVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CAPEVAL_ROOT not in sys.path:
    sys.path.insert(0, _CAPEVAL_ROOT)

from capeval.schema import CHECKLIST_KEYS, flatten_items
from capeval.util.caption_store import is_merged_basename

# ── defaults ─────────────────────────────────────────────────────────────

EVAL_MODEL_DEFAULT = "Qwen/Qwen2.5-72B-Instruct"

GT_CAPTION_DEFAULT = os.environ.get(
    "GT_CAPTION",
    os.environ.get(
        "GT_JSONL", os.path.join(_CAPEVAL_ROOT, "data", "gt_caption.jsonl")
    ),
)
CHECKLIST_JSONL_DEFAULT = os.environ.get(
    "CHECKLIST", os.path.join(_CAPEVAL_ROOT, "data", "checklist.jsonl")
)

CAPTION_ROOT_DEFAULT = os.environ.get(
    "CAPTION_ROOT",
    os.environ.get(
        "CAPEVAL_CAPTION_DIR", os.path.join(_CAPEVAL_ROOT, "outputs")
    ),
)
IMAGE_ROOT_DEFAULT = os.environ.get(
    "IMAGE_ROOT", os.path.join(_CAPEVAL_ROOT, "data", "image")
)

SINGLE_PASS_SYSTEM = """You are an expert image-caption judge for a research benchmark.

Behavior:
- Apply a moderately strict standard: reward clear, accurate coverage; penalize contradictions.
- When the caption is genuinely ambiguous about a checklist point, prefer "not_mentioned" over guessing "yes".
- Use only the caption text and the checklist metadata provided; do not invent scene facts from prior knowledge.
- Output must be one valid JSON object exactly in the form requested by the user—no markdown code fences, no text before or after."""


def image_id_from_row(row: dict, stats: Optional[dict] = None) -> str:
    i = row.get("id")
    if i is not None and str(i).strip():
        return str(i).strip()
    p = row.get("img_path")
    if p is not None and str(p).strip():
        if stats is not None:
            stats["fallback_img_path"] = stats.get("fallback_img_path", 0) + 1
        return str(p).strip()
    return ""


def load_jsonl(path: str, *, checklist_rows_only: bool = False) -> List[dict]:
    """Load JSON dicts from a file.

    Uses JSONDecoder.raw_decode sequentially so the file may contain strict JSONL or
    extra concatenated JSON values (e.g. trailing arrays). When checklist_rows_only is
    True (checklist file), keep only dicts that contain at least one non-empty checklist key.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lstrip("\ufeff").strip()
    if not text:
        return []
    dec = json.JSONDecoder()
    rows: List[dict] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path!r} at byte {idx}: {e}") from e
        idx = end
        if not isinstance(obj, dict):
            continue
        if checklist_rows_only:
            if not any(obj.get(k) for k in CHECKLIST_KEYS):
                continue
        rows.append(obj)
    return rows


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_jsonl_locked(path: str, obj: dict) -> None:
    """Append one JSONL record (cross-process safe when fcntl exists)."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _parse_semicolon_gpu_groups(s: str) -> List[str]:
    return [p.strip() for p in s.split(";") if p.strip()]


def _cuda_visible_ids() -> List[str]:
    return [x.strip() for x in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip()]


def _resolve_dp_gpu_groups(
    dp: int,
    tp: int,
    groups_csv: Optional[str],
    *,
    env_keys: Tuple[str, ...],
    label: str,
) -> List[str]:
    """Return ``dp`` comma-separated device lists, each with ``tp`` ids (for one vLLM engine)."""
    raw = (groups_csv or "").strip()
    if not raw:
        for k in env_keys:
            v = (os.environ.get(k) or "").strip()
            if v:
                raw = v
                break
    if raw:
        groups = _parse_semicolon_gpu_groups(raw)
        if len(groups) != dp:
            raise SystemExit(
                f"[{label}] expected {dp} GPU groups (semicolon-separated), got {len(groups)}: {raw!r}"
            )
        for i, g in enumerate(groups):
            ids = [x.strip() for x in g.split(",") if x.strip()]
            if len(ids) != tp:
                raise SystemExit(
                    f"[{label}] group {i} must list {tp} device id(s) for TP={tp}, got {len(ids)}: {g!r}"
                )
        return groups
    visible = _cuda_visible_ids()
    need = dp * tp
    if len(visible) != need:
        raise SystemExit(
            f"[{label}] {dp}×TP{tp} needs {need} GPU id(s) in CUDA_VISIBLE_DEVICES "
            f"or set env / --gpu-groups; got {len(visible)}: {visible!r}"
        )
    return [",".join(visible[j * tp : (j + 1) * tp]) for j in range(dp)]


def _split_list_for_dp(items: List[Any], dp: int) -> List[List[Any]]:
    buckets: List[List[Any]] = [[] for _ in range(dp)]
    for i, it in enumerate(items):
        buckets[i % dp].append(it)
    return buckets


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def caption_text_from_row(entry: dict) -> str:
    """Extract caption text from a JSONL row (GT or caption export)."""
    for key in ("gt_caption", "caption", "detailed_description"):
        val = entry.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _is_flat_filename_to_caption_map(row: dict) -> bool:
    """True when `row` is a merged export: {img_path -> caption string, ...} with no JSONL row fields."""
    if not row:
        return False
    row_field_keys = {
        "id",
        "img_path",
        "gt_caption",
        "caption",
        "overview_description",
        "detailed_description",
    }
    if row_field_keys & row.keys():
        return False
    return all(isinstance(k, str) for k in row) and all(
        isinstance(v, str) for v in row.values()
    )


def load_captions_by_image_id(path: str, stats: Optional[dict] = None) -> Dict[str, str]:
    """Map lookup key -> caption text.

    Supports:
    - JSONL rows with id/img_path + gt_caption (or legacy detailed_description)
    - Single JSON object mapping img_path (e.g. SO001.jpg) -> caption text
    """
    out: Dict[str, str] = {}
    for row in load_jsonl(path):
        if _is_flat_filename_to_caption_map(row):
            for k, v in row.items():
                cap = str(v).strip()
                if cap:
                    out[k] = cap
            continue
        cap = caption_text_from_row(row)
        if not cap:
            continue
        iid = image_id_from_row(row, stats)
        if not iid:
            continue
        out[iid] = cap
    return out


def build_checklist_items_with_index(entry: dict) -> List[dict]:
    flat = flatten_items(entry)
    items = []
    for idx, it in enumerate(flat):
        items.append(
            {
                "item_index": idx,
                "checklist_type": it["checklist_type"],
                "tags": it.get("tags", "") or "",
                "tag": it.get("tags", "") or "",
                "question": it["question"],
            }
        )
    return items


def default_eval_output_dir() -> str:
    """Default artifact directory for prepare / all (legacy flat scores/)."""
    return os.path.join(_CAPEVAL_ROOT, "outputs", "scores")


def _is_skipped_glm46v_full_caption_basename(name: str) -> bool:
    """True = do not eval this merged file (full zai-org/GLM-4.6V); keep GLM-4.6V-Flash.

    Matches caption output basenames like ``zai-org_GLM-4.6V.json`` but not ``...GLM-4.6V-Flash...``.
    Opt-in full model in eval: ``export CAPEVAL_EVAL_SKIP_GLM46V_FULL=0``.
    """
    v = (os.environ.get("CAPEVAL_EVAL_SKIP_GLM46V_FULL") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    stem = os.path.splitext(name)[0].lower()
    if "glm-4.6v" not in stem:
        return False
    if "flash" in stem:
        return False
    return True


def _use_caption_basename_for_eval(name: str) -> bool:
    if not is_merged_basename(name):
        return False
    if _is_skipped_glm46v_full_caption_basename(name):
        return False
    return True


def _list_merged_caption_files_in_dir(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out: List[str] = []
    n_skip_glm = 0
    for name in sorted(os.listdir(d)):
        if name.startswith("."):
            continue
        low = name.lower()
        if not (low.endswith(".json") or low.endswith(".jsonl")):
            continue
        if not is_merged_basename(name):
            continue
        if _is_skipped_glm46v_full_caption_basename(name):
            n_skip_glm += 1
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            out.append(p)
    if n_skip_glm:
        print(
            f"[prepare] skipped {n_skip_glm} full GLM-4.6V merged caption file(s) under {d!r} "
            "(CAPEVAL_EVAL_SKIP_GLM46V_FULL=1 default; GLM-4.6V-Flash kept; set CAPEVAL_EVAL_SKIP_GLM46V_FULL=0 to include full)."
        )
    return out


def discover_merged_caption_files(caption_root: str) -> List[str]:
    """Merged caption outputs (exclude *.shard*).

    Search order:
      1. New layout: ``{root}/<model>/caption/`` (all models, flattened)
      2. ``{caption_root}/prompt/`` (legacy)
      3. ``{caption_root}/``
      4. ``{caption_root}/captions/prompt/``
    """
    from capeval.util.paths import discover_model_score_jobs

    jobs = discover_model_score_jobs(caption_root)
    if jobs:
        out: List[str] = []
        for _mid, paths, _metrics in jobs:
            out.extend(paths)
        return out

    candidates = [
        os.path.join(caption_root, "prompt"),
        caption_root,
        os.path.join(caption_root, "captions", "prompt"),
    ]
    for d in candidates:
        found = _list_merged_caption_files_in_dir(d)
        if found:
            return found
    return []


def resolve_caption_paths(ns: argparse.Namespace) -> List[str]:
    caps = getattr(ns, "caption_paths", None)
    if caps:
        raw = list(caps)
        merged_only = [p for p in raw if _use_caption_basename_for_eval(os.path.basename(p))]
        dropped_shard = sum(
            1 for p in raw if not is_merged_basename(os.path.basename(p))
        )
        dropped_glm = sum(
            1
            for p in raw
            if is_merged_basename(os.path.basename(p))
            and _is_skipped_glm46v_full_caption_basename(os.path.basename(p))
        )
        if dropped_shard:
            print(
                f"[prepare] ignored {dropped_shard} shard file(s) (.shard in name); "
                f"eval uses merged captions only."
            )
        if dropped_glm:
            print(
                f"[prepare] ignored {dropped_glm} full GLM-4.6V path(s) "
                "(CAPEVAL_EVAL_SKIP_GLM46V_FULL=1; Flash paths kept)."
            )
        if not merged_only:
            raise SystemExit(
                "[prepare] --caption-paths: no merged caption files left after excluding *.shard* "
                "and optional full GLM-4.6V skips "
                "(merge shards first, or set CAPEVAL_EVAL_SKIP_GLM46V_FULL=0 to include full GLM-4.6V)."
            )
        return merged_only
    root = getattr(ns, "caption_root", CAPTION_ROOT_DEFAULT)
    found = discover_merged_caption_files(root)
    if not found:
        raise SystemExit(
            f"[prepare] No merged caption files under {root}/prompt/, {root}/, "
            f"or {root}/captions/prompt/ (need *.json / *.jsonl without '.shard'). "
            "Merge shards, adjust --caption-root, or pass --caption-paths."
        )
    print(f"[prepare] auto caption-paths: n_files={len(found)} (under {root})")
    return found

