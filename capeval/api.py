"""Public CAPEval API for external harnesses (VLMEvalKit, lmms-eval, …).

Typical flow for a host framework:
1. Run its own VLM inference with :func:`caption_prompt` on each image.
2. Collect ``{img_path: caption}`` (keys are filenames under ``data/image``).
3. Call :func:`score_caption_map` to compute C / P.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from capeval.config import apply_defaults, repo_root
from capeval.prompts import get_prompt

CaptionMap = Mapping[str, str]


def caption_prompt(name: str = "PROMPT") -> str:
    """Return the CAPEval caption instruction text."""
    return get_prompt(name)


def default_paths() -> Dict[str, Path]:
    """Resolved data / output paths after applying env defaults."""
    apply_defaults()
    from capeval.util.paths import output_root

    root = output_root()
    return {
        "repo": repo_root(),
        "image_root": Path(os.environ["IMAGE_ROOT"]),
        "gt_jsonl": Path(os.environ.get("GT_CAPTION") or os.environ["GT_JSONL"]),
        "checklist": Path(os.environ["CHECKLIST"]),
        "output_root": root,
        # Discovery roots (model-first layout lives under output_root)
        "caption_dir": Path(os.environ["CAPEVAL_CAPTION_DIR"]),
        "eval_dir": Path(os.environ["CAPEVAL_EVAL_DIR"]),
    }


def list_inference_rows() -> List[Dict[str, Any]]:
    """Rows for caption-only inference: one entry per image.

    Each dict has: ``index``, ``image_id``, ``img_path``, ``image`` (absolute path),
    ``question`` (caption prompt), ``category``.
    """
    paths = default_paths()
    prompt = caption_prompt()
    rows: List[Dict[str, Any]] = []
    with paths["checklist"].open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img_path = str(obj.get("img_path") or "").strip()
            if not img_path:
                continue
            abs_img = paths["image_root"] / img_path
            stem = Path(img_path).stem
            prefix = "".join(ch for ch in stem if ch.isalpha()) or "unknown"
            rows.append(
                {
                    "index": i,
                    "image_id": img_path,
                    "img_path": img_path,
                    "image": str(abs_img.resolve()),
                    "question": prompt,
                    "category": prefix.upper(),
                }
            )
    return rows


def _write_caption_json(captions: CaptionMap, path: Path) -> None:
    # Normalize keys to basenames (SO001.jpg)
    payload = {Path(str(k)).name: str(v) for k, v in captions.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def score_caption_map(
    captions: CaptionMap,
    *,
    model_id: str = "external",
    output_dir: Optional[Union[str, Path]] = None,
    eval_model: Optional[str] = None,
    tp_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    eval_dp_size: Optional[int] = None,
    save_every: Optional[int] = None,
    no_per_model_dirs: bool = True,
    keep_caption_json: bool = True,
) -> Dict[str, Any]:
    """Score a caption map with the CAPEval checklist judge.

    Args:
        captions: Mapping ``img_path -> caption`` (e.g. ``\"SO001.jpg\": \"...\"``).
        model_id: Name stored in metrics / artifacts.
        output_dir: Metrics directory (default ``outputs/<model>/metrics``).
        eval_model: Judge LLM id (default ``EVAL_MODEL`` / Qwen2.5-72B).
        tp_size / eval_batch_size / eval_dp_size / save_every: judge runtime knobs.
        keep_caption_json: Keep the staging caption JSON under ``caption/`` (default True).

    Returns:
        Dict with ``eval_model``, ``model_id``, ``summary`` (C/P on a 0–100 scale),
        ``output_dir``, and full ``metrics`` payload.
    """
    if not captions:
        raise ValueError("captions mapping is empty")

    apply_defaults()
    paths = default_paths()

    from capeval import judge as J
    from capeval.util.paths import model_caption_dir, model_metrics_dir

    out = Path(output_dir) if output_dir else model_metrics_dir(model_id)
    out.mkdir(parents=True, exist_ok=True)

    cap_dir = model_caption_dir(model_id)
    cap_path = cap_dir / "prompt.json"
    _write_caption_json(captions, cap_path)

    ns = argparse.Namespace(
        output_dir=str(out),
        image_root=str(paths["image_root"]),
        gt_jsonl=str(paths["gt_jsonl"]),
        checklist_jsonl=str(paths["checklist"]),
        caption_root=str(paths["caption_dir"]),
        caption_paths=[str(cap_path)],
        model_name=model_id,
        eval_model=eval_model or os.environ.get("EVAL_MODEL", J.EVAL_MODEL_DEFAULT),
        tp_size=int(tp_size if tp_size is not None else os.environ.get("TP_SIZE", "1")),
        eval_dp_size=int(
            eval_dp_size if eval_dp_size is not None else os.environ.get("EVAL_DP_SIZE", "1")
        ),
        eval_gpu_groups=os.environ.get("CAPEVAL_EVAL_GPU_GROUPS") or None,
        gpu_memory_utilization=float(os.environ.get("CAPEVAL_EVAL_GPU_MEM_UTIL", "0.9")),
        max_model_len=int(os.environ.get("CAPEVAL_EVAL_MAX_MODEL_LEN", "8192")),
        eval_batch_size=int(
            eval_batch_size
            if eval_batch_size is not None
            else os.environ.get("EVAL_BATCH_SIZE", "8")
        ),
        eval_max_tokens=int(os.environ.get("CAPEVAL_EVAL_MAX_TOKENS", "8192")),
        save_every=int(save_every if save_every is not None else os.environ.get("SAVE_EVERY", "50")),
        no_per_model_dirs=no_per_model_dirs,
    )

    J.cmd_prepare(ns)
    J.cmd_evaluate(ns)
    J.cmd_metrics(ns)
    J.cmd_report(ns)
    if not no_per_model_dirs:
        J.cmd_split_per_model(ns)

    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = metrics.get("models") or {}
    # Prefer exact model_id; else first entry
    summary = (models.get(model_id) or next(iter(models.values()), {})).get("summary", {})

    if not keep_caption_json and cap_path.is_file():
        cap_path.unlink()

    return {
        "eval_model": metrics.get("eval_model"),
        "model_id": model_id,
        "summary": summary,
        "output_dir": str(out),
        "metrics": metrics,
        "caption_json": str(cap_path),
    }


def export_inference_jsonl(out_path: Union[str, Path]) -> Path:
    """Write inference rows (no checklist) for lmms-eval / custom loaders."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list_inference_rows()
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def export_vlmeval_tsv(out_path: Union[str, Path]) -> Path:
    """Write a VLMEvalKit-style TSV (index / image / question / category).

    ``image`` is an absolute path; copy or re-root as needed when packaging for Hub.
    """
    import csv

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list_inference_rows()
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["index", "image", "question", "category", "image_id"],
            delimiter="\t",
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out
