"""Submission format for CAPEval one-click evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Union

CaptionMap = Dict[str, str]


def _normalize_image_id(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty image_id")
    return Path(s).name


def load_submission(path: Union[str, Path]) -> CaptionMap:
    """Load a submission file into ``{img_path: caption}``.

    Accepted shapes:
    - list of ``{"image_id"|"img_path", "caption"}``
    - dict mapping ``image_id -> caption``
    - dict with ``"captions"`` key holding either of the above
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"submission not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return parse_submission(data)


def parse_submission(data: Any) -> CaptionMap:
    if isinstance(data, dict) and "captions" in data:
        data = data["captions"]

    out: CaptionMap = {}
    if isinstance(data, list):
        if not data:
            raise ValueError("submission list is empty")
        for i, row in enumerate(data):
            if not isinstance(row, Mapping):
                raise ValueError(f"submission[{i}] must be an object")
            img = row.get("image_id", row.get("img_path"))
            cap = row.get("caption")
            if img is None or cap is None:
                raise ValueError(
                    f"submission[{i}] needs image_id (or img_path) and caption"
                )
            key = _normalize_image_id(img)
            if key in out:
                raise ValueError(f"duplicate image_id: {key}")
            text = str(cap).strip()
            if not text:
                raise ValueError(f"empty caption for {key}")
            out[key] = text
        return out

    if isinstance(data, dict):
        if not data:
            raise ValueError("submission map is empty")
        for k, v in data.items():
            key = _normalize_image_id(k)
            text = str(v).strip()
            if not text:
                raise ValueError(f"empty caption for {key}")
            out[key] = text
        return out

    raise ValueError("submission must be a JSON list or object")


def validate_against_benchmark(
    captions: Mapping[str, str],
    *,
    checklist_path: Union[str, Path],
    require_full: bool = False,
) -> Dict[str, Any]:
    """Check keys against checklist ``img_path`` values.

    Returns a report dict; raises ValueError on hard errors when require_full.
    """
    expected: List[str] = []
    with Path(checklist_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = str(obj.get("img_path") or "").strip()
            if img:
                expected.append(img)
    exp_set = set(expected)
    got = set(captions.keys())
    missing = sorted(exp_set - got)
    unknown = sorted(got - exp_set)
    report: Dict[str, Any] = {
        "n_submission": len(got),
        "n_benchmark": len(exp_set),
        "n_overlap": len(got & exp_set),
        "missing": missing,
        "unknown": unknown,
    }
    if require_full and missing:
        raise ValueError(
            f"submission missing {len(missing)} benchmark images "
            f"(e.g. {missing[:3]})"
        )
    if unknown:
        raise ValueError(
            f"submission has {len(unknown)} unknown image_id(s) "
            f"(e.g. {unknown[:3]})"
        )
    return report


def summary_from_metrics(
    metrics: Mapping[str, Any],
    *,
    model_id: str,
) -> MutableMapping[str, Any]:
    """Build a compact results_summary payload from metrics.json."""
    models = metrics.get("models") or {}
    block = models.get(model_id) or next(iter(models.values()), {})
    summary = dict(block.get("summary") or {})
    return {
        "model_id": model_id,
        "eval_model": metrics.get("eval_model"),
        "scale": metrics.get("scale", "percent"),
        "C": summary.get("C"),
        "P": summary.get("P"),
        "per_category": block.get("per_category") or {},
        "yes": summary.get("yes1"),
        "no": summary.get("no1"),
        "not_mentioned": summary.get("not_mentioned1"),
        "total": summary.get("total1"),
        "n_images": summary.get("n_images"),
        "n_error": summary.get("n_error", 0),
    }
