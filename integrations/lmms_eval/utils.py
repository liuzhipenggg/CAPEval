"""lmms-eval helpers for CAPEval.

Usage:
  export CAPEVAL_HOME=/path/to/CAPEval
  # generate inference jsonl once:
  python -m integrations.lmms_eval.prepare_data
  # then:
  python -m lmms_eval \\
    --model <your_model> \\
    --tasks capeval \\
    --include_path $CAPEVAL_HOME/integrations/lmms_eval \\
    --batch_size 1 \\
    --output_path ./lmms_capeval_out
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_CAPEVAL_HOME = Path(os.environ.get("CAPEVAL_HOME", _HERE.parents[1]))
if str(_CAPEVAL_HOME) not in sys.path:
    sys.path.insert(0, str(_CAPEVAL_HOME))

from capeval.api import caption_prompt, score_caption_map  # noqa: E402

# Aggregator state for one evaluation run (filled by process_results, read by aggregation).
_CAPTION_BAG: Dict[str, str] = {}
_MODEL_NAME = os.environ.get("CAPEVAL_LMMS_MODEL_NAME", "lmms_eval")


def capeval_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    """Return PIL image or path list for lmms-eval visual input."""
    from PIL import Image

    path = doc.get("image") or doc.get("img_path")
    if path is None:
        return []
    path = str(path)
    if not os.path.isabs(path):
        path = str(_CAPEVAL_HOME / "data" / "image" / Path(path).name)
    return [Image.open(path).convert("RGB")]


def capeval_doc_to_text(doc: Dict[str, Any]) -> str:
    return str(doc.get("question") or caption_prompt())


def capeval_doc_to_messages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chat-style messages for modern lmms-eval models."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": capeval_doc_to_text(doc)},
            ],
        }
    ]


def capeval_process_results(doc: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
    """Store caption for later aggregation; return placeholders for per-doc metrics."""
    pred = results[0] if results else ""
    if isinstance(pred, (list, tuple)):
        pred = pred[0] if pred else ""
    image_id = str(doc.get("image_id") or doc.get("img_path") or Path(str(doc.get("image", ""))).name)
    _CAPTION_BAG[Path(image_id).name] = str(pred)
    return {
        "capeval_c": {"image_id": image_id, "caption": str(pred)},
        "capeval_p": {"image_id": image_id, "caption": str(pred)},
    }


def _aggregate_score(metric_key: str) -> float:
    """Run CAPEval judge once over the collected bag; cache on first call."""
    cache_attr = "_CAPEVAL_SCORE_CACHE"
    cache = getattr(capeval_process_results, cache_attr, None)
    if cache is None:
        if not _CAPTION_BAG:
            return float("nan")
        out_dir = os.environ.get("CAPEVAL_LMMS_OUTPUT")
        if not out_dir:
            from capeval.util.paths import model_metrics_dir

            out_dir = str(model_metrics_dir(_MODEL_NAME, root=_CAPEVAL_HOME / "outputs"))
        result = score_caption_map(
            dict(_CAPTION_BAG),
            model_id=_MODEL_NAME,
            output_dir=out_dir,
            eval_model=os.environ.get("EVAL_MODEL"),
        )
        cache = result.get("summary") or {}
        setattr(capeval_process_results, cache_attr, cache)
        # Persist bag for debugging
        bag_path = Path(out_dir) / "lmms_caption_bag.json"
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        bag_path.write_text(json.dumps(_CAPTION_BAG, ensure_ascii=False, indent=2), encoding="utf-8")
    val = cache.get(metric_key)
    if val is None:
        return float("nan")
    return float(val)


def capeval_aggregate_c(results: List[Any]) -> float:
    del results
    return _aggregate_score("C")


def capeval_aggregate_p(results: List[Any]) -> float:
    del results
    return _aggregate_score("P")
