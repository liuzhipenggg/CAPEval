"""High-level CAPEval benchmark facade."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from capeval.api import caption_prompt, score_caption_map
from capeval.config import apply_defaults
from capeval.submission import (
    load_submission,
    parse_submission,
    summary_from_metrics,
    validate_against_benchmark,
)


class CAPEval:
    """Checklist-based Coverage / Precision evaluation.

    Example::

        from capeval import CAPEval

        ev = CAPEval()
        score = ev.evaluate_map({"SO001.jpg": "A detailed caption..."})
        print(score["C"], score["P"])
    """

    def __init__(
        self,
        *,
        judge: Optional[str] = None,
        model_id: str = "api",
    ) -> None:
        apply_defaults()
        self.judge = judge or os.environ.get("EVAL_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        self.model_id = model_id

    @staticmethod
    def prompt() -> str:
        return caption_prompt()

    def evaluate_map(
        self,
        captions: Mapping[str, str],
        *,
        model_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        require_full: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Score ``{img_path: caption}`` and return Coverage / Precision."""
        apply_defaults()
        mid = model_id or self.model_id
        validate_against_benchmark(
            captions,
            checklist_path=os.environ["CHECKLIST"],
            require_full=require_full,
        )
        raw = score_caption_map(
            captions,
            model_id=mid,
            output_dir=output_dir,
            eval_model=self.judge,
            **kwargs,
        )
        compact = summary_from_metrics(raw["metrics"], model_id=mid)
        out = Path(raw["output_dir"])
        summary_path = out / "results_summary.json"
        summary_path.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "C": compact.get("C"),
            "P": compact.get("P"),
            "per_category": compact.get("per_category") or {},
            "details": {
                "yes": compact.get("yes"),
                "no": compact.get("no"),
                "not_mentioned": compact.get("not_mentioned"),
                "total": compact.get("total"),
                "n_images": compact.get("n_images"),
            },
            "model_id": mid,
            "eval_model": compact.get("eval_model"),
            "output_dir": str(out),
            "results_summary": str(summary_path),
        }

    def evaluate_submission(
        self,
        path: Union[str, Path],
        *,
        model_id: Optional[str] = None,
        require_full: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        captions = load_submission(path)
        return self.evaluate_map(
            captions,
            model_id=model_id,
            require_full=require_full,
            **kwargs,
        )

    def evaluate(
        self,
        *,
        image_id: str,
        caption: str,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Score a single-image caption."""
        return self.evaluate_map(
            {Path(image_id).name: caption},
            model_id=model_id or self.model_id,
            require_full=False,
            **kwargs,
        )


def evaluate_submission_data(data: Any, **kwargs: Any) -> Dict[str, Any]:
    """Parse in-memory submission JSON and score via :class:`CAPEval`."""
    captions = parse_submission(data)
    return CAPEval().evaluate_map(captions, **kwargs)
