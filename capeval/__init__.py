"""
CAPEval — checklist Coverage / Precision evaluation.

  python evaluate.py --submission …
  python caption.py …
  python score.py …
  python pipeline.py …

  from capeval import CAPEval, caption_prompt, score_caption_map
"""

from capeval.api import caption_prompt, score_caption_map
from capeval.benchmark import CAPEval

__all__ = ["CAPEval", "caption_prompt", "score_caption_map"]
