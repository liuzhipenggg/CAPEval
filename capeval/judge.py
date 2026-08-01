#!/usr/bin/env python3
"""
CAPEval evaluation pipeline: checklist LLM judging only.

Compatibility facade for ``python -m capeval.judge`` and ``capeval.api``.
Implementation lives in ``capeval.score`` and ``capeval.metrics``.
"""

from __future__ import annotations

from capeval.metrics.aggregate import cmd_metrics
from capeval.metrics.report import cmd_report, cmd_split_per_model
from capeval.score.cli import cmd_all, main
from capeval.score.evaluate import cmd_evaluate, parse_single_pass_json
from capeval.score.io import EVAL_MODEL_DEFAULT
from capeval.score.prepare import cmd_prepare

__all__ = [
    "EVAL_MODEL_DEFAULT",
    "cmd_all",
    "cmd_evaluate",
    "cmd_metrics",
    "cmd_prepare",
    "cmd_report",
    "cmd_split_per_model",
    "main",
    "parse_single_pass_json",
]

if __name__ == "__main__":
    main()
