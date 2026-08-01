#!/usr/bin/env python3
"""Checklist score launcher (LLM judge only).

Usage:
  python score.py internvl1b
  EVAL_MODEL=Qwen/Qwen2.5-72B-Instruct python score.py qwen8b
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python score.py internvl1b
"""

from __future__ import annotations

from capeval.cli.score_launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
