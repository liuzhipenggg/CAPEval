#!/usr/bin/env python3
"""Multi-model caption launcher.

Usage:
  python caption.py internvl1b
  python caption.py qwen8b
  python caption.py qwen8b internvl8b
  python caption.py quick

GPU:
  CUDA_VISIBLE_DEVICES=0 python caption.py internvl1b
  CUDA_VISIBLE_DEVICES=0,1,2,3 python caption.py qwen32b
"""

from __future__ import annotations

from capeval.cli.caption_launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
