#!/usr/bin/env python3
"""Evaluate a CAPEval caption submission (Coverage / Precision).

Usage:
  python evaluate.py -s examples/submission.json --model-id my_captioner
  python evaluate.py -s examples/submission.json --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capeval.cli.evaluate_submission import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
