#!/usr/bin/env python3
"""Full CAPEval run: caption then score.

Usage:
  python pipeline.py internvl1b
  python pipeline.py qwen8b
  python pipeline.py quick
"""

from __future__ import annotations

import sys

from caption import main as caption_main
from score import main as score_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0 if args else 2
    rc = caption_main(args)
    if rc != 0:
        return rc
    return score_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
