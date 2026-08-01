"""CLI: evaluate a caption submission against CAPEval checklists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from capeval.api import score_caption_map
from capeval.config import apply_defaults
from capeval.submission import (
    load_submission,
    summary_from_metrics,
    validate_against_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score a CAPEval caption submission (Coverage / Precision).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python evaluate.py -s examples/submission.json --model-id my_captioner"
        ),
    )
    p.add_argument(
        "--submission",
        "-s",
        required=True,
        help="Path to submission JSON (list of {image_id, caption} or map)",
    )
    p.add_argument(
        "--model-id",
        default="submission",
        help="Name stored under outputs/<model_id>/ (default: submission)",
    )
    p.add_argument(
        "--eval-model",
        default=None,
        help="Judge LLM id (default: EVAL_MODEL / Qwen2.5-72B)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Metrics directory (default: outputs/<model_id>/metrics)",
    )
    p.add_argument(
        "--require-full",
        action="store_true",
        help="Require captions for all 300 benchmark images",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate submission schema and image ids without running the judge",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    apply_defaults()

    try:
        captions = load_submission(args.submission)
        report = validate_against_benchmark(
            captions,
            checklist_path=os.environ["CHECKLIST"],
            require_full=args.require_full,
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"Loaded {report['n_submission']} captions "
        f"({report['n_overlap']}/{report['n_benchmark']} benchmark images)."
    )
    if report["missing"] and not args.require_full:
        print(f"Missing {len(report['missing'])} images (partial evaluation).")

    if args.check_only:
        print("Submission is valid.")
        return 0

    result = score_caption_map(
        captions,
        model_id=args.model_id,
        output_dir=args.output_dir,
        eval_model=args.eval_model,
    )
    compact = summary_from_metrics(result["metrics"], model_id=args.model_id)
    out_dir = Path(result["output_dir"])
    summary_path = out_dir / "results_summary.json"
    summary_path.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"C: {compact.get('C')}   P: {compact.get('P')}")
    for cat, row in (compact.get("per_category") or {}).items():
        print(f"  [{cat}] C={row.get('C')}  P={row.get('P')}")
    print(f"Results: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
