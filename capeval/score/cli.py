"""CAPEval evaluation pipeline: checklist LLM judging only.

Steps: prepare -> evaluate -> metrics -> report

    python -m capeval.judge all
"""
from __future__ import annotations

import argparse

from capeval.metrics.aggregate import cmd_metrics
from capeval.metrics.report import cmd_report, cmd_split_per_model
from capeval.score.evaluate import cmd_evaluate
from capeval.score.io import (
    CAPTION_ROOT_DEFAULT,
    CHECKLIST_JSONL_DEFAULT,
    EVAL_MODEL_DEFAULT,
    GT_CAPTION_DEFAULT,
    IMAGE_ROOT_DEFAULT,
    default_eval_output_dir,
)
from capeval.score.prepare import cmd_prepare


def cmd_all(args: argparse.Namespace) -> None:
    if not getattr(args, "output_dir", None):
        args.output_dir = default_eval_output_dir()
        print(f"[all] default --output-dir -> {args.output_dir}")
    cmd_prepare(args)
    cmd_evaluate(args)
    cmd_metrics(args)
    cmd_report(args)
    if not getattr(args, "no_per_model_dirs", False):
        cmd_split_per_model(args)
    print(f"\n[all] finished under {args.output_dir!r}")


# ── argparse ─────────────────────────────────────────────────────────────


def add_output_dir_arg(p: argparse.ArgumentParser, *, required: bool = False) -> None:
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        required=required,
        help="Metrics artifact directory (score.py defaults to outputs/<model>/metrics/).",
    )


def add_image_root_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--image-root",
        type=str,
        default=IMAGE_ROOT_DEFAULT,
        help=f"GT img_path basenames live here (default: {IMAGE_ROOT_DEFAULT}).",
    )


def add_prepare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gt-caption",
        "--gt-jsonl",
        dest="gt_jsonl",
        type=str,
        default=GT_CAPTION_DEFAULT,
        help=f"GT caption JSONL (default: {GT_CAPTION_DEFAULT})",
    )
    p.add_argument(
        "--checklist-jsonl",
        type=str,
        default=CHECKLIST_JSONL_DEFAULT,
        help=f"default: {CHECKLIST_JSONL_DEFAULT}",
    )
    p.add_argument(
        "--caption-root",
        type=str,
        default=CAPTION_ROOT_DEFAULT,
        help=f"CAPEval caption root (default: {CAPTION_ROOT_DEFAULT}).",
    )
    p.add_argument(
        "--caption-paths",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Caption JSON/JSONL files; if omitted, auto-discover merged files under "
            "caption-root/prompt or caption-root."
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="If set with exactly one --caption-paths file, use this model_id instead of filename stem.",
    )


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-model", type=str, default=EVAL_MODEL_DEFAULT)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument(
        "--eval-dp-size",
        type=int,
        default=1,
        help=(
            "Number of parallel evaluate workers; each loads one vLLM engine with --tp-size GPUs. "
            "Example: 2 with TP4 ⇒ 8 GPUs (set CUDA_VISIBLE_DEVICES or --eval-gpu-groups)."
        ),
    )
    p.add_argument(
        "--eval-gpu-groups",
        type=str,
        default=None,
        help=(
            "Semicolon-separated CUDA device lists per worker, e.g. '0,1,2,3;4,5,6,7'. "
            "Default: split CUDA_VISIBLE_DEVICES into eval-dp-size chunks of tp-size ids each. "
            "Env: CAPEVAL_EVAL_GPU_GROUPS."
        ),
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-max-tokens", type=int, default=8192)
    p.add_argument("--save-every", type=int, default=50)


def add_metrics_metadata_args(p: argparse.ArgumentParser) -> None:
    """eval-model stored in metrics.json (metrics subcommand only)."""
    p.add_argument("--eval-model", type=str, default=EVAL_MODEL_DEFAULT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Step 1: build prepared.jsonl")
    add_output_dir_arg(p_prep)
    add_image_root_arg(p_prep)
    add_prepare_args(p_prep)
    p_prep.set_defaults(func=cmd_prepare)

    p_ev = sub.add_parser("evaluate", help="Step 2: LLM single-pass")
    add_output_dir_arg(p_ev, required=True)
    add_eval_args(p_ev)
    p_ev.set_defaults(func=cmd_evaluate)

    p_met = sub.add_parser("metrics", help="Step 3: compute metrics.json")
    add_output_dir_arg(p_met, required=True)
    add_metrics_metadata_args(p_met)
    p_met.set_defaults(func=cmd_metrics)

    p_rep = sub.add_parser("report", help="Step 4: CSV + markdown from metrics.json")
    add_output_dir_arg(p_rep, required=True)
    p_rep.set_defaults(func=cmd_report)

    p_all = sub.add_parser("all", help="Run prepare through report")
    add_output_dir_arg(p_all)
    add_image_root_arg(p_all)
    add_prepare_args(p_all)
    add_eval_args(p_all)
    p_all.add_argument(
        "--no-per-model-dirs",
        action="store_true",
        help="Skip creating <output_dir>/<model_id>/ (one result tree per model).",
    )
    p_all.set_defaults(func=cmd_all)

    p_spm = sub.add_parser(
        "split-per-model",
        help="From an existing output directory (with metrics.json), populate <model_id>/ subdirs.",
    )
    add_output_dir_arg(p_spm, required=True)
    add_metrics_metadata_args(p_spm)
    p_spm.set_defaults(func=cmd_split_per_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
