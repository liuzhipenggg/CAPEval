#!/usr/bin/env python3
"""Checklist score launcher (LLM judge only).

Usage:
  python score.py internvl1b
  EVAL_MODEL=Qwen/Qwen2.5-72B-Instruct python score.py qwen8b
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python score.py internvl1b

Writes metrics under ``outputs/<model>/metrics/`` for each selected model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from capeval.config import apply_defaults, python_bin, repo_root
from capeval.util.paths import resolve_score_jobs


def _run_judge(
    *,
    out_dir: Path,
    caption_paths: list[Path],
    cwd: Path,
    model_name: str | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin(),
        "-m",
        "capeval.judge",
        "all",
        "--gt-caption",
        os.environ.get("GT_CAPTION") or os.environ["GT_JSONL"],
        "--checklist-jsonl",
        os.environ["CHECKLIST"],
        "--image-root",
        os.environ["IMAGE_ROOT"],
        "--output-dir",
        str(out_dir),
        "--eval-model",
        os.environ["EVAL_MODEL"],
        "--tp-size",
        os.environ["TP_SIZE"],
        "--save-every",
        os.environ["SAVE_EVERY"],
        "--eval-batch-size",
        os.environ["EVAL_BATCH_SIZE"],
        "--eval-dp-size",
        os.environ["EVAL_DP_SIZE"],
        "--no-per-model-dirs",
        "--caption-paths",
        *[str(p) for p in caption_paths],
    ]
    if model_name and len(caption_paths) == 1:
        cmd.extend(["--model-name", model_name])

    env = os.environ.copy()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_LIST", "0")
    print(f"[score] EVAL_MODEL={os.environ['EVAL_MODEL']} TP_SIZE={os.environ['TP_SIZE']}")
    print(f"[score] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def main(argv: list[str] | None = None) -> int:
    apply_defaults()
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    cwd = repo_root()

    image_root = os.environ.get("IMAGE_ROOT") or os.environ.get("CAPTION_LAUNCHER_INPUT_DIR", "")
    os.environ["IMAGE_ROOT"] = image_root
    if not image_root or not Path(image_root).is_dir():
        print("ERROR: Set IMAGE_ROOT or CAPTION_LAUNCHER_INPUT_DIR to an existing image directory.")
        return 1
    gt_path = os.environ.get("GT_CAPTION") or os.environ.get("GT_JSONL", "")
    if not gt_path or not Path(gt_path).is_file():
        print(f"ERROR: Set GT_CAPTION to an existing file (got {gt_path!r}).")
        return 1
    os.environ["GT_CAPTION"] = gt_path
    os.environ["GT_JSONL"] = gt_path
    cl_path = os.environ.get("CHECKLIST", "")
    if not cl_path or not Path(cl_path).is_file():
        print(f"ERROR: Set CHECKLIST to an existing file (got {cl_path!r}).")
        return 1

    jobs = resolve_score_jobs(args)
    rc = 0
    for model_id, cap_paths, metrics_dir in jobs:
        print(f"[score] model={model_id} captions={len(cap_paths)} → {metrics_dir}")
        model_name = model_id if len(cap_paths) == 1 else None
        one = _run_judge(
            out_dir=Path(metrics_dir),
            caption_paths=[Path(p) for p in cap_paths],
            cwd=cwd,
            model_name=model_name,
        )
        if one != 0:
            rc = one
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
