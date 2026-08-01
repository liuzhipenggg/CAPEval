"""Output layout helpers: ``outputs/<model>/{caption,metrics}/``."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from capeval.util.names import model_safe

PathLike = Union[str, Path]


def output_root(default: Optional[PathLike] = None) -> Path:
    raw = (os.environ.get("CAPEVAL_OUTPUT_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser()
    if default is not None:
        return Path(default)
    # Fallback when apply_defaults() has not run yet
    return Path(__file__).resolve().parents[2] / "outputs"


def model_run_dir(model_id: str, *, root: Optional[PathLike] = None) -> Path:
    """``outputs/<model_safe>/``."""
    return (Path(root) if root is not None else output_root()) / model_safe(model_id)


def model_caption_dir(model_id: str, *, root: Optional[PathLike] = None) -> Path:
    """``outputs/<model_safe>/caption/``."""
    return model_run_dir(model_id, root=root) / "caption"


def model_metrics_dir(model_id: str, *, root: Optional[PathLike] = None) -> Path:
    """``outputs/<model_safe>/metrics/``."""
    return model_run_dir(model_id, root=root) / "metrics"


def caption_json_path(
    model_id: str,
    prompt: str = "PROMPT",
    *,
    root: Optional[PathLike] = None,
) -> Path:
    """``outputs/<model_safe>/caption/<prompt.lower()>.json``."""
    return model_caption_dir(model_id, root=root) / f"{(prompt or 'PROMPT').lower()}.json"


def model_id_from_caption_path(path: PathLike) -> str:
    """Infer model id from a caption file path.

    Prefers the new layout ``.../<model>/caption/<file>.json``.
    Falls back to the legacy basename stem (``.../prompt/<model>.json``).
    """
    p = Path(path)
    parent = p.parent
    if parent.name.lower() in {"caption", "captions"} and parent.parent.name:
        return parent.parent.name
    return p.stem


def _caption_files_under(model_dir: Path) -> List[str]:
    from capeval.score.io import _list_merged_caption_files_in_dir

    cap_dir = model_dir / "caption"
    if not cap_dir.is_dir():
        alt = model_dir / "captions"
        cap_dir = alt if alt.is_dir() else cap_dir
    if not cap_dir.is_dir():
        return []
    return _list_merged_caption_files_in_dir(str(cap_dir))


def discover_model_score_jobs(
    root: Optional[PathLike] = None,
) -> List[Tuple[str, List[str], str]]:
    """Discover ``(model_id, caption_paths, metrics_dir)`` under the output root.

    New layout only: ``<root>/<model>/caption/*.{json,jsonl}`` (merged files).
    """
    base = Path(root) if root is not None else output_root()
    if not base.is_dir():
        return []

    jobs: List[Tuple[str, List[str], str]] = []
    skip = {"captions", "scores", ".git"}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if model_dir.name.lower() in skip:
            continue
        files = _caption_files_under(model_dir)
        if not files:
            continue
        metrics = str(model_dir / "metrics")
        jobs.append((model_dir.name, files, metrics))
    return jobs


def resolve_score_jobs(
    targets: List[str],
    *,
    root: Optional[PathLike] = None,
) -> List[Tuple[str, List[str], str]]:
    """Resolve explicit score targets to ``(model_dir_name, caption_paths, metrics_dir)``.

    Accepts the same aliases / HF ids / paths as ``caption.py``, or an existing
    ``outputs/<model_safe>/`` directory name.
    """
    import sys

    from capeval.models import expand_targets, resolve_spec

    if not targets:
        print(
            "No model specified for scoring.\n"
            "Pass the same target used for captioning, e.g.\n"
            "  python score.py internvl1b\n"
            "  python score.py Qwen/Qwen2.5-VL-7B-Instruct",
            file=sys.stderr,
        )
        raise SystemExit(2)

    base = Path(root) if root is not None else output_root()
    expanded = expand_targets(targets)
    jobs: List[Tuple[str, List[str], str]] = []
    seen: set[str] = set()

    for t in expanded:
        candidates: List[Path] = []
        # Exact outputs/<name>/
        candidates.append(base / t)
        candidates.append(base / model_safe(t))
        try:
            spec = resolve_spec(t)
            candidates.append(base / model_safe(spec.model_id))
        except SystemExit:
            pass

        model_dir: Optional[Path] = None
        for cand in candidates:
            if cand.is_dir() and _caption_files_under(cand):
                model_dir = cand
                break
        if model_dir is None:
            print(
                f"ERROR: No captions for {t!r} under {base}/<model>/caption/.\n"
                f"Run: python caption.py {t}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        key = model_dir.name
        if key in seen:
            continue
        seen.add(key)
        files = _caption_files_under(model_dir)
        jobs.append((key, files, str(model_dir / "metrics")))
    return jobs
