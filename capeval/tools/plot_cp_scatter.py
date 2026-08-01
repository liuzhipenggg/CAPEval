"""Plot Coverage vs Precision for one or more metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def _load_points(paths: List[Path]) -> List[Tuple[str, float, float]]:
    points: List[Tuple[str, float, float]] = []
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models") or {}
        for mid, block in models.items():
            summary = block.get("summary") or {}
            c, p = summary.get("C"), summary.get("P")
            if c is None or p is None:
                continue
            # metrics.json stores percent (0–100); tolerate legacy fractions
            cf, pf = float(c), float(p)
            if cf <= 1.0 and pf <= 1.0:
                cf, pf = cf * 100.0, pf * 100.0
            points.append((str(mid), cf, pf))
    return points


def plot_cp_scatter(
    metrics_paths: List[Path],
    out_path: Path,
    *,
    title: str = "CAPEval captioner profile (C vs P)",
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. "
            "pip install matplotlib"
        ) from e

    points = _load_points(metrics_paths)
    if not points:
        raise SystemExit("No C/P points found in the given metrics.json files")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, c, p in points:
        ax.scatter([c], [p], s=80)
        ax.annotate(name, (c, p), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Coverage C (%)")
    ax.set_ylabel("Precision P (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more metrics.json paths (shell globs expanded by your shell)",
    )
    ap.add_argument(
        "--out",
        default="cp_scatter.png",
        help="Output PNG path",
    )
    ap.add_argument("--title", default="CAPEval captioner profile (C vs P)")
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.metrics]
    out = plot_cp_scatter(paths, Path(args.out), title=args.title)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
