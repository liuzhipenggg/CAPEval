"""Write CSV / markdown reports and per-model ranking sidecars."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os

from typing import Any, Dict, List, Optional

from capeval.metrics.aggregate import cmd_metrics
from capeval.score.io import iter_jsonl
from capeval.submission import summary_from_metrics
from capeval.util.names import safe_model_subdir


def cmd_report(args: argparse.Namespace) -> None:
    path = os.path.join(args.output_dir, "metrics.json")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}; run metrics first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", {})
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    tag_csv = os.path.join(args.output_dir, "per_tag.csv")
    cat_csv = os.path.join(args.output_dir, "per_category.csv")
    md_path = os.path.join(args.output_dir, "report.md")

    cols = ["model", "C", "P", "n_images", "n_error"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name in sorted(models.keys()):
            s = models[name]["summary"]
            w.writerow(
                [
                    name,
                    s.get("C", ""),
                    s.get("P", ""),
                    s.get("n_images", ""),
                    s.get("n_error", 0),
                ]
            )

    with open(tag_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "tag", "yes1", "total1"])
        for name in sorted(models.keys()):
            for tg, row in sorted(models[name].get("per_tag", {}).items()):
                w.writerow([name, tg, row["yes1"], row["total1"]])

    with open(cat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "C", "P", "n_images", "yes1", "no1", "total1"])
        for name in sorted(models.keys()):
            for cat, row in (models[name].get("per_category") or {}).items():
                w.writerow(
                    [
                        name,
                        cat,
                        row.get("C", ""),
                        row.get("P", ""),
                        row.get("n_images", ""),
                        row.get("yes1", ""),
                        row.get("no1", ""),
                        row.get("total1", ""),
                    ]
                )

    lines = [
        "# CAPEval eval report",
        "",
        "Scores use a **0–100** (percent) scale. Micro-averaged at the model level.",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| C | `100 * (yes + no) / total` |",
        "| P | `100 * yes / (yes + no)` |",
        "",
        f"Eval model (reference): `{data.get('eval_model', '')}`",
        "",
        "## Summary",
        "",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for name in sorted(models.keys()):
        s = models[name]["summary"]
        lines.append(f"| {name} | {s.get('C','')} | {s.get('P','')} | {s.get('n_images','')} | {s.get('n_error', 0)} |")
    lines.append("")
    lines.append("## Per category (C / P)")
    lines.append("")
    lines.append("| model | category | C | P | n_images |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in sorted(models.keys()):
        for cat, row in (models[name].get("per_category") or {}).items():
            lines.append(
                f"| {name} | {cat} | {row.get('C','')} | {row.get('P','')} | "
                f"{row.get('n_images','')} |"
            )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    write_ranking_json_sidecars(args.output_dir)
    summary_paths = write_results_summaries(args.output_dir, data)
    wrote = f"{summary_csv}, {tag_csv}, {cat_csv}, {md_path}"
    if summary_paths:
        wrote += ", " + ", ".join(summary_paths)
    print(f"[report] wrote {wrote}")


def write_results_summaries(
    output_dir: str, data: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Write ``results_summary.json`` (single model) or per-model sidecars."""
    if data is None:
        mp = os.path.join(output_dir, "metrics.json")
        if not os.path.isfile(mp):
            return []
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
    models = data.get("models") or {}
    if not models:
        return []
    names = sorted(models.keys())
    written: List[str] = []
    multi = len(names) > 1
    for name in names:
        compact = dict(summary_from_metrics(data, model_id=name))
        if multi:
            path = os.path.join(output_dir, f"{safe_model_subdir(name)}_results_summary.json")
        else:
            path = os.path.join(output_dir, "results_summary.json")
        with open(path, "w", encoding="utf-8") as fo:
            fo.write(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
        written.append(path)
    return written


def _filter_jsonl_by_model_id(src: str, dst: str, model_id: str) -> int:
    n = 0
    if not os.path.isfile(src):
        return 0
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fout:
        for row in iter_jsonl(src):
            if row.get("model_id") == model_id:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def write_ranking_json_sidecars(output_dir: str) -> None:
    """Write one ``<model>.json`` summary file per model key."""
    mp = os.path.join(output_dir, "metrics.json")
    if not os.path.isfile(mp):
        return
    with open(mp, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, m in (data.get("models") or {}).items():
        safe = safe_model_subdir(name)
        p = os.path.join(output_dir, f"{safe}.json")
        with open(p, "w", encoding="utf-8") as fo:
            fo.write(
                json.dumps(
                    {
                        "summary": m.get("summary", {}),
                        "per_category": m.get("per_category", {}),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )


def cmd_split_per_model(args: argparse.Namespace) -> None:
    """Under the output directory, create ``<model_id>/`` with only that model's artifacts."""
    out = args.output_dir
    if not out or not os.path.isdir(out):
        raise SystemExit("[split-per-model] --output-dir must be an existing directory.")
    agg = os.path.join(out, "metrics.json")
    if not os.path.isfile(agg):
        raise SystemExit(f"[split-per-model] missing {agg}; run metrics first.")
    with open(agg, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = sorted((data.get("models") or {}).keys())
    if not models:
        print("[split-per-model] no models in metrics.json")
        return
    prepared = os.path.join(out, "prepared.jsonl")
    evaluate = os.path.join(out, "evaluate.jsonl")
    for mid in models:
        sub_name = safe_model_subdir(mid)
        sub = os.path.join(out, sub_name)
        if os.path.isfile(sub):
            raise SystemExit(f"[split-per-model] expected directory but file exists: {sub!r}")
        os.makedirs(sub, exist_ok=True)
        _filter_jsonl_by_model_id(prepared, os.path.join(sub, "prepared.jsonl"), mid)
        _filter_jsonl_by_model_id(evaluate, os.path.join(sub, "evaluate.jsonl"), mid)
        ns = copy.copy(args)
        ns.output_dir = sub
        cmd_metrics(ns)
        cmd_report(ns)
    print(f"[split-per-model] wrote {len(models)} model directories under {out!r}")
