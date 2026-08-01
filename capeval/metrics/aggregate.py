"""Aggregate evaluate.jsonl into metrics.json (C / P)."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from capeval.score.io import iter_jsonl
from capeval.tags import split_raw_tag_field

# Filename prefix → super-category display name
CATEGORY_NAMES = {
    "SO": "Scene & Object",
    "PA": "People & Activity",
    "TI": "Text & Interface",
    "DK": "Design & Knowledge",
}


def _cp_percent(yes: int, no: int, total: int) -> Tuple[float, float]:
    """Return Coverage / Precision on a 0–100 scale."""
    c = 100.0 * (yes + no) / total if total else 0.0
    mentioned = yes + no
    p = 100.0 * yes / mentioned if mentioned else 0.0
    return c, p


def _category_name(domain: Optional[str]) -> str:
    d = (domain or "unknown").strip().upper()
    return CATEGORY_NAMES.get(d, domain or "unknown")


def _empty_bucket() -> dict:
    return {
        "yes1": 0,
        "no1": 0,
        "not_mentioned1": 0,
        "total1": 0,
        "n_images": 0,
        "n_error": 0,
    }


def _summary_block(bucket: dict) -> Dict[str, Any]:
    sy, sn, st = bucket["yes1"], bucket["no1"], bucket["total1"]
    c, p = _cp_percent(sy, sn, st)
    out: Dict[str, Any] = {
        "C": round(c, 4),
        "P": round(p, 4),
        "yes1": sy,
        "no1": sn,
        "not_mentioned1": bucket["not_mentioned1"],
        "total1": st,
        "n_images": bucket["n_images"],
    }
    if bucket.get("n_error"):
        out["n_error"] = int(bucket["n_error"])
    return out


def cmd_metrics(args: argparse.Namespace) -> None:
    """Compute CAPEval metrics: C / P (0–100) overall and per super-category."""
    eval_path = os.path.join(args.output_dir, "evaluate.jsonl")
    out_path = os.path.join(args.output_dir, "metrics.json")
    if not os.path.isfile(eval_path):
        raise SystemExit(f"Missing {eval_path}")

    by_model: Dict[str, dict] = defaultdict(
        lambda: {
            **_empty_bucket(),
            "tag_yes": defaultdict(int),
            "tag_total": defaultdict(int),
            "by_category": defaultdict(_empty_bucket),
        }
    )

    for row in iter_jsonl(eval_path):
        m = row.get("model_id")
        if not m:
            continue
        if row.get("status") != "ok":
            by_model[m]["n_error"] += 1
            continue
        g = by_model[m]
        y1 = int(row.get("yes1", 0) or 0)
        n1 = int(row.get("no1", 0) or 0)
        nm1 = int(row.get("not_mentioned1", 0) or 0)
        t1 = int(row.get("total1", 0) or 0)
        g["yes1"] += y1
        g["no1"] += n1
        g["not_mentioned1"] += nm1
        g["total1"] += t1
        g["n_images"] += 1

        cat = _category_name(row.get("domain"))
        bc = g["by_category"][cat]
        bc["yes1"] += y1
        bc["no1"] += n1
        bc["not_mentioned1"] += nm1
        bc["total1"] += t1
        bc["n_images"] += 1

        items = row.get("checklist_items") or []
        verdicts = row.get("gt_verdicts") or []
        vmap = {v["item_index"]: v.get("verdict") for v in verdicts if isinstance(v, dict)}

        for it in items:
            idx = it.get("item_index")
            ver = vmap.get(idx, "not_mentioned")
            is_yes = 1 if ver == "yes" else 0
            tags = split_raw_tag_field(str(it.get("tags", "") or ""))
            if not tags:
                tags = ["_untagged_"]
            for tg in tags:
                g["tag_yes"][tg] += is_yes
                g["tag_total"][tg] += 1

    metrics_models: Dict[str, Any] = {}
    for m, g in by_model.items():
        per_tag_out = {
            tg: {
                "yes1": g["tag_yes"][tg],
                "total1": g["tag_total"][tg],
            }
            for tg in sorted(g["tag_total"].keys())
        }
        # Stable category order
        order = list(CATEGORY_NAMES.values())
        cats = sorted(
            g["by_category"].keys(),
            key=lambda x: (order.index(x) if x in order else 99, x),
        )
        per_category = {name: _summary_block(g["by_category"][name]) for name in cats}

        metrics_models[m] = {
            "summary": _summary_block(g),
            "per_category": per_category,
            "per_tag": per_tag_out,
        }

    payload = {
        "eval_model": args.eval_model,
        "scale": "percent",
        "formulas": {
            "C": "100 * (yes + no) / total",
            "P": "100 * yes / (yes + no)",
        },
        "models": metrics_models,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[metrics] wrote {out_path}")
