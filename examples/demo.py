#!/usr/bin/env python3
"""CAPEval Gradio demo.

  pip install gradio
  python examples/demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_image_choices() -> List[str]:
    img_dir = ROOT / "data" / "image"
    return sorted(
        p.name
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _find_row(evaluate_jsonl: Path, image_id: str) -> Optional[Dict[str, Any]]:
    if not evaluate_jsonl.is_file():
        return None
    want = Path(image_id).name
    with evaluate_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img = str(row.get("img_path") or row.get("image_id") or "")
            if Path(img).name == want:
                return row
    return None


def _format_feedback(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    y = int(row.get("yes1") or 0)
    n = int(row.get("no1") or 0)
    t = int(row.get("total1") or 0)
    c = (y + n) / t if t else 0.0
    p = y / (y + n) if (y + n) else 0.0

    items = row.get("checklist_items") or []
    verdicts = row.get("gt_verdicts") or []
    vmap = {
        v.get("item_index"): v.get("verdict")
        for v in verdicts
        if isinstance(v, dict)
    }

    missing: List[str] = []
    wrong: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = it.get("item_index")
        q = str(it.get("question") or it.get("Question") or "").strip()
        ver = vmap.get(idx, "not_mentioned")
        if ver == "not_mentioned" and q:
            missing.append(f"• {q}")
        elif ver == "no" and q:
            wrong.append(f"• {q}")

    miss_txt = "\n".join(missing[:40]) if missing else "—"
    wrong_txt = "\n".join(wrong[:40]) if wrong else "—"
    if len(missing) > 40:
        miss_txt += f"\n(+{len(missing) - 40} more)"
    if len(wrong) > 40:
        wrong_txt += f"\n(+{len(wrong) - 40} more)"
    return f"{c * 100:.1f}%", f"{p * 100:.1f}%", miss_txt, wrong_txt


def build_app(default_eval: Optional[Path] = None):
    import gradio as gr

    choices = _load_image_choices()

    def run(image_id: str, caption: str, mode: str, eval_jsonl: str):
        if not image_id:
            return None, "—", "—", "Select an image.", "—"
        img_path = ROOT / "data" / "image" / image_id
        preview = str(img_path) if img_path.is_file() else None

        if mode == "Replay results":
            path = Path(eval_jsonl) if eval_jsonl.strip() else default_eval
            if path is None or not Path(path).is_file():
                return preview, "—", "—", "Set evaluate.jsonl to an existing metrics file.", "—"
            row = _find_row(Path(path), image_id)
            if row is None:
                return preview, "—", "—", f"No result for {image_id}.", "—"
            return preview, *_format_feedback(row)

        if not (caption or "").strip():
            return preview, "—", "—", "Enter a caption to score.", "—"

        from capeval import CAPEval

        result = CAPEval(model_id="demo").evaluate(image_id=image_id, caption=caption)
        c = result.get("coverage")
        p = result.get("precision")
        c_pct = f"{float(c) * 100:.1f}%" if c is not None else "—"
        p_pct = f"{float(p) * 100:.1f}%" if p is not None else "—"
        row = _find_row(Path(result["output_dir"]) / "evaluate.jsonl", image_id)
        if row:
            _, _, miss, wrong = _format_feedback(row)
            return preview, c_pct, p_pct, miss, wrong
        return preview, c_pct, p_pct, "—", "—"

    with gr.Blocks(title="CAPEval") as demo:
        gr.Markdown("# CAPEval\nCoverage / Precision checklist feedback.")
        with gr.Row():
            image_id = gr.Dropdown(
                choices=choices,
                label="Image",
                value=choices[0] if choices else None,
            )
            mode = gr.Radio(
                ["Replay results", "Live judge"],
                value="Replay results",
                label="Mode",
            )
        eval_jsonl = gr.Textbox(
            label="evaluate.jsonl (Replay)",
            value=str(default_eval) if default_eval else "",
            placeholder="outputs/<model>/metrics/evaluate.jsonl",
        )
        caption = gr.Textbox(label="Caption (Live judge)", lines=8)
        btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            preview = gr.Image(label="Image", type="filepath")
            with gr.Column():
                cov = gr.Textbox(label="Coverage (C)")
                prec = gr.Textbox(label="Precision (P)")
        missing = gr.Textbox(label="Not mentioned", lines=12)
        wrong = gr.Textbox(label="Incorrect (no)", lines=8)
        btn.click(
            run,
            inputs=[image_id, caption, mode, eval_jsonl],
            outputs=[preview, cov, prec, missing, wrong],
        )
    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--evaluate-jsonl", default="")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    default_eval = Path(args.evaluate_jsonl) if args.evaluate_jsonl else None
    if default_eval is None and (ROOT / "outputs").is_dir():
        cands = sorted((ROOT / "outputs").glob("*/metrics/evaluate.jsonl"))
        default_eval = cands[0] if cands else None

    build_app(default_eval).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
