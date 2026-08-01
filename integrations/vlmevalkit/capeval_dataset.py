"""VLMEvalKit dataset adapter for CAPEval.

Installation (pick one):
  1. Copy this file into ``VLMEvalKit/vlmeval/dataset/capeval.py`` and register it
     in ``vlmeval/dataset/__init__.py`` (add to ``IMAGE_DATASET`` / import).
  2. Or keep it here and set ``PYTHONPATH`` so VLMEvalKit can import
     ``integrations.vlmevalkit.capeval_dataset`` (advanced).

Environment:
  CAPEVAL_HOME   — path to the CAPEval repo (default: inferred from this file)
  EVAL_MODEL     — judge LLM (default Qwen/Qwen2.5-72B-Instruct)
  CUDA_VISIBLE_DEVICES — GPUs for the judge stage inside ``evaluate``

Inference uses CAPEval's caption prompt only; checklist scoring runs in ``evaluate``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── locate CAPEval repo ──────────────────────────────────────────────────

_HERE = Path(__file__).resolve()
_CAPEVAL_CANDIDATES = [
    Path(os.environ.get("CAPEVAL_HOME", "")),
    _HERE.parents[2],  # integrations/vlmevalkit -> repo root
    _HERE.parents[1],
]
CAPEVAL_HOME = next((p for p in _CAPEVAL_CANDIDATES if p and (p / "capeval" / "api.py").is_file()), None)
if CAPEVAL_HOME is None:
    raise ImportError(
        "Cannot find CAPEval repo. Set CAPEVAL_HOME to the CAPEval root "
        "(directory that contains capeval/api.py)."
    )
if str(CAPEVAL_HOME) not in sys.path:
    sys.path.insert(0, str(CAPEVAL_HOME))

from capeval.api import caption_prompt, default_paths, list_inference_rows, score_caption_map  # noqa: E402

# Optional VLMEvalKit imports — file still documents the contract if absent.
try:
    from vlmeval.dataset.image_base import ImageBaseDataset
    from vlmeval.smp import dump, load
except ImportError:  # pragma: no cover - template mode
    ImageBaseDataset = object  # type: ignore
    load = None  # type: ignore
    dump = None  # type: ignore


class CAPEvalDataset(ImageBaseDataset):
    """Caption → checklist judge benchmark.

    Dataset name for ``run.py --data``: ``CAPEval``.
    """

    TYPE = "VQA"
    DATASET_URL = {"CAPEval": ""}
    DATASET_MD5 = {"CAPEval": ""}

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["CAPEval"]

    def load_data(self, dataset: str = "CAPEval"):
        """Build a DataFrame-like list from local CAPEval data (no TSV download)."""
        import pandas as pd

        rows = list_inference_rows()
        # VLMEvalKit expects columns: index, image, question, …
        df = pd.DataFrame(rows)
        # Prefer basename in `image` for dump_image compatibility when using paths
        return df

    def build_prompt(self, line) -> List[Dict[str, Any]]:
        if isinstance(line, int):
            line = self.data.iloc[line]
        img = line["image"] if "image" in line else line["img_path"]
        # Resolve relative paths against CAPEval image root
        img_path = str(img)
        if not os.path.isabs(img_path):
            img_path = str(default_paths()["image_root"] / Path(img_path).name)
        question = line["question"] if "question" in line else caption_prompt()
        return [
            dict(type="image", value=img_path),
            dict(type="text", value=str(question)),
        ]

    def evaluate(self, eval_file: str, **judge_kwargs) -> Any:
        """Score model captions with CAPEval checklist judge.

        ``eval_file`` is VLMEvalKit's ``{model}_{dataset}.xlsx`` (or TSV/JSON)
        with columns including ``prediction`` and ``image_id`` or ``image``.
        """
        if load is None:
            raise ImportError("vlmeval is required to call CAPEvalDataset.evaluate")

        data = load(eval_file)
        # pandas DataFrame
        captions: Dict[str, str] = {}
        for _, row in data.iterrows():
            pred = row.get("prediction", "")
            if pred is None:
                continue
            key = row.get("image_id") or row.get("img_path")
            if key is None or (isinstance(key, float) and str(key) == "nan"):
                img = row.get("image", "")
                key = Path(str(img)).name if img is not None else None
            if not key:
                continue
            captions[str(Path(str(key)).name)] = str(pred)

        model_id = judge_kwargs.get("model") or Path(eval_file).stem
        # Strip dataset suffix if present: ModelName_CAPEval -> ModelName
        if str(model_id).endswith("_CAPEval"):
            model_id = str(model_id)[: -len("_CAPEval")]

        out_dir = judge_kwargs.get("capeval_output_dir")
        if not out_dir:
            from capeval.util.paths import model_metrics_dir

            out_dir = str(model_metrics_dir(str(model_id), root=Path(CAPEVAL_HOME) / "outputs"))
        result = score_caption_map(
            captions,
            model_id=str(model_id),
            output_dir=out_dir,
            eval_model=judge_kwargs.get("eval_model") or os.environ.get("EVAL_MODEL"),
            tp_size=judge_kwargs.get("tp_size"),
        )
        summary = result.get("summary") or {}
        import pandas as pd

        ret = pd.DataFrame(
            [
                {
                    "C": summary.get("C"),
                    "P": summary.get("P"),
                }
            ]
        )
        suffix = eval_file.split(".")[-1]
        result_file = eval_file.replace(f".{suffix}", "_capeval_score.csv")
        if dump is not None:
            dump(ret, result_file)
        return ret


# Registration snippet for VLMEvalKit maintainers:
#   from vlmeval.dataset.capeval import CAPEvalDataset
#   IMAGE_DATASET.append(CAPEvalDataset)
