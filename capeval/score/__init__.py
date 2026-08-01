"""CAPEval scoring: prepare + evaluate."""

from capeval.score.evaluate import cmd_evaluate, parse_single_pass_json
from capeval.score.io import (
    EVAL_MODEL_DEFAULT,
    discover_merged_caption_files,
    load_captions_by_image_id,
    load_jsonl,
)
from capeval.score.prepare import cmd_prepare

__all__ = [
    "EVAL_MODEL_DEFAULT",
    "cmd_evaluate",
    "cmd_prepare",
    "discover_merged_caption_files",
    "load_captions_by_image_id",
    "load_jsonl",
    "main",
    "parse_single_pass_json",
]


def __getattr__(name: str):
    if name == "main":
        from capeval.score.cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
