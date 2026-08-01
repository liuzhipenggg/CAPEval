"""CAPEval caption engine package.

Public entry: ``main`` (``python -m capeval.caption``).
"""
from __future__ import annotations

__all__ = [
    "caption_images",
    "caption_images_single_pass",
    "detect_model_backend",
    "generate_caption",
    "list_images_in_dir",
    "main",
    "resize_image_for_api",
]


def __getattr__(name: str):
    if name == "main":
        from capeval.caption.cli import main

        return main
    if name in {
        "caption_images",
        "caption_images_single_pass",
        "detect_model_backend",
        "generate_caption",
    }:
        from capeval.caption import engine

        return getattr(engine, name)
    if name in {"list_images_in_dir", "resize_image_for_api"}:
        from capeval.caption import images

        return getattr(images, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
