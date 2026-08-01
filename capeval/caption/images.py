"""Image listing, resize, and decode helpers for captioning."""
from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from PIL import Image

from capeval.io import encode_image
from capeval.llm import _downscale_pil_max_edge

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def _decode_image_paths_rgb_parallel(
    paths: List[str], workers: int, max_image_edge: int = 0
) -> List[Any]:
    """Decode image paths to PIL RGB; use threads to overlap I/O and libjpeg decode."""

    def _one(p: str) -> Any:
        im = Image.open(p).convert("RGB")
        return _downscale_pil_max_edge(im, max_image_edge) if max_image_edge > 0 else im

    if len(paths) <= 1:
        return [_one(p) for p in paths]
    w = workers if workers > 0 else min(16, len(paths))
    if w <= 1:
        return [_one(p) for p in paths]
    with ThreadPoolExecutor(max_workers=min(w, len(paths))) as ex:
        return list(ex.map(_one, paths))



def resize_image_for_api(
    img_path: str,
    *,
    max_base64_bytes: int = 5 * 1024 * 1024,
    min_side: int = 512,
    max_side: int = 2048,
    max_attempts: int = 8
) -> str:
    """
    Return base64-encoded JPEG image data under a size limit.
    Primarily used for Claude, which has strict payload size limits.
    """
    img = Image.open(img_path)
    img = img.convert("RGB")

    def _encode_jpeg(pil_img: Image.Image, quality: int) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    w, h = img.size
    scale = 1.0
    quality = 85

    for _ in range(max_attempts):
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        if min(new_w, new_h) < min_side:
            # Don't shrink below minimum detail; rely on quality reduction instead.
            new_w, new_h = (w, h)

        if max(new_w, new_h) > max_side:
            ratio = max_side / float(max(new_w, new_h))
            new_w = max(1, int(new_w * ratio))
            new_h = max(1, int(new_h * ratio))

        resized = img if (new_w, new_h) == img.size else img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        jpeg_bytes = _encode_jpeg(resized, quality=quality)
        b64 = base64.b64encode(jpeg_bytes)
        if len(b64) <= max_base64_bytes:
            return b64.decode("utf-8")

        # Try shrinking and reducing quality.
        scale *= 0.85
        quality = max(40, int(quality * 0.85))

    # Last resort: return whatever we have (may still be too large).
    jpeg_bytes = _encode_jpeg(img, quality=40)
    return base64.b64encode(jpeg_bytes).decode("utf-8")


def _is_image_path(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in _IMAGE_EXTS


def list_images_in_dir(input_dir: str) -> List[str]:
    """Recursively list image files under input_dir."""
    files: List[str] = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            full = os.path.join(root, name)
            if os.path.isfile(full) and _is_image_path(full):
                files.append(full)
    files.sort()
    return files


def _local_image_key(img_path: str, input_dir: str) -> str:
    """Stable key for local images: path relative to --input-dir."""
    rel = os.path.relpath(img_path, input_dir)
    return rel.replace(os.sep, "/")

