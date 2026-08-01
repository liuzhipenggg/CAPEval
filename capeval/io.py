"""Small I/O helpers for captioning."""

from __future__ import annotations

import base64
import json
from io import BytesIO


def load_json(json_file):
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            byte_data = image_file.read()
    else:
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
    return base64.b64encode(byte_data).decode("utf-8")
