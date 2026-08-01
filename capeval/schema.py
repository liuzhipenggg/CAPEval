"""Checklist JSONL row schema shared by CAPEval prepare (evaluate)."""

from __future__ import annotations

from typing import Any, Dict, List

CHECKLIST_KEYS = [
    "instance_checklist",
    "attribute_checklist",
    "relation_checklist",
    "image_checklist",
    "text_checklist",
    "human_checklist",
    "ui_checklist",
    "world_knowledge_checklist",
]


def flatten_items(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return flat list of {checklist_type, tags, question} for one image row."""
    items: List[Dict[str, Any]] = []
    for ck in CHECKLIST_KEYS:
        for item in entry.get(ck, []) or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get("Question", "")).strip()
            if q:
                items.append(
                    {
                        "checklist_type": ck,
                        "tags": item.get("Tags", ""),
                        "question": q,
                    }
                )
    return items
