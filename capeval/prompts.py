"""Built-in caption prompts for CAPEval."""

from __future__ import annotations

from typing import Dict

# Default CAPEval caption prompt (+ short alias)
CAPTION_PROMPT_TEXT = "Analyze the image in a comprehensive and detailed manner."

CAPTION_PROMPTS: Dict[str, str] = {
    "PROMPT": CAPTION_PROMPT_TEXT,
    "SIMPLE": CAPTION_PROMPT_TEXT,
}


def get_prompt(prompt_name: str | None = None) -> str:
    """Return a built-in prompt by name (default: PROMPT)."""
    key = (prompt_name or "PROMPT").upper()
    return CAPTION_PROMPTS.get(key, CAPTION_PROMPTS["PROMPT"])


def list_available_prompts() -> None:
    print("Available prompts:")
    print("=================")
    for name, prompt_text in CAPTION_PROMPTS.items():
        first_line = prompt_text.split("\n")[0].strip()
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        print(f"  {name}: {first_line}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List and preview CAPEval caption prompts")
    parser.add_argument("--show-prompt", type=str, default=None)
    args = parser.parse_args()
    if args.show_prompt:
        key = args.show_prompt.upper()
        if key not in CAPTION_PROMPTS:
            raise SystemExit(f"Unknown prompt: {args.show_prompt}")
        print(CAPTION_PROMPTS[key])
    else:
        list_available_prompts()
