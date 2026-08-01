"""Caption model aliases + passthrough for arbitrary HF / local model ids.

- Built-in short names (``qwen8b``, ``internvl1b``, …) map to common Hub checkpoints.
- Any other token is treated as a Hugging Face id or local path and still runs
  through the caption launcher (backend inferred heuristically).
- Caption launcher requires an explicit target (no default model).
- Scoring does **not** require an alias: any caption JSON keyed by ``img_path``
  can be judged via ``score.py`` / ``capeval.api.score_caption_map``.
"""

from __future__ import annotations

from dataclasses import dataclass

from capeval.util.names import model_safe


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    backend: str = "vllm"  # vllm | transformers
    temperature: float = 0.7
    large_vlm: bool = False
    glm_flash_nx1: bool = False
    is_glm: bool = False


DEFAULT_ALIAS = "internvl1b"

# Convenience groups (never the implicit default).
META: dict[str, list[str]] = {
    "quick": ["internvl1b", "qwen8b", "llava8b"],
    "qwen": ["qwen8b"],
    "qwen25vl": ["qwen25vl7b"],
    "qwen3vl": ["qwen8b"],
    "internvl": ["internvl1b"],
    "llava": ["llava8b"],
    "glm": ["glm46vflash"],
    "minicpm": ["minicpmv46"],
    "ovis": ["ovis2_8b"],
    "phi": ["phi35vision"],
    "molmo": ["molmo7b"],
}


def _vllm(alias: str, model_id: str, *, large: bool = False, temp: float = 0.7) -> ModelSpec:
    return ModelSpec(alias, model_id, backend="vllm", temperature=temp, large_vlm=large)


def _tf(
    alias: str,
    model_id: str,
    *,
    large: bool = False,
    temp: float = 0.0,
    is_glm: bool = False,
    glm_flash_nx1: bool = False,
) -> ModelSpec:
    return ModelSpec(
        alias,
        model_id,
        backend="transformers",
        temperature=temp,
        large_vlm=large,
        is_glm=is_glm,
        glm_flash_nx1=glm_flash_nx1,
    )


_SPECS: dict[str, ModelSpec] = {
    # ── Qwen3-VL ─────────────────────────────────────────────────────────
    "qwen2b": _vllm("qwen2b", "Qwen/Qwen3-VL-2B-Instruct"),
    "qwen4b": _vllm("qwen4b", "Qwen/Qwen3-VL-4B-Instruct"),
    "qwen8b": _vllm("qwen8b", "Qwen/Qwen3-VL-8B-Instruct"),
    "qwen32b": _vllm("qwen32b", "Qwen/Qwen3-VL-32B-Instruct", large=True),
    # ── Qwen2.5-VL ───────────────────────────────────────────────────────
    "qwen25vl3b": _vllm("qwen25vl3b", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "qwen25vl7b": _vllm("qwen25vl7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwen25vl32b": _vllm("qwen25vl32b", "Qwen/Qwen2.5-VL-32B-Instruct", large=True),
    "qwen25vl72b": _vllm("qwen25vl72b", "Qwen/Qwen2.5-VL-72B-Instruct", large=True),
    # ── Qwen2-VL ─────────────────────────────────────────────────────────
    "qwen2vl2b": _vllm("qwen2vl2b", "Qwen/Qwen2-VL-2B-Instruct"),
    "qwen2vl7b": _vllm("qwen2vl7b", "Qwen/Qwen2-VL-7B-Instruct"),
    "qwen2vl72b": _vllm("qwen2vl72b", "Qwen/Qwen2-VL-72B-Instruct", large=True),
    # ── InternVL 3.5 / 3 / 2.5 ───────────────────────────────────────────
    "internvl1b": _vllm("internvl1b", "OpenGVLab/InternVL3_5-1B"),
    "internvl4b": _vllm("internvl4b", "OpenGVLab/InternVL3_5-4B"),
    "internvl8b": _vllm("internvl8b", "OpenGVLab/InternVL3_5-8B"),
    "internvl38b": _vllm("internvl38b", "OpenGVLab/InternVL3_5-38B", large=True),
    "internvl3_8b": _vllm("internvl3_8b", "OpenGVLab/InternVL3-8B"),
    "internvl3_14b": _vllm("internvl3_14b", "OpenGVLab/InternVL3-14B", large=True),
    "internvl3_38b": _vllm("internvl3_38b", "OpenGVLab/InternVL3-38B", large=True),
    "internvl3_78b": _vllm("internvl3_78b", "OpenGVLab/InternVL3-78B", large=True),
    "internvl25_8b": _vllm("internvl25_8b", "OpenGVLab/InternVL2_5-8B"),
    "internvl25_26b": _vllm("internvl25_26b", "OpenGVLab/InternVL2_5-26B", large=True),
    "internvl25_38b": _vllm("internvl25_38b", "OpenGVLab/InternVL2_5-38B", large=True),
    "internvl25_78b": _vllm("internvl25_78b", "OpenGVLab/InternVL2_5-78B", large=True),
    "internvl2_8b": _vllm("internvl2_8b", "OpenGVLab/InternVL2-8B"),
    "internvl2_26b": _vllm("internvl2_26b", "OpenGVLab/InternVL2-26B", large=True),
    "internvl2_40b": _vllm("internvl2_40b", "OpenGVLab/InternVL2-40B", large=True),
    # ── LLaVA / OneVision ────────────────────────────────────────────────
    "llava4b": _tf("llava4b", "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"),
    "llava8b": _tf("llava8b", "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"),
    "llavaov7b": _tf("llavaov7b", "lmms-lab/llava-onevision-qwen2-7b-ov"),
    "llavaov72b": _tf("llavaov72b", "lmms-lab/llava-onevision-qwen2-72b-ov", large=True),
    "llava16_7b": _tf("llava16_7b", "liuhaotian/llava-v1.6-vicuna-7b"),
    "llava16_13b": _tf("llava16_13b", "liuhaotian/llava-v1.6-vicuna-13b", large=True),
    # ── GLM-4V ───────────────────────────────────────────────────────────
    "glm46vflash": _tf(
        "glm46vflash",
        "zai-org/GLM-4.6V-Flash",
        is_glm=True,
        glm_flash_nx1=True,
    ),
    "glm46v": _tf("glm46v", "zai-org/GLM-4.6V", large=True, is_glm=True),
    "glm4v9b": _vllm("glm4v9b", "THUDM/glm-4v-9b"),
    # ── MiniCPM-V / o ────────────────────────────────────────────────────
    "minicpmv26": _vllm("minicpmv26", "openbmb/MiniCPM-V-2_6"),
    "minicpmv46": _vllm("minicpmv46", "openbmb/MiniCPM-V-4_5"),
    "minicpmo26": _tf("minicpmo26", "openbmb/MiniCPM-o-2_6"),
    # ── Ovis ─────────────────────────────────────────────────────────────
    "ovis2_4b": _vllm("ovis2_4b", "AIDC-AI/Ovis2-4B"),
    "ovis2_8b": _vllm("ovis2_8b", "AIDC-AI/Ovis2-8B"),
    "ovis2_16b": _vllm("ovis2_16b", "AIDC-AI/Ovis2-16B", large=True),
    "ovis2_34b": _vllm("ovis2_34b", "AIDC-AI/Ovis2-34B", large=True),
    # ── Phi / Microsoft ──────────────────────────────────────────────────
    "phi35vision": _tf("phi35vision", "microsoft/Phi-3.5-vision-instruct"),
    "phi4mm": _tf("phi4mm", "microsoft/Phi-4-multimodal-instruct", large=True),
    # ── Molmo (Ai2) ──────────────────────────────────────────────────────
    "molmo7b": _tf("molmo7b", "allenai/Molmo-7B-D-0924"),
    "molmo72b": _tf("molmo72b", "allenai/Molmo-72B-0924", large=True),
    # ── SmolVLM / Idefics ────────────────────────────────────────────────
    "smolvlm": _tf("smolvlm", "HuggingFaceTB/SmolVLM-Instruct"),
    "smolvlm2": _tf("smolvlm2", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    "idefics2_8b": _tf("idefics2_8b", "HuggingFaceM4/idefics2-8b"),
    "idefics3_8b": _tf("idefics3_8b", "HuggingFaceM4/Idefics3-8B-Llama3"),
    # ── DeepSeek / Aria / others ─────────────────────────────────────────
    "deepseekvl2_tiny": _tf("deepseekvl2_tiny", "deepseek-ai/deepseek-vl2-tiny"),
    "deepseekvl2_small": _tf("deepseekvl2_small", "deepseek-ai/deepseek-vl2-small"),
    "deepseekvl2": _tf("deepseekvl2", "deepseek-ai/deepseek-vl2", large=True),
    "aria": _tf("aria", "rhymes-ai/Aria", large=True),
    "pixtral12b": _vllm("pixtral12b", "mistralai/Pixtral-12B-2409", large=True),
    "gemma3_4b": _tf("gemma3_4b", "google/gemma-3-4b-it"),
    "gemma3_12b": _tf("gemma3_12b", "google/gemma-3-12b-it", large=True),
    "gemma3_27b": _tf("gemma3_27b", "google/gemma-3-27b-it", large=True),
    "llama32_11b_vision": _tf(
        "llama32_11b_vision", "meta-llama/Llama-3.2-11B-Vision-Instruct", large=True
    ),
    "llama32_90b_vision": _tf(
        "llama32_90b_vision", "meta-llama/Llama-3.2-90B-Vision-Instruct", large=True
    ),
    "kimi_vl_a3b": _tf("kimi_vl_a3b", "moonshotai/Kimi-VL-A3B-Instruct"),
}


def expand_targets(targets: list[str]) -> list[str]:
    """Expand group aliases; requires at least one explicit target (no default model)."""
    if not targets:
        import sys

        print(
            "No caption model specified.\n"
            "Pass an alias, group, Hugging Face id, or local path, e.g.\n"
            "  python caption.py internvl1b\n"
            "  python caption.py Qwen/Qwen2.5-VL-7B-Instruct\n"
            "  python caption.py --list-models",
            file=sys.stderr,
        )
        raise SystemExit(2)
    out: list[str] = []
    for t in targets:
        if t in META:
            out.extend(META[t])
        else:
            out.append(t)
    return out


def _looks_like_model_id(name: str) -> bool:
    """HF Hub id or local path — not a bare typo that should fail as unknown alias."""
    if name in _SPECS or name in META:
        return False
    if "/" in name or name.startswith(".") or name.startswith("/"):
        return True
    from pathlib import Path

    # Existing local relative path (e.g. ./ckpt or my_model/)
    if Path(name).exists():
        return True
    return False


def _infer_custom_spec(model_ref: str) -> ModelSpec:
    mid = model_ref.strip()
    low = mid.lower()
    alias = model_safe(mid)

    large = any(
        tok in low
        for tok in (
            "32b",
            "34b",
            "38b",
            "40b",
            "70b",
            "72b",
            "78b",
            "90b",
            "110b",
            "-27b",
            "26b",
        )
    )
    is_glm = ("glm" in low and ("vl" in low or "4v" in low)) or "glm-4v" in low
    glm_flash = is_glm and "flash" in low

    # Prefer transformers for families that are awkward on vLLM / need special processors.
    tf_markers = (
        "llava",
        "molmo",
        "smolvlm",
        "idefics",
        "phi-3",
        "phi-4",
        "phi4",
        "deepseek-vl",
        "deepseek_vl",
        "aria",
        "gemma-3",
        "gemma3",
        "llama-3.2",
        "llama3.2",
        "kimi-vl",
        "minicpm-o",
    )
    use_tf = any(m in low for m in tf_markers) or glm_flash
    # Explicit user override via env is handled in caption launcher; default here:
    backend = "transformers" if use_tf else "vllm"
    temp = 0.0 if backend == "transformers" else 0.7

    return ModelSpec(
        alias=alias,
        model_id=mid,
        backend=backend,
        temperature=temp,
        large_vlm=large,
        is_glm=is_glm,
        glm_flash_nx1=glm_flash,
    )


def resolve_spec(alias: str) -> ModelSpec:
    if alias in _SPECS:
        return _SPECS[alias]
    if _looks_like_model_id(alias):
        return _infer_custom_spec(alias)
    raise SystemExit(
        "Unknown target: "
        f"{alias}\n"
        f"Built-in aliases: {' '.join(sorted(_SPECS))}\n"
        f"Groups: {' '.join(sorted(META))}\n"
        f"Or pass a Hugging Face id / local path, e.g. org/model-name\n"
        f"See: python caption.py --list-models"
    )


def list_aliases() -> list[str]:
    return sorted(_SPECS)


def format_model_table() -> str:
    """Human-readable alias table for --list-models / docs."""
    lines = [
        f"{'alias':<22} {'backend':<14} {'model_id'}",
        "-" * 88,
    ]
    for a in list_aliases():
        s = _SPECS[a]
        lines.append(f"{s.alias:<22} {s.backend:<14} {s.model_id}")
    lines.append("")
    lines.append("Groups: " + ", ".join(f"{k}→{v}" for k, v in sorted(META.items())))
    lines.append(
        "Custom: pass any Hugging Face id or local checkpoint path, e.g. "
        "`python caption.py Qwen/Qwen2.5-VL-7B-Instruct`"
    )
    return "\n".join(lines)
