"""Qwen-VL / LLaVA-style Transformers backends."""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

from capeval.llm.common import (
    OptionalDependencyError,
    _downscale_pil_max_edge,
    _env_capeval,
    _open_pil_rgb,
    _transformers,
)

@dataclass
class AMDQwenVLClient:
    model: Any
    processor: Any
    device: str


def AMD_qwenvl_client(
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    *,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AMDQwenVLClient:
    """Create a local Transformers client for Qwen-VL models.

    Works on CPU, CUDA, or Apple MPS depending on availability.
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq  # type: ignore
    except Exception as e:
        raise OptionalDependencyError(
            f"transformers/torch are unavailable ({type(e).__name__}: {e}). "
            "pip install 'transformers>=4.48,<5' 'huggingface-hub>=0.34,<1' torch. "
            "If hub>=1 is installed with transformers 4.x, import fails."
        ) from e

    # pick device
    chosen_device = device
    if chosen_device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            chosen_device = "mps"
        elif torch.cuda.is_available():  # type: ignore
            chosen_device = "cuda"
        else:
            chosen_device = "cpu"

    processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
    # Decoder-only VLMs: batched processor(..., padding=True) must left-pad so generation
    # aligns on the last real token (Transformers warns on right-padding + generate).
    _pt_tok = getattr(processor, "tokenizer", None)
    if _pt_tok is not None and hasattr(_pt_tok, "padding_side"):
        _pt_tok.padding_side = "left"
    # Use low memory dtype by default on CPU/MPS
    torch_dtype = "auto"
    try:
        import torch  # type: ignore
        if chosen_device in {"cpu", "mps"}:
            torch_dtype = torch.float16 if chosen_device == "mps" else torch.float32
    except Exception:
        pass

    try:
        import accelerate  # noqa: F401
        _has_accelerate = True
    except ImportError:
        _has_accelerate = False

    # device_map requires accelerate. For ``cuda:N``, prefer ``device_map=cuda:N`` instead of
    # ``device_map=None`` + ``.to(cuda:N)``: LLaVA-OneVision-1.5 (bf16) can degenerate to all
    # token id 0 / ``!`` spam on the latter path (verified vs ``device_map=\"cuda:0\"``).
    _cd = str(chosen_device or "")
    use_device_map = _has_accelerate and (_cd == "cuda" or _cd.startswith("cuda:"))
    load_kw = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if use_device_map:
        load_kw["device_map"] = _cd if _cd.startswith("cuda:") else "auto"

    ml = (model or "").lower()
    # lmms-lab/LLaVA-OneVision-1.5-* registers only on AutoModelForCausalLM, not Vision2Seq
    if "onevision-1.5" in ml or "onevision_1.5" in ml.replace("-", "_"):
        model_obj = AutoModelForCausalLM.from_pretrained(model, **load_kw)
    else:
        model_obj = AutoModelForVision2Seq.from_pretrained(model, **load_kw)

    if not use_device_map:
        model_obj.to(chosen_device)
    model_obj.eval()
    return AMDQwenVLClient(model=model_obj, processor=processor, device=chosen_device)


_RE_HF_PIPE_TOKEN = re.compile(r"<\|[^>\n]*?\|>")
# Legacy name kept for any external imports (prefer ``_RE_HF_PIPE_TOKEN``).
_RE_QWENVL_PIPE_TAG = _RE_HF_PIPE_TOKEN


def _qwenvl_strip_hf_pipe_fragments(text: str) -> str:
    """Remove HF chat/control spans ``<|...|>`` and truncated ``<|`` prefixes.

    LLaVA + chat templates often emit ``<|im_end|>`` or cut mid-token (``<|pattern``).
    A buggy ``<\\|[^>]*\\|>``-style regex does **not** match real ``<|name|>`` closers (``|>``),
    leaving literal ``<|`` in captions — fix is ``[^>\\n]*?`` then ``|>`` (see ``_RE_HF_PIPE_TOKEN``).
    """
    if not text:
        return text
    t = _RE_HF_PIPE_TOKEN.sub("", text)
    t = re.sub(r"^(?:<\|)+", "", t)
    return t.strip()


def _qwenvl_llava_refusal_placeholder() -> str:
    if ("CAPEVAL_LLAVA_REFUSAL_PLACEHOLDER" in os.environ or "CAPTIONQA_LLAVA_REFUSAL_PLACEHOLDER" in os.environ):
        return (_env_capeval("LLAVA_REFUSAL_PLACEHOLDER") or "").strip()
    return "[caption_refused]"


def _qwenvl_decode_generated_ids(tokenizer: Any, gen_ids: Any, *, llava_like: bool) -> str:
    """Decode **new** tokens from ``model.generate``.

    LLaVA checkpoints may emit only control tokens such as ``<|im_end|>`` for
    policy-filtered inputs; ``skip_special_tokens=True`` then yields an empty string
    even though generation ran. Strip ``<|...|>`` fragments and optionally substitute
    a placeholder so caption jobs can persist a stable string (see env below).

    ``CAPEVAL_LLAVA_REFUSAL_PLACEHOLDER``: if **unset**, refusal-only generations use
    ``[caption_refused]``. If set to empty, keep strict behaviour (caller may raise).
    """
    raw_skip = tokenizer.decode(gen_ids, skip_special_tokens=True)
    raw_keep = tokenizer.decode(gen_ids, skip_special_tokens=False)
    without_tags = _RE_HF_PIPE_TOKEN.sub("", raw_keep or "")
    if (raw_skip or "").strip():
        out = raw_skip
    elif (without_tags or "").strip():
        out = (without_tags or "").strip()
    else:
        out = raw_skip
    out = _qwenvl_strip_hf_pipe_fragments(out)
    if (out or "").strip():
        return out
    if not llava_like:
        return raw_skip
    _n = int(gen_ids.numel()) if hasattr(gen_ids, "numel") else len(gen_ids)
    if _n == 0:
        return raw_skip
    if ("CAPEVAL_LLAVA_REFUSAL_PLACEHOLDER" in os.environ or "CAPTIONQA_LLAVA_REFUSAL_PLACEHOLDER" in os.environ):
        refusal_ph = (_env_capeval("LLAVA_REFUSAL_PLACEHOLDER") or "").strip()
        return refusal_ph
    return _qwenvl_llava_refusal_placeholder()




def _qwenvl_try_qwen_chat_assistant_suffix_from_full_seq(seq_row: Any, *, tok: Any) -> Optional[str]:
    """Decode full ``generate`` row and take text after the last Qwen-style ``<|im_start|>assistant`` span.

    HF model cards often trim with ``out_ids[len(in_ids):]`` then ``batch_decode``. For some VLMs the
    trimmed span can still decode to junk (single letters / ``<``) while the **full** sequence decodes
    cleanly after the assistant marker — so use this as the primary path for matching checkpoints when
    the marker is present.
    """
    try:
        # ``skip_special_tokens=True`` strips Qwen chat markers like ``<|im_start|>assistant``, breaking split.
        full_t = tok.decode(seq_row, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        full_t = tok.decode(seq_row, skip_special_tokens=False)
    marker = "<|im_start|>assistant"
    if marker not in full_t:
        return None
    cand = full_t.split(marker)[-1].strip()
    if "<|im_start|>" in cand:
        cand = cand.split("<|im_start|>", 1)[0].strip()
    cand = _qwenvl_strip_hf_pipe_fragments(cand)
    cand = _qwenvl_clean_assistant_text(cand).strip()
    return cand or None


def _qwenvl_clean_assistant_text(text: str) -> str:
    """Strip common chat template leader so mostly assistant reply remains."""
    cleaned = text
    for marker in (
        "<|im_start|>assistant",
        "<|assistant|>",
        "\nassistant\n",
        "\nassistant:",
        "assistant\n",
        "assistant:",
    ):
        idx = cleaned.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker) :].lstrip()
            break
    return cleaned.strip()


def _qwenvl_normalize_greedy_sampling_kwargs(gen_kw: Dict[str, Any]) -> None:
    """Align greedy ``generate`` kwargs with transformers ``GenerationConfig.validate`` (>=5.x).

    Pretrained configs often set ``temperature`` / ``top_p`` from ``generation_config.json``.
    For ``do_sample=False``, newer Transformers treat ``temperature`` as invalid (warns and may
    ignore); omit sampling kwargs so greedy merges do not carry stale values.
    """
    if gen_kw.get("do_sample") is not False:
        return
    for _k in ("temperature", "top_p", "top_k", "typical_p"):
        gen_kw.pop(_k, None)


def _qwenvl_apply_sampling_to_gen_kw(
    client: AMDQwenVLClient,
    gen_kw: Dict[str, Any],
    *,
    temperature: Optional[float],
) -> str:
    """Pick ``do_sample`` / ``temperature`` / ``repetition_penalty`` before ``model.generate``.

    ``lmms-lab/LLaVA-OneVision-1.5-*`` Hugging Face checkpoints ship a ``generation_config`` with
    ``do_sample=True`` and ``temperature`` on the order of ``1e-6`` (near-deterministic sampling).
    The previous behaviour forced ``do_sample=False`` whenever callers passed ``temperature<=0``,
    which diverges from that checkpoint and — on local greedy runs — consistently produced
    ultra-short assistant spans ending on ``<|im_end|>`` (tokenizer ``eos_token_id``).

    Default now: for OV1.5 with ``temperature<=0`` (or ``None``), copy those fields from
    ``model.generation_config``. Set ``CAPEVAL_OV15_GREEDY=1`` to restore strict greedy decoding.
    """
    ov15_ck = _qwenvl_llava_onevision_15_hf_checkpoint(client)
    t = float(temperature) if temperature is not None else 0.0

    if ov15_ck and _env_capeval("OV15_GREEDY", "").strip() == "1":
        if temperature is None or t <= 0:
            gen_kw["do_sample"] = False
        else:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = t
        return "ov15_forced_greedy"
    if ov15_ck and temperature is not None and t > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = t
        return "ov15_user_temperature"
    if ov15_ck:
        _gc = getattr(client.model, "generation_config", None)
        if _gc is not None:
            if getattr(_gc, "do_sample", None) is not None:
                gen_kw["do_sample"] = bool(_gc.do_sample)
            _gt = getattr(_gc, "temperature", None)
            if _gt is not None and gen_kw.get("do_sample", False):
                gen_kw["temperature"] = float(_gt)
            _grp = getattr(_gc, "repetition_penalty", None)
            if _grp is not None:
                gen_kw["repetition_penalty"] = float(_grp)
        else:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = 1e-6
        return "ov15_checkpoint_generation_config"
    if temperature is None or t <= 0:
        gen_kw["do_sample"] = False
        return "generic_greedy"
    gen_kw["do_sample"] = True
    gen_kw["temperature"] = t
    return "generic_user_temperature"


def _qwenvl_llava_onevision_15_hf_checkpoint(client: AMDQwenVLClient) -> bool:
    """True for ``lmms-lab/LLaVA-OneVision-1.5-*`` (HF README Quick Start model ids)."""
    cfg = getattr(client.model, "config", None)
    mp = str(getattr(cfg, "_name_or_path", "") or getattr(cfg, "name_or_path", "") or "").lower()
    if "llava-onevision-1.5" in mp:
        return True
    if "onevision" in mp and "1.5" in mp.replace("_", "-"):
        return True
    return False


def _qwenvl_ov15_round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def _qwenvl_ov15_floor_by_factor(number: float, factor: int) -> int:
    import math

    return int(math.floor(number / factor) * factor)


def _qwenvl_ov15_ceil_by_factor(number: float, factor: int) -> int:
    import math

    return int(math.ceil(number / factor) * factor)


def _qwenvl_ov15_smart_resize(height: int, width: int, *, factor: int = 28) -> tuple[int, int]:
    """Same defaults as ``qwen_vl_utils.vision_process.smart_resize`` (Qwen-VL grid / OneVision-1.5)."""
    import math

    min_pixels = 4 * 28 * 28
    max_pixels = 16384 * 28 * 28
    max_ratio = 200
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}")
    h_bar = max(factor, _qwenvl_ov15_round_by_factor(height, factor))
    w_bar = max(factor, _qwenvl_ov15_round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _qwenvl_ov15_floor_by_factor(height / beta, factor)
        w_bar = _qwenvl_ov15_floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _qwenvl_ov15_ceil_by_factor(height * beta, factor)
        w_bar = _qwenvl_ov15_ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def _qwenvl_process_vision_info_pil_only(messages: List[dict]) -> tuple[Optional[List[Any]], Any]:
    """Subset of ``qwen_vl_utils.process_vision_info``: PIL ``image`` only (local caption paths)."""
    from PIL import Image as PILImage  # type: ignore

    image_inputs: List[Any] = []
    try:
        resample = PILImage.Resampling.BICUBIC  # type: ignore[attr-defined]
    except AttributeError:
        resample = PILImage.BICUBIC  # type: ignore[attr-defined]
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for ele in content:
            if not isinstance(ele, dict):
                continue
            raw = ele.get("image")
            if raw is None:
                continue
            if isinstance(raw, PILImage.Image):
                img = raw.convert("RGB")
                w, h = img.size
                rh, rw = _qwenvl_ov15_smart_resize(h, w)
                image_inputs.append(img.resize((rw, rh), resample))
            else:
                raise OptionalDependencyError(
                    "LLaVA-OneVision-1.5 expects PIL.Image in message ``image`` fields for local inference; "
                    "install ``qwen-vl-utils`` and use URL/path content if you need remote images."
                )
    if not image_inputs:
        return None, None
    return image_inputs, None


def _qwenvl_sanitize_inputs_for_generate(
    batch: Dict[str, Any], *, keep_mm_token_type_ids: bool = False
) -> Dict[str, Any]:
    """Remove processor outputs that ``model.generate`` rejects for some VLMs (e.g. older stacks).

    LLaVA-OneVision-1.5 Hugging Face `Quick Start` passes the processor batch (including
    ``mm_token_type_ids`` when present) straight to ``generate`` — set *keep_mm_token_type_ids*
    for that checkpoint family.

    Optional: ``CAPEVAL_QWENVL_DROP_GENERATE_KEYS`` — comma-separated extra keys to pop before ``generate``.
    """
    import os

    out = dict(batch)
    drop: set[str] = set()
    if not keep_mm_token_type_ids:
        drop.add("mm_token_type_ids")
    extra = (_env_capeval("QWENVL_DROP_GENERATE_KEYS") or "").strip()
    if extra:
        drop |= {x.strip() for x in extra.split(",") if x.strip()}
    for k in list(out.keys()):
        if k in drop:
            out.pop(k, None)
    return out


def _qwenvl_should_move_inputs_to_model_device(device: Any) -> bool:
    """True when processor tensors should be moved to the model device (``mps`` or any ``cuda:*``)."""
    s = str(device or "")
    return s == "mps" or s.startswith("cuda")


def _qwenvl_build_processor_inputs(
    client: AMDQwenVLClient,
    images: List[Any],
    prompt_text: str,
) -> Dict[str, Any]:
    """Tokenize ``images`` + templated ``prompt_text`` for ``AMD_qwenvl_call`` / retries."""
    import torch  # type: ignore

    if _qwenvl_llava_onevision_15_hf_checkpoint(client):
        # HF README Quick Start (lmms-lab/LLaVA-OneVision-1.5-*): ``process_vision_info`` +
        # ``processor(text=[text], images=..., videos=..., padding=True, return_tensors="pt")``.
        # Using ``processor(images=..., text=...)`` alone can succeed but misaligns vision tokens,
        # producing 1-token / ``<`` / degenerate decodes (verified on local caption JSON).
        user_msg = {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": prompt_text}],
        }
        qwen_messages = [user_msg]
        if not hasattr(client.processor, "apply_chat_template"):
            raise RuntimeError("processor.apply_chat_template is required for LLaVA-OneVision-1.5")
        text_for_model = client.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            image_inputs, video_inputs = process_vision_info(qwen_messages)
        except ImportError:
            image_inputs, video_inputs = _qwenvl_process_vision_info_pil_only(qwen_messages)
        _vis_wh: List[Any] = []
        if image_inputs:
            for _im in image_inputs:
                if hasattr(_im, "size"):
                    _vis_wh.append(list(_im.size))
        inputs = client.processor(
            text=[text_for_model],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt_text}]),
            }
        ]
        if hasattr(client.processor, "apply_chat_template"):
            text_for_model = client.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif hasattr(client.processor, "tokenizer") and hasattr(client.processor.tokenizer, "apply_chat_template"):
            text_for_model = client.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text_for_model = prompt_text

        try:
            inputs = client.processor(
                images=images if len(images) > 1 else images[0],
                text=text_for_model,
                return_tensors="pt",
            )
        except Exception:
            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
            except ImportError:
                raise OptionalDependencyError(
                    "This Qwen-VL checkpoint needs qwen_vl_utils. pip install qwen-vl-utils"
                ) from None
            user_msg = {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in images]
                + [{"type": "text", "text": prompt_text}],
            }
            qwen_messages = [user_msg]
            text_for_model = client.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(qwen_messages)
            inputs = client.processor(
                text=[text_for_model],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
    _ov15 = _qwenvl_llava_onevision_15_hf_checkpoint(client)
    # HF README: ``inputs = inputs.to("cuda")`` on the processor batch before ``generate``.
    if _ov15 and _qwenvl_should_move_inputs_to_model_device(client.device):
        if not isinstance(inputs, dict) and hasattr(inputs, "to"):
            try:
                inputs = inputs.to(str(client.device))
            except Exception:
                pass
    if not isinstance(inputs, dict):
        try:
            inputs = {k: v for k, v in inputs.items()}
        except Exception:
            inputs = dict(inputs)
    inputs = _qwenvl_sanitize_inputs_for_generate(inputs, keep_mm_token_type_ids=_ov15)
    if _qwenvl_should_move_inputs_to_model_device(client.device):
        inputs = {k: (v.to(client.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return inputs


def AMD_qwenvl_call(
    client: AMDQwenVLClient,
    image_paths: Iterable[str],
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_image_edge: int = 0,
) -> str:
    """Generate caption with Qwen-VL using local Transformers.

    Supports one or multiple images.
    """
    from PIL import Image  # type: ignore
    import torch  # type: ignore

    images: List[Any] = []
    load_errors: List[str] = []
    _dbg_first_path = ""
    for p in image_paths:
        if not _dbg_first_path:
            _dbg_first_path = str(p)
        try:
            img = Image.open(p).convert("RGB")
            if max_image_edge and max_image_edge > 0:
                img = _downscale_pil_max_edge(img, int(max_image_edge))
            images.append(img)
        except Exception as e:
            load_errors.append(f"{p!s}: {e!r}")
            continue
    if not images:
        tail = "; ".join(load_errors[:12])
        more = f" …(+{len(load_errors) - 12} more)" if len(load_errors) > 12 else ""
        raise RuntimeError(f"no image could be decoded from paths={list(image_paths)!r}; errors: {tail}{more}")


    inputs = _qwenvl_build_processor_inputs(client, images, prompt)


    gen_kw: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    model_hint = (
        str(
            getattr(getattr(client.model, "config", None), "_name_or_path", "")
            or getattr(getattr(client.model, "config", None), "name_or_path", "")
            or ""
        ).lower()
        + " "
        + type(client.model).__name__.lower()
    )
    llava_like = "llava" in model_hint or "onevision" in model_hint
    tok = getattr(client.processor, "tokenizer", None)
    if tok is not None:
        _ei = getattr(tok, "eos_token_id", None)
        if _ei is not None:
            gen_kw.setdefault("eos_token_id", int(_ei))
        _pi = getattr(tok, "pad_token_id", None)
        if _pi is not None:
            gen_kw.setdefault("pad_token_id", int(_pi))
    ov15_ck = _qwenvl_llava_onevision_15_hf_checkpoint(client)
    _dec_mode = _qwenvl_apply_sampling_to_gen_kw(client, gen_kw, temperature=temperature)
    if gen_kw.get("do_sample") is False:
        _qwenvl_normalize_greedy_sampling_kwargs(gen_kw)


    def _decode_from_generate_output(raw_out_obj: Any) -> str:
        if hasattr(raw_out_obj, "sequences"):
            seq_tensor = raw_out_obj.sequences
        elif isinstance(raw_out_obj, (list, tuple)) and raw_out_obj and isinstance(raw_out_obj[0], torch.Tensor):
            seq_tensor = raw_out_obj[0]
        else:
            seq_tensor = raw_out_obj
        if not isinstance(seq_tensor, torch.Tensor):
            raise TypeError(f"Unexpected generate() return type {type(raw_out_obj)}")
        seq_row = seq_tensor[0]
        if "input_ids" not in inputs:
            raise RuntimeError("processor batch missing input_ids; cannot locate generated span")
        # Match HF README: ``out_ids[len(in_ids) :]`` per row (same as ``shape[1]`` when batch=1 and no ragged padding).
        prompt_row = inputs["input_ids"][0]
        prompt_len = int(prompt_row.shape[-1])
        gen_ids_i = seq_row[prompt_len:]
        if isinstance(gen_ids_i, torch.Tensor):
            if int(gen_ids_i.numel()) == 0:
                raise ValueError("generate returned no new tokens")
        elif len(gen_ids_i) == 0:
            raise ValueError("generate returned no new tokens")
        if tok is None:
            raise RuntimeError("processor.tokenizer is required for multimodal decode")
        if llava_like and hasattr(client.processor, "batch_decode"):
            try:
                raw_t = client.processor.batch_decode(
                    [gen_ids_i],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception:
                raw_t = ""
            if not (raw_t or "").strip():
                raw_t = _qwenvl_decode_generated_ids(tok, gen_ids_i, llava_like=True)
        else:
            raw_t = _qwenvl_decode_generated_ids(tok, gen_ids_i, llava_like=llava_like)
        if ov15_ck:
            # Hugging Face model card: decode **only** the trimmed new-token span with ``processor.batch_decode``;
            # skip extra chat cleaners / assistant-suffix heuristics that are not in the official snippet.
            out_i = _qwenvl_strip_hf_pipe_fragments((raw_t or "").strip())
            if not out_i and (raw_t or "").strip():
                out_i = (raw_t or "").strip()
            alt = None
            _applied_alt = False
        else:
            cleaned_t = _qwenvl_clean_assistant_text(raw_t)
            out_i = (cleaned_t or "").strip()
            if not out_i and (raw_t or "").strip():
                out_i = (raw_t or "").strip()
            alt: Optional[str] = None
            _applied_alt = False
            if tok is not None:
                alt = _qwenvl_try_qwen_chat_assistant_suffix_from_full_seq(seq_row, tok=tok)
                if alt:
                    base = (out_i or "").strip()
                    if not base:
                        out_i = alt
                        _applied_alt = True
                    elif len(alt) > len(base):
                        out_i = alt
                        _applied_alt = True
        if not out_i and llava_like:
            marker = "<|im_start|>assistant"
            full_t = tok.decode(seq_row, skip_special_tokens=True)
            if marker in full_t:
                out_i = full_t.split(marker)[-1].strip()
        return out_i

    with torch.no_grad():
        raw_out = client.model.generate(**inputs, **gen_kw)
    out_stripped = _qwenvl_strip_hf_pipe_fragments(_decode_from_generate_output(raw_out))

    if not (out_stripped or "").strip():
        if llava_like:
            ph = _qwenvl_llava_refusal_placeholder()
            if ph:
                return ph
        raise ValueError("model returned empty assistant text after decode/clean")
    return out_stripped


def AMD_qwenvl_call_batch(
    client: AMDQwenVLClient,
    image_paths: List[str],
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_image_edge: int = 0,
) -> List[str]:
    """Batched generate: one image per row, same text prompt (LLaVA / Qwen-VL HF path).

    Falls back to sequential single calls if the processor/model does not accept batched VLM inputs.
    """
    from PIL import Image  # type: ignore
    import torch  # type: ignore

    paths = [p for p in image_paths if p]
    if not paths:
        return []
    if len(paths) == 1:
        return [
            AMD_qwenvl_call(
                client,
                paths,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_image_edge=max_image_edge,
            )
        ]

    if len(paths) > 1 and _qwenvl_llava_onevision_15_hf_checkpoint(client):
        # Batched ``processor(text=[...], images=pil_list)`` omits ``process_vision_info`` alignment; use single-image official path per row.
        return [
            AMD_qwenvl_call(
                client,
                [p],
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_image_edge=max_image_edge,
            )
            for p in paths
        ]

    pil_list: List[Any] = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            if max_image_edge and max_image_edge > 0:
                img = _downscale_pil_max_edge(img, int(max_image_edge))
            pil_list.append(img)
        except Exception:
            pil_list.append(None)
    if any(im is None for im in pil_list):
        return [
            AMD_qwenvl_call(
                client,
                [p],
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_image_edge=max_image_edge,
            )
            for p in paths
        ]

    text_for_models: List[str] = []
    for _im in pil_list:
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"}]
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ]
        if hasattr(client.processor, "apply_chat_template"):
            text_for_models.append(
                client.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        elif hasattr(client.processor, "tokenizer") and hasattr(
            client.processor.tokenizer, "apply_chat_template"
        ):
            text_for_models.append(
                client.processor.tokenizer.apply_chat_template(  # type: ignore
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            text_for_models.append(prompt)

    inputs: Dict[str, Any]
    try:
        inputs = client.processor(  # type: ignore[call-arg]
            text=text_for_models,
            images=pil_list,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        return [
            AMD_qwenvl_call(
                client,
                [p],
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_image_edge=max_image_edge,
            )
            for p in paths
        ]

    ov15_b = _qwenvl_llava_onevision_15_hf_checkpoint(client)
    if ov15_b and _qwenvl_should_move_inputs_to_model_device(client.device):
        if not isinstance(inputs, dict) and hasattr(inputs, "to"):
            try:
                inputs = inputs.to(str(client.device))
            except Exception:
                pass
    if not isinstance(inputs, dict):
        try:
            inputs = {k: v for k, v in inputs.items()}
        except Exception:
            inputs = dict(inputs)
    inputs = _qwenvl_sanitize_inputs_for_generate(inputs, keep_mm_token_type_ids=ov15_b)
    if _qwenvl_should_move_inputs_to_model_device(client.device):
        inputs = {
            k: (v.to(client.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

    gen_kw: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
    )
    model_hint_b = (
        str(
            getattr(getattr(client.model, "config", None), "_name_or_path", "")
            or getattr(getattr(client.model, "config", None), "name_or_path", "")
            or ""
        ).lower()
        + " "
        + type(client.model).__name__.lower()
    )
    llava_like_b = "llava" in model_hint_b or "onevision" in model_hint_b
    _tok_b = getattr(client.processor, "tokenizer", None)
    if _tok_b is not None:
        _bei = getattr(_tok_b, "eos_token_id", None)
        if _bei is not None:
            gen_kw.setdefault("eos_token_id", int(_bei))
        _bpi = getattr(_tok_b, "pad_token_id", None)
        if _bpi is not None:
            gen_kw.setdefault("pad_token_id", int(_bpi))
    _qwenvl_apply_sampling_to_gen_kw(client, gen_kw, temperature=temperature)
    if gen_kw.get("do_sample") is False:
        _qwenvl_normalize_greedy_sampling_kwargs(gen_kw)

    with torch.no_grad():
        raw_out = client.model.generate(**inputs, **gen_kw)

    if hasattr(raw_out, "sequences"):
        seq_tensor = raw_out.sequences
    elif isinstance(raw_out, (list, tuple)) and raw_out and isinstance(raw_out[0], torch.Tensor):
        seq_tensor = raw_out[0]
    else:
        seq_tensor = raw_out
    if not isinstance(seq_tensor, torch.Tensor):
        raise TypeError(f"Unexpected generate() return type {type(raw_out)}")

    input_ids = inputs["input_ids"]
    batch_size = int(input_ids.shape[0])
    out_list: List[str] = []
    tok = getattr(client.processor, "tokenizer", None)
    for i in range(batch_size):
        row = seq_tensor[i]
        in_row = input_ids[i]
        prompt_len_i = int(in_row.shape[-1])
        gen_ids = row[prompt_len_i:]
        if isinstance(gen_ids, torch.Tensor) and int(gen_ids.numel()) == 0:
            out_list.append("")
            continue
        if tok is not None:
            if llava_like_b and hasattr(client.processor, "batch_decode"):
                try:
                    raw_txt = client.processor.batch_decode(
                        [gen_ids],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                except Exception:
                    raw_txt = ""
                if not (raw_txt or "").strip():
                    raw_txt = _qwenvl_decode_generated_ids(tok, gen_ids, llava_like=True)
            else:
                raw_txt = _qwenvl_decode_generated_ids(tok, gen_ids, llava_like=llava_like_b)
        elif hasattr(client.processor, "decode"):
            raw_txt = client.processor.decode(gen_ids, skip_special_tokens=True)
        else:
            raw_txt = ""
        cleaned = _qwenvl_clean_assistant_text(raw_txt)
        out_stripped = (cleaned or "").strip() or (raw_txt or "").strip()
        if ov15_b and tok is not None:
            alt_b = _qwenvl_try_qwen_chat_assistant_suffix_from_full_seq(row, tok=tok)
            if alt_b:
                base_b = (out_stripped or "").strip()
                if not base_b:
                    out_stripped = alt_b
                elif len(alt_b) > len(base_b):
                    out_stripped = alt_b
        if not out_stripped and llava_like_b:
            marker = "<|im_start|>assistant"
            full_txt = (
                tok.decode(row, skip_special_tokens=True)
                if tok is not None
                else (
                    client.processor.decode(row, skip_special_tokens=True)
                    if hasattr(client.processor, "decode")
                    else ""
                )
            )
            if marker in full_txt:
                out_stripped = full_txt.split(marker)[-1].strip()
        out_stripped = _qwenvl_strip_hf_pipe_fragments(out_stripped)
        if not (out_stripped or "").strip() and llava_like_b:
            out_stripped = _qwenvl_llava_refusal_placeholder()
        out_list.append(out_stripped)
    return out_list
