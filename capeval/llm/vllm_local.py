"""Local vLLM chat / multimodal backends."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from capeval.llm.common import (
    OptionalDependencyError,
    _downscale_pil_max_edge,
    _env_capeval,
    _item_has_predecoded_pils,
    _load_pil_images_for_mm_item,
    _open_pil_rgb,
    _transformers,
    _transformers_import_error,
    _vllm,
    _vllm_import_error,
    require_optional,
)

def _patch_vllm_internvl_tokenizer_kwargs() -> None:
    """
    vLLM's built-in InternVLProcessor.__call__ has a fixed signature (no **kwargs) and
    calls ``self.tokenizer(text)`` without forwarding tokenizer options. vLLM therefore
    drops ``max_length``, ``truncation``, ``add_special_tokens``, etc. from merged
    mm/tokenization kwargs and logs func_utils warnings.

    Patch BaseInternVLProcessor and InternVLProcessor so extra keywords are passed to
    ``self.tokenizer(text, **tokenizer_kwargs)``. Same pattern helps H2OVL / NVLM /
    Eagle2.5-VL processors that inherit BaseInternVLProcessor without overriding
    __call__.

    Disable with CAPEVAL_DISABLE_INTERNVL_TOK_KW_PATCH=1.
    """
    if _vllm is None or _transformers is None:
        return
    if _env_capeval("DISABLE_INTERNVL_TOK_KW_PATCH", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        from vllm.model_executor.models.internvl import (
            BaseInternVLProcessor,
            InternVLProcessor,
        )
    except Exception:
        return

    marker = "_capeval_patched_internvl_tokenizer_kwargs"
    if getattr(BaseInternVLProcessor, marker, False):
        return

    BatchFeature = _transformers.BatchFeature

    def _base_call(
        self,
        text=None,
        images=None,
        min_dynamic_patch=None,
        max_dynamic_patch=None,
        dynamic_image_size=None,
        return_tensors=None,
        **tokenizer_kwargs: Any,
    ):
        text, images = [self._make_batch_input(x) for x in (text, images)]
        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )
        text_inputs = self.tokenizer(text, **tokenizer_kwargs)
        combined_outputs = {**text_inputs, **image_inputs}
        return BatchFeature(combined_outputs, tensor_type=return_tensors)

    def _internvl_call(
        self,
        text=None,
        images=None,
        videos=None,
        min_dynamic_patch=None,
        max_dynamic_patch=None,
        dynamic_image_size=None,
        return_tensors=None,
        **tokenizer_kwargs: Any,
    ):
        text, images, videos = [
            self._make_batch_input(x) for x in (text, images, videos)
        ]
        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )
        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
            dynamic_image_size=dynamic_image_size,
        )
        text_inputs = self.tokenizer(text, **tokenizer_kwargs)
        combined_outputs = {**text_inputs, **image_inputs, **video_inputs}
        return BatchFeature(combined_outputs, tensor_type=return_tensors)

    BaseInternVLProcessor.__call__ = _base_call  # type: ignore[method-assign, assignment]
    InternVLProcessor.__call__ = _internvl_call  # type: ignore[method-assign, assignment]
    setattr(BaseInternVLProcessor, marker, True)
    setattr(InternVLProcessor, marker, True)


def _resolve_hf_repo_to_local_snapshot(model: str) -> str:
    """
    If `model` is a Hub repo id and a complete snapshot exists under HF_HOME, return the
    local directory path so vLLM lists files on disk instead of calling list_repo_files
    (avoids hf-mirror 429 when many shards start).
    Set CAPEVAL_VLLM_DISABLE_LOCAL_RESOLVE=1 to always pass through the original string.
    """
    raw = (model or "").strip()
    if not raw:
        return model
    p = Path(raw)
    if p.is_dir() and p.exists():
        return str(p.resolve())
    if _env_capeval("VLLM_DISABLE_LOCAL_RESOLVE", "").lower() in ("1", "true", "yes"):
        return raw
    if "/" not in raw or "://" in raw:
        return raw
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return raw
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    try:
        resolved = snapshot_download(
            repo_id=raw,
            local_files_only=True,
            cache_dir=cache_dir,
        )
        rp = Path(resolved)
        if rp.is_dir():
            return str(rp.resolve())
    except Exception:
        pass
    return raw


@dataclass
class AMDvLLMClient:
    tokenizer: Any
    llm: Any
    processor: Optional[Any] = None  # For multimodal models


def AMD_vllm_chat_client(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    *,
    tp_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    trust_remote_code: bool = True,
    **llm_kwargs: Any,
) -> AMDvLLMClient:
    """
    Create vLLM client for both text-only and multimodal models.
    
    For multimodal models (LLaVA, Qwen-VL, LLaMA-Vision, InternVL, etc.),
    use this same function - it will work with AMD_vllm_multimodal_call.

    InternVL (vLLM): optional tokenizer flags are merged into the engine preprocessor
    kwargs; pass them via ``mm_processor_kwargs`` (and/or CAPEval's
    ``--vllm-mm-processor-kwargs`` / ``CAPEVAL_VLLM_MM_PROCESSOR_KWARGS``), e.g.
    ``max_length``, ``truncation``, ``add_special_tokens``. A small patch forwards
    these from InternVLProcessor into ``tokenizer(...)`` (see
    ``_patch_vllm_internvl_tokenizer_kwargs``).

    Example:
        # Text-only model
        client = AMD_vllm_chat_client(model="Qwen/Qwen2.5-7B-Instruct")
        
        # Multimodal model
        client = AMD_vllm_chat_client(model="Qwen/Qwen2-VL-7B-Instruct")
    """
    require_optional(
        "vllm",
        _vllm,
        _vllm_import_error,
        install_hint="pip install a CUDA-matched 'vllm' (see requirements.txt).",
    )
    require_optional(
        "transformers",
        _transformers,
        _transformers_import_error,
        install_hint="pip install 'transformers>=4.48,<5' 'huggingface-hub>=0.34,<1'.",
    )

    import os

    _patch_vllm_internvl_tokenizer_kwargs()

    AutoTokenizer = _transformers.AutoTokenizer
    LLM = _vllm.LLM

    model_path = _resolve_hf_repo_to_local_snapshot(model)
    if model_path != model:
        print(f"[vLLM] resolved Hub id to local snapshot (skip remote repo listing): {model_path}")

    # When running in explicit offline mode, force Transformers to only use local cached files.
    # This avoids probing remote endpoints (e.g. hf-mirror) and makes cache failures deterministic.
    offline = os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"
    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=offline,
        cache_dir=cache_dir,
    )
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    # Decoder-only LMs: left padding keeps true tokens at the end for generation.
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"

    # Try to load processor for multimodal models
    processor = None
    # Skip by default (CAPEVAL_VLLM_SKIP_PROCESSOR=1): saves Hub traffic; multimodal_call falls back to tokenizer.
    skip_processor = _env_capeval("VLLM_SKIP_PROCESSOR", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    # NOTE: In strict offline environments, AutoProcessor may still attempt hub lookups for
    # processor_config.json on some model repos (even if preprocessor_config.json exists),
    # causing long retries/noise. vLLM multimodal models can run without an external processor
    # here, so we skip processor loading when offline.
    if not offline and not skip_processor:
        try:
            AutoProcessor = _transformers.AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
        except Exception:
            # Processor not available, that's fine for text-only models
            processor = None

    # GLM-4.6V-Flash (vLLM glm4_1v): multimodal prompts need HF processor chat template; allow local cache when offline.
    if processor is None and not skip_processor:
        _mp = str(model_path).lower()
        if "glm-4.6v" in _mp or "glm4.6v" in _mp:
            try:
                AutoProcessor = _transformers.AutoProcessor
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    local_files_only=offline,
                    cache_dir=cache_dir,
                )
            except Exception:
                processor = None

    # Optional: skip torch.compile + cudagraph capture (long silent periods, 0% GPU util).
    if _env_capeval("VLLM_ENFORCE_EAGER", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        if "enforce_eager" not in llm_kwargs:
            llm_kwargs["enforce_eager"] = True
            print(
                "[vLLM] CAPEVAL_VLLM_ENFORCE_EAGER: enforce_eager=True "
                "(faster cold start; slower steady-state decode)"
            )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        # max_model_len=32000,
        # max_num_batched_tokens=320000,
        # max_num_seqs=64,
        **llm_kwargs,
    )
    return AMDvLLMClient(tokenizer=tok, llm=llm, processor=processor)


# Backward compatibility alias


# ---------- generic prompt builder ----------
def _build_prompt(question: str, tokenizer: Any, system: str = "You are a helpful assistant.") -> str:
    # Prefer model-provided chat template
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except TypeError:
            # Qwen-style content list fallback
            msgs = [
                {"role": "system", "content": system},
                {"role": "user",   "content": [{"type": "text", "text": question}]},
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Fallback for base models with no template
    return f"{system}\n\nUser: {question}\nAssistant:"


def AMD_vllm_text_chat_call(
    client: AMDvLLMClient,
    items: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = 1.0,
    stop_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_tqdm: bool = False,
    system: str = "You are a helpful assistant.",
    n: int = 1,               # number of samples per prompt
    return_all: bool = False, # True -> return List[List[str]] of all candidates
) -> Union[List[str], List[List[str]]]:
    require_optional(
        "vllm",
        _vllm,
        _vllm_import_error,
        install_hint="pip install a CUDA-matched 'vllm' (see requirements.txt).",
    )

    SamplingParams = _vllm.SamplingParams

    # Normalize batch
    batch: List[Union[str, Dict[str, str]]] = items if isinstance(items, list) else [items]

    # Build prompts
    prompts: List[str] = []
    for it in batch:
        if isinstance(it, str):
            prompts.append(_build_prompt(it, client.tokenizer, system))
        elif isinstance(it, dict):
            if "prompt" in it and it["prompt"] is not None:
                prompts.append(str(it["prompt"]))
            elif "question" in it and it["question"] is not None:
                prompts.append(_build_prompt(str(it["question"]), client.tokenizer, system))
            else:
                raise ValueError("Each dict item must contain 'question' or 'prompt'.")
        else:
            raise TypeError(f"Unsupported item type: {type(it)}")

    # Sampling params
    sampling_kwargs: dict = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        n=max(1, int(n)),
    )
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if repetition_penalty is not None and repetition_penalty != 1.0:
        sampling_kwargs["repetition_penalty"] = repetition_penalty
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = stop_token_ids

    sampling = SamplingParams(**sampling_kwargs)

    outs = client.llm.generate(prompts, sampling, use_tqdm=use_tqdm)

    if sampling.n == 1 and not return_all:
        return [o.outputs[0].text.strip() if getattr(o, "outputs", None) else "" for o in outs]

    all_texts: List[List[str]] = []
    for o in outs:
        candidates = [cand.text.strip() for cand in (getattr(o, "outputs", None) or [])]
        while len(candidates) < sampling.n:
            candidates.append("")
        all_texts.append(candidates)
    return all_texts


def _build_multimodal_prompt_with_processor(
    text: str,
    images: List[Any],  # PIL Images
    processor: Any,
    system: str = "You are a helpful assistant."
) -> str:
    """
    Build multimodal prompt for vLLM using processor.
    """
    placeholders = [{"type": "image", "image": img} for img in images]
    
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": text},
            ],
        },
    ]
    
    # Use processor's apply_chat_template (not tokenizer's)
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback: Try tokenizer if processor doesn't have the method
        print("Fallback: Try tokenizer if processor doesn't have the method")
        # if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
        #     return processor.tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        # Last resort: Phi4 multimodal format
        image_tokens = "".join([f"<|image_{i+1}|>" for i in range(len(images))])
        return f"<|user|>{image_tokens}{text}<|end|><|assistant|>"


def AMD_vllm_multimodal_call(
    client: AMDvLLMClient,
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = 1.0,
    stop_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_tqdm: bool = False,
    system: str = "You are a helpful assistant.",
    n: int = 1,
    return_all: bool = False,
    image_load_workers: int = 0,
    max_image_edge: int = 0,
) -> Union[List[str], List[List[str]]]:
    """
    Call vLLM with multimodal inputs (text + images).
    
    Supports all vLLM multimodal models including:
      - LLaVA variants (llava-1.5-7b-hf, llava-v1.6-vicuna-7b, etc.)
      - Qwen-VL variants (Qwen-VL-Chat, Qwen2-VL, etc.)
      - LLaMA 3.2 Vision
      - InternVL, MiniCPM-V, Phi-3-Vision, and others
    
    Each item should be a dict with:
      - 'text' or 'question' or 'prompt': the text prompt
      - 'image_paths' or 'images' or 'image': image path(s), and/or
      - 'pil_images': pre-loaded PIL image(s) (list or single image) to skip disk decode
    
    Disk-backed items in a batch are decoded in parallel (ThreadPoolExecutor) when
    ``image_load_workers`` is 0 (default: min(16, batch size)) or >1.
    
    Example:
        # Single image
        result = AMD_vllm_multimodal_call(
            client,
            {"text": "Describe this image", "image_paths": ["img.jpg"]},
        )
        
        # Multiple images per prompt
        result = AMD_vllm_multimodal_call(
            client,
            {"text": "Compare these images", "image_paths": ["img1.jpg", "img2.jpg"]},
        )
        
        # Batch processing
        results = AMD_vllm_multimodal_call(
            client,
            [
                {"question": "What's in this?", "images": ["photo1.jpg"]},
                {"text": "Describe the scene", "image_paths": ["photo2.jpg", "photo3.jpg"]}
            ],
            use_tqdm=True
        )
    """
    require_optional(
        "vllm",
        _vllm,
        _vllm_import_error,
        install_hint="pip install a CUDA-matched 'vllm' (see requirements.txt).",
    )

    SamplingParams = _vllm.SamplingParams
    
    # Normalize to list
    batch: List[Dict[str, Any]] = items if isinstance(items, list) else [items]

    texts: List[str] = []
    for it in batch:
        if not isinstance(it, dict):
            raise TypeError(f"Each item must be a dict, got {type(it)}")
        text = None
        for key in ("text", "question", "prompt"):
            if key in it and it[key] is not None:
                text = str(it[key])
                break
        if text is None:
            raise ValueError("Each item must contain 'text', 'question', or 'prompt'")
        texts.append(text)

    nw = int(image_load_workers or 0)
    if nw < 1:
        nw = min(16, max(1, len(batch)))
    try:
        all_pre = all(_item_has_predecoded_pils(it) for it in batch)
        if all_pre or len(batch) <= 1 or nw == 1:
            pil_lists = [_load_pil_images_for_mm_item(it) for it in batch]
        else:
            with ThreadPoolExecutor(max_workers=min(nw, len(batch))) as ex:
                pil_lists = list(ex.map(_load_pil_images_for_mm_item, batch))
    except ImportError:
        raise OptionalDependencyError("PIL is required for multimodal. pip install 'Pillow'")

    prompts: List[str] = []
    multi_modal_data: List[Dict[str, Any]] = []
    for text, pil_images in zip(texts, pil_lists):
        if int(max_image_edge or 0) > 0:
            pil_images = [_downscale_pil_max_edge(im, int(max_image_edge)) for im in pil_images]
        if client.processor is not None:
            prompt = _build_multimodal_prompt_with_processor(text, pil_images, client.processor, system)
        else:
            prompt = _build_multimodal_prompt_with_processor(text, pil_images, client.tokenizer, system)
        prompts.append(prompt)
        multi_modal_data.append({"image": pil_images if len(pil_images) > 1 else pil_images[0]})
    
    # Sampling params
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        n=max(1, int(n)),
    )
    
    # Generate with multimodal data (vLLM 0.8.x format)
    # Build inputs as list of dicts with "prompt" and "multi_modal_data" keys
    inputs = []
    for prompt, mm_data in zip(prompts, multi_modal_data):
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
    
    outs = client.llm.generate(inputs, sampling, use_tqdm=use_tqdm)
    
    # Extract results
    if sampling.n == 1 and not return_all:
        return [o.outputs[0].text.strip() if getattr(o, "outputs", None) else "" for o in outs]
    
    all_texts: List[List[str]] = []
    for o in outs:
        candidates = [cand.text.strip() for cand in (getattr(o, "outputs", None) or [])]
        while len(candidates) < sampling.n:
            candidates.append("")
        all_texts.append(candidates)
    return all_texts

