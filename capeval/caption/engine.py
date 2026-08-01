"""Caption generation engine (single-pass + multi-backend)."""
from __future__ import annotations

import os
import sys

_CAPEVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CAPEVAL_ROOT not in sys.path:
    sys.path.insert(0, _CAPEVAL_ROOT)

import argparse
import base64
import io
import json
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from capeval.caption.images import (
    _IMAGE_EXTS,
    _decode_image_paths_rgb_parallel,
    _is_image_path,
    _local_image_key,
    list_images_in_dir,
    resize_image_for_api,
)
from capeval.caption.store import (
    _append_caption_jsonl,
    _backup_output_path,
    _caption_store_fmt,
    _load_caption_store,
    _parse_caption_shard,
    _write_results,
)
from capeval.io import encode_image
from capeval.llm import (
    AMD_claude_call,
    AMD_claude_client,
    AMD_gemini_call,
    AMD_gemini_client,
    AMD_openai_call,
    AMD_openai_client,
    AMD_transformers_caption_call,
    AMD_transformers_caption_call_batch,
    AMD_transformers_caption_client,
    AMD_vllm_chat_client,
    AMD_vllm_multimodal_call,
    AMD_vllm_server_client,
    AMD_vllm_server_multimodal_call,
    _downscale_pil_max_edge,
)
from capeval.prompts import get_prompt, list_available_prompts
from capeval.util.env import env_capeval as _env_capeval

try:
    import openai as _openai
except ImportError:  # optional until an OpenAI/Azure backend is selected
    _openai = None


def _configure_caption_stack_console_noise() -> None:
    """Optional: reduce HF Transformers log + Python warning noise during long caption runs.

    Environment:
    - ``CAPEVAL_TRANSFORMERS_LOG_LEVEL``: unset → ``error``; or ``warning`` / ``info`` /
      ``debug`` / ``none`` / ``off`` (last two skip ``set_verbosity``).
    - ``CAPEVAL_SUPPRESS_TF_FUTURE_WARNINGS``: unset or truthy → suppress known
      transformers ``FutureWarning`` spam; ``0`` / ``false`` / ``no`` to keep them.
    """
    import warnings

    _lvl_raw = _env_capeval("TRANSFORMERS_LOG_LEVEL")
    if _lvl_raw is None:
        raw = "error"
    else:
        raw = _lvl_raw.strip().lower()
    if raw not in ("none", "off"):
        try:
            from transformers import logging as tf_logging

            _v = {
                "error": tf_logging.ERROR,
                "warning": tf_logging.WARNING,
                "info": tf_logging.INFO,
                "debug": tf_logging.DEBUG,
            }.get(raw, tf_logging.ERROR)
            tf_logging.set_verbosity(_v)
        except Exception:
            pass

    _suppress = (_env_capeval("SUPPRESS_TF_FUTURE_WARNINGS", "1") or "1").strip().lower()
    if _suppress in ("0", "false", "no"):
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*rope_config_validation.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*AttentionMaskConverter.*",
        category=FutureWarning,
    )


def _sleep_backoff(attempt: int, base: float = 0.5, factor: float = 2.0, jitter: float = 0.25) -> None:
    """Sleep with exponential backoff and jitter."""
    time.sleep(base * (factor ** attempt) + random.uniform(0, max(0.0, jitter)))


def detect_model_backend(model: str) -> str:
    """Detect which API backend to use based on model name."""
    model_lower = model.lower()
    if 'gemini' in model_lower:
        return 'gemini'
    elif 'claude' in model_lower or 'anthropic' in model_lower:
        return 'claude'
    elif any(
        marker in model_lower
        for marker in (
            "qwen",
            "llama",
            "mistral",
            "pixtral",
            "phi",
            "ovis",
            "llava",
            "internvl",
            "minicpm",
            "cogvlm",
            "fuyu",
            "glm",
            "molmo",
            "smolvlm",
            "idefics",
            "deepseek",
            "gemma",
            "aria",
            "kimi",
            "vila",
            "bunny",
            "cambrian",
            "moondream",
        )
    ):
        return "vllm"
    else:
        return "openai"



def generate_caption(
    client: Any,
    model: str,
    image_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    retries: int = 5,
    backend: str = "openai",
    *,
    max_image_edge: int = 0,
) -> Optional[str]:
    """Generate caption for image(s) using the appropriate LLM backend."""
    
    for attempt in range(retries + 1):
        try:
            if backend == 'gemini':
                # Gemini uses image file paths directly
                completion = AMD_gemini_call(
                    client,
                    model,
                    messages=prompt,
                    image_paths=image_paths,
                    temperature=temperature
                )
                caption = completion.text.strip()
                return caption
                
            elif backend == 'claude':
                # Claude uses base64 encoded images (with 5 MB limit after encoding)
                content = [{"type": "text", "text": prompt}]
                
                # Add images with base64 encoding (resize_image_for_api handles size checking)
                for img_path in image_paths:
                    # This function checks if resizing is needed and returns base64 encoded string
                    # If resizing occurs, it converts to JPEG
                    image_data = resize_image_for_api(img_path)
                    
                    # resize_image_for_api always returns base64 JPEG bytes
                    mime_type = "image/jpeg"
                    
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_data
                        }
                    })
                
                messages = [{"role": "user", "content": content}]
                
                completion = AMD_claude_call(
                    client,
                    model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                caption = completion.content[0].text.strip()
                return caption
                
            elif backend == 'vllm':
                # Use vLLM multimodal API
                result = AMD_vllm_multimodal_call(
                    client,
                    {"text": prompt, "image_paths": image_paths},
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_image_edge=max_image_edge,
                )
                if isinstance(result, list) and len(result) > 0:
                    cap = (result[0] or "").strip()
                    if cap:
                        return cap
                raise ValueError("empty caption from vLLM multimodal call")
            elif backend == 'transformers':
                text = AMD_transformers_caption_call(
                    client,
                    image_paths,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    max_image_edge=max_image_edge,
                )
                cap = (text or "").strip()
                if cap:
                    return cap
                raise ValueError("empty caption from transformers backend")
            elif backend == 'vllm_server':
                # Use vLLM server (OpenAI-compatible HTTP) multimodal API
                # Some vLLM server deployments (e.g., NVLM-D-72B) require top_p in (0, 1]
                # and may default to 0. Set a safe default only for this model.
                if str(model).strip().lower() == "nvidia/nvlm-d-72b":
                    result = AMD_vllm_server_multimodal_call(
                        client,
                        {"text": prompt, "image_paths": image_paths},
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=1.0 #add this
                    )
                else:
                    result = AMD_vllm_server_multimodal_call(
                        client,
                        {"text": prompt, "image_paths": image_paths},
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                if isinstance(result, list) and len(result) > 0:
                    cap = (result[0] or "").strip()
                    if cap:
                        return cap
                raise ValueError("empty caption from vLLM server multimodal call")
            else:  # OpenAI backend
                # Encode all images for OpenAI
                encoded_images = [encode_image(img_path) for img_path in image_paths]
                
                # Create content list with all images
                content_items = []
                for encoded_image in encoded_images:
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    })
                content_items.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content_items}]
                
                completion = AMD_openai_call(
                    client,
                    model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                    max_tokens=max_tokens
                )
                
                caption = completion.choices[0].message.content.strip()
                return caption
            
        except Exception as e:
            label = (
                "api_error"
                if _openai is not None and isinstance(e, _openai.OpenAIError)
                else "unknown_error"
            )
            print(f"[{label}] attempt {attempt + 1}: {e}")
            if attempt < retries:
                _sleep_backoff(attempt)
            continue

    return None



def _caption_output_fully_done(
    args,
    out_path: str,
    *,
    local_image_paths: Optional[List[str]],
) -> bool:
    """True if out_path already contains captions for every expected local image key."""
    if getattr(args, "overwrite", False) or not os.path.isfile(out_path):
        return False
    if local_image_paths is None or not getattr(args, "input_dir", None):
        return False
    try:
        fmt = _caption_store_fmt(args)
        pre = _load_caption_store(out_path, fmt)
    except Exception:
        return False
    input_root = os.path.abspath(args.input_dir)
    keys_all = [_local_image_key(os.path.abspath(p), input_root) for p in local_image_paths]
    if not keys_all:
        return False
    return all(k in pre for k in keys_all)


def _resolve_prompt_text(args) -> str:
    """Return the user/system prompt string for one caption pass."""
    prompt_name = (args.prompt or "PROMPT").upper()
    print(f"Using prompt: {prompt_name}")
    return get_prompt(prompt_name)


def caption_images_single_pass(
    args,
    client: Any,
    backend: str,
    prompt_text: str,
    output_path: str,
    *,
    max_tokens: int,
    local_image_paths: List[str],
):
    """Run one captioning job (one prompt + one output JSON / JSONL)."""
    start_time = time.time()
    fmt = _caption_store_fmt(args)
    if args.overwrite and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass
        results = {}
        print(f"[overwrite] cleared {output_path}")
    elif os.path.exists(output_path):
        results = _load_caption_store(output_path, fmt)
        print(f"Loaded existing results from {output_path} ({len(results)} captions, format={fmt})")
    else:
        results = {}

    print(f"Output: {output_path}")
    print(f"max_tokens={max_tokens}  caption_store={fmt}  caption_max_image_edge={getattr(args, 'caption_max_image_edge', 0) or 0}")
    print(prompt_text[:500] + ("…" if len(prompt_text) > 500 else ""))

    me_cap = int(getattr(args, "caption_max_image_edge", 0) or 0)

    if not getattr(args, "input_dir", None):
        raise ValueError("--input-dir is unset (default should be IMAGE_ROOT / data/image)")
    input_root = os.path.abspath(args.input_dir)
    keys_all = [_local_image_key(os.path.abspath(p), input_root) for p in local_image_paths]
    existing_cnt = sum(1 for k in keys_all if k in results)
    pending_cnt = len(keys_all) - existing_cnt if not args.overwrite else len(keys_all)
    print(
        f"[resume] local total={len(keys_all)} existing={existing_cnt} "
        f"pending={pending_cnt} overwrite={args.overwrite} only_missing={getattr(args, 'only_missing', False)}"
    )
    vllm_bs = int(getattr(args, "vllm_batch_size", 1) or 1)
    tf_bs = int(getattr(args, "transformers_batch_size", 1) or 1)
    # Batched vLLM: one engine call per chunk (much faster than 1 image per generate()).
    if backend == "vllm" and vllm_bs > 1:
        pending: List[Tuple[str, str]] = []
        for img_path in local_image_paths:
            image_key = _local_image_key(os.path.abspath(img_path), input_root)
            if image_key in results and not args.overwrite:
                continue
            pending.append((image_key, img_path))
        if not pending:
            print("All local images already have captions (use --overwrite to redo).")
        else:
            print(
                f"vLLM batch mode: batch_size={vllm_bs}, "
                f"{len(pending)} images to caption"
            )
            # Reduce JSON I/O overhead by flushing every N chunks.
            flush_every_chunks = max(1, int(getattr(args, "save_every_chunks", 1) or 1))
            dirty_chunks = 0
            load_workers = int(getattr(args, "vllm_image_load_workers", 0) or 0)
            use_prefetch = (
                not getattr(args, "no_vllm_prefetch_next_chunk", False) and vllm_bs > 1
            )

            def _run_one_chunk(
                chunk: List[Tuple[str, str]],
                chunk_idx: int,
                items_arg: List[Dict[str, Any]],
                ilw: int,
            ) -> List[str]:
                try:
                    raw = AMD_vllm_multimodal_call(
                        client,
                        items_arg,
                        temperature=args.temperature,
                        max_tokens=max_tokens,
                        image_load_workers=ilw,
                        max_image_edge=me_cap,
                    )
                    if isinstance(raw, list):
                        return [str(x or "").strip() for x in raw][: len(chunk)]
                    return []
                except Exception as e:
                    print(
                        f"[vLLM batch] chunk {chunk_idx} failed ({e}); "
                        "falling back to single-image calls for this chunk."
                    )
                    caps_fb: List[str] = []
                    for _, p in chunk:
                        one = generate_caption(
                            client=client,
                            model=args.model,
                            image_paths=[p],
                            prompt=prompt_text,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            retries=args.retries,
                            backend=backend,
                            max_image_edge=me_cap,
                        )
                        caps_fb.append((one or "").strip())
                    return caps_fb

            jsonl_staged: Dict[str, str] = {}
            with tqdm(total=len(pending), desc="Captioning (vLLM batch)") as pbar:
                if use_prefetch:
                    with ThreadPoolExecutor(max_workers=1) as prefetch_ex:
                        next_fut = None
                        for i in range(0, len(pending), vllm_bs):
                            chunk = pending[i : i + vllm_bs]
                            paths = [p for _, p in chunk]
                            if next_fut is not None:
                                pils = next_fut.result()
                            else:
                                pils = _decode_image_paths_rgb_parallel(
                                    paths, load_workers, me_cap
                                )
                            nxt = i + vllm_bs
                            if nxt < len(pending):
                                npaths = [p for _, p in pending[nxt : nxt + vllm_bs]]
                                next_fut = prefetch_ex.submit(
                                    _decode_image_paths_rgb_parallel,
                                    npaths,
                                    load_workers,
                                    me_cap,
                                )
                            else:
                                next_fut = None
                            items = [{"text": prompt_text, "pil_images": [im]} for im in pils]
                            caps = _run_one_chunk(chunk, i // vllm_bs, items, 1)
                            while len(caps) < len(chunk):
                                caps.append("")
                            for (image_key, _), cap in zip(chunk, caps):
                                if cap:
                                    results[image_key] = cap
                                    jsonl_staged[image_key] = cap
                                else:
                                    print(f"Failed to generate caption for {image_key}")
                                pbar.update(1)
                                dirty_chunks += 1
                                if dirty_chunks >= flush_every_chunks:
                                    if fmt == "jsonl":
                                        _append_caption_jsonl(output_path, jsonl_staged, args)
                                        jsonl_staged.clear()
                                    else:
                                        _write_results(output_path, results, args)
                                    dirty_chunks = 0
                else:
                    for i in range(0, len(pending), vllm_bs):
                        chunk = pending[i : i + vllm_bs]
                        items = [
                            {"text": prompt_text, "image_paths": [p]} for _, p in chunk
                        ]
                        caps = _run_one_chunk(
                            chunk, i // vllm_bs, items, load_workers
                        )
                        while len(caps) < len(chunk):
                            caps.append("")
                        for (image_key, _), cap in zip(chunk, caps):
                            if cap:
                                results[image_key] = cap
                                jsonl_staged[image_key] = cap
                            else:
                                print(f"Failed to generate caption for {image_key}")
                            pbar.update(1)
                            dirty_chunks += 1
                            if dirty_chunks >= flush_every_chunks:
                                if fmt == "jsonl":
                                    _append_caption_jsonl(output_path, jsonl_staged, args)
                                    jsonl_staged.clear()
                                else:
                                    _write_results(output_path, results, args)
                                dirty_chunks = 0
            if fmt == "jsonl" and jsonl_staged:
                _append_caption_jsonl(output_path, jsonl_staged, args)
            elif fmt == "json" and dirty_chunks > 0:
                _write_results(output_path, results, args)
    elif backend == "transformers" and tf_bs > 1:
        pending_tf: List[Tuple[str, str]] = []
        for img_path in local_image_paths:
            image_key_tf = _local_image_key(os.path.abspath(img_path), input_root)
            if image_key_tf in results and not args.overwrite:
                continue
            pending_tf.append((image_key_tf, img_path))
        if not pending_tf:
            print("All local images already have captions (use --overwrite to redo).")
        else:
            print(
                f"Transformers batch mode: batch_size={tf_bs}, "
                f"{len(pending_tf)} images to caption"
            )
            flush_every_tf = max(1, int(getattr(args, "save_every_chunks", 1) or 1))
            dirty_tf = 0

            def _run_transformers_chunk(
                chunk: List[Tuple[str, str]],
                chunk_idx: int,
            ) -> List[str]:
                try:
                    paths_only = [p for _, p in chunk]
                    raw_tb = AMD_transformers_caption_call_batch(
                        client,
                        paths_only,
                        prompt_text,
                        max_new_tokens=max_tokens,
                        temperature=args.temperature,
                        max_image_edge=me_cap,
                    )
                    if isinstance(raw_tb, list):
                        return [str(x or "").strip() for x in raw_tb][: len(chunk)]
                    return []
                except Exception as e:
                    print(
                        f"[transformers batch] chunk {chunk_idx} failed ({e}); "
                        "falling back to single-image calls for this chunk."
                    )
                    caps_fb_tf: List[str] = []
                    for _, p in chunk:
                        one_tf = generate_caption(
                            client=client,
                            model=args.model,
                            image_paths=[p],
                            prompt=prompt_text,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            retries=args.retries,
                            backend=backend,
                            max_image_edge=me_cap,
                        )
                        caps_fb_tf.append((one_tf or "").strip())
                    return caps_fb_tf

            jsonl_staged_tf: Dict[str, str] = {}
            with tqdm(total=len(pending_tf), desc="Captioning (transformers batch)") as pbar:
                for i in range(0, len(pending_tf), tf_bs):
                    chunk = pending_tf[i : i + tf_bs]
                    caps = _run_transformers_chunk(chunk, i // tf_bs)
                    while len(caps) < len(chunk):
                        caps.append("")
                    for (image_key_tf, _), cap in zip(chunk, caps):
                        if cap:
                            results[image_key_tf] = cap
                            jsonl_staged_tf[image_key_tf] = cap
                        else:
                            print(f"Failed to generate caption for {image_key_tf}")
                        pbar.update(1)
                        dirty_tf += 1
                        if dirty_tf >= flush_every_tf:
                            if fmt == "jsonl":
                                _append_caption_jsonl(output_path, jsonl_staged_tf, args)
                                jsonl_staged_tf.clear()
                            else:
                                _write_results(output_path, results, args)
                            dirty_tf = 0
            if fmt == "jsonl" and jsonl_staged_tf:
                _append_caption_jsonl(output_path, jsonl_staged_tf, args)
            elif fmt == "json" and dirty_tf > 0:
                _write_results(output_path, results, args)
    else:
        jsonl_staged_single: Dict[str, str] = {}
        flush_n = max(1, int(getattr(args, "save_every_chunks", 1) or 1))
        since_flush = 0
        if backend == "transformers" and local_image_paths:
            print(
                "[caption] Transformers: progress bar stays at 0/N until the **first** image "
                "returns from generate (full GLM-4.6V can be many minutes per image, especially "
                "if layers are CPU-offloaded — see accelerate warnings above). "
                "The output JSON gains keys only after each successful caption.",
                flush=True,
            )
        for img_path in tqdm(local_image_paths, desc="Captioning"):
            image_key = _local_image_key(os.path.abspath(img_path), input_root)

            if image_key in results and not args.overwrite:
                continue

            caption = generate_caption(
                client=client,
                model=args.model,
                image_paths=[img_path],
                prompt=prompt_text,
                temperature=args.temperature,
                max_tokens=max_tokens,
                retries=args.retries,
                backend=backend,
                max_image_edge=me_cap,
            )

            if caption:
                results[image_key] = caption
                if fmt == "jsonl":
                    jsonl_staged_single[image_key] = caption
                    since_flush += 1
                    if since_flush >= flush_n:
                        _append_caption_jsonl(output_path, jsonl_staged_single, args)
                        jsonl_staged_single.clear()
                        since_flush = 0
                else:
                    _write_results(output_path, results, args)
            else:
                print(f"Failed to generate caption for {image_key}")
        if fmt == "jsonl" and jsonl_staged_single:
            _append_caption_jsonl(output_path, jsonl_staged_single, args)
    try:
        if fmt == "json":
            _write_results(output_path, results, args)
    except Exception as e:
        print(f"Error writing results to {output_path}: {e}")
    
    print(f"Captioning pass complete! Results saved to {output_path}")
    print(f"Total captions in this file: {len(results)}")
    elapsed = time.time() - start_time
    _mins, _secs = divmod(int(elapsed), 60)
    _hours, _mins = divmod(_mins, 60)
    print(f"Pass time: {_hours:02d}:{_mins:02d}:{_secs:02d} ({elapsed:.2f}s)")


def caption_images(args):
    """Caption local images under ``--input-dir`` into one JSON/JSONL store."""
    _configure_caption_stack_console_noise()
    start_total = time.time()

    if not getattr(args, "input_dir", None):
        raise ValueError("--input-dir is unset (default should be IMAGE_ROOT / data/image)")
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"--input-dir must be a directory: {args.input_dir}")

    local_image_paths_full = list_images_in_dir(args.input_dir)
    local_image_paths = list(local_image_paths_full)
    sp = getattr(args, "caption_shard", None)
    if sp is not None:
        k, n = sp
        before = len(local_image_paths)
        local_image_paths = [p for i, p in enumerate(local_image_paths) if i % n == k]
        print(
            f"Caption shard {k}/{n}: {len(local_image_paths)} / {before} images "
            f"(sorted list, index %% {n} == {k})"
        )
    print(
        f"Shard has {len(local_image_paths)} image paths under {args.input_dir} "
        f"(not all may need work — existing captions are skipped after output file is loaded)"
    )
    if getattr(args, "output_path", None):
        op = args.output_path
        if os.path.isfile(op):
            try:
                fmt = _caption_store_fmt(args)
                pre = _load_caption_store(op, fmt)
                input_root = os.path.abspath(args.input_dir)
                keys_all = [_local_image_key(os.path.abspath(p), input_root) for p in local_image_paths]
                existing_cnt = sum(1 for k in keys_all if k in pre)
                print(
                    f"[resume:pre-vllm] output={op} keys_in_file={len(pre)} "
                    f"shard_paths={len(keys_all)} already_captioned={existing_cnt} "
                    f"pending_in_shard={len(keys_all) - existing_cnt}"
                )
            except Exception as ex:
                print(f"[resume:pre-vllm] WARN: could not read {op}: {ex}")
        else:
            print(f"[resume:pre-vllm] no file yet (first run for this shard): {op}")

    if getattr(args, "output_path", None) and _caption_output_fully_done(
        args,
        args.output_path,
        local_image_paths=local_image_paths,
    ):
        print(
            "[resume:skip] single-pass output already complete; skipping model load -> "
            f"{args.output_path}"
        )
        elapsed = time.time() - start_total
        _mins, _secs = divmod(int(elapsed), 60)
        _hours, _mins = divmod(_mins, 60)
        print(
            f"\nAll caption jobs finished. Wall time: "
            f"{_hours:02d}:{_mins:02d}:{_secs:02d} ({elapsed:.2f}s)"
        )
        return

    backend = detect_model_backend(args.model)

    if getattr(args, "backend", None) == "transformers":
        if backend in ("gemini", "claude", "openai"):
            raise ValueError(
                f"--backend transformers is for local HF VLMs; model {args.model!r} "
                f"would use {backend}. Use a HuggingFace vision model id instead."
            )
        backend = "transformers"

    if getattr(args, "vllm_server_url", None) and backend != "transformers":
        backend = "vllm_server"
    print(f"Using {backend} backend for model {args.model}")

    if backend == "gemini":
        client = AMD_gemini_client()
    elif backend == "claude":
        client = AMD_claude_client()
    elif backend == "vllm_server":
        client = AMD_vllm_server_client(
            base_url=args.vllm_server_url, model=args.model, tensor_parallel_size=args.tp_size
        )
    elif backend == "transformers":
        print("Loading model with Hugging Face Transformers (no vLLM)...")
        client = AMD_transformers_caption_client(args.model)
    elif backend == "vllm":
        vllm_kw: Dict[str, Any] = {}
        # Qwen3-VL defaults to huge max_model_len (e.g. 262k) → KV/buffer budget can OOM at init;
        # captioning only needs image tokens + a few k decode.
        mml = getattr(args, "vllm_max_model_len", 0) or 0
        if mml > 0:
            vllm_kw["max_model_len"] = int(mml)
        mns = getattr(args, "vllm_max_num_seqs", 0) or 0
        if mns > 0:
            vllm_kw["max_num_seqs"] = int(mns)
        mnbt = getattr(args, "vllm_max_num_batched_tokens", 0) or 0
        if mnbt > 0:
            vllm_kw["max_num_batched_tokens"] = int(mnbt)
        # PCIe TP: vLLM disables custom AR anyway; set explicitly to silence warnings.
        vllm_kw["disable_custom_all_reduce"] = True
        gmu = float(getattr(args, "vllm_gpu_memory_utilization", 0.95))

        def _merge_mm_processor_kwargs_dict(extra: Dict[str, Any]) -> None:
            cur = vllm_kw.get("mm_processor_kwargs")
            base: Dict[str, Any] = dict(cur) if isinstance(cur, dict) else {}
            base.update(extra)
            vllm_kw["mm_processor_kwargs"] = base

        env_mm = (_env_capeval("VLLM_MM_PROCESSOR_KWARGS", "") or "").strip()
        if env_mm:
            try:
                parsed = json.loads(env_mm)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in CAPEVAL_VLLM_MM_PROCESSOR_KWARGS: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError("CAPEVAL_VLLM_MM_PROCESSOR_KWARGS must be a JSON object")
            _merge_mm_processor_kwargs_dict(parsed)
        cli_mm = getattr(args, "vllm_mm_processor_kwargs", None)
        if cli_mm:
            try:
                parsed = json.loads(cli_mm)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --vllm-mm-processor-kwargs: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("--vllm-mm-processor-kwargs must be a JSON object")
            _merge_mm_processor_kwargs_dict(parsed)

        ep_env = (_env_capeval("VLLM_ENABLE_EXPERT_PARALLEL", "") or "").lower()
        if getattr(args, "vllm_enable_expert_parallel", False) or ep_env in (
            "1",
            "true",
            "yes",
        ):
            vllm_kw["enable_expert_parallel"] = True

        cogb = float(getattr(args, "vllm_cpu_offload_gb", 0) or 0)
        if cogb <= 0:
            _raw_cogb = (_env_capeval("VLLM_CPU_OFFLOAD_GB", "") or "").strip()
            if _raw_cogb:
                cogb = float(_raw_cogb)
        if cogb > 0:
            vllm_kw["cpu_offload_gb"] = cogb

        pp_cli = int(getattr(args, "vllm_pipeline_parallel_size", 0) or 0)
        pp_eff = pp_cli
        if pp_eff <= 0:
            _raw_pp = (os.environ.get("GLM46V_VLLM_PIPELINE_PARALLEL_SIZE") or "").strip()
            if _raw_pp:
                pp_eff = int(_raw_pp)
        if pp_eff > 1:
            vllm_kw["pipeline_parallel_size"] = pp_eff

        client = AMD_vllm_chat_client(
            model=args.model,
            tp_size=args.tp_size,
            gpu_memory_utilization=gmu,
            **vllm_kw,
        )
        print(
            f"[vLLM] tp_size={args.tp_size} gpu_memory_utilization={gmu} "
            f"engine_kw={vllm_kw} caption_batch_size={getattr(args, 'vllm_batch_size', 1)}"
        )
    else:
        client = AMD_openai_client(model_id=args.model)

    prompt_text = _resolve_prompt_text(args)
    if _caption_output_fully_done(
        args, args.output_path, local_image_paths=local_image_paths
    ):
        print(
            f"[resume:skip] single-pass output already complete -> {args.output_path}"
        )
    else:
        print(prompt_text[:1200] + ("…" if len(prompt_text) > 1200 else ""))
        caption_images_single_pass(
            args,
            client,
            backend,
            prompt_text,
            args.output_path,
            max_tokens=args.max_tokens,
            local_image_paths=local_image_paths,
        )

    elapsed = time.time() - start_total
    _mins, _secs = divmod(int(elapsed), 60)
    _hours, _mins = divmod(_mins, 60)
    print(
        f"\nAll caption jobs finished. Wall time: "
        f"{_hours:02d}:{_mins:02d}:{_secs:02d} ({elapsed:.2f}s)"
    )
