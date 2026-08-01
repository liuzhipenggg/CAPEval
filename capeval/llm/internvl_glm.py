"""InternVL and GLM-4V Transformers backends."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from capeval.llm.common import (
    OptionalDependencyError,
    _downscale_pil_max_edge,
    _open_pil_rgb,
    _transformers,
)

def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _internvl_load_image_tensor(image_file: str, input_size: int = 448, max_num: int = 12):
    """Load one image to InternVL pixel tensor (CPU), shape [num_patches, C, H, W]."""
    import torch
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    transform = _internvl_build_transform(input_size=input_size)
    patches = _internvl_dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = torch.stack([transform(im) for im in patches])
    return pixel_values


@dataclass
class AMDInternVLTransformersClient:
    model: Any
    tokenizer: Any
    device: str


def AMD_internvl_transformers_client(
    model_path: str,
    *,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AMDInternVLTransformersClient:
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:
        raise OptionalDependencyError(
            f"transformers/torch/torchvision unavailable ({type(e).__name__}: {e}). "
            "pip install transformers torch torchvision; keep huggingface-hub>=0.34,<1 with transformers 4.x."
        ) from e
    chosen = device
    if chosen is None:
        if torch.cuda.is_available():
            chosen = "cuda"
        else:
            chosen = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_fast=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    kwargs = dict(
        torch_dtype=torch.bfloat16 if chosen == "cuda" else torch.float32,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if chosen == "cuda":
        model = AutoModel.from_pretrained(model_path, **kwargs).eval().cuda()
    else:
        model = AutoModel.from_pretrained(model_path, **kwargs).eval()
    return AMDInternVLTransformersClient(model=model, tokenizer=tokenizer, device=chosen)


def AMD_internvl_transformers_call(
    client: AMDInternVLTransformersClient,
    image_paths: Iterable[str],
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_num_patches: int = 12,
) -> str:
    import torch  # type: ignore

    paths = [p for p in image_paths]
    if not paths:
        return ""
    input_size = getattr(client.model.config, "vision_config", None)
    input_size = getattr(input_size, "image_size", 448) if input_size else 448

    if len(paths) == 1:
        pv = _internvl_load_image_tensor(paths[0], input_size=input_size, max_num=max_num_patches)
        if str(client.device).startswith("cuda"):
            pv = pv.to(client.device).to(torch.bfloat16)
        else:
            pv = pv.to(client.device).to(torch.float32)
        num_patches_list = [pv.size(0)]
    else:
        lst = []
        npl = []
        for p in paths:
            t = _internvl_load_image_tensor(p, input_size=input_size, max_num=max_num_patches)
            if str(client.device).startswith("cuda"):
                t = t.to(client.device).to(torch.bfloat16)
            else:
                t = t.to(client.device).to(torch.float32)
            npl.append(t.size(0))
            lst.append(t)
        pv = torch.cat(lst, dim=0)
        num_patches_list = npl

    gen = dict(
        do_sample=temperature is not None and float(temperature) > 0,
        temperature=float(temperature) if temperature else 0.0,
        max_new_tokens=int(max_new_tokens),
    )
    if not gen["do_sample"]:
        gen.pop("temperature", None)

    with torch.no_grad():
        try:
            out = client.model.chat(
                client.tokenizer,
                pixel_values=pv,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=gen,
                verbose=False,
            )
        except TypeError:
            out = client.model.chat(
                client.tokenizer,
                pixel_values=pv,
                question=prompt,
                generation_config=gen,
                verbose=False,
            )
    return (out or "").strip()


@dataclass
class AMDGlm4VTransformersClient:
    model: Any
    processor: Any
    device: str


def AMD_glm4v_transformers_client(
    model_path: str,
    *,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AMDGlm4VTransformersClient:
    try:
        import torch  # type: ignore
        import importlib.util
        import importlib.metadata
        import warnings
        from transformers import AutoProcessor  # type: ignore
    except Exception as e:
        raise OptionalDependencyError(
            f"transformers/torch unavailable ({type(e).__name__}: {e}). "
            "pip install 'transformers>=4.51,<5' torch; keep huggingface-hub>=0.34,<1."
        ) from e

    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
    except ImportError:
        raise OptionalDependencyError(
            "GLM-4.xV needs transformers with AutoModelForImageTextToText (e.g. transformers>=4.51 / 5.x)"
        )

    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    # GLM-4.6V requires a real multimodal processor; on incompatible transformers
    # versions AutoProcessor may silently fall back to a plain tokenizer.
    if not hasattr(processor, "image_processor"):
        try:
            tf_ver = importlib.metadata.version("transformers")
        except Exception:
            tf_ver = "unknown"
        proc_name = processor.__class__.__name__
        raise OptionalDependencyError(
            "Loaded processor is not multimodal "
            f"({proc_name}, transformers=={tf_ver}). "
            "GLM-4.6V needs Glm46VProcessor support. "
            "Please upgrade transformers to 5.x (or at least a version that includes "
            "Glm46VProcessor) and re-download an intact processor cache."
        )
    dtype_env = os.environ.get("GLM4V_DTYPE", "").strip().lower()
    if chosen == "cuda":
        if dtype_env in ("fp16", "float16", "half"):
            dtype = torch.float16
        elif dtype_env in ("bf16", "bfloat16", ""):
            dtype = torch.bfloat16
        else:
            warnings.warn(
                f"Unknown GLM4V_DTYPE={dtype_env!r}; fallback to bfloat16.",
                RuntimeWarning,
            )
            dtype = torch.bfloat16
    else:
        dtype = torch.float32
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    n_cuda = int(torch.cuda.device_count()) if chosen == "cuda" else 0
    # Data-parallel caption: each shard sets CUDA_VISIBLE_DEVICES to one GPU → n_cuda==1.
    # device_map + max_memory is model-parallel style and adds overhead; Flash ~9B fits on 48GB.
    # Set GLM4V_USE_DEVICE_MAP=1 to restore accelerate sharding (e.g. full GLM-4.6V on one box).
    use_device_map = (
        chosen == "cuda"
        and has_accelerate
        and n_cuda > 1
        and os.environ.get("GLM4V_USE_DEVICE_MAP", "").strip().lower() in ("1", "true", "yes")
    )
    if chosen == "cuda" and not use_device_map and has_accelerate and n_cuda == 1:
        pass  # intentional: full model on cuda:0
    elif chosen == "cuda" and not has_accelerate:
        warnings.warn(
            "accelerate is not installed; loading GLM on a single CUDA device without device_map.",
            RuntimeWarning,
        )

    base_load_kw: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if use_device_map:
        # GLM-4.6V MoE: load-time CONVERSION (torch.cat) can spike VRAM on one GPU; without a
        # per-GPU cap, ~48GB cards OOM (~41GiB weights + ~2.75GiB cat + fragmentation).
        # Default max_memory leaves ~6–7GiB headroom per card on 48GB to limit CPU offload
        # (CPU-offloaded layers make captioning extremely slow). Tighten with GLM4V_MAX_MEMORY_GB=39
        # if load OOM; loosen up to ~43 if you have headroom. GLM4V_MAX_MEMORY_GB=0 disables cap.
        # device_map=balanced spreads blocks more evenly than auto (often evens util across 0–7).
        dm = os.environ.get("GLM4V_DEVICE_MAP", "balanced").strip() or "balanced"
        base_load_kw["device_map"] = dm
        offload_dir = os.environ.get("GLM4V_OFFLOAD_DIR", "").strip()
        if offload_dir:
            base_load_kw["offload_folder"] = offload_dir
        mm_raw = os.environ.get("GLM4V_MAX_MEMORY_GB", "").strip()
        mm_lower = mm_raw.lower()
        max_mem_off = mm_lower in ("0", "off", "false", "no")
        if not max_mem_off:
            try:
                cap_gib = float(mm_raw) if mm_raw else 41.0
            except ValueError:
                cap_gib = 41.0
            if cap_gib > 0:
                cpu_raw = os.environ.get("GLM4V_CPU_OFFLOAD_GB", "256").strip()
                try:
                    cpu_gib = float(cpu_raw)
                except ValueError:
                    cpu_gib = 256.0
                cap_s = f"{cap_gib:g}GiB"
                cpu_s = f"{cpu_gib:g}GiB"
                n_dev = torch.cuda.device_count()
                max_mem: Dict[Union[int, str], str] = {i: cap_s for i in range(n_dev)}
                max_mem["cpu"] = cpu_s
                base_load_kw["max_memory"] = max_mem

    attn_impl = os.environ.get("GLM4V_ATTN_IMPLEMENTATION", "").strip()
    if attn_impl:
        attn_candidates: List[Optional[str]] = [attn_impl]
    elif chosen == "cuda":
        # Prefer Flash-Attention 2 when installed; fall back to SDPA / HF default.
        attn_candidates = ["flash_attention_2", "sdpa", None]
    else:
        attn_candidates = [None]

    def _load_glm4v_model(load_kw: Dict[str, Any]) -> Any:
        try:
            return AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=dtype,
                **load_kw,
            )
        except TypeError:
            # Backward compatibility for transformers versions that still use torch_dtype.
            return AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                **load_kw,
            )

    model = None
    last_exc: Optional[BaseException] = None
    for att in attn_candidates:
        kw = dict(base_load_kw)
        if att is not None:
            kw["attn_implementation"] = att
        try:
            model = _load_glm4v_model(kw)
            break
        except Exception as e:
            last_exc = e
            if att is not None:
                warnings.warn(
                    f"GLM from_pretrained failed with attn_implementation={att!r}: {e}; retrying…",
                    RuntimeWarning,
                )
    if model is None:
        raise last_exc if last_exc is not None else RuntimeError("GLM model load failed")

    if chosen == "cuda" and not use_device_map:
        model = model.to("cuda:0")
        dev_str = "cuda:0"
    elif chosen != "cuda":
        model = model.to(chosen)
        dev_str = chosen
    else:
        dev_str = str(getattr(model, "device", None) or "cuda:0")
    if (
        not use_device_map
        and os.environ.get("GLM4V_TORCH_COMPILE", "").strip().lower() in ("1", "true", "yes")
    ):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception as _e:
            warnings.warn(f"GLM4V_TORCH_COMPILE requested but torch.compile failed: {_e}", RuntimeWarning)
    model.eval()
    return AMDGlm4VTransformersClient(model=model, processor=processor, device=dev_str)


def _glm4v_load_one_path(path: str, max_image_edge: int) -> Any:
    """Decode one image path to RGB PIL or ``None`` on failure."""
    from PIL import Image  # type: ignore

    try:
        img = Image.open(path).convert("RGB")
        if max_image_edge and max_image_edge > 0:
            img = _downscale_pil_max_edge(img, int(max_image_edge))
        return img
    except Exception:
        return None


def _glm4v_load_pils(paths: List[str], max_image_edge: int) -> List[Any]:
    """Return list of PIL images aligned with ``paths``; failed paths become ``None``.

    Uses ``ThreadPoolExecutor`` (``GLM4V_IMAGE_LOAD_WORKERS``, default 4) so JPEG decode
    overlaps like a DataLoader with ``num_workers>0``.
    """
    workers_raw = os.environ.get("GLM4V_IMAGE_LOAD_WORKERS", "4").strip()
    try:
        nw = int(workers_raw)
    except ValueError:
        nw = 4
    nw = max(1, nw)
    if nw <= 1 or len(paths) <= 1:
        return [_glm4v_load_one_path(p, max_image_edge) for p in paths]
    out: List[Any] = [None] * len(paths)
    cap = min(nw, len(paths))
    with ThreadPoolExecutor(max_workers=cap) as ex:
        futs: List[tuple[int, Any]] = [
            (i, ex.submit(_glm4v_load_one_path, paths[i], max_image_edge))
            for i in range(len(paths))
        ]
        for i, fut in futs:
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = None
    return out


def _glm4v_forward_pils(
    client: AMDGlm4VTransformersClient,
    pil_list: List[Any],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """One caption per PIL row (same ``prompt``). ``pil_list`` must be non-empty RGB images."""
    import torch  # type: ignore

    text_for_models: List[str] = []
    for _im in pil_list:
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image"}] + [{"type": "text", "text": prompt}]),
            }
        ]
        if hasattr(client.processor, "apply_chat_template"):
            try:
                text_for_models.append(
                    client.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                text_for_models.append(prompt)
        else:
            text_for_models.append(prompt)

    inputs = client.processor(  # type: ignore[call-arg]
        text=text_for_models,
        images=pil_list,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        k: (v.to(client.device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }
    tok = getattr(client.processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("GLM processor has no tokenizer")
    gen_kw: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature is not None and float(temperature) > 0,
    )
    if gen_kw["do_sample"]:
        gen_kw["temperature"] = float(temperature)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        gen_kw["pad_token_id"] = tok.eos_token_id
    with torch.no_grad():
        outputs = client.model.generate(**inputs, **gen_kw)
    in_len = int(inputs["input_ids"].shape[1])
    batch_size = int(inputs["input_ids"].shape[0])
    out_list: List[str] = []
    for i in range(batch_size):
        gen_ids = outputs[i, in_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        out_list.append((text or "").strip())
    return out_list


def AMD_glm4v_transformers_call(
    client: AMDGlm4VTransformersClient,
    image_paths: Iterable[str],
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    paths = list(image_paths)
    if not paths:
        return ""
    images = [Image.open(p).convert("RGB") for p in paths]
    img_arg = images[0] if len(images) == 1 else images

    # Prefer a chat-template prompt so special image placeholders are aligned.
    text_for_model = prompt
    if hasattr(client.processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]),
            }
        ]
        try:
            text_for_model = client.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text_for_model = prompt

    # Some GLM processor variants accept scalar image/text; others expect batched format.
    try:
        inputs = client.processor(images=img_arg, text=text_for_model, return_tensors="pt")
    except TypeError as e:
        if "unexpected keyword argument 'images'" in str(e):
            raise RuntimeError(
                "Current GLM processor does not accept image inputs. "
                "This usually means AutoProcessor fell back to a tokenizer "
                "(incompatible transformers version or broken processor cache)."
            ) from e
        inputs = client.processor(
            text=[text_for_model],
            images=images,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        inputs = client.processor(
            text=[text_for_model],
            images=images,
            padding=True,
            return_tensors="pt",
        )
    inputs = {
        k: (v.to(client.device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }
    tok = getattr(client.processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("GLM processor has no tokenizer")
    gen_kw: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature is not None and float(temperature) > 0,
    )
    if gen_kw["do_sample"]:
        gen_kw["temperature"] = float(temperature)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        gen_kw["pad_token_id"] = tok.eos_token_id
    with torch.no_grad():
        out_ids = client.model.generate(**inputs, **gen_kw)
    # Strip prompt tokens from output if needed
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0, in_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def AMD_glm4v_transformers_call_batch(
    client: AMDGlm4VTransformersClient,
    image_paths: List[str],
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_image_edge: int = 0,
) -> List[str]:
    """GLM-4.6V / Flash: prefer **one** ``model.generate`` per caption chunk (like LLaVA).

    ``GLM_TRANSFORMERS_MICRO_BATCH``: unset / empty → **8** images per ``generate`` (batch
    throughput). ``0`` / ``full`` / ``auto`` → one forward for the whole chunk from
    ``caption.py`` (``--transformers-batch-size``). Any other positive int caps images per
    forward if VRAM is tight.

    If a full-chunk forward raises, retry once using slices of size
    ``GLM_TRANSFORMERS_MICRO_BATCH_FALLBACK`` (default 12), then per-image on slice failure.
    """
    paths = [p for p in image_paths if p]
    if not paths:
        return []
    if len(paths) == 1:
        return [
            AMD_glm4v_transformers_call(
                client,
                paths,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        ]

    n = len(paths)
    raw_mb = (os.environ.get("GLM_TRANSFORMERS_MICRO_BATCH") or "").strip()
    if raw_mb.lower() in ("0", "full", "auto"):
        micro = n
    elif raw_mb == "":
        micro = min(8, n)
    else:
        try:
            micro = max(1, int(raw_mb))
        except ValueError:
            micro = min(8, n)
    micro = min(micro, n)

    pil_list = _glm4v_load_pils(paths, max_image_edge)
    if any(im is None for im in pil_list):
        return [
            AMD_glm4v_transformers_call(
                client,
                [p],
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for p in paths
        ]

    def _forward_slices(slice_sz: int) -> List[str]:
        acc: List[str] = []
        for start in range(0, n, slice_sz):
            end = min(start + slice_sz, n)
            sub_pils = pil_list[start:end]
            sub_paths = paths[start:end]
            try:
                acc.extend(
                    _glm4v_forward_pils(
                        client,
                        sub_pils,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                )
            except Exception:
                acc.extend(
                    [
                        AMD_glm4v_transformers_call(
                            client,
                            [p],
                            prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                        for p in sub_paths
                    ]
                )
        return acc

    if micro >= n:
        try:
            return _glm4v_forward_pils(
                client,
                pil_list,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception:
            fb_raw = (os.environ.get("GLM_TRANSFORMERS_MICRO_BATCH_FALLBACK") or "12").strip()
            try:
                fb = max(1, int(fb_raw))
            except ValueError:
                fb = 12
            fb = min(fb, n)
            return _forward_slices(fb)

    return _forward_slices(micro)
