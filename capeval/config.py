"""Runtime paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _setdefault_int(name: str, value: int) -> None:
    os.environ.setdefault(name, str(value))


def _detect_visible_gpu_count() -> int:
    """Count GPUs visible to this process (CUDA_VISIBLE_DEVICES or torch)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd != "-1":
        parts = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        return max(1, len(parts))
    try:
        import torch

        n = int(torch.cuda.device_count())
        if n > 0:
            return n
    except Exception:
        pass
    return 1


def apply_defaults() -> Path:
    """Populate os.environ with CAPEval defaults. Returns repo root."""
    root = repo_root()
    os.environ.setdefault("CAPEVAL_HOME", str(root))
    os.environ.setdefault("PROJECT_DIR", str(root))

    data = _env_path(
        "CAPEVAL_DATA_ROOT",
        _env_path("CAPEVAL_DATASET_ROOT", root / "data"),
    )
    os.environ["CAPEVAL_DATA_ROOT"] = str(data)
    os.environ["CAPEVAL_DATASET_ROOT"] = str(data)

    out = _env_path("CAPEVAL_OUTPUT_ROOT", root / "outputs")
    os.environ["CAPEVAL_OUTPUT_ROOT"] = str(out)

    image = _env_path("IMAGE_ROOT", data / "image")
    os.environ.setdefault("IMAGE_ROOT", str(image))
    os.environ.setdefault("CAPTION_LAUNCHER_INPUT_DIR", os.environ["IMAGE_ROOT"])
    os.environ.setdefault("GT_CAPTION", str(data / "gt_caption.jsonl"))
    # Backward-compatible alias used by score.py / older env files
    os.environ.setdefault("GT_JSONL", os.environ["GT_CAPTION"])
    os.environ.setdefault("CHECKLIST", str(data / "checklist.jsonl"))

    # Layout: outputs/<model_safe>/{caption,metrics}/
    # CAPEVAL_CAPTION_DIR / CAPEVAL_EVAL_DIR default to the output root (discovery);
    # per-model paths are built via capeval.util.paths.
    os.environ.setdefault("CAPEVAL_CAPTION_DIR", str(out))
    os.environ.setdefault("CAPEVAL_EVAL_DIR", str(out))
    os.environ.setdefault("CAPTION_ROOT", os.environ["CAPEVAL_CAPTION_DIR"])

    os.environ.setdefault("PYTHON_BIN", "python3")
    # Prefer CUDA_VISIBLE_DEVICES; derive GPU_LIST when the user did not set it.
    if "GPU_LIST" not in os.environ:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and cvd != "-1":
            n = len([x for x in cvd.split(",") if x.strip() != ""])
            os.environ["GPU_LIST"] = ",".join(str(i) for i in range(max(1, n)))
        else:
            n = _detect_visible_gpu_count()
            os.environ["GPU_LIST"] = ",".join(str(i) for i in range(n))

    os.environ.setdefault("PROMPT_NAME", "PROMPT")
    # Multi-GPU caption data sharding is opt-in; default is one process on all visible GPUs.
    os.environ.setdefault("TWO_SHARD_MODE", "0")
    gpus = [x.strip() for x in os.environ["GPU_LIST"].split(",") if x.strip()]
    os.environ.setdefault("NUM_SHARDS", "1")
    os.environ.setdefault("SHARD_GPU_GROUPS", ",".join(gpus) if gpus else "0")
    os.environ.setdefault("SHARD_TP_SIZE", str(max(1, len(gpus))))
    os.environ.setdefault("SHARD_CAPTION_EXT", ".json")

    _setdefault_int("CAPEVAL_LARGE_VLM_TP_SIZE", max(1, len(gpus)))
    _setdefault_int("CAPTION_VLLM_SHARD_STAGGER_SEC", 0)

    _setdefault_int("CAPEVAL_CAPTION_MAX_TOKENS", 1024)
    _setdefault_int("CAPEVAL_CAPTION_SHARD_SAVE_EVERY_CHUNKS", 100)

    _setdefault_int("CAPEVAL_CAPTION_SMALL_VLLM_MAX_MODEL_LEN", 32768)
    _setdefault_int("CAPEVAL_CAPTION_SMALL_VLLM_BATCH_SIZE", 18)
    _setdefault_int("CAPEVAL_CAPTION_SMALL_VLLM_MAX_NUM_SEQS", 30)
    _setdefault_int("CAPEVAL_CAPTION_SMALL_VLLM_MAX_NUM_BATCHED_TOKENS", 196608)
    os.environ.setdefault("CAPEVAL_CAPTION_SMALL_VLLM_GPU_MEM_UTIL", "0.95")

    _setdefault_int("CAPEVAL_CAPTION_LARGE_VLLM_MAX_MODEL_LEN", 32768)
    _setdefault_int("CAPEVAL_CAPTION_LARGE_VLLM_BATCH_SIZE", 2)
    _setdefault_int("CAPEVAL_CAPTION_LARGE_VLLM_MAX_NUM_SEQS", 4)
    _setdefault_int("CAPEVAL_CAPTION_LARGE_VLLM_MAX_NUM_BATCHED_TOKENS", 88304)
    os.environ.setdefault("CAPEVAL_CAPTION_LARGE_VLLM_GPU_MEM_UTIL", "0.90")

    # Score / judge defaults
    os.environ.setdefault("EVAL_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    os.environ.setdefault("TP_SIZE", str(max(1, len(gpus))))
    os.environ.setdefault("SAVE_EVERY", "50")
    os.environ.setdefault("EVAL_BATCH_SIZE", "8")
    os.environ.setdefault("EVAL_DP_SIZE", "1")

    return root


def gpu_list() -> str:
    return os.environ.get("GPU_LIST", "0")


def parse_gpu_list(gpus: str | None = None) -> list[str]:
    g = (gpus or gpu_list()).strip()
    if not g:
        return []
    return [x.strip() for x in g.split(",") if x.strip()]


def tp_size_from_gpu_list(gpus: str | None = None) -> int:
    return max(1, len(parse_gpu_list(gpus)))


def python_bin(*, glm: bool = False) -> str:
    if glm:
        explicit = os.environ.get("PYTHON_BIN_GLM", "").strip()
        base = os.environ.get("PYTHON_BIN", "python3")
        if explicit and explicit != base:
            return explicit
        glm_py = os.environ.get("CAPEVAL_GLM_PYTHON", "").strip()
        if glm_py and Path(glm_py).exists():
            return glm_py
    return os.environ.get("PYTHON_BIN", "python3")
