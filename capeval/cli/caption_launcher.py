#!/usr/bin/env python3
"""Multi-model caption launcher.

Usage:
  python caption.py internvl1b
  python caption.py qwen8b internvl8b
  python caption.py quick            # small convenience group
  python caption.py Qwen/Qwen2.5-VL-7B-Instruct

GPU:
  CUDA_VISIBLE_DEVICES=0 python caption.py internvl1b
  CUDA_VISIBLE_DEVICES=0,1,2,3 python caption.py qwen32b
  # Optional: GPU_LIST=0,1 overrides / sets CUDA_VISIBLE_DEVICES for the run
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from capeval.config import apply_defaults, parse_gpu_list, python_bin, repo_root, tp_size_from_gpu_list
from capeval.models import expand_targets, format_model_table, resolve_spec
from capeval.util.caption_store import load_caption_dict, write_caption_json
from capeval.util.paths import model_caption_dir


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _extra_args() -> list[str]:
    return shlex.split(os.environ.get("EXTRA_ARGS", ""))


def _caption_cmd(
    *,
    model_id: str,
    backend: str,
    temperature: float,
    tp_size: int,
    input_dir: str,
    output_dir: str,
    prompt: str,
    extra: list[str],
    py: str,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "capeval.caption",
        "--model",
        model_id,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--temperature",
        str(temperature),
        "--prompt",
        prompt,
    ]
    if backend == "transformers":
        cmd.extend(["--backend", "transformers"])
    else:
        cmd.extend(["--tp-size", str(tp_size)])
    cmd.extend(extra)
    return cmd


def _run_proc(cmd: list[str], *, gpu_list: str | None, cwd: Path, dry_run: bool) -> int:
    env = os.environ.copy()
    if gpu_list is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_list
    print(f"[caption] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')} {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _merge_shard_files(shards: list[Path], out: Path) -> None:
    merged: dict = {}
    for path in shards:
        data = load_caption_dict(path)
        overlap = set(merged) & set(data)
        if overlap:
            print(f"warn: {path}: {len(overlap)} keys already present; shard values win")
        merged.update(data)
    write_caption_json(out, merged)
    print(f"Wrote {len(merged)} entries -> {out}")


def _shard_vllm_extra(*, large: bool) -> list[str]:
    if large and os.environ.get("APPLY_CAPEVAL_LARGE_VLLM_PRESET", "1") == "1":
        bs = os.environ["CAPEVAL_CAPTION_LARGE_VLLM_BATCH_SIZE"]
        mns = os.environ["CAPEVAL_CAPTION_LARGE_VLLM_MAX_NUM_SEQS"]
        mnbt = os.environ["CAPEVAL_CAPTION_LARGE_VLLM_MAX_NUM_BATCHED_TOKENS"]
        gmu = os.environ["CAPEVAL_CAPTION_LARGE_VLLM_GPU_MEM_UTIL"]
        mml = os.environ["CAPEVAL_CAPTION_LARGE_VLLM_MAX_MODEL_LEN"]
    else:
        bs = os.environ.get(
            "SHARD_VLLM_BATCH_SIZE", os.environ["CAPEVAL_CAPTION_SMALL_VLLM_BATCH_SIZE"]
        )
        mns = os.environ.get(
            "SHARD_VLLM_MAX_NUM_SEQS",
            os.environ["CAPEVAL_CAPTION_SMALL_VLLM_MAX_NUM_SEQS"],
        )
        mnbt = os.environ.get(
            "SHARD_VLLM_MAX_NUM_BATCHED_TOKENS",
            os.environ["CAPEVAL_CAPTION_SMALL_VLLM_MAX_NUM_BATCHED_TOKENS"],
        )
        gmu = os.environ.get(
            "SHARD_VLLM_GPU_MEM_UTIL",
            os.environ["CAPEVAL_CAPTION_SMALL_VLLM_GPU_MEM_UTIL"],
        )
        mml = os.environ.get(
            "SHARD_VLLM_MAX_MODEL_LEN",
            os.environ["CAPEVAL_CAPTION_SMALL_VLLM_MAX_MODEL_LEN"],
        )
    save = os.environ.get(
        "SHARD_VLLM_SAVE_EVERY_CHUNKS",
        os.environ["CAPEVAL_CAPTION_SHARD_SAVE_EVERY_CHUNKS"],
    )
    max_tokens = os.environ.get("SHARD_MAX_TOKENS", os.environ["CAPEVAL_CAPTION_MAX_TOKENS"])
    return [
        "--max-tokens",
        str(max_tokens),
        "--vllm-max-model-len",
        str(mml),
        "--vllm-batch-size",
        str(bs),
        "--vllm-max-num-seqs",
        str(mns),
        "--vllm-max-num-batched-tokens",
        str(mnbt),
        "--vllm-gpu-memory-utilization",
        str(gmu),
        "--save-every-chunks",
        str(save),
    ]


def _large_layout(gpus: list[str]) -> tuple[int, int, list[str]]:
    ng = len(gpus)
    tp_per = _env_int("CAPEVAL_LARGE_VLM_TP_SIZE", max(1, ng))
    if tp_per <= 0 or tp_per >= ng:
        return 1, ng, [",".join(gpus)]
    if ng % tp_per != 0:
        raise SystemExit(
            f"ERROR: large VLM: |GPU_LIST|={ng} not divisible by CAPEVAL_LARGE_VLM_TP_SIZE={tp_per}"
        )
    groups = []
    for i in range(0, ng, tp_per):
        groups.append(",".join(gpus[i : i + tp_per]))
    return ng // tp_per, tp_per, groups


def run_shards(
    *,
    model_id: str,
    backend: str,
    temperature: float,
    groups: list[str],
    shard_tp: int,
    large_vllm: bool,
    cwd: Path,
    dry_run: bool,
    save_every_chunks: int | None = None,
) -> int:
    prompt = os.environ.get("PROMPT_NAME", "PROMPT")
    input_dir = os.environ["CAPTION_LAUNCHER_INPUT_DIR"]
    output_dir = os.environ.get("OUTPUT_DIR") or str(model_caption_dir(model_id))
    ext = os.environ.get("SHARD_CAPTION_EXT", ".json")
    if not ext.startswith("."):
        ext = f".{ext}"
    prompt_stem = prompt.lower()
    out_dir = Path(output_dir)
    num_shards = len(groups)
    py = python_bin(glm=backend == "transformers" and "GLM" in model_id.upper())
    extra = list(_extra_args())
    if backend != "transformers":
        shard_extra = _shard_vllm_extra(large=large_vllm)
        if save_every_chunks is not None:
            if "--save-every-chunks" in shard_extra:
                i = shard_extra.index("--save-every-chunks")
                shard_extra[i + 1] = str(save_every_chunks)
            else:
                shard_extra.extend(["--save-every-chunks", str(save_every_chunks)])
        extra = extra + shard_extra

    stagger = _env_int("CAPTION_VLLM_SHARD_STAGGER_SEC", 0)
    print(
        f"[multi-shard] model={model_id} num_shards={num_shards} tp={shard_tp} groups={';'.join(groups)}"
    )
    if dry_run:
        return 0

    procs: list[subprocess.Popen] = []
    for i, gpu_group in enumerate(groups):
        if stagger > 0 and i > 0:
            time.sleep(stagger)
        shard_extra = list(extra) + ["--caption-shard", f"{i}/{num_shards}"]
        cmd = _caption_cmd(
            model_id=model_id,
            backend=backend,
            temperature=temperature,
            tp_size=shard_tp,
            input_dir=input_dir,
            output_dir=output_dir,
            prompt=prompt,
            extra=shard_extra,
            py=py,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_group
        print(f"[caption][shard {i}/{num_shards}] GPUs={gpu_group}")
        procs.append(subprocess.Popen(cmd, cwd=str(cwd), env=env))

    failed = False
    for p in procs:
        if p.wait() != 0:
            failed = True
    if failed:
        print(f"[multi-shard] ERROR: one or more shards failed (model={model_id})")
        return 1

    shard_files = [
        out_dir / f"{prompt_stem}.shard{i}of{num_shards}{ext}" for i in range(num_shards)
    ]
    for sp in shard_files:
        if not sp.is_file():
            print(f"[multi-shard] skip merge (missing shard file): {sp}")
            return 1
    _merge_shard_files(shard_files, out_dir / f"{prompt_stem}.json")
    return 0


def run_single(
    *,
    model_id: str,
    backend: str,
    temperature: float,
    cwd: Path,
    dry_run: bool,
) -> int:
    gpus = os.environ.get("GPU_LIST", "0")
    tp = int(os.environ.get("TP_SIZE") or tp_size_from_gpu_list(gpus))
    prompt = os.environ.get("PROMPT_NAME", "PROMPT")
    input_dir = os.environ["CAPTION_LAUNCHER_INPUT_DIR"]
    output_dir = os.environ.get("OUTPUT_DIR") or str(model_caption_dir(model_id))
    py = python_bin(glm=backend == "transformers" and "GLM" in model_id.upper())
    extra = list(_extra_args())
    if "--max-tokens" not in extra:
        extra.extend(
            [
                "--max-tokens",
                os.environ.get("CAPEVAL_CAPTION_MAX_TOKENS", "1024"),
            ]
        )
    cmd = _caption_cmd(
        model_id=model_id,
        backend=backend,
        temperature=temperature,
        tp_size=tp,
        input_dir=input_dir,
        output_dir=output_dir,
        prompt=prompt,
        extra=extra,
        py=py,
    )
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_arg = None if cvd else gpus
    return _run_proc(cmd, gpu_list=gpu_arg, cwd=cwd, dry_run=dry_run)


def run_one(alias: str, *, cwd: Path, dry_run: bool) -> int:
    spec = resolve_spec(alias)
    two_shard = os.environ.get("TWO_SHARD_MODE", "0") == "1"
    if not two_shard:
        return run_single(
            model_id=spec.model_id,
            backend=spec.backend,
            temperature=spec.temperature,
            cwd=cwd,
            dry_run=dry_run,
        )

    gpus = parse_gpu_list()
    if spec.glm_flash_nx1:
        groups = list(gpus) if gpus else ["0"]
        return run_shards(
            model_id=spec.model_id,
            backend=spec.backend,
            temperature=spec.temperature,
            groups=groups,
            shard_tp=1,
            large_vllm=False,
            cwd=cwd,
            dry_run=dry_run,
            save_every_chunks=1,
        )

    if spec.large_vlm:
        num_shards, shard_tp, groups = _large_layout(gpus or ["0"])
        print(
            f"[multi-shard] large VLM layout: NUM_SHARDS={num_shards} "
            f"SHARD_TP_SIZE={shard_tp} groups={';'.join(groups)}"
        )
        return run_shards(
            model_id=spec.model_id,
            backend=spec.backend,
            temperature=spec.temperature,
            groups=groups,
            shard_tp=shard_tp,
            large_vllm=spec.backend != "transformers",
            cwd=cwd,
            dry_run=dry_run,
        )

    groups_env = os.environ.get("SHARD_GPU_GROUPS", "")
    if groups_env.strip():
        groups = [g.strip() for g in groups_env.split(";") if g.strip()]
    else:
        groups = [",".join(gpus)] if gpus else ["0"]
    num_shards = _env_int("NUM_SHARDS", len(groups))
    if len(groups) != num_shards:
        raise SystemExit(
            f"ERROR: NUM_SHARDS={num_shards} but SHARD_GPU_GROUPS has {len(groups)} groups"
        )
    shard_tp = _env_int("SHARD_TP_SIZE", 1)
    return run_shards(
        model_id=spec.model_id,
        backend=spec.backend,
        temperature=spec.temperature,
        groups=groups,
        shard_tp=shard_tp,
        large_vllm=False,
        cwd=cwd,
        dry_run=dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    apply_defaults()
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        print(__doc__)
        print("\nBuilt-in models:\n")
        print(format_model_table())
        return 0
    if args and args[0] in {"--list-models", "list-models"}:
        print(format_model_table())
        return 0
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    targets = expand_targets(args)
    print(f"Selected targets: {' '.join(args)}")
    print(f"Expanded run order: {' '.join(targets)}")
    print(
        f"Tip: PROMPT_NAME={os.environ.get('PROMPT_NAME')} "
        f"GPU_LIST={os.environ.get('GPU_LIST')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"TWO_SHARD_MODE={os.environ.get('TWO_SHARD_MODE')}"
    )
    print("Tip: unknown names are treated as HF ids / local paths (see --list-models).")
    input_dir = os.environ.get("CAPTION_LAUNCHER_INPUT_DIR", "")
    if not input_dir or not Path(input_dir).is_dir():
        print(
            f"[capeval][WARN] CAPTION_LAUNCHER_INPUT_DIR missing or not a directory: "
            f"{input_dir or '<unset>'}"
        )

    cwd = repo_root()
    for i, t in enumerate(targets, 1):
        print(f"===== [{i}/{len(targets)}] {t} =====")
        rc = run_one(t, cwd=cwd, dry_run=dry_run)
        if rc != 0:
            return rc
    print("All requested targets finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
