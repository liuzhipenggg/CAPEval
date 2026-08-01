"""CLI for ``python -m capeval.caption``."""
from __future__ import annotations

import argparse
import os

from capeval.prompts import get_prompt, list_available_prompts


def main():
    from capeval.config import apply_defaults

    apply_defaults()
    default_input = os.environ.get("CAPTION_LAUNCHER_INPUT_DIR") or os.environ.get(
        "IMAGE_ROOT", ""
    )
    parser = argparse.ArgumentParser(
        description="Generate CAPEval image captions from a local directory"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=default_input or None,
        help=(
            "Directory of images to caption recursively "
            f"(default: IMAGE_ROOT / data/image = {default_input or '<unset>'})."
        ),
    )
    parser.add_argument(
        "--caption-shard",
        type=str,
        default=None,
        metavar="K/N",
        help=(
            "Take shard K of N by sorted filename order (i %% N == K). "
            "Runs in parallel with different K on disjoint GPUs; output file gets .shardKofN suffix. "
            "Merge JSONs when all shards finish."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR") or None,
        help=(
            "Caption directory for this model: writes OUTPUT_DIR/<prompt.lower()>.json "
            "(default: outputs/<model_safe>/caption/)."
        ),
    )
    
    # Prompt configuration
    parser.add_argument(
        "--prompt",
        type=str,
        default="PROMPT",
        help=(
            "Prompt name (default: PROMPT = "
            "Analyze the image in a comprehensive and detailed manner.)"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to caption with (Hugging Face id, local path, or resolved alias id).",
    )
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens to generate (default: 1024)")
    
    # Processing options
    parser.add_argument("--retries", type=int, default=5,
                       help="Number of retries for failed API calls (default 5 => up to 6 attempts)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing captions (disables smart resume for this run).")
    parser.add_argument(
        "--only-missing",
        dest="only_missing",
        action="store_true",
        help="Resume mode: print explicit total/existing/pending stats (default: on).",
    )
    parser.add_argument(
        "--no-only-missing",
        dest="only_missing",
        action="store_false",
        help="Disable explicit resume stats lines (still skips existing keys unless --overwrite).",
    )
    parser.set_defaults(only_missing=True)
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Optional backup root; each write also copies output JSON to BACKUP_DIR (same relative layout under --output-dir).",
    )
    
    # Utility options
    parser.add_argument("--list-prompts", action="store_true",
                       help="List all available prompts and exit")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                       help="Base URL for vLLM server (OpenAI-compatible), e.g. http://127.0.0.1:8000")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallel size for vLLM inference (default: 1)")
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=32768,
        help=(
            "Cap vLLM context length (0 = use model default, often 256k and can OOM on 48G). "
            "Captioning: 16k–32k is usually enough."
        ),
    )
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=64,
        help=(
            "For --input-dir + in-process vLLM only: images per llm.generate() call. "
            "TP=8 / roomy GPUs: try 64–96; TP=4: often 32–48. Lower on OOM."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=320,
        help=(
            "vLLM scheduler: max concurrent sequences (0 = omit). "
            "Should be >= --vllm-batch-size."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=294912,
        help=(
            "vLLM scheduler max batched tokens (0 = omit). "
            "Qwen3-VL: 3× deepstack buffers ∝ this; TP=8 often allows 294912 vs 262144 on TP=4."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.95,
        help="vLLM GPU memory fraction for weights/KV (default 0.95). Lower if OOM at init.",
    )
    parser.add_argument(
        "--vllm-enable-expert-parallel",
        action="store_true",
        help=(
            "vLLM MoE: set enable_expert_parallel=True so routed experts are sharded across "
            "tensor-parallel ranks (vLLM FusedMoEParallelConfig). Often required for large MoE "
            "on 48GB GPUs (e.g. GLM-4.6V with TP=4). Same as CAPEVAL_VLLM_ENABLE_EXPERT_PARALLEL=1."
        ),
    )
    parser.add_argument(
        "--vllm-cpu-offload-gb",
        type=float,
        default=0.0,
        help=(
            "vLLM CacheConfig.cpu_offload_gb: GiB budget to pin early layer weights on CPU "
            "(see vLLM make_layers/maybe_offload_to_cpu). Helps tight 48GB fits for very large "
            "LMs. CLI overrides CAPEVAL_VLLM_CPU_OFFLOAD_GB when this value is > 0."
        ),
    )
    parser.add_argument(
        "--vllm-pipeline-parallel-size",
        type=int,
        default=0,
        help=(
            "vLLM ParallelConfig.pipeline_parallel_size (0 = omit; use env "
            "GLM46V_VLLM_PIPELINE_PARALLEL_SIZE if set). Use with tensor parallel to occupy "
            "PP×TP GPUs (e.g. GLM-4.6V: TP=4 and PP=2 on 8×48G). TP=8 alone is invalid for this vision stack."
        ),
    )
    parser.add_argument(
        "--vllm-mm-processor-kwargs",
        type=str,
        default=None,
        help=(
            "JSON object for vLLM LLM(mm_processor_kwargs=...). "
            "For InternVL, tokenizer options such as max_length / truncation / "
            "add_special_tokens are forwarded into the built-in processor after "
            "InternVL tokenizer kwargs patch (see capeval.llm._patch_vllm_internvl_tokenizer_kwargs). "
            "Merged with CAPEVAL_VLLM_MM_PROCESSOR_KWARGS when both are set (CLI wins on key conflict)."
        ),
    )
    parser.add_argument(
        "--save-every-chunks",
        type=int,
        default=50,
        help=(
            "Flush output to disk every N completed captions: "
            "for vLLM / Transformers batch mode (each image in a batch counts as one); "
            "for vLLM batch, previously counted whole chunks (now per-caption). "
            "JSON store: rewrite file every N new keys; JSONL: append every N keys. "
            "Default 50 to reduce I/O; use 1 to update watchers every caption."
        ),
    )
    parser.add_argument(
        "--vllm-image-load-workers",
        type=int,
        default=0,
        help=(
            "Thread count for parallel JPEG decode per vLLM batch (0 = min(16, batch size)). "
            "Also used while prefetching the next batch."
        ),
    )
    parser.add_argument(
        "--no-vllm-prefetch-next-chunk",
        action="store_true",
        help=(
            "Disable decoding the next image batch on CPU while vLLM.generate runs on GPU "
            "(default: prefetch enabled when --vllm-batch-size > 1)."
        ),
    )
    parser.add_argument(
        "--caption-results-format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help=(
            "json: rewrite full dict on flush (slow once you have ~1M keys). "
            "jsonl: append one JSON object per line (fast crash-resume)."
        ),
    )
    parser.add_argument(
        "--caption-max-image-edge",
        type=int,
        default=0,
        help=(
            "If >0, resize images so max(width,height)<=this before the vision encoder "
            "(often large GPU/throughput win; try 896–1024; 0=full resolution)."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["transformers"],
        help="If 'transformers', run local HF inference (Qwen-VL, InternVL, GLM-4.6V, LLaVA via AutoModelForVision2Seq) instead of vLLM.",
    )
    parser.add_argument(
        "--transformers-batch-size",
        type=int,
        default=1,
        help=(
            "For --input-dir + --backend transformers: chunk size for one caption client call "
            "(Qwen-VL / LLaVA-OneVision: one batched generate per chunk). "
            "GLM-4.6V: GLM_TRANSFORMERS_MICRO_BATCH unset defaults to 8 images per generate; "
            "0/full/auto = one generate per chunk (like LLaVA); other ints cap batch if OOM. "
            "This flag sets caption chunk size / flush cadence. "
            "InternVL stays one image per internal call."
        ),
    )

    args = parser.parse_args()
    if args.overwrite:
        args.only_missing = False
    elif args.only_missing:
        args.overwrite = False

    if args.list_prompts:
        list_available_prompts()
        return

    if not args.input_dir:
        parser.error(
            "--input-dir is empty; set IMAGE_ROOT / CAPTION_LAUNCHER_INPUT_DIR or pass --input-dir"
        )
    if not args.output_dir:
        from capeval.util.paths import model_caption_dir

        args.output_dir = str(model_caption_dir(args.model or ""))
    if not os.path.isdir(args.input_dir):
        parser.error(f"--input-dir is not a directory: {args.input_dir}")

    if args.caption_shard:
        from capeval.caption.store import _parse_caption_shard

        try:
            args.caption_shard = _parse_caption_shard(args.caption_shard)
        except ValueError as e:
            parser.error(str(e))
    else:
        args.caption_shard = None

    prompt_stem = (args.prompt or "PROMPT").lower()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_path = os.path.join(args.output_dir, f"{prompt_stem}.json")
    if args.caption_shard is not None:
        k, n = args.caption_shard
        root, ext = os.path.splitext(args.output_path)
        args.output_path = f"{root}.shard{k}of{n}{ext}"
    if getattr(args, "caption_results_format", "json") == "jsonl" and args.output_path.endswith(
        ".json"
    ):
        args.output_path = args.output_path[: -len(".json")] + ".jsonl"
    print(f"Saving outputs to {args.output_path}...")

    from capeval.caption.engine import caption_images

    caption_images(args)


if __name__ == "__main__":
    main()
