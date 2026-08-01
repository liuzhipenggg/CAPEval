"""Evaluate step: LLM single-pass checklist judging."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from capeval.llm import (
    AMD_vllm_chat_client,
    AMD_vllm_text_chat_call,
)
from capeval.score.io import (
    SINGLE_PASS_SYSTEM,
    append_jsonl,
    append_jsonl_locked,
    iter_jsonl,
    _parse_semicolon_gpu_groups,
    _cuda_visible_ids,
    _resolve_dp_gpu_groups,
    _split_list_for_dp,
)

def build_single_pass_user_prompt(caption: str, checklist_items: List[dict]) -> str:
    lines = [
        "Evaluate the CAPTION against each checklist item below.",
        "",
        "For each item_index, based ONLY on the caption, assign exactly one verdict:",
        '  - "yes": the caption correctly covers the factual content that this checklist item is asking about—'
        "the caption supports the proposition in the question without contradiction.",
        '  - "no": the caption engages with the same topic as the question but contradicts it or is clearly inconsistent.',
        '  - "not_mentioned": the caption does not address this checklist point (or is too vague to decide).',
        "If the question is phrased negatively (e.g. whether something is absent), "
        '"yes" means the caption is consistent with that negative claim—not that you are answering the word "yes" to English grammar.',
        "",
        'For every verdict you MUST include a short "reasoning" field (one concise sentence) explaining',
        "the evidence from the caption (for calibration and auditing).",
        "",
        "CAPTION:",
        caption,
        "",
        "CHECKLIST (metadata is for context; the Question text is primary):",
        "  - item_index: stable id you must echo in gt_verdicts.",
        "  - tag: fine-grained label for downstream analysis (e.g. color, spatial); do not replace the Question.",
        "  - type: which checklist channel this question belongs to (e.g. attribute vs relation); use the Question as ground truth.",
        "",
    ]
    for it in checklist_items:
        lines.append(
            f"  item_index={it['item_index']}  tag={it.get('tags', '')!r}  "
            f"type={it['checklist_type']}  Q: {it['question']}"
        )
    lines.extend(
        [
            "",
            "Return ONLY a JSON object (no markdown) with exactly this structure:",
            '{"gt_verdicts":['
            '{"item_index":<int>,"verdict":"yes"|"no"|"not_mentioned","reasoning":"<one short sentence>"},'
            "...]}",
            "Rules:",
            "- Every item_index from the checklist must appear exactly once in gt_verdicts.",
            "- Every gt_verdicts entry must include non-empty reasoning (at least a few words).",
        ]
    )
    return "\n".join(lines)


def _strip_think(text: str) -> str:
    """Drop leading chain-of-thought blocks before JSON parse (Qwen-style tags)."""
    # Built via concat so editors/filters do not mangle the closing tags.
    for sep in ("</" + "think>", "</" + "thinking>"):
        if sep in text:
            text = text.split(sep, 1)[-1]
    return text.strip()


def parse_single_pass_json(
    raw: str, n_items: int
) -> Tuple[Optional[dict], Optional[str]]:
    text = _strip_think(raw)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    decode_err: Optional[str] = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        decode_err = f"json_decode: {e}"
        data = None
    if data is None:
        # Fallback for slightly malformed JSON: recover per-item records directly.
        by_idx: Dict[int, str] = {}
        by_reason: Dict[int, str] = {}
        item_pat = re.compile(
            r'\{[^{}]*?"item_index"\s*:\s*(\d+)[^{}]*?"verdict"\s*:\s*"([^"]*)"'
            r'[^{}]*?"reasoning"\s*:\s*"([^"]*)"[^{}]*?\}',
            re.S,
        )
        for mm in item_pat.finditer(text):
            ii = int(mm.group(1))
            ver = mm.group(2).lower().strip()
            if ver in ("yes", "y", "correct", "correctly_mentioned", "b"):
                ver = "yes"
            elif ver in ("no", "n", "incorrect", "incorrectly_mentioned", "c"):
                ver = "no"
            else:
                ver = "not_mentioned"
            by_idx[ii] = ver
            by_reason[ii] = mm.group(3).strip()
        if by_idx:
            ordered = [by_idx.get(i, "not_mentioned") for i in range(n_items)]
            ordered_reason = [by_reason.get(i, "") for i in range(n_items)]
            return {"verdicts": ordered, "reasonings": ordered_reason}, None
        return None, decode_err or "json_decode_unknown"
    if not isinstance(data, dict):
        return None, "root_not_object"
    verdicts = data.get("gt_verdicts")
    if not isinstance(verdicts, list):
        return None, "missing_gt_verdicts"

    by_idx: Dict[int, str] = {}
    by_reason: Dict[int, str] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("item_index")
        if idx is None:
            continue
        try:
            ii = int(idx)
        except (TypeError, ValueError):
            continue
        ver = str(v.get("verdict", "")).lower().strip()
        if ver in ("yes", "y", "correct", "correctly_mentioned", "b"):
            ver = "yes"
        elif ver in ("no", "n", "incorrect", "incorrectly_mentioned", "c"):
            ver = "no"
        elif ver in ("not_mentioned", "notmentioned", "nm", "a", "unknown"):
            ver = "not_mentioned"
        else:
            ver = "not_mentioned"
        by_idx[ii] = ver
        by_reason[ii] = str(v.get("reasoning", "") or "").strip()

    ordered = [by_idx.get(i, "not_mentioned") for i in range(n_items)]
    ordered_reason = [by_reason.get(i, "") for i in range(n_items)]
    return {"verdicts": ordered, "reasonings": ordered_reason}, None


def normalize_verdict_list(verdicts: List[str]) -> Tuple[int, int, int]:
    y = n = nm = 0
    for v in verdicts:
        if v == "yes":
            y += 1
        elif v == "no":
            n += 1
        else:
            nm += 1
    return y, n, nm


def load_done_evaluate(path: str) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not os.path.isfile(path):
        return done
    for row in iter_jsonl(path):
        if row.get("status") == "ok":
            done.add((row.get("model_id", ""), row.get("image_id", "")))
    return done


def _evaluate_process_todo(
    client: Any,
    todo: List[dict],
    out_path: str,
    append_rec: Callable[[str, dict], None],
    *,
    batch_size: int,
    save_every: int,
    eval_max_tokens: int,
    desc: str,
) -> None:
    buf: List[dict] = []
    pending_flush = 0

    def flush_records(records: List[dict]) -> None:
        for rec in records:
            append_rec(out_path, rec)
        nonlocal pending_flush
        pending_flush = 0

    i = 0
    pbar = tqdm(total=len(todo), desc=desc)
    while i < len(todo):
        batch = todo[i : i + batch_size]
        prompts = [
            build_single_pass_user_prompt(u["caption"], u["checklist_items"]) for u in batch
        ]
        outs = AMD_vllm_text_chat_call(
            client,
            prompts,
            temperature=0.0,
            max_tokens=eval_max_tokens,
            n=1,
            return_all=False,
            use_tqdm=False,
            system=SINGLE_PASS_SYSTEM,
        )
        if outs and isinstance(outs[0], list):
            texts = [lst[0] if lst else "" for lst in outs]
        else:
            texts = [o if isinstance(o, str) else "" for o in (outs or [])]
        while len(texts) < len(batch):
            texts.append("")

        records: List[dict] = []
        for u, text in zip(batch, texts):
            n_it = len(u["checklist_items"])
            parsed, err = parse_single_pass_json(text, n_it)
            rec: dict = {
                "model_id": u["model_id"],
                "image_id": u["image_id"],
                "image_path": u["image_path"],
                "absolute_image_path": u["absolute_image_path"],
                "domain": u["domain"],
                "caption": u["caption"],
                "checklist_items": u["checklist_items"],
                "raw_response": text,
            }
            if parsed is None:
                rec["status"] = "error"
                rec["error"] = err
                rec["yes1"] = rec["no1"] = rec["not_mentioned1"] = rec["total1"] = 0
            else:
                y, n, nm = normalize_verdict_list(parsed["verdicts"])
                rec["status"] = "ok"
                rec["yes1"] = y
                rec["no1"] = n
                rec["not_mentioned1"] = nm
                rec["total1"] = n_it
                reasons = parsed.get("reasonings") or [""] * n_it
                while len(reasons) < n_it:
                    reasons.append("")
                rec["gt_verdicts"] = [
                    {
                        "item_index": j,
                        "verdict": parsed["verdicts"][j],
                        "reasoning": reasons[j],
                    }
                    for j in range(n_it)
                ]
            records.append(rec)

        buf.extend(records)
        pending_flush += len(records)
        if pending_flush >= save_every:
            flush_records(buf)
            buf.clear()
        pbar.update(len(batch))
        i += batch_size

    if buf:
        flush_records(buf)
    pbar.close()


def _mp_evaluate_worker_entry(payload: dict) -> None:
    """Non-daemon child entry (vLLM must spawn its own workers; Pool workers are daemon)."""
    try:
        _mp_evaluate_worker(payload)
    except BaseException:
        traceback.print_exc()
        raise


def _mp_evaluate_worker(payload: dict) -> int:
    """Spawn target: one vLLM engine on ``payload['gpu_group']`` (TP = payload['tp_size'])."""
    todo = payload["todo"]
    if not todo:
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = payload["gpu_group"]
    print(
        f"[evaluate][dp] pid={os.getpid()} CUDA_VISIBLE_DEVICES={payload['gpu_group']} "
        f"units={len(todo)}",
        flush=True,
    )
    client = AMD_vllm_chat_client(
        payload["eval_model"],
        tp_size=payload["tp_size"],
        gpu_memory_utilization=payload["gpu_memory_utilization"],
        trust_remote_code=True,
        max_model_len=payload["max_model_len"],
    )
    _evaluate_process_todo(
        client,
        todo,
        payload["out_path"],
        append_jsonl_locked,
        batch_size=payload["eval_batch_size"],
        save_every=payload["save_every"],
        eval_max_tokens=payload["eval_max_tokens"],
        desc=str(payload.get("desc", "evaluate")),
    )
    return len(todo)


def cmd_evaluate(args: argparse.Namespace) -> None:
    prepared_path = os.path.join(args.output_dir, "prepared.jsonl")
    out_path = os.path.join(args.output_dir, "evaluate.jsonl")
    if not os.path.isfile(prepared_path):
        raise SystemExit(f"Missing {prepared_path}; run prepare first.")

    units = list(iter_jsonl(prepared_path))
    done = load_done_evaluate(out_path)
    todo = [u for u in units if (u.get("model_id"), u.get("image_id")) not in done]
    print(f"[evaluate] total={len(units)} done={len(done)} todo={len(todo)}")

    if not todo:
        print("[evaluate] nothing to do")
        return

    batch_size = max(1, args.eval_batch_size)
    save_every = max(1, args.save_every)
    dp = max(1, int(getattr(args, "eval_dp_size", 1)))

    if dp <= 1:
        client = AMD_vllm_chat_client(
            args.eval_model,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )
        _evaluate_process_todo(
            client,
            todo,
            out_path,
            append_jsonl,
            batch_size=batch_size,
            save_every=save_every,
            eval_max_tokens=args.eval_max_tokens,
            desc="evaluate",
        )
        print(f"[evaluate] appended results -> {out_path}")
        return

    groups = _resolve_dp_gpu_groups(
        dp,
        args.tp_size,
        getattr(args, "eval_gpu_groups", None),
        env_keys=("CAPEVAL_EVAL_GPU_GROUPS",),
        label="evaluate",
    )
    buckets = _split_list_for_dp(todo, dp)
    payloads: List[dict] = []
    for rank in range(dp):
        if not buckets[rank]:
            continue
        payloads.append(
            {
                "gpu_group": groups[rank],
                "todo": buckets[rank],
                "out_path": out_path,
                "eval_model": args.eval_model,
                "tp_size": args.tp_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "eval_batch_size": batch_size,
                "save_every": save_every,
                "eval_max_tokens": args.eval_max_tokens,
                "desc": f"evaluate[dp{rank}]",
            }
        )
    print(
        f"[evaluate] data-parallel: {len(payloads)} worker(s) × TP {args.tp_size} "
        f"(target {dp}×TP{args.tp_size})"
    )
    if not payloads:
        print(f"[evaluate] appended results -> {out_path}")
        return
    ctx = multiprocessing.get_context("spawn")
    procs: List[Any] = []
    for pay in payloads:
        p = ctx.Process(target=_mp_evaluate_worker_entry, args=(pay,), daemon=False)
        p.start()
        procs.append(p)
    bad: Optional[int] = None
    for p in procs:
        p.join()
        if p.exitcode != 0:
            bad = p.exitcode
    if bad is not None:
        raise SystemExit(f"[evaluate] worker failed with exit code {bad}")
    print(f"[evaluate] appended results -> {out_path}")

