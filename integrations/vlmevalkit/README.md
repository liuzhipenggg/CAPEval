# CAPEval × VLMEvalKit

## Setup

1. Install [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and CAPEval.
2. Register the dataset — copy `capeval_dataset.py` into VLMEvalKit:

```bash
cp integrations/vlmevalkit/capeval_dataset.py \
   /path/to/VLMEvalKit/vlmeval/dataset/capeval.py
```

Then in `vlmeval/dataset/__init__.py`, import and append `CAPEvalDataset` to the image dataset list (same pattern as other custom datasets).

3. Set env:

```bash
export CAPEVAL_HOME=/path/to/CAPEval
export EVAL_MODEL=Qwen/Qwen2.5-72B-Instruct   # judge for evaluate()
```

`CAPEvalDataset.load_data` reads images via CAPEval’s `list_inference_rows()` (no separate JSONL required).

Optional packaging export (TSV / lmms-eval JSONL):

```bash
cd /path/to/CAPEval
python -m integrations.lmms_eval.prepare_data
# TSV also available via: python -c "from capeval.api import export_vlmeval_tsv; export_vlmeval_tsv('capeval.tsv')"
```

## Run

```bash
# Inference + CAPEval judge (evaluate uses CAPEval API)
cd /path/to/VLMEvalKit
CUDA_VISIBLE_DEVICES=0 python run.py --data CAPEval --model InternVL2-8B --verbose

# Or inference only, then score later with CAPEval API
CUDA_VISIBLE_DEVICES=0 python run.py --data CAPEval --model InternVL2-8B --mode infer
```

GPU usage matches VLMEvalKit conventions (`CUDA_VISIBLE_DEVICES` / `python` vs `torchrun`).
The judge stage inside `evaluate()` also respects `CUDA_VISIBLE_DEVICES` and `EVAL_MODEL`.
