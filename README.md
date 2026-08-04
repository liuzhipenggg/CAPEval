<p align="center">
  <img src="assets/logo.png" alt="CAPEval" width="520"/>
</p>

<h2 align="center">A Decoupled Caption Evaluation across Understanding and Generation</h2>

<p align="center">
  <a href="#-leaderboard">🏆 Leaderboard</a> ·
  <a href="https://huggingface.co/datasets/LiuzhipengUCAS/CAPEval">🤗 Hugging Face</a> ·
  <a href="#-downstream-evaluation">🔗 Downstream</a> ·
  <a href="#-resources">📄 Paper</a>
</p>

> A checklist-based benchmark that decouples caption quality into **Coverage (C)** and **Precision (P)** (0–100), and studies how each profile transfers to VLM understanding and T2I generation.

## 🔥 News

- **[2026-08]** 📄 Paper on arXiv: https://arxiv.org/abs/2608.02589
- **[2026-07]** 🏆 Project page & leaderboard: https://liuzhipenggg.github.io/CAPEval/
- **[2026-07]** 💻 Code released: captioning, checklist judging, and C / P evaluation.

## 📎 Resources

- 📄 **Paper**: [arXiv:2608.02589](https://arxiv.org/abs/2608.02589)
- 🌐 **Project page**: https://liuzhipenggg.github.io/CAPEval/
- 🤗 **Hugging Face Dataset**: [LiuzhipengUCAS/CAPEval](https://huggingface.co/datasets/LiuzhipengUCAS/CAPEval)
- 🏆 **Leaderboard**: https://liuzhipenggg.github.io/CAPEval/leaderboard/

## 🚀 Installation

```bash
git clone https://github.com/liuzhipenggg/CAPEval.git
cd CAPEval
pip install -r requirements.txt
```

`requirements.txt` pins light deps and keeps **transformers 4.x** with `huggingface-hub<1` (hub 1.x breaks `import transformers` on that line). Install a CUDA-matched `torch` / `vllm` pair separately, then check:

```bash
python -c "import vllm, transformers, huggingface_hub as h; print('ok', h.__version__)"
```

See [`.env.example`](.env.example) for optional environment variables.

## 📊 Evaluation

CAPEval judges each caption against atomic checklist items (`yes` / `no` / `not_mentioned`) and reports **percent** scores (0–100):

| Metric | Definition |
|--------|------------|
| **C** | `100 × (yes + no) / total` — coverage |
| **P** | `100 × yes / (yes + no)` — precision |

Also reported: **per-category C / P** for Scene & Object · People & Activity · Text & Interface · Design & Knowledge.

Default judge: `Qwen/Qwen2.5-72B-Instruct`.

### Step 1: Generate captions

Official prompt: `Analyze the image in a comprehensive and detailed manner.`

**Option A — built-in captioner**

```bash
python caption.py internvl1b
```

**Option B — your own captions**

Write a submission JSON (see [`examples/submission.json`](examples/submission.json)):

```json
[
  {"image_id": "SO001.jpg", "caption": "A detailed caption..."},
  {"image_id": "SO002.jpg", "caption": "..."}
]
```

Keys are image basenames under [`data/image/`](data/image/).

### Step 2: Score captions (C / P)

**Option A — built-in captions** (after `caption.py`):

```bash
python score.py internvl1b
```

**Option B — your own submission JSON**:

```bash
# Validate format only
python evaluate.py -s examples/submission.json --check-only

# Full checklist judge
python evaluate.py -s path/to/submission.json --model-id my_captioner
```

**Option C — caption + score together**:

```bash
python pipeline.py internvl1b
```

### Step 3: Read results

```text
outputs/<model_id>/
  caption/prompt.json
  metrics/metrics.json
  metrics/results_summary.json
```

```python
from capeval import CAPEval

score = CAPEval().evaluate_map({"SO001.jpg": "A detailed caption..."})
print(score["C"], score["P"])
print(score["per_category"])
```

## 📦 Dataset

300 images · **14,965** atomic checklist items · [`data/`](data/) (`image/`, `gt_caption.jsonl`, `checklist.jsonl`; join key `img_path`) · [Hugging Face](https://huggingface.co/datasets/LiuzhipengUCAS/CAPEval)

Per-category C / P use the filename prefix (`SO` / `PA` / `TI` / `DK`). Optional tables in [`data/meta/`](data/meta/) list category / subcategory labels (not required by the scorer).

<table width="100%">
<thead>
<tr>
<th align="left">Super-category</th>
<th align="center">Prefix</th>
<th align="right">Images</th>
<th align="right">Checklist items</th>
<th align="right">Avg. / image</th>
</tr>
</thead>
<tbody>
<tr><td>Scene &amp; Object</td><td align="center"><code>SO</code></td><td align="right">148</td><td align="right">7,039</td><td align="right">47.6</td></tr>
<tr><td>People &amp; Activity</td><td align="center"><code>PA</code></td><td align="right">60</td><td align="right">3,164</td><td align="right">52.7</td></tr>
<tr><td>Text &amp; Interface</td><td align="center"><code>TI</code></td><td align="right">40</td><td align="right">2,146</td><td align="right">53.6</td></tr>
<tr><td>Design &amp; Knowledge</td><td align="center"><code>DK</code></td><td align="right">52</td><td align="right">2,616</td><td align="right">50.3</td></tr>
<tr><td><b>Total</b></td><td align="center"></td><td align="right"><b>300</b></td><td align="right"><b>14,965</b></td><td align="right"><b>49.9</b></td></tr>
</tbody>
</table>

## 🔗 Downstream evaluation

Paper downstream numbers use official stacks:

- **Understanding:** [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) — MME, MMBench (EN/CN), ScienceQA, OCRBench, RealWorldQA, MMStar, MMVP, POPE, HallusionBench, AMBER, CV-Bench (2D/3D), HR-Bench (4K/8K), BLINK, AI2D, SEED-Bench, MMMU
- **Generation:** [GenEval](https://github.com/djghosh13/geneval) · [DPG-Bench (ELLA)](https://github.com/TencentQQGYLab/ELLA) · [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench)

## 🏆 Leaderboard

Official site: https://liuzhipenggg.github.io/CAPEval/leaderboard/  
(Source: [`docs/leaderboard/`](docs/leaderboard/). Preliminary public numbers below.)

| Model | C ↑ | P |
|-------|----:|----:|
| Gemini-3.1-Pro | 80.2 | 82.5 |
| Gemini-3.5-Flash | 78.1 | 81.4 |
| Gemini-2.5-Pro | 75.5 | 80.1 |
| GPT-5.5 | 72.8 | 79.0 |
| Qwen3-VL-32B | 68.5 | 86.1 |
| Qwen3-VL-8B | 63.2 | 83.5 |
| GLM-4.6V-Flash | 62.0 | 85.4 |
| Qwen3-VL-4B | 60.2 | 81.1 |
| InternVL3.5-38B | 48.6 | 77.7 |
| InternVL3.5-1B | 48.3 | 65.2 |
| InternVL3.5-8B | 46.5 | 72.6 |
| InternVL3.5-4B | 45.0 | 73.5 |
| LLaVA-OV-1.5-4B | 43.5 | 45.1 |
| LLaVA-OV-1.5-8B | 41.2 | 50.4 |

## 📚 Citation

```bibtex
@misc{liu2026capeval,
  title         = {CAPEval: A Decoupled Caption Evaluation across Understanding and Generation},
  author        = {Zhipeng Liu, Haochen Wang, Zhaoxiang Zhang},
  year          = {2026},
  eprint        = {2608.02589},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.02589},
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
