# CAPEval cluster runbook

Apple Mac 本地不跑模型；在 **CUDA + vLLM** 集群上按下面做。

## 0. 环境

```bash
git clone https://github.com/liuzhipenggg/CAPEval.git
cd CAPEval
# 或: git pull

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt   # pins transformers 4.x + huggingface-hub<1
# torch / vllm: install a CUDA-matched pair, then verify they import together:
#   python -c "import vllm, transformers, huggingface_hub as h; print('ok', h.__version__)"
# Do NOT `pip install -U huggingface_hub` into hub 1.x while on transformers 4.x —
# import fails and CAPEval used to mis-report that as "transformers is not installed".
```

数据二选一：

```bash
# A) 仓库自带 data/
# B) 从 HF 拉（keep hub on 0.x for transformers 4.x）
pip install 'huggingface_hub>=0.34,<1'
hf download LiuzhipengUCAS/CAPEval --repo-type dataset --local-dir ./capeval_data
export IMAGE_ROOT=$PWD/capeval_data/image
export GT_CAPTION=$PWD/capeval_data/gt_caption.jsonl
export CHECKLIST=$PWD/capeval_data/checklist.jsonl
```

选卡：

```bash
export CUDA_VISIBLE_DEVICES=0          # 按实际改
# 多卡 judge 示例: export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 1. 冒烟（最小 captioner）

**Caption 最小别名：`internvl1b`**（`OpenGVLab/InternVL3_5-1B`，vLLM）

**Judge 默认很大**（`Qwen/Qwen2.5-72B-Instruct`）。冒烟可先换成小文本模型，例如：

```bash
export EVAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct   # 集群若可用；否则用 7B/72B
export TP_SIZE=1
```

### 1.1 只生成 caption

```bash
python caption.py internvl1b
# 输出: outputs/OpenGVLab_InternVL3_5-1B/caption/prompt.json
```

### 1.2 对已有 caption 打分（C/P + 四大类）

```bash
python score.py internvl1b
# 或
python score.py OpenGVLab_InternVL3_5-1B
```

### 1.3 一条龙

```bash
python pipeline.py internvl1b
```

### 1.4 看结果

```bash
cat outputs/OpenGVLab_InternVL3_5-1B/metrics/results_summary.json
# 或
cat outputs/OpenGVLab_InternVL3_5-1B/metrics/metrics.json
cat outputs/OpenGVLab_InternVL3_5-1B/metrics/per_category.csv
cat outputs/OpenGVLab_InternVL3_5-1B/metrics/report.md
```

分数为 **0–100**：

| 字段 | 含义 |
|------|------|
| `C` / `P` | Coverage / Precision（百分制） |
| `per_category` | `Scene & Object` / `People & Activity` / `Text & Interface` / `Design & Knowledge` 各自的 C、P |
| `n_error` | judge 失败的图像数（不计入 C/P） |

---

## 2. 外部 submission（README 主路径）

```bash
# 校验格式
python evaluate.py -s examples/submission.json --check-only

# 全量 300 张 caption 写成 submission.json 后：
python evaluate.py -s path/to/submission.json --model-id my_captioner
```

`submission.json`：

```json
[
  {"image_id": "SO001.jpg", "caption": "..."},
  {"image_id": "SO002.jpg", "caption": "..."}
]
```

---

## 3. 正式评测建议

| 步骤 | 建议 |
|------|------|
| Caption | 目标 VLM（`python caption.py --list-models`） |
| Judge | `EVAL_MODEL=Qwen/Qwen2.5-72B-Instruct`，多卡设 `TP_SIZE`=可见 GPU 数 |
| 跑法 | `pipeline.py <alias>` 或 caption → score |
| 产出 | `outputs/<model_safe>/metrics/` |

多卡 judge 示例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_MODEL=Qwen/Qwen2.5-72B-Instruct
export TP_SIZE=4
python score.py qwen8b
```

---

## 4. 常见问题

1. **`vllm` / CUDA 报错**  
   本流程假定 Linux + NVIDIA。确认 `nvidia-smi` 与 `pip show vllm`。

2. **OOM**  
   减小 `EVAL_BATCH_SIZE`（默认 8）、换更小 `EVAL_MODEL`，或减少 `TP` 占用的并行。

3. **score 找不到 caption**  
   必须先有 `outputs/<model>/caption/prompt.json`；`score.py` 参数与 caption 时同一 alias / `model_safe` 名。

4. **只测几张图**  
   把少量图拷到临时目录并 `export IMAGE_ROOT=...`，再 caption + score。score 按 caption 里出现的 key 做 **partial** 评测（未覆盖的图不计入 C/P）。若用 `evaluate.py` 提交 JSON 并要求全集，加 `--require-full`。

5. **`per_category` 键名**  
   结果里是完整类名：`Scene & Object` / `People & Activity` / `Text & Interface` / `Design & Knowledge`（不是 `SO`/`PA`/`TI`/`DK` 前缀）。

6. **HF 数据**  
   https://huggingface.co/datasets/LiuzhipengUCAS/CAPEval  

---

## 5. 推荐最短命令清单（复制用）

```bash
cd CAPEval && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export EVAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct   # 冒烟；正式改回 72B
export TP_SIZE=1

python caption.py internvl1b
python score.py internvl1b
python -c "import json; print(json.load(open('outputs/OpenGVLab_InternVL3_5-1B/metrics/results_summary.json')))"
```

正式：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_MODEL=Qwen/Qwen2.5-72B-Instruct
export TP_SIZE=4
python pipeline.py <your_model_alias>
```
