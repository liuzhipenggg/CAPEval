# Framework integrations

CAPEval keeps checklist judging in-repo. Host harnesses only **generate captions**, then call `capeval.api.score_caption_map`.

| Path | Framework |
|------|-----------|
| [`vlmevalkit/`](vlmevalkit/) | [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) |
| [`lmms_eval/`](lmms_eval/) | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) |

Shared API: [`capeval/api.py`](../capeval/api.py).

```python
from capeval.api import caption_prompt, score_caption_map

prompt = caption_prompt()  # "Analyze the image in a comprehensive and detailed manner."
# ... run your VLM ...
result = score_caption_map({"SO001.jpg": "..."}, model_id="my_model")
print(result["summary"])  # C, P (0–100)
```
