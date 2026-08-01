"""Smoke tests for CAPEval layout and schema."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LayoutSmokeTest(unittest.TestCase):
    def test_core_files(self) -> None:
        for rel in (
            "caption.py",
            "score.py",
            "pipeline.py",
            "evaluate.py",
            "capeval/__init__.py",
            "capeval/config.py",
            "capeval/models.py",
            "capeval/caption/__init__.py",
            "capeval/judge.py",
            "capeval/schema.py",
            "capeval/tags.py",
            "capeval/llm/__init__.py",
            "capeval/score/__init__.py",
            "capeval/metrics/__init__.py",
            "capeval/util/__init__.py",
            "capeval/cli/__init__.py",
            "capeval/cli/evaluate_submission.py",
            "capeval/io.py",
            "capeval/prompts.py",
            "capeval/tools/merge_shards.py",
            "capeval/tools/plot_cp_scatter.py",
            "capeval/api.py",
            "capeval/benchmark.py",
            "capeval/submission.py",
            "examples/submission.json",
            "examples/demo.py",
            "assets/logo.png",
        ):
            self.assertTrue((ROOT / rel).is_file(), msg=rel)
        self.assertFalse((ROOT / "assets" / "overview.png").exists())
        self.assertFalse((ROOT / "assets" / "dataset.png").exists())
        for gone in ("leaderboard", "demo"):
            self.assertFalse((ROOT / gone).exists(), msg=gone)
        self.assertTrue((ROOT / "docs" / "index.html").is_file())
        self.assertTrue((ROOT / "docs" / "styles.css").is_file())
        self.assertTrue((ROOT / "docs" / "assets" / "figure1.png").is_file())
        self.assertTrue((ROOT / "docs" / "assets" / "figure2.png").is_file())
        self.assertTrue((ROOT / "assets" / "logo.png").is_file())
        self.assertFalse((ROOT / "docs" / "assets" / "logo.png").exists())
        self.assertTrue((ROOT / "docs" / "leaderboard" / "index.html").is_file())
        self.assertTrue((ROOT / "docs" / "leaderboard" / "data.json").is_file())
        self.assertFalse((ROOT / "hf_dataset").exists())

    def test_no_legacy_tops(self) -> None:
        for name in (
            "env.sh",
            "scripts",
            "pipeline",
            "captioner",
            "checklist",
            "dataset",
            "track1",
            "bin",
            "paper",
            "capeval/__main__.py",
            "capeval/run_caption.py",
            "capeval/run_score.py",
            "rank.py",
            "capeval/rank.py",
            "run_caption.sh",
            "run_eval.sh",
            "run_pipeline.sh",
        ):
            self.assertFalse((ROOT / name).exists(), msg=name)

    def test_data_counts(self) -> None:
        images = [
            p
            for p in (ROOT / "data" / "image").iterdir()
            if p.is_file() and p.name != ".DS_Store"
        ]
        self.assertEqual(len(images), 300)
        gt = [
            ln
            for ln in (ROOT / "data" / "gt_caption.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        chk = [
            ln
            for ln in (ROOT / "data" / "checklist.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        self.assertEqual(len(gt), 300)
        self.assertEqual(len(chk), 300)
        row = json.loads(gt[0])
        self.assertEqual(set(row.keys()), {"id", "img_path", "gt_caption"})
        self.assertTrue(str(row["img_path"]).startswith(("SO", "PA", "TI", "DK")))
        self.assertTrue(row["gt_caption"].strip())
        self.assertNotIn("lzp_data", row["id"])
        cl = json.loads(chk[0])
        self.assertNotIn("detailed_description", cl)
        self.assertNotIn("overview_description", cl)
        self.assertNotIn("category", cl)
        self.assertEqual(row["id"], cl["id"])
        self.assertEqual(row["img_path"], cl["img_path"])


class SchemaSmokeTest(unittest.TestCase):
    def test_import_flatten(self) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from capeval.schema import CHECKLIST_KEYS, flatten_items

        self.assertEqual(len(CHECKLIST_KEYS), 8)
        items = flatten_items(
            {
                "img_path": "SO001.jpg",
                "instance_checklist": [
                    {"Question": "Does the caption mention a cat?", "Tags": "object"}
                ],
            }
        )
        self.assertEqual(len(items), 1)

    def test_public_api(self) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from capeval import CAPEval, caption_prompt, score_caption_map
        from capeval.api import list_inference_rows
        from capeval.metrics.aggregate import _cp_percent
        from capeval.metrics.report import write_results_summaries
        from capeval.submission import load_submission, summary_from_metrics

        self.assertEqual(
            caption_prompt(),
            "Analyze the image in a comprehensive and detailed manner.",
        )
        self.assertTrue(callable(score_caption_map))
        self.assertTrue(callable(CAPEval))
        c, p = _cp_percent(yes=80, no=20, total=200)
        self.assertAlmostEqual(c, 50.0)
        self.assertAlmostEqual(p, 80.0)
        fake = {
            "eval_model": "unit-test-judge",
            "scale": "percent",
            "models": {
                "demo": {
                    "summary": {
                        "C": 50.0,
                        "P": 80.0,
                        "yes1": 80,
                        "no1": 20,
                        "not_mentioned1": 100,
                        "total1": 200,
                        "n_images": 2,
                        "n_error": 1,
                    },
                    "per_category": {},
                }
            },
        }
        compact = summary_from_metrics(fake, model_id="demo")
        self.assertEqual(compact["C"], 50.0)
        self.assertEqual(compact["P"], 80.0)
        self.assertEqual(compact["n_error"], 1)
        self.assertNotIn("U", compact)
        self.assertNotIn("G", compact)
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            paths = write_results_summaries(td, fake)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("results_summary.json"))
            loaded = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["C"], 50.0)
            self.assertEqual(loaded["P"], 80.0)

            slashy = {
                "eval_model": "unit-test-judge",
                "scale": "percent",
                "models": {
                    "Org/Model": {
                        "summary": {
                            "C": 1.0,
                            "P": 2.0,
                            "yes1": 1,
                            "no1": 0,
                            "not_mentioned1": 0,
                            "total1": 1,
                            "n_images": 1,
                        },
                        "per_category": {},
                    },
                    "../escape": {
                        "summary": {
                            "C": 3.0,
                            "P": 4.0,
                            "yes1": 1,
                            "no1": 0,
                            "not_mentioned1": 0,
                            "total1": 1,
                            "n_images": 1,
                        },
                        "per_category": {},
                    },
                },
            }
            mp = Path(td) / "metrics.json"
            mp.write_text(json.dumps(slashy), encoding="utf-8")
            from capeval.metrics.report import write_ranking_json_sidecars

            write_ranking_json_sidecars(td)
            self.assertTrue((Path(td) / "Org_Model.json").is_file())
            self.assertTrue((Path(td) / "escape.json").is_file())
            for p in Path(td).iterdir():
                if p.suffix == ".json":
                    self.assertEqual(p.resolve().parent, Path(td).resolve())
        rows = list_inference_rows()
        self.assertEqual(len(rows), 300)
        self.assertIn("image_id", rows[0])
        self.assertIn("question", rows[0])
        caps = load_submission(ROOT / "examples" / "submission.json")
        self.assertGreaterEqual(len(caps), 1)
        self.assertTrue((ROOT / "integrations" / "vlmevalkit" / "capeval_dataset.py").is_file())
        self.assertTrue((ROOT / "integrations" / "lmms_eval" / "capeval.yaml").is_file())

    def test_custom_model_passthrough(self) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from capeval.models import list_aliases, resolve_spec

        self.assertGreaterEqual(len(list_aliases()), 40)
        spec = resolve_spec("Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(spec.model_id, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(spec.backend, "vllm")
        local = resolve_spec("/tmp/my-vlm-ckpt")
        self.assertEqual(local.model_id, "/tmp/my-vlm-ckpt")



class CliSmokeTest(unittest.TestCase):
    def test_caption_requires_model(self) -> None:
        proc = subprocess.run(
            [sys.executable, "caption.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "DRY_RUN": "1"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("No caption model specified", proc.stderr + proc.stdout)

    def test_score_requires_model(self) -> None:
        proc = subprocess.run(
            [sys.executable, "score.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("No model specified for scoring", proc.stderr + proc.stdout)

    def test_evaluate_check_only(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "evaluate.py",
                "--submission",
                "examples/submission.json",
                "--check-only",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ},
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        self.assertIn("Submission is valid", proc.stdout)

    def test_caption_help_dry_run(self) -> None:
        out = subprocess.check_output(
            [sys.executable, "caption.py", "internvl1b"],
            cwd=ROOT,
            text=True,
            env={**os.environ, "DRY_RUN": "1"},
        )
        self.assertIn("Expanded run order", out)
        self.assertIn("internvl1b", out)

    def test_config_defaults(self) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        for k in (
            "CAPEVAL_DATA_ROOT",
            "CAPEVAL_DATASET_ROOT",
            "CAPEVAL_OUTPUT_ROOT",
            "IMAGE_ROOT",
            "GT_JSONL",
            "GT_CAPTION",
            "CHECKLIST",
            "CAPEVAL_CAPTION_DIR",
            "CAPEVAL_EVAL_DIR",
            "PROMPT_NAME",
            "EVAL_MODEL",
            "TWO_SHARD_MODE",
            "GPU_LIST",
        ):
            os.environ.pop(k, None)
        from capeval.config import apply_defaults

        root = apply_defaults()
        self.assertEqual(root, ROOT)
        self.assertTrue(Path(os.environ["IMAGE_ROOT"]).is_dir())
        self.assertTrue(Path(os.environ["GT_CAPTION"]).is_file())
        self.assertEqual(Path(os.environ["GT_CAPTION"]).name, "gt_caption.jsonl")
        self.assertTrue(Path(os.environ["CHECKLIST"]).is_file())
        self.assertEqual(Path(os.environ["CHECKLIST"]).name, "checklist.jsonl")
        self.assertEqual(os.environ["GT_JSONL"], os.environ["GT_CAPTION"])
        self.assertEqual(
            os.environ["CAPEVAL_CAPTION_DIR"],
            os.environ["CAPEVAL_OUTPUT_ROOT"],
        )
        self.assertEqual(
            os.environ["CAPEVAL_EVAL_DIR"],
            os.environ["CAPEVAL_OUTPUT_ROOT"],
        )
        from capeval.util.paths import caption_json_path, model_metrics_dir

        self.assertTrue(
            str(caption_json_path("OpenGVLab/InternVL3_5-1B")).endswith(
                "/OpenGVLab_InternVL3_5-1B/caption/prompt.json"
            )
        )
        self.assertTrue(
            str(model_metrics_dir("my_vlm")).endswith("/my_vlm/metrics")
        )
        self.assertEqual(os.environ["PROMPT_NAME"], "PROMPT")
        self.assertEqual(os.environ["EVAL_MODEL"], "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(os.environ["TWO_SHARD_MODE"], "0")
        self.assertNotIn("VLM_MODEL", os.environ)


if __name__ == "__main__":
    unittest.main()
