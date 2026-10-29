"""Budget orchestration checks; runnable without torch or downloaded model weights."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from proj6405 import budget, common


class BudgetTests(unittest.TestCase):
    def test_core_and_optional_long_are_matched(self):
        core = budget.make_plan("core", "h6")
        long = budget.make_plan("long", "h6")
        self.assertEqual(len(core), 9)
        self.assertEqual(len(long), 3)
        self.assertEqual(len({r["name"] for r in core + long}), 12)
        for dataset in ("ag_news", "imdb", "scotus"):
            models = [r["model"] for r in core if r["dataset"] == dataset]
            self.assertEqual(set(models), {"svm", "roberta", "mamba"})
        for model in ("mamba", "longformer"):
            lengths = {r["length"] for r in core + long if r["dataset"] == "scotus" and r["model"] == model}
            self.assertEqual(lengths, {512, 2048})
        for run in core + long:
            options = budget.train_options(run)
            self.assertEqual((options["limit_train"], options["limit_val"], options["epochs"], options["seed"]), (1000, 200, 2, 42))
            self.assertEqual(options["batch_size"] * options["grad_accum"], 16)

    def test_full_test_set_is_not_capped(self):
        for stage in ("core", "long"):
            for _, cmd in budget.make_commands(stage, "evaluate", "h6"):
                self.assertNotIn("--limit", cmd)
            for _, cmd in budget.make_commands(stage, "prepare", "h6"):
                self.assertNotIn("--limit-test", cmd)
                self.assertIn("--limit-train", cmd)

    def test_model_filter_applies_to_every_phase(self):
        for phase in ("train", "evaluate", "benchmark"):
            commands = budget.make_commands("core", phase, "h6", ["svm", "roberta"])
            self.assertEqual(len(commands), 6)
            self.assertFalse(any(run["model"] == "mamba" for run, _ in commands))

    def test_elapsed_time_is_shared_between_commands(self):
        # The second command receives 30s, not a fresh 60s allocation.
        ticks = iter([100, 101, 130])
        calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            budget.execute([(None, ["first"]), (None, ["second"])], "prepare", 1,
                           runner=lambda cmd, **kw: calls.append(kw["timeout"]), clock=lambda: next(ticks))
        self.assertEqual(calls, [59, 30])

    def test_exhausted_budget_starts_no_command(self):
        ticks = iter([100, 161])
        calls = []
        with self.assertRaises(TimeoutError):
            budget.execute([(None, ["unused"])], "prepare", 1,
                           runner=lambda *a, **kw: calls.append(a), clock=lambda: next(ticks))
        self.assertEqual(calls, [])

    def test_real_child_process_is_stopped_on_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "should_not_exist.txt"
            code = "import sys,time,pathlib; time.sleep(5); pathlib.Path(sys.argv[1]).write_text('too late')"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired):
                budget.execute([(None, [sys.executable, "-c", code, str(marker)])], "prepare", 0.005)
            self.assertFalse(marker.exists())

    def test_resume_rejects_changed_data_or_parameters(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(common, "RUNS", Path(temp) / "runs"), patch.object(common, "DATA", Path(temp) / "data"):
            run = budget.make_plan("core", "h6")[0]
            config = budget.train_options(run)
            config.update(status="completed", max_length=None, precision="fp32", device="cpu", data_signature="v1",
                          gradient_checkpointing=False, pooling=None, dropout=0.1, weight_decay=0.01,
                          warmup_ratio=0.1, attention_backend="sdpa", allow_slow_mamba=False,
                          svm_c=1.0, ngram_max=2, max_features=100000)
            common.write_json(common.run_path(run["name"]) / "config.json", config)
            manifest_path = common.data_path(run["dataset"], run["view"]) / "manifest.json"
            common.write_json(manifest_path, {"signature": "v1"})
            self.assertTrue(budget.is_completed(run))
            config["epochs"] = 3
            common.write_json(common.run_path(run["name"]) / "config.json", config)
            with self.assertRaises(ValueError):
                budget.is_completed(run)
            config["epochs"] = 2
            common.write_json(common.run_path(run["name"]) / "config.json", config)
            common.write_json(manifest_path, {"signature": "v2"})
            with self.assertRaises(ValueError):
                budget.is_completed(run)


if __name__ == "__main__":
    unittest.main()
