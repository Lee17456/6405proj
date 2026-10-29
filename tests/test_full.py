"""Full-campaign protocol checks with temporary metadata; no model downloads."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proj6405 import common, full


class FullPlanTests(unittest.TestCase):
    def setUp(self):
        self.settings = full.load_settings(common.ROOT / "configs/full.json")
        self.plan = full.make_plan(self.settings)

    def test_full_matrix_and_controlled_length_settings(self):
        self.assertEqual(len(self.plan), 20)
        self.assertEqual(len({r["options"]["name"] for r in self.plan}), 20)
        self.assertEqual({r["options"]["seed"] for r in self.plan}, {42})
        self.assertEqual(sum(r["options"]["model"] == "svm" for r in self.plan), 6)
        for run in self.plan:
            opts = run["options"]
            self.assertEqual((opts["limit_train"], opts["limit_val"]), (0, 0))
            self.assertEqual(opts["batch_size"] * opts["grad_accum"], 16)
            if run["stage"] == "long":
                self.assertEqual(opts["dataset"], "scotus")
                self.assertTrue(opts["gradient_checkpointing"])
                self.assertEqual(opts["batch_size"], 1)
        for length in (512, 1024, 2048, 4096):
            runs = [r for r in self.plan if r["stage"] == "long" and r["options"]["max_length"] == length]
            self.assertEqual(len(runs), 2)

    def test_preparation_evaluation_benchmark_and_separate_reports(self):
        stages = ["core", "long"]
        prepare = full.jobs_for("prepare", self.plan, "full", stages, 1, [42])
        self.assertEqual(len(prepare), 6)
        self.assertTrue(all("--limit-train" not in cmd for _, _, cmd in prepare))
        evaluation = full.jobs_for("evaluate", self.plan, "full", stages, 1, [42])
        self.assertEqual(len(evaluation), 20)
        self.assertTrue(all("--limit" not in cmd for _, _, cmd in evaluation))
        benchmarks = full.jobs_for("benchmark", self.plan, "full", stages, 1, [42])
        self.assertEqual(len(benchmarks), 20)
        self.assertTrue(all(run["options"]["seed"] == 42 for run, _, _ in benchmarks))
        reports = full.jobs_for("report", self.plan, "full", stages, 1, [42])
        self.assertNotEqual(reports[0][2][-1], reports[1][2][-1])

    def test_frozen_campaign_rejects_settings_and_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            full.freeze_plan(dest, self.settings, self.plan, {"dataset": "v1"})
            full.freeze_plan(dest, self.settings, self.plan, {"dataset": "v1"})
            with self.assertRaises(ValueError):
                full.freeze_plan(dest, self.settings, self.plan, {"dataset": "v2"})
            changed = copy.deepcopy(self.settings)
            changed["epochs"] = 9
            with self.assertRaises(ValueError):
                full.freeze_plan(dest, changed, full.make_plan(changed), {"dataset": "v1"})

    def test_mutual_exclusion_lock_is_cleaned_after_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with self.assertRaises(ValueError):
                with full.campaign_lock(dest):
                    with self.assertRaises(RuntimeError):
                        with full.campaign_lock(dest):
                            self.fail("Second campaign acquired lock")
                    raise ValueError("simulated failure")
            self.assertFalse((dest / "active.lock").exists())

    def test_full_inputs_and_completed_run_must_match(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, "DATA", Path(tmp) / "data"), patch.object(common, "RUNS", Path(tmp) / "runs"):
            run = self.plan[0]  # CPU SVM, native input.
            base = common.data_path("ag_news")
            manifest = {"signature": "v1", "splits": {s: {"n": 10} for s in ("train", "validation", "test")}}
            common.write_json(base / "manifest.json", manifest)
            self.assertEqual(full.matching_state(run), "missing")
            config = {**full.expected_config(run), "status": "completed", "data_signature": "v1"}
            common.write_json(common.run_path(config["name"]) / "config.json", config)
            self.assertEqual(full.matching_state(run), "completed")
            manifest["subset_limits"] = {"train": 5, "validation": 0, "test": 0}
            common.write_json(base / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                full.matching_state(run)
            manifest.pop("subset_limits")
            manifest["signature"] = "v2"
            common.write_json(base / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                full.matching_state(run)

    def test_retry_archive_preserves_files_and_stays_in_runs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, "RUNS", Path(tmp) / "runs"), patch.object(full, "RUNS", Path(tmp) / "runs"):
            path = common.run_path("interrupted")
            common.write_json(path / "config.json", {"status": "training"})
            (path / "checkpoint.txt").write_text("keep checkpoint")
            moved = full.archive_incomplete("interrupted")
            self.assertFalse(path.exists())
            self.assertEqual((moved / "checkpoint.txt").read_text(), "keep checkpoint")
            self.assertTrue(moved.resolve().is_relative_to(common.RUNS.resolve()))
            with self.assertRaises(ValueError):
                full.archive_incomplete("../outside")


if __name__ == "__main__":
    unittest.main()
