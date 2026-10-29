import sys

import pytest

from proj6405 import common


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
    monkeypatch.setattr(common, "DATA", tmp_path / "data")
    monkeypatch.setattr(common, "RUNS", tmp_path / "runs")
    features = Features({"id": Value("string"), "text": Value("string"), "label": ClassLabel(names=["negative", "positive"])})
    splits = {}
    for split, count in (("train", 40), ("validation", 12), ("test", 12)):
        splits[split] = Dataset.from_dict({"id": [f"{split}:{i}" for i in range(count)],
                                          "text": [f"bad boring poor movie {i}" if i % 2 == 0 else f"good excellent lovely movie {i}" for i in range(count)],
                                          "label": [i % 2 for i in range(count)]}, features=features)
    base = common.data_path("ag_news")
    DatasetDict(splits).save_to_disk(str(base / "dataset"))
    manifest = {"dataset": "ag_news", "view": "native", "label_names": ["negative", "positive"], "num_labels": 2,
                "splits": {s: {"n": len(d)} for s, d in splits.items()}, "signature": "synthetic-test-fixture"}
    common.write_json(base / "manifest.json", manifest)
    return tmp_path


def call_main(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()


def test_subset_is_fixed_across_calls(fixture_data):
    a, _ = common.load_rows("ag_news", limit=20)
    b, _ = common.load_rows("ag_news", limit=20)
    test, _ = common.load_rows("ag_news", split="test")
    assert a["id"] == b["id"]
    assert set(a["id"]).isdisjoint(test["id"])
    assert set(a["label"]) == {0, 1}


def test_singleton_class_can_be_sampled_without_losing_it():
    from datasets import Dataset
    rows = Dataset.from_dict({"id": list(range(16)), "label": [0] * 10 + [1] * 5 + [2]})
    a = common.subset_rows(rows, 8)
    b = common.subset_rows(rows, 8)
    assert len(a) == 8
    assert a["id"] == b["id"]
    assert set(a["label"]) == {0, 1, 2}
    assert len(set(a["id"])) == 8
    with pytest.raises(ValueError):
        common.subset_rows(rows, 2)


def test_budget_views_keep_test_full_and_match_training_ids(fixture_data, monkeypatch):
    from proj6405 import views, train as training, evaluate
    class CharacterTokenizer:
        def __call__(self, text, **kwargs):
            offsets = [(i, i + 1) for i in range(len(text))]
            if kwargs.get("truncation"):
                offsets = offsets[:kwargs["max_length"] - 2]
            return {"input_ids": list(range(len(offsets) + 2)), "offset_mapping": [(0, 0), *offsets, (0, 0)]}
    models = fixture_data / "models"
    for name in ("roberta", "mamba", "longformer"):
        common.write_json(models / name / "download_manifest.json", {"revision": "test"})
    monkeypatch.setattr(views, "MODELS", models)
    monkeypatch.setattr(views, "view_tokenizers", lambda manifest: [CharacterTokenizer()])
    selected, _ = common.load_rows("ag_news", limit=20)
    for length in (512, 2048):
        view = f"h6_shared{length}"
        call_main(monkeypatch, views, "--dataset", "ag_news", "--budget", str(length), "--name", view,
                  "--limit-train", "20", "--limit-val", "6")
        train, manifest = common.load_rows("ag_news", view)
        val, _ = common.load_rows("ag_news", view, "validation")
        test, _ = common.load_rows("ag_news", view, "test")
        assert train["id"] == selected["id"]
        assert len(val) == 6
        assert len(test) == manifest["splits"]["test"]["n"] == 12
        # Original counts remain available; this is a bounded training view.
        assert manifest["splits"]["train"]["n"] == 40
        prepared = common.read_json(common.data_path("ag_news", view) / "prepared_splits.json")
        assert prepared["train"]["n"] == 20
        assert set(train["id"]).isdisjoint(test["id"])
    call_main(monkeypatch, training, "--model", "svm", "--dataset", "ag_news", "--name", "h6_fixture",
              "--view", "h6_shared512", "--limit-train", "20", "--limit-val", "6", "--epochs", "2")
    call_main(monkeypatch, evaluate, "--run", "h6_fixture", "--device", "cpu")
    metrics = common.read_json(common.run_path("h6_fixture") / "metrics_test.json")
    config = common.read_json(common.run_path("h6_fixture") / "config.json")
    assert metrics["full_split"] is True and metrics["n"] == 12
    assert config["n_train"] == 20 and config["n_validation"] == 6


def test_svm_evaluation_benchmark_and_api(fixture_data, monkeypatch):
    from fastapi.testclient import TestClient
    from proj6405 import api, benchmark, evaluate, train
    call_main(monkeypatch, train, "--model", "svm", "--dataset", "ag_news", "--name", "fixture_svm")
    path = common.run_path("fixture_svm")
    assert not (path / "metrics_test.json").exists()
    call_main(monkeypatch, evaluate, "--run", "fixture_svm", "--device", "cpu")
    assert common.read_json(path / "metrics_test.json")["macro_f1"] == 1.0
    call_main(monkeypatch, evaluate, "--run", "fixture_svm", "--device", "cpu", "--limit", "6")
    assert common.read_json(path / "metrics_test.json")["n"] == 12
    assert (path / "metrics_test_subset6.json").exists()
    call_main(monkeypatch, benchmark, "--run", "fixture_svm", "--device", "cpu", "--samples", "12", "--warmup", "1", "--repeats", "2")
    result = common.read_json(next(path.glob("benchmark_*.json")))
    assert result["peak_allocated_mb"] is None
    assert result["model_documents_per_second"] > 0
    assert result["measured_documents_including_repeats"] == 24
    monkeypatch.setattr(api, "RUNS", common.RUNS)
    api.cache.update(name=None, predictor=None)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/runs").json()) == 1
        response = client.post("/predict", json={"run": "fixture_svm", "text": "excellent lovely movie"})
        assert response.status_code == 200
        assert response.json()["labels"] == ["positive"]
        assert client.post("/predict", json={"run": "fixture_svm", "text": "  "}).status_code == 422
        assert client.post("/predict", json={"run": "unknown", "text": "hello"}).status_code == 404
        assert client.get("/results").json()[0]["test_metrics"]["full_split"] is True
    api.cache.update(name=None, predictor=None)
    with pytest.raises(FileExistsError):
        call_main(monkeypatch, train, "--model", "svm", "--dataset", "ag_news", "--name", "fixture_svm")


def test_report_single_seed_no_fake_std(fixture_data, monkeypatch):
    from proj6405 import evaluate, report, train
    call_main(monkeypatch, train, "--model", "svm", "--dataset", "ag_news", "--name", "report_svm")
    call_main(monkeypatch, evaluate, "--run", "report_svm", "--device", "cpu")
    monkeypatch.setattr(report, "RUNS", common.RUNS)
    monkeypatch.setattr(report, "REPORTS", fixture_data / "reports")
    call_main(monkeypatch, report)
    import pandas as pd
    row = pd.read_csv(fixture_data / "reports" / "summary.csv").iloc[0]
    assert row.n_seeds == 1
    assert pd.isna(row.macro_f1_std)


def test_full_commands_train_all_fixture_rows_and_evaluate(fixture_data, monkeypatch):
    from proj6405 import full, train, evaluate
    settings = full.load_settings(common.ROOT / "configs/full.json")
    run = full.make_plan(settings)[0]
    cmd = full.command("train", run["options"])
    call_main(monkeypatch, train, *cmd[3:])
    assert full.matching_state(run) == "completed"
    config = common.read_json(common.run_path(run["options"]["name"]) / "config.json")
    assert config["n_train"] == 40 and config["n_validation"] == 12
    call_main(monkeypatch, evaluate, "--run", config["name"], "--device", "cpu")
    metrics = common.read_json(common.run_path(config["name"]) / "metrics_test.json")
    assert metrics["full_split"] is True and metrics["n"] == 12


def test_report_three_seeds_and_output_directory(fixture_data, monkeypatch):
    import pandas as pd
    from proj6405 import full, report
    settings = full.load_settings(common.ROOT / "configs/full.json")
    settings["seeds"] = [42, 43, 44]  # Explicit fixture for optional multi-seed reporting.
    plan = full.make_plan(settings)
    runs = [r for r in plan if r["stage"] == "long" and r["options"]["model"] == "mamba" and r["options"]["max_length"] == 512]
    for run, f1 in zip(runs, [0.5, 0.6, 0.7]):
        c = {**full.expected_config(run), "status": "completed", "n_train": 40,
             "data_signature": "synthetic-v1", "train_ids_signature": "synthetic-train", "train_seconds": 10.0}
        base = common.run_path(c["name"])
        common.write_json(base / "config.json", c)
        common.write_json(base / "metrics_test.json", {"precision": "bf16", "evaluated_ids_signature": "synthetic-test",
                          "accuracy": f1, "macro_precision": f1, "macro_recall": f1, "macro_f1": f1,
                          "n": 12, "confusion_matrix": [[4, 2], [2, 4]], "full_split": True})
    monkeypatch.setattr(report, "RUNS", common.RUNS)
    monkeypatch.setattr(report, "REPORTS", fixture_data / "reports")
    default = fixture_data / "reports/results.csv"
    default.parent.mkdir()
    default.write_text("preserve default report")
    dest = fixture_data / "reports/full/long"
    call_main(monkeypatch, report, "--prefix", "full_long_", "--output-dir", str(dest))
    row = pd.read_csv(dest / "summary.csv").iloc[0]
    assert row.n_seeds == 3
    assert row.macro_f1_mean == pytest.approx(0.6)
    assert row.macro_f1_std == pytest.approx(0.1)
    assert row.macro_precision_std == pytest.approx(0.1)
    assert (dest / "length_f1_scotus.png").exists()
    assert (dest / "f1_summary_scotus.png").exists()
    assert default.read_text() == "preserve default report"


def test_api_campaign_filter(fixture_data, monkeypatch):
    from proj6405 import api
    for name in ("full_core_fixture", "pilot_fixture", "h6_fixture"):
        common.write_json(common.run_path(name) / "config.json", {"name": name, "status": "completed", "model": "svm"})
    monkeypatch.setattr(api, "RUNS", common.RUNS)
    monkeypatch.setenv("RUN_PREFIX", "full_")
    assert [r["name"] for r in api.completed_runs()] == ["full_core_fixture"]


@pytest.mark.parametrize("kind", ["roberta", "mamba", "longformer"])
def test_neural_training_loop(fixture_data, monkeypatch, kind):
    pytest.importorskip("torch")
    from transformers import AutoModel, MambaConfig, LongformerConfig, RobertaConfig, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from proj6405 import train, evaluate
    base = dict(vocab_size=16, hidden_size=16, num_hidden_layers=1, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    if kind == "mamba":
        config = MambaConfig(**base, state_size=4, expand=2)
    else:
        base.update(num_attention_heads=2, intermediate_size=32, max_position_embeddings=66)
        config = RobertaConfig(**base) if kind == "roberta" else LongformerConfig(**base, attention_window=[4])
    source = fixture_data / "models" / kind
    AutoModel.from_config(config).save_pretrained(source)
    vocabulary = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "good": 4, "bad": 5, "movie": 6, "excellent": 7, "poor": 8}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    raw.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", bos_token="[CLS]", eos_token="[SEP]", unk_token="[UNK]")
    tokenizer.save_pretrained(source)
    common.write_json(source / "download_manifest.json", {"revision": "random-local-fixture"})
    monkeypatch.setattr(train, "MODELS", fixture_data / "models")
    name = f"fixture_{kind}"
    call_main(monkeypatch, train, "--model", kind, "--dataset", "ag_news", "--name", name,
              "--device", "cpu", "--precision", "fp32", "--max-length", "16", "--epochs", "1",
              "--batch-size", "6", "--grad-accum", "3", "--allow-slow-mamba", "--gradient-checkpointing")
    config = common.read_json(common.run_path(name) / "config.json")
    assert config["status"] == "completed"
    assert config["best_epoch"] == 1
    call_main(monkeypatch, evaluate, "--run", name, "--device", "cpu", "--precision", "fp32")
    metrics = common.read_json(common.run_path(name) / "metrics_test.json")
    assert metrics["n"] == 12
    assert 0 <= metrics["macro_f1"] <= 1
