from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from streamlit.testing.v1 import AppTest


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_ui_empty_state():
    with patch("requests.get", return_value=Response([])):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert not app.exception
        assert len(app.info) == 1


def test_ui_prediction_and_results():
    run = {"name": "fixture", "model": "svm", "dataset": "ag_news", "view": "native", "max_length": None,
           "seed": 42, "n_train": 40, "label_names": ["negative", "positive"]}
    metrics = dict(accuracy=0.8, macro_precision=0.8, macro_recall=0.8, macro_f1=0.8)

    def get(url, **kwargs):
        if url.endswith("/runs"):
            return Response([run])
        if url.endswith("/results"):
            return Response([{**run, "test_metrics": metrics, "benchmarks": []}])
        return Response([])

    predicted = dict(labels=["positive"], input_tokens=[None], model_ms=1.0, preprocess_ms=1.0,
                     local_pipeline_ms=2.0, load_and_warmup_ms=3.0)
    with patch("requests.get", side_effect=get), patch("requests.post", return_value=Response(predicted)):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert not app.exception
        app.button(key="example_ag_news").click().run()
        app.button(key="compare").click().run()
        assert not app.exception
        assert any("Prediction" in frame.value.columns for frame in app.dataframe)


def fixture_run(model="mamba", dataset="ag_news", view="shared512", seed=42, stage="core"):
    return {"name": f"full_{stage}_{dataset}_{model}_{view}_s{seed}", "model": model,
            "dataset": dataset, "view": view, "seed": seed, "max_length": 512,
            "n_train": 40, "label_names": ["1", "2"] if dataset == "scotus" else ["negative", "positive"]}


def fixture_get(runs, errors=None, results_fail=False):
    def get(url, **kwargs):
        if url.endswith("/runs"):
            return Response(runs)
        if url.endswith("/results"):
            if results_fail:
                raise requests.ConnectionError("simulated results endpoint failure")
            metrics = dict(accuracy=0.8, macro_precision=0.8, macro_recall=0.8, macro_f1=0.8, n=10, full_split=True)
            return Response([{**r, "test_metrics": metrics, "benchmarks": []} for r in runs])
        return Response(errors or [])
    return get


def test_default_models_and_results_follow_dataset_and_input_view():
    runs = [fixture_run("mamba", seed=s) for s in (42, 43, 44)]
    runs += [fixture_run("svm"), fixture_run("roberta"), fixture_run("mamba", stage="long"),
             fixture_run("svm", view="native"), fixture_run("mamba", dataset="imdb")]
    with patch("requests.get", side_effect=fixture_get(runs)):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert not app.exception
        selected = app.multiselect[0].value
        assert len(selected) == 3
        assert {next(r["model"] for r in runs if r["name"] == name) for name in selected} == {"svm", "roberta", "mamba"}
        assert all(name.endswith("s42") and "_core_" in name for name in selected)
        app.selectbox(key="view_ag_news").select("native").run()
        scores = next(d.value for d in app.dataframe if "Macro-F1" in d.value.columns)
        assert scores["Run ID"].tolist() == [fixture_run("svm", view="native")["name"]]
        app.selectbox(key="dataset").select("imdb").run()
        assert not app.exception
        scores = next(d.value for d in app.dataframe if "Macro-F1" in d.value.columns)
        assert scores["Run ID"].tolist() == [fixture_run("mamba", dataset="imdb")["name"]]


def test_predictions_persist_and_stale_input_is_marked_without_repeating_requests():
    run = fixture_run()
    predicted = dict(labels=["positive"], input_tokens=[8], model_ms=1.0, preprocess_ms=1.0,
                     local_pipeline_ms=2.0, load_and_warmup_ms=3.0)
    with patch("requests.get", side_effect=fixture_get([run])), patch("requests.post", return_value=Response(predicted)) as post:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert app.button(key="compare").disabled
        app.button(key="example_ag_news").click().run()
        assert app.text_area[0].value
        app.button(key="compare").click().run()
        app.selectbox(key="quality_metric").select("Accuracy").run()
        assert post.call_count == 1
        assert any("Prediction" in d.value.columns for d in app.dataframe)
        app.text_area[0].set_value("A different input.").run()
        assert not app.exception
        assert any("previous request" in i.value for i in app.info)
        assert post.call_count == 1


def test_scotus_english_labels_and_raw_codes_in_prediction_and_errors():
    run = fixture_run(dataset="scotus")
    errors = [{"id": "scotus:test:0", "true_label": "1", "predicted_label": "999", "text_preview": "A legal opinion."}]
    predicted = dict(labels=["1"], input_tokens=[8], model_ms=1.0, preprocess_ms=1.0,
                     local_pipeline_ms=2.0, load_and_warmup_ms=3.0)
    with patch("requests.get", side_effect=fixture_get([run], errors)), patch("requests.post", return_value=Response(predicted)):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.button(key="example_scotus").click().run()
        app.button(key="compare").click().run()
        assert not app.exception
        prediction = next(d.value for d in app.dataframe if "Prediction" in d.value.columns)
        assert prediction.iloc[0]["Prediction"] == "Criminal Procedure · code 1"
        mistakes = next(d.value for d in app.dataframe if "Expected label" in d.value.columns)
        assert mistakes.iloc[0]["Predicted label"] == "Unknown category · code 999"


def test_results_failure_does_not_prevent_prediction():
    predicted = dict(labels=["positive"], input_tokens=[8], model_ms=1.0, preprocess_ms=1.0,
                     local_pipeline_ms=2.0, load_and_warmup_ms=3.0)
    with patch("requests.get", side_effect=fixture_get([fixture_run()], results_fail=True)), patch("requests.post", return_value=Response(predicted)):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.button(key="example_ag_news").click().run()
        app.button(key="compare").click().run()
        assert not app.exception
        assert any("Evaluation results could not be loaded" in e.value for e in app.error)
        assert any("Prediction" in d.value.columns for d in app.dataframe)


def test_disconnected_backend_has_recovery_action():
    with patch("requests.get", side_effect=requests.ConnectionError("fixture offline")):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert not app.exception
        assert app.button(key="retry_connection")
        assert any("inference service is unavailable" in e.value for e in app.error)


def test_upload_is_explicit_and_pasted_draft_survives_mode_changes():
    upload = SimpleNamespace(size=18, getvalue=lambda: b"Uploaded document.")
    predicted = dict(labels=["positive"], input_tokens=[8], model_ms=1.0, preprocess_ms=1.0,
                     local_pipeline_ms=2.0, load_and_warmup_ms=3.0)
    with patch("requests.get", side_effect=fixture_get([fixture_run()])), patch("requests.post", return_value=Response(predicted)) as post:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.text_area[0].set_value("Keep this pasted draft.").run()
        with patch("streamlit.file_uploader", return_value=upload):
            app.radio[0].set_value("Upload file").run()
            app.button(key="compare").click().run()
            assert post.call_args.kwargs["json"]["text"] == "Uploaded document."
        app.radio[0].set_value("Paste text").run()
        assert not app.exception
        assert app.text_area[0].value == "Keep this pasted draft."
        invalid = SimpleNamespace(size=2, getvalue=lambda: b"\xff\xfe")
        with patch("streamlit.file_uploader", return_value=invalid):
            app.radio[0].set_value("Upload file").run()
            assert not app.exception
            assert app.button(key="compare").disabled
            assert any("UTF-8" in e.value for e in app.error)


def test_benchmark_filters_preserve_measured_units_and_cpu_missing_memory():
    run = fixture_run("svm")
    bench = dict(batch_size=1, precision="fp32", device="cpu", model_ms_per_document=2.0,
                 pipeline_documents_per_second=100.0, peak_allocated_mb=None,
                 mean_actual_input_tokens=None, sample_ids_signature="fixture")
    def get(url, **kwargs):
        if url.endswith("/runs"):
            return Response([run])
        if url.endswith("/results"):
            return Response([{**run, "test_metrics": None, "benchmarks": [bench, {**bench, "batch_size": 4}]}])
        return Response([])
    with patch("requests.get", side_effect=get):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.selectbox(key="benchmark_batch").select(4).run()
        assert not app.exception
        table = next(d.value for d in app.dataframe if "Model time (ms/doc)" in d.value.columns)
        assert table["Batch size"].tolist() == [4]
        assert table["Peak GPU memory (MiB)"].isna().all()
