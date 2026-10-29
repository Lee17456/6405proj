"""Local-only API. Run exactly one worker; GPU access is serialized by a lock."""
import gc
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import RUNS, read_json, run_path

app = FastAPI(title="6405 Text Classification Lab", version="1.0")
lock = threading.Lock()
cache = {"name": None, "predictor": None}


def completed_runs():
    records = []
    for file in sorted(RUNS.glob("*/config.json")):
        if not file.parent.name.startswith(os.environ.get("RUN_PREFIX", "")):
            continue
        config = read_json(file)
        if config.get("status") == "completed":
            records.append({k: config.get(k) for k in
                            ("name", "model", "dataset", "view", "max_length", "seed", "label_names",
                             "limit_train", "n_train", "precision", "best_validation_macro_f1")})
    return records


class PredictRequest(BaseModel):
    run: str = Field(min_length=1, max_length=150)
    text: str = Field(min_length=1, max_length=400000)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_run": cache["name"], "completed_runs": len(completed_runs())}


@app.get("/runs")
def runs():
    return completed_runs()


@app.get("/results")
def results():
    output = []
    for run in completed_runs():
        path = run_path(run["name"])
        metrics = path / "metrics_test.json"
        output.append({**run, "test_metrics": read_json(metrics) if metrics.exists() else None,
                       "benchmarks": [read_json(f) for f in sorted(path.glob("benchmark_*.json"))]})
    return output


@app.get("/errors/{name}")
def errors(name: str):
    import csv
    if name not in {r["name"] for r in completed_runs()}:
        raise HTTPException(404, "Unknown completed run")
    file = run_path(name) / "predictions_test.csv"
    if not file.exists():
        return []
    with file.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["correct"] == "False"][:30]


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(422, "Please provide nonempty English text")
    if request.run not in {r["name"] for r in completed_runs()}:
        raise HTTPException(404, "Unknown completed run")
    queued = time.perf_counter()
    with lock:
        entered = time.perf_counter()
        load_ms = 0.0
        try:
            if cache["name"] != request.run:
                cache.update(name=None, predictor=None)
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                from .inference import Predictor
                started = time.perf_counter()
                predictor = Predictor(request.run, os.environ.get("MODEL_DEVICE", "cuda"), os.environ.get("MODEL_PRECISION") or None)
                # Explicit warm-up outside the displayed prediction timer.
                predictor.predict_batch([request.text])
                predictor.synchronize()
                load_ms = (time.perf_counter() - started) * 1000
                cache.update(name=request.run, predictor=predictor)
            output = cache["predictor"].predict_batch([request.text])
            return {"run": request.run, **output, "queue_ms": (entered - queued) * 1000,
                    "load_and_warmup_ms": load_ms,
                    "note": "Single live observation; use offline benchmark for reported performance."}
        except Exception as exc:
            # Preserve a clear server-side traceback, return a concise client error.
            import logging
            logging.exception("Inference failed")
            cache.update(name=None, predictor=None)
            raise HTTPException(500, f"Inference failed: {type(exc).__name__}. Check the API log.") from exc
