import time

import numpy as np

from .common import read_json, run_path
from .views import transform_live_text, view_tokenizers


class Predictor:
    def __init__(self, name, device="cuda", precision=None):
        self.path = run_path(name)
        self.config = read_json(self.path / "config.json")
        if self.config["status"] != "completed":
            raise ValueError("Only completed runs can be loaded")
        self.manifest = read_json(self.path / "data_manifest.json")
        self.kind = self.config["model"]
        self.device = "cpu" if self.kind == "svm" else device
        self.precision = "fp32" if self.kind == "svm" else (precision or self.config["precision"])
        self.labels = self.config["label_names"]
        self.shared_tokenizers = None
        if self.kind == "svm":
            import joblib
            self.model = joblib.load(self.path / "svm.joblib")
        else:
            from safetensors.torch import load_file
            from .models import TextClassifier, tokenizer_from, mamba_fast_available
            if self.kind == "mamba" and (not device.startswith("cuda") or not mamba_fast_available()) and not self.config["allow_slow_mamba"]:
                raise RuntimeError("Optimized CUDA Mamba unavailable; this run requires it")
            self.tokenizer = tokenizer_from(self.path / "tokenizer")
            self.model = TextClassifier(self.path / "backbone", self.kind, len(self.labels), self.config["pooling"],
                                        self.config["dropout"], self.config["attention_backend"])
            self.model.head.load_state_dict(load_file(str(self.path / "head.safetensors")))
            self.model.to(device).eval()

    def synchronize(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()

    def prepare(self, texts, already_view=False):
        if not already_view and self.manifest["view"] != "native":
            if self.shared_tokenizers is None:
                self.shared_tokenizers = view_tokenizers(self.manifest)
            texts = [transform_live_text(t, self.manifest, self.shared_tokenizers) for t in texts]
        if self.kind == "svm":
            return self.model.named_steps["tfidf"].transform(texts), [None] * len(texts)
        encoded = self.tokenizer(texts, truncation=True, max_length=self.config["max_length"], padding=True,
                                 return_tensors="pt", return_token_type_ids=False)
        return encoded, encoded["attention_mask"].sum(1).tolist()

    def transfer(self, encoded):
        return encoded if self.kind == "svm" else {k: v.to(self.device) for k, v in encoded.items()}

    def forward(self, encoded):
        if self.kind == "svm":
            return self.model.named_steps["svm"].predict(encoded)
        import torch
        from .models import autocast_context
        with torch.inference_mode(), autocast_context(self.device, self.precision):
            return self.model(**encoded)

    def decode(self, output):
        if self.kind == "svm":
            return output.tolist()
        return output.argmax(-1).cpu().tolist()

    def predict_batch(self, texts, already_view=False):
        started = time.perf_counter()
        encoded, lengths = self.prepare(texts, already_view)
        prepared = time.perf_counter()
        encoded = self.transfer(encoded)
        self.synchronize()
        transferred = time.perf_counter()
        output = self.forward(encoded)
        self.synchronize()
        forwarded = time.perf_counter()
        predicted = self.decode(output)
        ended = time.perf_counter()
        return {"predicted_ids": predicted, "labels": [self.labels[i] for i in predicted], "input_tokens": lengths,
                "preprocess_ms": (prepared - started) * 1000, "transfer_ms": (transferred - prepared) * 1000,
                "model_ms": (forwarded - transferred) * 1000, "postprocess_ms": (ended - forwarded) * 1000,
                "local_pipeline_ms": (ended - started) * 1000}


def latency_summary(records):
    total_docs = sum(len(r["predicted_ids"]) for r in records)
    model_ms = sum(r["model_ms"] for r in records)
    pipeline_ms = sum(r["local_pipeline_ms"] for r in records)
    valid_tokens = [n for r in records for n in r["input_tokens"] if n is not None]
    return {"measured_documents_including_repeats": total_docs,
            "model_ms_per_document": model_ms / total_docs,
            "model_documents_per_second": total_docs / (model_ms / 1000),
            "pipeline_documents_per_second": total_docs / (pipeline_ms / 1000),
            "batch_latency_p50_ms": float(np.percentile([r["model_ms"] for r in records], 50)),
            "batch_latency_p95_ms": float(np.percentile([r["model_ms"] for r in records], 95)),
            "input_tokens_per_second": sum(valid_tokens) / (model_ms / 1000) if valid_tokens else None,
            "mean_actual_input_tokens": float(np.mean(valid_tokens)) if valid_tokens else None}
