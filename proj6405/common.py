import hashlib
import importlib.metadata
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
MODEL_IDS = {
    "roberta": "FacebookAI/roberta-base",
    "mamba": "state-spaces/mamba-130m-hf",
    "longformer": "allenai/longformer-base-4096",
}
DATA_IDS = {"ag_news": ("fancyzhx/ag_news", "default"),
            "imdb": ("stanfordnlp/imdb", "plain_text"),
            "scotus": ("coastalcph/lex_glue", "scotus")}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def valid_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or ".." in name:
        raise ValueError("Use a simple name containing letters, digits, _, - or .")
    return name


def data_path(dataset, view="native"):
    if dataset not in DATA_IDS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATA / dataset if view == "native" else DATA / "views" / dataset / valid_name(view)


def run_path(name):
    return RUNS / valid_name(name)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "scikit-learn", "mamba-ssm", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "packages": packages}
    try:
        import torch
        result.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return result


def load_rows(dataset, view="native", split="train", limit=0):
    from datasets import load_from_disk
    base = data_path(dataset, view)
    manifest = read_json(base / "manifest.json")
    rows = load_from_disk(str(base / "dataset"))[split]
    return subset_rows(rows, limit), manifest


def subset_rows(rows, limit=0):
    if limit < 0:
        raise ValueError("Subset limits must be nonnegative")
    if limit and limit < len(rows):
        from collections import Counter
        counts = Counter(rows["label"])
        if limit < len(counts):
            raise ValueError("Subset must have at least one slot per observed class")
        if min(counts.values()) < 2 or len(rows) - limit < len(counts):
            # train_test_split requires >=2 per class and space in both parts.
            # For singleton classes, keep one each and proportionally allocate the rest.
            import random
            labels = sorted(counts)
            denominator = len(rows) - len(labels)
            extra = limit - len(labels)
            quotas = {label: 1 + extra * (counts[label] - 1) // denominator for label in labels}
            ranked = sorted(labels, key=lambda label: (-(extra * (counts[label] - 1) % denominator), label))
            for label in ranked[:limit - sum(quotas.values())]:
                quotas[label] += 1
            generator = random.Random(2026)
            selected = []
            groups = {label: [] for label in labels}
            for index, label in enumerate(rows["label"]):
                groups[label].append(index)
            for label in labels:
                selected.extend(generator.sample(groups[label], quotas[label]))
            return rows.select(sorted(selected))
        from sklearn.model_selection import train_test_split
        # Fixed data seed across ALL model seeds. Never redefine the official splits.
        selected, _ = train_test_split(list(range(len(rows))), train_size=limit,
                                       stratify=rows["label"], random_state=2026)
        rows = rows.select(sorted(selected))
    return rows


def score_labels(y_true, y_pred, label_names):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    labels = list(range(len(label_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f), "n": len(y_true),
            "per_class": classification_report(y_true, y_pred, labels=labels, target_names=label_names,
                                               output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def split_batches(items, size):
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]
