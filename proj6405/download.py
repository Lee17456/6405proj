"""All network downloads occur here, on the Linux server."""
import argparse
import hashlib
from collections import Counter

from .common import DATA_IDS, MODEL_IDS, MODELS, data_path, read_json, signature, write_json


def download_model(name):
    from huggingface_hub import HfApi, snapshot_download
    destination = MODELS / name
    lock = destination / "download_manifest.json"
    if lock.exists():
        print(f"Already downloaded: {name}; revision={read_json(lock)['revision']}")
        return
    info = HfApi().model_info(MODEL_IDS[name])
    revision = info.sha
    has_safe = any(f.rfilename.endswith(".safetensors") for f in info.siblings)
    weight_patterns = ["*.safetensors"] if has_safe else ["pytorch_model*.bin"]
    snapshot_download(MODEL_IDS[name], revision=revision, local_dir=str(destination),
                      allow_patterns=["*.json", "*.txt", "*.model", "README.md", *weight_patterns])
    if not list(destination.glob("*.safetensors")) and not list(destination.glob("pytorch_model*.bin")):
        raise RuntimeError(f"No model weights found in {destination}")
    write_json(lock, {"repo_id": MODEL_IDS[name], "revision": revision,
                      "weight_format": "safetensors" if has_safe else "official pytorch bin"})
    print(f"Model ready: {destination}")


def download_data(name):
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import HfApi
    base = data_path(name)
    if (base / "manifest.json").exists():
        print(f"Already prepared: {name}")
        return
    if (base / "dataset").exists():
        raise RuntimeError(f"Incomplete dataset at {base}. Inspect it and rename the directory before retrying.")
    repo, config = DATA_IDS[name]
    revision = HfApi().dataset_info(repo).sha
    raw = load_dataset(repo, config, revision=revision, trust_remote_code=False)
    splits = {}
    for split in ("train", "validation", "test"):
        if split in raw:
            ds = raw[split].select_columns(["text", "label"])
            splits[split] = ds.add_column("id", [f"{name}:{split}:{i}" for i in range(len(ds))])
    if "validation" not in splits:
        separated = splits["train"].train_test_split(test_size=0.1, seed=2026, stratify_by_column="label")
        splits["train"], splits["validation"] = separated["train"], separated["test"]
    label_names = list(splits["train"].features["label"].names)
    hashes, stats = {}, {}
    for split, ds in splits.items():
        observed = set(ds["label"])
        if not observed.issubset(set(range(len(label_names)))):
            raise ValueError(f"Invalid labels in {name}/{split}: {observed}")
        hashes[split] = {hashlib.sha256(t.strip().encode()).hexdigest() for t in ds["text"]}
        stats[split] = {"n": len(ds), "labels": dict(Counter(ds["label"])), "fingerprint": ds._fingerprint}
    overlaps = {f"{a}__{b}": len(hashes[a] & hashes[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    manifest = {"dataset": name, "repo_id": repo, "revision": revision, "config": config,
                "view": "native", "split_seed": 2026, "label_names": label_names,
                "num_labels": len(label_names), "splits": stats, "exact_text_overlap_counts": overlaps,
                "split_policy": "official temporal splits" if name == "scotus" else "official test; 10% stratified train held out"}
    manifest["signature"] = signature(manifest)
    DatasetDict(splits).save_to_disk(str(base / "dataset"))
    write_json(base / "manifest.json", manifest)
    print(f"Data ready: {name}, classes={len(label_names)}, counts={stats}, overlaps={overlaps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", choices=list(DATA_IDS), default=list(DATA_IDS))
    parser.add_argument("--models", nargs="*", choices=list(MODEL_IDS), default=list(MODEL_IDS))
    args = parser.parse_args()
    for name in args.datasets:
        download_data(name)
    for name in args.models:
        download_model(name)


if __name__ == "__main__":
    main()
