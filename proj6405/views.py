"""Build identical raw-text prefixes that fit every participating tokenizer."""
import argparse

from .common import DATA_IDS, MODELS, data_path, read_json, signature, subset_rows, valid_name, write_json


def common_prefix(text, tokenizers, budget):
    if budget < 8:
        raise ValueError("budget must be at least 8")
    prefix = text
    # Token boundaries can change when a string is shortened. Recheck until it fits.
    for _ in range(32):
        cuts = []
        for tok in tokenizers:
            encoded = tok(prefix, add_special_tokens=True, return_offsets_mapping=True,
                          truncation=True, max_length=budget)
            end = max((end for _, end in encoded["offset_mapping"]), default=0)
            count = len(tok(prefix, add_special_tokens=True, truncation=False)["input_ids"])
            if count > budget:
                cuts.append(end)
        if not cuts:
            return prefix
        next_end = min(cuts)
        if next_end >= len(prefix):
            next_end = len(prefix) - 1
        prefix = prefix[:next_end]
        if not prefix.strip():
            raise ValueError("Unable to produce a nonempty shared prefix")
    raise RuntimeError("Shared-prefix truncation did not converge")


def view_tokenizers(manifest):
    from transformers import AutoTokenizer
    return [AutoTokenizer.from_pretrained(MODELS / name, local_files_only=True, use_fast=True)
            for name in manifest["participating_models"]]


def transform_live_text(text, manifest, tokenizers=None):
    if manifest["view"] == "native":
        return text
    return common_prefix(text, tokenizers or view_tokenizers(manifest), manifest["token_budget"])


def main():
    from datasets import DatasetDict, load_from_disk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATA_IDS), required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--name", help="Separate view name for a bounded-data experiment")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    args = parser.parse_args()
    if args.budget not in (256, 512, 1024, 2048, 4096):
        parser.error("Use 256, 512, 1024, 2048, or 4096")
    if min(args.limit_train, args.limit_val) < 0:
        parser.error("Subset limits must be nonnegative")
    if (args.limit_train or args.limit_val) and not args.name:
        parser.error("Bounded views require --name to keep the full-data views separate")
    participants = ["roberta", "mamba", "longformer"] if args.budget <= 512 else ["mamba", "longformer"]
    base = data_path(args.dataset)
    manifest = read_json(base / "manifest.json")
    manifest.update(view=valid_name(args.name) if args.name else f"shared{args.budget}", token_budget=args.budget,
                    participating_models=participants, parent_signature=manifest["signature"],
                    model_revisions={n: read_json(MODELS / n / "download_manifest.json")["revision"] for n in participants})
    if args.limit_train or args.limit_val:
        manifest["subset_limits"] = {"train": args.limit_train, "validation": args.limit_val, "test": 0}
    manifest.pop("signature")
    manifest["signature"] = signature(manifest)
    target = data_path(args.dataset, manifest["view"])
    if (target / "manifest.json").exists():
        if read_json(target / "manifest.json")["signature"] != manifest["signature"]:
            raise RuntimeError("Existing view has a different source revision; do not overwrite")
        print(f"Already prepared: {target}")
        return
    if (target / "dataset").exists():
        raise RuntimeError(f"Incomplete output exists at {target}; inspect and rename it before retrying")
    tokenizers = view_tokenizers(manifest)
    ds = load_from_disk(str(base / "dataset"))
    if args.limit_train or args.limit_val:
        # Select BEFORE tokenization; keep the entire official test split.
        ds = DatasetDict({s: subset_rows(rows, manifest["subset_limits"].get(s, 0)) for s, rows in ds.items()})

    def convert(row):
        text = common_prefix(row["text"], tokenizers, args.budget)
        return {"text": text, "original_chars": len(row["text"]),
                "coverage_chars": len(text) / max(len(row["text"]), 1)}

    ds = ds.map(convert, desc=f"Preparing {manifest['view']}", load_from_cache_file=False)
    ds.save_to_disk(str(target / "dataset"))
    write_json(target / "manifest.json", manifest)
    write_json(target / "prepared_splits.json", {
        s: {"n": len(rows), "ids_signature": signature(rows["id"])} for s, rows in ds.items()
    })
    print(target)


if __name__ == "__main__":
    main()
