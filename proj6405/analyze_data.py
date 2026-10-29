import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .common import DATA_IDS, MODEL_IDS, MODELS, REPORTS, load_rows, write_json


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description="Measure raw lengths and truncation rates without changing datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(DATA_IDS), default=list(DATA_IDS))
    parser.add_argument("--models", nargs="+", choices=list(MODEL_IDS), default=["roberta", "mamba", "longformer"])
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--limit", type=int, default=0, help="0=full split; sampling is labeled in output")
    args = parser.parse_args()
    output = []
    for dataset in args.datasets:
        rows, manifest = load_rows(dataset, split=args.split, limit=args.limit)
        for model in args.models:
            tok = AutoTokenizer.from_pretrained(MODELS / model, local_files_only=True, use_fast=True)
            lengths = []
            for start in tqdm(range(0, len(rows), 32), desc=f"{dataset}/{model}"):
                lengths.extend(len(x) for x in tok(rows[start:start + 32]["text"], truncation=False)["input_ids"])
            values = np.asarray(lengths)
            row = {"dataset": dataset, "tokenizer": model, "split": args.split, "limit": args.limit,
                   "n": len(rows), "num_labels": manifest["num_labels"], "mean_tokens": float(values.mean()),
                   "median_tokens": float(np.median(values)), "p90_tokens": float(np.percentile(values, 90)),
                   "max_tokens": int(values.max())}
            row.update({f"fraction_over_{length}": float((values > length).mean()) for length in (512, 1024, 2048, 4096)})
            output.append(row)
    REPORTS.mkdir(exist_ok=True)
    suffix = f"{args.split}_{args.limit or 'full'}"
    pd.DataFrame(output).to_csv(REPORTS / f"lengths_{suffix}.csv", index=False)
    write_json(REPORTS / f"lengths_{suffix}.json", output)
    print(pd.DataFrame(output).to_string(index=False))


if __name__ == "__main__":
    main()
