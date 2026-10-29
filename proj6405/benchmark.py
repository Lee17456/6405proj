import argparse

import psutil
from tqdm.auto import tqdm

from .common import environment, load_rows, run_path, signature, write_json
from .inference import Predictor, latency_summary


def main():
    parser = argparse.ArgumentParser(description="Isolated, warmed-up classifier benchmark; excludes loading and network")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.warmup, args.repeats) < 1:
        parser.error("All sample/batch/warmup/repeat values must be positive")
    predictor = Predictor(args.run, args.device, args.precision)
    rows, manifest = load_rows(predictor.config["dataset"], predictor.config["view"], args.split, args.samples)
    if manifest["signature"] != predictor.config["data_signature"]:
        raise ValueError("Dataset signature mismatch")
    texts = rows["text"]
    for _ in range(args.warmup):
        predictor.predict_batch(texts[:args.batch_size], already_view=True)
    gpu = predictor.device.startswith("cuda")
    if gpu:
        import torch
        torch.cuda.reset_peak_memory_stats()
    records = []
    for _ in tqdm(range(args.repeats), desc="Benchmark repeats"):
        for start in range(0, len(texts), args.batch_size):
            records.append(predictor.predict_batch(texts[start:start + args.batch_size], already_view=True))
    result = {**vars(args), **latency_summary(records), "precision": predictor.precision,
              "device": predictor.device, "environment": environment(),
              "train_max_length": predictor.config["max_length"], "view": predictor.config["view"],
              "data_signature": manifest["signature"], "sample_ids_signature": signature(rows["id"]),
              "unique_documents": len(rows), "dynamic_padding": True,
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if gpu else None,
              "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if gpu else None,
              "process_rss_after_mb": psutil.Process().memory_info().rss / 2**20,
              "scope": "preprocess=tokenization/TFIDF; model=GPU-resident forward or CPU SVM decision; pipeline also includes H2D and argmax; excludes loading/network"}
    filename = f"benchmark_{args.split}_b{args.batch_size}_{predictor.precision}_n{len(rows)}_r{args.repeats}.json"
    write_json(run_path(args.run) / filename, result)
    print(result)


if __name__ == "__main__":
    main()
