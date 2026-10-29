import argparse
import csv

from tqdm.auto import tqdm

from .common import load_rows, run_path, score_labels, signature, write_json
from .inference import Predictor


def main():
    parser = argparse.ArgumentParser(description="Evaluate one completed run on a held-out split")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("Invalid batch size or limit")
    predictor = Predictor(args.run, args.device, args.precision)
    config = predictor.config
    rows, manifest = load_rows(config["dataset"], config["view"], args.split, args.limit)
    if manifest["signature"] != config["data_signature"]:
        raise ValueError("Dataset differs from the one used for training")
    truth, predicted, results = [], [], []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Evaluating"):
        batch = rows[start:start + args.batch_size]
        output = predictor.predict_batch(batch["text"], already_view=True)
        truth.extend(batch["label"])
        predicted.extend(output["predicted_ids"])
        for i, label in enumerate(output["predicted_ids"]):
            results.append({"id": batch["id"][i], "true_id": batch["label"][i], "predicted_id": label,
                            "true_label": predictor.labels[batch["label"][i]], "predicted_label": predictor.labels[label],
                            "correct": label == batch["label"][i], "input_tokens": output["input_tokens"][i],
                            "input_chars": len(batch["text"][i]),
                            "coverage_chars": batch.get("coverage_chars", [None] * len(batch["text"]))[i],
                            "text_preview": batch["text"][i][:600]})
    metrics = score_labels(truth, predicted, predictor.labels)
    metrics.update(split=args.split, limit=args.limit, data_signature=manifest["signature"],
                   evaluated_ids_signature=signature(rows["id"]), precision=predictor.precision,
                   full_split=(len(rows) == manifest["splits"][args.split]["n"]))
    suffix = args.split if metrics["full_split"] else f"{args.split}_subset{len(rows)}"
    write_json(run_path(args.run) / f"metrics_{suffix}.json", metrics)
    with (run_path(args.run) / f"predictions_{suffix}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print({k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "n", "full_split")})


if __name__ == "__main__":
    main()
