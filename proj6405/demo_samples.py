import argparse

from .common import DATA_IDS, REPORTS, load_rows, write_json


def main():
    parser = argparse.ArgumentParser(description="Export fixed test documents for the live demonstration")
    parser.add_argument("--dataset", choices=list(DATA_IDS), required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    rows, manifest = load_rows(args.dataset, split="test")
    if args.count < 1:
        parser.error("count must be positive")
    target = REPORTS / "demo_samples" / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(min(args.count, len(rows))):
        row = rows[i]
        (target / f"sample_{i}.txt").write_text(row["text"], encoding="utf-8")
        index.append({"file": f"sample_{i}.txt", "id": row["id"], "true_label": manifest["label_names"][row["label"]],
                      "chars": len(row["text"]), "selection": "first official test documents, not selected for correctness"})
    write_json(target / "index.json", index)
    print(target)


if __name__ == "__main__":
    main()
