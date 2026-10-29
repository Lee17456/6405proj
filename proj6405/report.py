"""Export measured results only; do not fabricate missing runs or error bars."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import REPORTS, RUNS, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Only include run names with this prefix, e.g. main_")
    parser.add_argument("--output-dir", type=Path, help="Separate output folder for a campaign/study")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else REPORTS
    accuracy, timing = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if not path.is_dir() or not path.name.startswith(args.prefix) or not (path / "config.json").exists():
            continue
        c = read_json(path / "config.json")
        if c["status"] != "completed":
            continue
        base = {"run": path.name, "model": c["model"], "dataset": c["dataset"], "view": c["view"],
                "max_length": c["max_length"], "seed": c["seed"], "n_train": c["n_train"],
                "data_signature": c["data_signature"], "train_ids_signature": c["train_ids_signature"],
                "lr": c["lr"], "pooling": c["pooling"], "epochs": c["epochs"], "batch_size": c["batch_size"],
                "grad_accum": c["grad_accum"], "svm_c": c["svm_c"], "ngram_max": c["ngram_max"],
                "max_features": c["max_features"], "weight_decay": c["weight_decay"],
                "dropout": c["dropout"], "attention_backend": c["attention_backend"],
                "gradient_checkpointing": c["gradient_checkpointing"], "limit_val": c["limit_val"],
                "train_seconds": c["train_seconds"]}
        metric_file = path / "metrics_test.json"
        if metric_file.exists():
            m = read_json(metric_file)
            accuracy.append({**base, "precision": m["precision"], "test_ids_signature": m["evaluated_ids_signature"],
                             **{k: m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "n")}})
            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(m["confusion_matrix"], cmap="Blues")
            ax.set(xlabel="Predicted label ID", ylabel="True label ID", title=path.name)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(output_dir / f"confusion_{path.name}.png", dpi=160)
            plt.close(fig)
        for file in sorted(path.glob("benchmark_*.json")):
            b = read_json(file)
            timing.append({**base, "benchmark_file": file.name,
                           **{k: b[k] for k in ("batch_size", "precision", "device", "model_ms_per_document",
                                               "model_documents_per_second", "pipeline_documents_per_second", "peak_allocated_mb",
                                               "mean_actual_input_tokens", "input_tokens_per_second", "sample_ids_signature")}})
    pd.DataFrame(timing).to_csv(output_dir / "benchmarks.csv", index=False)
    if not accuracy:
        print("No full test results found. Benchmarks exported; no invented scores or plots.")
        return
    df = pd.DataFrame(accuracy)
    df.to_csv(output_dir / "results.csv", index=False)
    keys = ["dataset", "model", "view", "max_length", "precision", "n_train", "data_signature",
            "train_ids_signature", "test_ids_signature", "lr", "pooling", "epochs", "batch_size", "grad_accum",
            "svm_c", "ngram_max", "max_features", "weight_decay", "dropout", "attention_backend",
            "gradient_checkpointing", "limit_val"]
    # Reject duplicate seeds within a configuration instead of inflating n.
    if df.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate model seeds within a configuration. Use --prefix to select exactly one experiment campaign.")
    summary = df.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "nunique"), macro_precision_mean=("macro_precision", "mean"),
        macro_precision_std=("macro_precision", "std"), macro_recall_mean=("macro_recall", "mean"),
        macro_recall_std=("macro_recall", "std"), macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"), accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"), train_seconds_mean=("train_seconds", "mean"),
        train_seconds_std=("train_seconds", "std")).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    for dataset, group in summary.groupby("dataset"):
        group = group.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(max(8, len(group) * 0.7), 5))
        labels = [f"{r.model}/{r.view}\nn={r.n_seeds}" for r in group.itertuples()]
        ax.bar(range(len(group)), group["macro_f1_mean"], tick_label=labels)
        multiple = group[group["n_seeds"] > 1]
        if not multiple.empty:
            ax.errorbar(multiple.index, multiple["macro_f1_mean"], yerr=multiple["macro_f1_std"],
                        fmt="none", ecolor="black", capsize=4)
        ax.set(ylabel="Test Macro-F1 (mean; bars = sample SD across seeds)", ylim=(0, 1), title=dataset)
        ax.tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        fig.savefig(output_dir / f"f1_summary_{dataset}.png", dpi=160)
        plt.close(fig)
    for dataset, group in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(max(8, len(group) * 0.4), 5))
        ax.bar(group["run"], group["macro_f1"])
        ax.set(ylabel="Test Macro-F1", ylim=(0, 1), title=dataset)
        ax.tick_params(axis="x", labelrotation=75)
        fig.tight_layout()
        fig.savefig(output_dir / f"f1_{dataset}.png", dpi=160)
        plt.close(fig)
        long_runs = summary[(summary["dataset"] == dataset) & (summary["dataset"] == "scotus") &
                            (summary["model"].isin(["mamba", "longformer"])) & summary["view"].str.contains(r"(?:^|_)shared")]
        if not long_runs.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for model, g in long_runs.groupby("model"):
                ax.scatter(g["max_length"], g["macro_f1_mean"], label=model)
                repeated = g[g["n_seeds"] > 1]
                if not repeated.empty:
                    ax.errorbar(repeated["max_length"], repeated["macro_f1_mean"], yerr=repeated["macro_f1_std"],
                                fmt="none", capsize=4)
            ax.set(xlabel="Shared-prefix token budget (train and test)", ylabel="Test Macro-F1 (mean +/- seed SD)", title="SCOTUS length analysis")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "length_f1_scotus.png", dpi=160)
            plt.close(fig)
    # Fixed batch and precision panels prevent mixing incomparable throughput settings.
    if timing:
        bench = pd.DataFrame(timing)
        paired = bench.merge(df[["run", "macro_f1"]], on="run", how="inner")
        for (dataset, batch), group in paired.groupby(["dataset", "batch_size"]):
            fig, ax = plt.subplots(figsize=(8, 5))
            for (model, precision), g in group.groupby(["model", "precision"]):
                # Pipeline latency includes TF-IDF/tokenization for a system-level comparison.
                x = 1000 / g["pipeline_documents_per_second"]
                ax.scatter(x, g["macro_f1"], label=f"{model}/{precision}")
                for xi, (_, row) in zip(x, g.iterrows()):
                    ax.annotate(row["view"], (xi, row["macro_f1"]), fontsize=7)
            ax.set(xlabel="Local pipeline ms/document (lower is better)", ylabel="Test Macro-F1", title=f"{dataset}; batch={batch}; CPU SVM / GPU neural")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"quality_cost_{dataset}_b{batch}.png", dpi=160)
            plt.close(fig)
        for (dataset, batch, precision), group in bench.groupby(["dataset", "batch_size", "precision"]):
            neural = group[group["model"] != "svm"]
            if neural.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for (model, view), g in neural.groupby(["model", "view"]):
                axes[0].scatter(g["mean_actual_input_tokens"], g["model_ms_per_document"], label=f"{model}/{view}")
                axes[1].scatter(g["mean_actual_input_tokens"], g["peak_allocated_mb"], label=f"{model}/{view}")
            axes[0].set(xlabel="Mean actual input tokens", ylabel="Model ms/document")
            axes[1].set(xlabel="Mean actual input tokens", ylabel="Peak GPU allocated MiB")
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{dataset}; batch={batch}; {precision}")
            fig.tight_layout()
            fig.savefig(output_dir / f"scaling_{dataset}_b{batch}_{precision}.png", dpi=160)
            plt.close(fig)
    print(f"Exported measured results to {output_dir}; std is blank when only one seed exists.")


if __name__ == "__main__":
    main()
