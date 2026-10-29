"""Run a small, explicit matrix sequentially on one GPU. Train/test remain separate."""
import argparse
import shlex
import subprocess
import sys

from .common import REPORTS, read_json, run_path, valid_name, write_json


def make_plan(stage, prefix, seeds, lengths):
    plan = []
    if stage == "pilot":
        for model in ("svm", "roberta", "mamba"):
            plan.append(dict(model=model, dataset="ag_news", view="native", max_length=256, seed=seeds[0]))
    elif stage == "core":
        for dataset in ("ag_news", "imdb", "scotus"):
            plan.append(dict(model="svm", dataset=dataset, view="native", max_length=512, seed=seeds[0]))
            for model in ("svm", "roberta", "mamba"):
                for seed in (seeds[:1] if model == "svm" else seeds):
                    plan.append(dict(model=model, dataset=dataset, view="shared512", max_length=512, seed=seed))
    else:
        for length in lengths:
            for model in ("mamba", "longformer"):
                for seed in seeds:
                    plan.append(dict(model=model, dataset="scotus", view=f"shared{length}", max_length=length, seed=seed))
    for p in plan:
        p["name"] = f"{prefix}_{p['dataset']}_{p['model']}_{p['view']}_s{p['seed']}"
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "core", "long"], required=True)
    parser.add_argument("--phase", choices=["train", "evaluate", "benchmark"], required=True)
    parser.add_argument("--prefix", default="main")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--lengths", nargs="+", type=int, choices=[512, 1024, 2048, 4096], default=[2048])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--benchmark-batch", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    valid_name(args.prefix)
    plan = make_plan(args.stage, args.prefix, args.seeds, args.lengths)
    if args.stage == "pilot":
        args.epochs, args.limit_train, args.limit_val = 1, 1000, 200
        if args.phase == "evaluate" and not args.eval_limit:
            args.eval_limit = 200
    write_json(REPORTS / f"plan_{args.prefix}_{args.stage}_{args.phase}.json", {"arguments": vars(args), "runs": plan})
    for p in plan:
        if args.phase == "train":
            command = [sys.executable, "-m", "proj6405.train"]
            for key, value in p.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            for key in ("epochs", "lr", "batch_size", "grad_accum", "limit_train", "limit_val"):
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            if p["max_length"] > 512:
                command += ["--gradient-checkpointing", "--eval-batch-size", "1"]
            existing = run_path(p["name"]) / "config.json"
            if existing.exists():
                c = read_json(existing)
                expected = {**p, **{k: getattr(args, k) for k in ("epochs", "lr", "batch_size", "grad_accum", "limit_train", "limit_val")}}
                expected.update(eval_batch_size=1 if p["max_length"] > 512 else 2,
                                gradient_checkpointing=p["max_length"] > 512,
                                pooling=None, dropout=0.1, weight_decay=0.01, warmup_ratio=0.1,
                                attention_backend="sdpa", allow_slow_mamba=False,
                                svm_c=1.0, ngram_max=2, max_features=100000,
                                precision="bf16", device="cuda")
                if p["model"] == "svm":
                    expected.update(max_length=None, precision="fp32", device="cpu")
                if c.get("status") == "completed" and all(c.get(k) == v for k, v in expected.items()):
                    print(f"Skip completed matching run: {p['name']}")
                    continue
                raise ValueError(f"Existing run differs or is incomplete: {p['name']}. Use a NEW --prefix; inspect old logs first.")
        elif args.phase == "evaluate":
            command = [sys.executable, "-m", "proj6405.evaluate", "--run", p["name"], "--batch-size", "1",
                       "--limit", str(args.eval_limit)]
        else:
            # Speed does not need to be remeasured for every training seed.
            if p["seed"] != args.seeds[0]:
                continue
            command = [sys.executable, "-m", "proj6405.benchmark", "--run", p["name"],
                       "--batch-size", str(args.benchmark_batch)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
