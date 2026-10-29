"""Small-data experiment with a wall-clock cap per invocation; does not stop the Pod."""
import argparse
import shlex
import subprocess
import sys
import time

from .common import REPORTS, ROOT, data_path, read_json, run_path, valid_name, write_json


def make_plan(stage, prefix):
    valid_name(prefix)
    if stage == "core":
        settings = [(d, m, 512) for m in ("svm", "roberta", "mamba") for d in ("ag_news", "imdb", "scotus")]
    else:
        # Reuse the core Mamba-512 result; add a matched Longformer-512 baseline.
        settings = [("scotus", "longformer", 512), ("scotus", "mamba", 2048), ("scotus", "longformer", 2048)]
    return [{"dataset": d, "model": m, "length": length, "view": f"{prefix}_shared{length}",
             "name": f"{prefix}_{d}_{m}_shared{length}_s42"} for d, m, length in settings]


def train_options(run):
    long_input = run["length"] > 512
    return {"model": run["model"], "dataset": run["dataset"], "name": run["name"], "view": run["view"],
            "max_length": run["length"], "seed": 42, "epochs": 2, "limit_train": 1000, "limit_val": 200,
            "batch_size": 1 if long_input else 2, "grad_accum": 16 if long_input else 8,
            "eval_batch_size": 1 if long_input else 2, "lr": 2e-5, "precision": "bf16", "device": "cuda"}


def command(module, options, *flags):
    result = [sys.executable, "-m", f"proj6405.{module}"]
    for key, value in options.items():
        result += ["--" + key.replace("_", "-"), str(value)]
    return result + list(flags)


def make_commands(stage, phase, prefix, models=None):
    runs = make_plan(stage, prefix)
    if models:
        runs = [r for r in runs if r["model"] in models]
    if not runs:
        raise ValueError("No runs match this stage/model selection")
    if phase == "prepare":
        views = sorted({(r["dataset"], r["length"], r["view"]) for r in runs})
        return [(None, command("views", {"dataset": d, "budget": length, "name": view,
                                        "limit_train": 1000, "limit_val": 200})) for d, length, view in views]
    commands = []
    for run in runs:
        if phase == "train":
            flags = ["--gradient-checkpointing"] if run["length"] > 512 else []
            cmd = command("train", train_options(run), *flags)
        elif phase == "evaluate":
            cmd = command("evaluate", {"run": run["name"], "batch_size": 1 if run["length"] > 512 else 2})
        else:
            cmd = command("benchmark", {"run": run["name"], "batch_size": 1, "samples": 64, "warmup": 5, "repeats": 3})
        commands.append((run, cmd))
    return commands


def is_completed(run):
    config_file = run_path(run["name"]) / "config.json"
    if not config_file.exists():
        return False
    config = read_json(config_file)
    expected = train_options(run)
    expected.update(gradient_checkpointing=run["length"] > 512, pooling=None, dropout=0.1,
                    weight_decay=0.01, warmup_ratio=0.1, attention_backend="sdpa", allow_slow_mamba=False,
                    svm_c=1.0, ngram_max=2, max_features=100000)
    if run["model"] == "svm":
        expected.update(max_length=None, precision="fp32", device="cpu")
    manifest = read_json(data_path(run["dataset"], run["view"]) / "manifest.json")
    expected["data_signature"] = manifest["signature"]
    if config.get("status") == "completed" and all(config.get(k) == v for k, v in expected.items()):
        return True
    raise ValueError(f"Incomplete or different existing run: {run['name']}. Inspect it; use a NEW --prefix for a fresh campaign.")


def execute(commands, phase, minutes, runner=subprocess.run, clock=time.monotonic):
    started = clock()
    for run, cmd in commands:
        if phase == "train" and is_completed(run):
            print(f"Skip matching completed run: {run['name']}", flush=True)
            continue
        remaining = minutes * 60 - (clock() - started)
        if remaining <= 0:
            raise TimeoutError("Phase time budget exhausted before the next command")
        print(shlex.join(cmd), flush=True)
        runner(cmd, check=True, cwd=ROOT, timeout=remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["core", "long"], default="core")
    parser.add_argument("--phase", choices=["prepare", "train", "evaluate", "benchmark"], required=True)
    parser.add_argument("--prefix", default="h6")
    parser.add_argument("--models", nargs="+", choices=["svm", "roberta", "mamba", "longformer"])
    parser.add_argument("--minutes", type=float, default=90, help="Cap THIS invocation, including all its child commands")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.minutes <= 360:
        parser.error("--minutes must be greater than 0 and at most 360")
    commands = make_commands(args.stage, args.phase, args.prefix, args.models)
    if args.dry_run:
        for _, cmd in commands:
            print(shlex.join(cmd))
        return
    record = {"arguments": vars(args), "commands": [cmd for _, cmd in commands], "status": "running",
              "note": "Per-invocation cap, not a rental countdown; does not stop or terminate the Pod."}
    destination = REPORTS / f"budget_{args.prefix}_{args.stage}_{args.phase}.json"
    write_json(destination, record)
    started = time.monotonic()
    try:
        execute(commands, args.phase, args.minutes)
        record["status"] = "completed"
    except (TimeoutError, subprocess.TimeoutExpired):
        record["status"] = "timed_out"
        print("Time cap reached. Completed runs remain usable. An interrupted training run is NOT a completed experiment.", file=sys.stderr)
        raise SystemExit(124)
    except (Exception, KeyboardInterrupt):
        record["status"] = "failed_or_interrupted"
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(destination, record)


if __name__ == "__main__":
    main()
