"""Full-data campaign. Separate core/length studies; no automatic test-based tuning."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from .common import MODELS, REPORTS, ROOT, RUNS, data_path, read_json, run_path, valid_name, write_json


def load_settings(path):
    settings = read_json(path)
    if set(settings) != {"seeds", "epochs", "learning_rates", "svm_c", "core", "long"}:
        raise ValueError("Unexpected or missing full configuration fields")
    seeds = settings["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    if type(settings["epochs"]) is not int or settings["epochs"] < 1 or settings["svm_c"] <= 0:
        raise ValueError("Invalid epochs or SVM C")
    if set(settings["learning_rates"]) != {"roberta", "mamba", "longformer"}:
        raise ValueError("Provide one learning rate per neural model")
    if any(not 0 < value < 1 for value in settings["learning_rates"].values()):
        raise ValueError("Invalid learning rate")
    for stage in ("core", "long"):
        cfg = settings[stage]
        expected = {"batch_size", "grad_accum", "gradient_checkpointing"} | ({"lengths"} if stage == "long" else set())
        if set(cfg) != expected or type(cfg["gradient_checkpointing"]) is not bool:
            raise ValueError(f"Invalid {stage} configuration")
        if any(type(cfg[k]) is not int or cfg[k] < 1 for k in ("batch_size", "grad_accum")):
            raise ValueError("Batch sizes and accumulation must be positive integers")
    lengths = settings["long"]["lengths"]
    if not lengths or len(set(lengths)) != len(lengths) or any(n not in (512, 1024, 2048, 4096) for n in lengths):
        raise ValueError("Invalid or repeated long-study lengths")
    return settings


def make_plan(settings, prefix="full"):
    valid_name(prefix)
    specs = []
    seeds = settings["seeds"]
    for dataset in ("ag_news", "imdb", "scotus"):
        for view in ("native", "shared512"):
            specs.append(("core", dataset, "svm", view, 512, seeds[0]))
        for model in ("roberta", "mamba"):
            specs.extend(("core", dataset, model, "shared512", 512, seed) for seed in seeds)
    for length in settings["long"]["lengths"]:
        for model in ("mamba", "longformer"):
            specs.extend(("long", "scotus", model, f"shared{length}", length, seed) for seed in seeds)
    plan = []
    for stage, dataset, model, view, length, seed in specs:
        opts = dict(model=model, dataset=dataset, view=view, max_length=length, seed=seed,
                    name=f"{prefix}_{stage}_{dataset}_{model}_{view}_s{seed}", epochs=settings["epochs"],
                    lr=settings["learning_rates"].get(model, 2e-5), batch_size=settings[stage]["batch_size"],
                    grad_accum=settings[stage]["grad_accum"], eval_batch_size=1 if stage == "long" else 2,
                    gradient_checkpointing=settings[stage]["gradient_checkpointing"] if model != "svm" else False,
                    limit_train=0, limit_val=0, precision="bf16", device="cuda", pooling=None,
                    weight_decay=0.01, warmup_ratio=0.1, dropout=0.1, attention_backend="sdpa",
                    allow_slow_mamba=False, svm_c=settings["svm_c"], ngram_max=2, max_features=100000)
        plan.append({"stage": stage, "options": opts})
    return plan


def command(module, options):
    cmd = [sys.executable, "-m", f"proj6405.{module}"]
    for key, value in options.items():
        if value is None or value is False:
            continue
        cmd.append("--" + key.replace("_", "-"))
        if value is not True:
            cmd.append(str(value))
    return cmd


def expected_config(run):
    expected = dict(run["options"])
    if expected["model"] == "svm":
        expected.update(max_length=None, precision="fp32", device="cpu")
    return expected


def verify_inputs(run):
    opts = run["options"]
    base = data_path(opts["dataset"], opts["view"])
    manifest = read_json(base / "manifest.json")
    if any(manifest.get("subset_limits", {}).values()):
        raise ValueError("Full campaign requires full-data views; h6 views cannot be reused")
    prepared_file = base / "prepared_splits.json"
    if prepared_file.exists():
        prepared = read_json(prepared_file)
        if any(prepared[s]["n"] != manifest["splits"][s]["n"] for s in ("train", "validation", "test")):
            raise ValueError("Prepared view is not the complete source split")
    for model, revision in manifest.get("model_revisions", {}).items():
        if read_json(MODELS / model / "download_manifest.json")["revision"] != revision:
            raise ValueError("Tokenizer revision changed since view preparation")
    return manifest


def matching_state(run):
    opts = run["options"]
    manifest = verify_inputs(run)
    path = run_path(opts["name"])
    if not path.exists():
        return "missing"
    if not (path / "config.json").exists():
        raise ValueError(f"Existing directory has no config: {path}; inspect it manually")
    config = read_json(path / "config.json")
    expected = {**expected_config(run), "data_signature": manifest["signature"]}
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError(f"Existing run has different settings/data: {opts['name']}; use a new campaign prefix")
    if opts["model"] != "svm":
        if config.get("model_download") != read_json(MODELS / opts["model"] / "download_manifest.json"):
            raise ValueError("Pretrained model revision differs from the saved run")
    return "completed" if config.get("status") == "completed" else "incomplete"


def archive_incomplete(name):
    root = RUNS.resolve()
    source = run_path(name)
    if source.is_symlink() or source.resolve().parent != root:
        raise ValueError("Refusing to move a run outside the runs directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = RUNS / "_interrupted" / f"{name}-{stamp}"
    if not target.resolve().is_relative_to(root):
        raise ValueError("Archive target is outside the runs directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    return target


def freeze_plan(destination, settings, plan, sources=None):
    snapshot = {"settings": settings, "runs": plan, "sources": sources or {}}
    path = destination / "plan.json"
    if path.exists() and read_json(path) != snapshot:
        raise ValueError("Campaign configuration is already frozen. Use --prefix full_v2 for changed settings.")
    if not path.exists():
        write_json(path, snapshot)


@contextmanager
def campaign_lock(destination):
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "active.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
    except FileExistsError as exc:
        raise RuntimeError(f"Campaign is already running or was killed: {path}. Check its PID before manually removing a stale lock.") from exc
    try:
        yield
    finally:
        path.unlink()


def run_logged(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {shlex.join(cmd)}\nLog: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {shlex.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def jobs_for(phase, selected, prefix, stages, batch_size, seeds):
    if phase == "prepare":
        views = sorted({(r["options"]["dataset"], r["options"]["max_length"]) for r in selected if r["options"]["view"] != "native"})
        return [(None, f"prepare_{d}_{n}", command("views", {"dataset": d, "budget": n})) for d, n in views]
    if phase == "report":
        return [(None, f"report_{stage}", command("report", {"prefix": f"{prefix}_{stage}_", "output_dir": str(REPORTS / prefix / stage)})) for stage in stages]
    jobs = []
    for run in selected:
        opts = run["options"]
        if phase == "train":
            cmd = command("train", opts)
        elif phase == "evaluate":
            cmd = command("evaluate", {"run": opts["name"], "batch_size": 1 if run["stage"] == "long" else 2})
        else:
            if opts["seed"] != seeds[0]:
                continue
            cmd = command("benchmark", {"run": opts["name"], "batch_size": batch_size, "samples": 128, "warmup": 10, "repeats": 3})
        jobs.append((run, opts["name"], cmd))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["plan", "prepare", "train", "evaluate", "benchmark", "report", "status"], required=True)
    parser.add_argument("--stage", choices=["core", "long", "all"], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full.json")
    parser.add_argument("--prefix", default="full")
    parser.add_argument("--models", nargs="+", choices=["svm", "roberta", "mamba", "longformer"])
    parser.add_argument("--lengths", nargs="+", type=int, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--benchmark-batch", type=int, default=1)
    parser.add_argument("--retry-incomplete", action="store_true", help="Archive an interrupted matching run and retrain it from scratch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    valid_name(args.prefix)
    if args.benchmark_batch < 1:
        parser.error("Benchmark batch must be positive")
    settings = load_settings(args.config)
    plan = make_plan(settings, args.prefix)
    stages = ["core", "long"] if args.stage == "all" else [args.stage]
    selected = [r for r in plan if r["stage"] in stages and
                (not args.models or r["options"]["model"] in args.models) and
                (not args.lengths or r["options"]["max_length"] in args.lengths)]
    if not selected:
        parser.error("No configurations match the filters")
    if args.phase == "plan":
        for stage in stages:
            runs = [r for r in selected if r["stage"] == stage]
            print(f"{stage}: {len(runs)} training runs; {sum(r['options']['model'] != 'svm' for r in runs)} neural")
        print(json.dumps(settings, indent=2))
        return
    if args.phase == "status":
        for stage in stages:
            runs = [r for r in selected if r["stage"] == stage]
            completed = tested = 0
            for run in runs:
                path = run_path(run["options"]["name"])
                done = (path / "config.json").exists() and read_json(path / "config.json").get("status") == "completed"
                completed += bool(done)
                tested += bool(done and (path / "metrics_test.json").exists())
            print(f"{stage}: trained {completed}/{len(runs)}; full-test files {tested}/{len(runs)}")
        return
    jobs = jobs_for(args.phase, selected, args.prefix, stages, args.benchmark_batch, settings["seeds"])
    if args.dry_run:
        for _, _, cmd in jobs:
            print(shlex.join(cmd))
        return
    destination = REPORTS / args.prefix
    with campaign_lock(destination):
        # Preparation may precede validation-only tuning. Freeze before formal runs.
        if args.phase != "prepare":
            sources = {"datasets": {d: read_json(data_path(d) / "manifest.json")["signature"]
                                    for d in ("ag_news", "imdb", "scotus")},
                       "models": {m: read_json(MODELS / m / "download_manifest.json")
                                  for m in ("roberta", "mamba", "longformer")}}
            freeze_plan(destination, settings, plan, sources)
        for index, (run, name, cmd) in enumerate(jobs, 1):
            progress = {"phase": args.phase, "job": name, "index": index, "total": len(jobs), "status": "running"}
            write_json(destination / "progress.json", progress)
            try:
                if run is not None:
                    state = matching_state(run)
                    if args.phase == "train" and state == "completed":
                        print(f"Skip matching completed run: {name}", flush=True)
                        progress["status"] = "skipped_completed"
                        continue
                    if args.phase == "train" and state == "incomplete":
                        if not args.retry_incomplete:
                            raise ValueError("Interrupted run found. Inspect logs, then use --retry-incomplete to archive and retrain.")
                        print(f"Preserved interrupted run at {archive_incomplete(name)}", flush=True)
                    elif args.phase != "train" and state != "completed":
                        raise ValueError(f"Cannot {args.phase} a missing/incomplete run: {name}")
                run_logged(cmd, ROOT / "logs" / args.prefix / f"{name}_{args.phase}.log")
                progress["status"] = "completed"
            except BaseException as exc:
                progress.update(status="failed_or_interrupted", error=str(exc))
                raise
            finally:
                write_json(destination / "progress.json", progress)
    print(f"Requested {args.phase} jobs finished. Run --phase status to inspect campaign completeness.")


if __name__ == "__main__":
    main()
