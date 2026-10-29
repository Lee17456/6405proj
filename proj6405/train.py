import argparse
import json
import math
import random
import time

import numpy as np

from .common import DATA_IDS, MODEL_IDS, MODELS, environment, load_rows, run_path, score_labels, signature, write_json


def train_svm(args, train, validation, manifest, path):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, args.ngram_max), min_df=2, max_features=args.max_features,
                                  sublinear_tf=True, dtype=np.float32)),
        ("svm", LinearSVC(C=args.svm_c, random_state=args.seed, max_iter=10000)),
    ])
    started = time.perf_counter()
    pipeline.fit(train["text"], train["label"])
    elapsed = time.perf_counter() - started
    metrics = score_labels(validation["label"], pipeline.predict(validation["text"]).tolist(), manifest["label_names"])
    joblib.dump(pipeline, path / "svm.joblib")
    write_json(path / "validation.json", metrics)
    return {"svm_fit_seconds": elapsed, "best_validation_macro_f1": metrics["macro_f1"], "best_epoch": None}


def train_neural(args, train, validation, manifest, path):
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup
    from .models import BatchCollator, TextClassifier, autocast_context, encode_rows, mamba_fast_available, tokenizer_from

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use CPU fp32 only for smoke testing, not the GPU benchmark.")
    if args.model == "mamba" and (not args.device.startswith("cuda") or not mamba_fast_available()) and not args.allow_slow_mamba:
        raise RuntimeError("Mamba fast CUDA path unavailable. Run doctor and fix extensions; --allow-slow-mamba is diagnostic only.")
    tokenizer = tokenizer_from(MODELS / args.model)
    model = TextClassifier(MODELS / args.model, args.model, manifest["num_labels"], args.pooling,
                           args.dropout, args.attention_backend).to(args.device)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_encoded = encode_rows(train, tokenizer, args.max_length)
    val_encoded = encode_rows(validation, tokenizer, args.max_length)
    collator = BatchCollator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_encoded, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collator)
    val_loader = DataLoader(val_encoded, batch_size=args.eval_batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -1.0
    best_epoch = None
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with (path / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, examples = 0.0, 0
            for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                group_start = (step // args.grad_accum) * args.grad_accum
                # Correct scaling also for the last partial accumulation group.
                group_size = min(args.grad_accum, len(loader) - group_start)
                with autocast_context(args.device, args.precision):
                    logits = model(**batch)
                    loss = loss_fn(logits.float(), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss. Inspect precision, learning rate and CUDA kernel compatibility.")
                total_loss += float(loss.detach()) * len(labels)
                examples += len(labels)
                scaler.scale(loss / group_size).backward()
                if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            model.eval()
            truth, predicted = [], []
            with torch.inference_mode():
                for batch in val_loader:
                    truth.extend(batch.pop("labels").tolist())
                    with autocast_context(args.device, args.precision):
                        logits = model(**{k: v.to(args.device) for k, v in batch.items()})
                    predicted.extend(logits.argmax(-1).cpu().tolist())
            metrics = score_labels(truth, predicted, manifest["label_names"])
            entry = {"epoch": epoch, "train_loss": total_loss / examples,
                     "validation_macro_f1": metrics["macro_f1"], "elapsed_seconds": time.perf_counter() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(entry, flush=True)
            if metrics["macro_f1"] > best:
                best, best_epoch = metrics["macro_f1"], epoch
                model.save(path, tokenizer)
                write_json(path / "validation.json", metrics)
    return {"epochs_validation_checkpoint_seconds": time.perf_counter() - started, "best_validation_macro_f1": best,
            "best_epoch": best_epoch, "train_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "mamba_fast_path": mamba_fast_available() if args.model == "mamba" else None}


def main():
    parser = argparse.ArgumentParser(description="Train on train; choose best checkpoint by validation Macro-F1. Test is separate.")
    parser.add_argument("--model", choices=["svm", *MODEL_IDS], required=True)
    parser.add_argument("--dataset", choices=list(DATA_IDS), required=True)
    parser.add_argument("--name", required=True, help="New run directory name; existing runs are never overwritten")
    parser.add_argument("--view", default="native")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["first", "last", "mean"])
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-slow-mamba", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-features", type=int, default=100000)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.grad_accum, args.max_length) < 1:
        parser.error("Epochs, batches, accumulation and length must be positive")
    if min(args.limit_train, args.limit_val) < 0:
        parser.error("Subset limits must be nonnegative")
    if args.model == "roberta" and args.max_length > 512:
        parser.error("RoBERTa supports at most 512 input tokens; use Longformer for longer inputs")
    if args.model == "longformer" and args.max_length > 4096:
        parser.error("This Longformer checkpoint supports at most 4096 tokens")
    random.seed(args.seed)
    np.random.seed(args.seed)
    train, manifest = load_rows(args.dataset, args.view, "train", args.limit_train)
    validation, _ = load_rows(args.dataset, args.view, "validation", args.limit_val)
    if args.view != "native" and args.model != "svm":
        if args.model not in manifest["participating_models"] or args.max_length < manifest["token_budget"]:
            parser.error("Model/length would truncate the shared view again; choose a compatible shared view")
    path = run_path(args.name)
    path.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "status": "training", "num_labels": manifest["num_labels"],
              "label_names": manifest["label_names"], "data_signature": manifest["signature"],
              "train_ids_signature": signature(train["id"]), "val_ids_signature": signature(validation["id"]),
              "n_train": len(train), "n_validation": len(validation),
              "environment": environment(), "model_source": None if args.model == "svm" else str(MODELS / args.model)}
    if args.model != "svm":
        config["model_download"] = json.loads((MODELS / args.model / "download_manifest.json").read_text(encoding="utf-8"))
    else:
        config.update(precision="fp32", device="cpu", max_length=None)
    write_json(path / "config.json", config)
    write_json(path / "data_manifest.json", manifest)
    started = time.perf_counter()
    result = train_svm(args, train, validation, manifest, path) if args.model == "svm" else train_neural(args, train, validation, manifest, path)
    result["train_seconds"] = time.perf_counter() - started
    result["train_timing_scope"] = "model initialization/local loading, feature preparation, fit, validation and checkpoint writes; excludes dataset loading and downloads"
    config.update(result, status="completed")
    write_json(path / "config.json", config)
    print(f"Completed: {path}. Test evaluation has NOT been run.")


if __name__ == "__main__":
    main()
