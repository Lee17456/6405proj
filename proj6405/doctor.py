import argparse
import json
import shutil

from .common import MODEL_IDS, MODELS, REPORTS, environment, write_json


def main():
    parser = argparse.ArgumentParser(description="Inspect runtime; optionally exercise a real forward/backward pass")
    parser.add_argument("--smoke", choices=list(MODEL_IDS))
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = parser.parse_args()
    report = environment()
    report.update(nvcc=shutil.which("nvcc"), nvidia_smi=shutil.which("nvidia-smi"), tmux=shutil.which("tmux"))
    try:
        from .models import mamba_fast_available
        report["mamba_fast_path"] = mamba_fast_available()
    except Exception as exc:
        report["mamba_error"] = repr(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    write_json(REPORTS / "environment.json", report)
    if args.smoke:
        import time
        import torch
        from .models import TextClassifier, autocast_context, tokenizer_from
        if not torch.cuda.is_available():
            raise RuntimeError("GPU smoke requires CUDA")
        if args.smoke == "mamba" and not report.get("mamba_fast_path"):
            raise RuntimeError("Mamba fast path is unavailable")
        model = TextClassifier(MODELS / args.smoke, args.smoke, 2).cuda().train()
        tokenizer = tokenizer_from(MODELS / args.smoke)
        encoded = tokenizer(["A test sentence about a movie. " * args.length], truncation=True,
                            max_length=args.length, return_tensors="pt", return_token_type_ids=False)
        encoded = {k: v.cuda() for k, v in encoded.items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context("cuda", args.precision):
            loss = torch.nn.functional.cross_entropy(model(**encoded).float(), torch.tensor([1], device="cuda"))
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Non-finite smoke loss or gradient")
        result = {"model": args.smoke, "tokens": encoded["input_ids"].shape[1], "loss": float(loss),
                  "forward_backward_seconds_first_call": time.perf_counter() - started,
                  "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                  "note": "No optimizer state allocated. Not a training-memory or speed benchmark."}
        write_json(REPORTS / f"smoke_{args.smoke}_{args.length}.json", result)
        print(result)


if __name__ == "__main__":
    main()
