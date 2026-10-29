from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if torch.device(device).type != "cuda":
        raise ValueError("Use fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use --precision fp16 or fp32")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def mamba_fast_available():
    from transformers.models.mamba import modeling_mamba
    return bool(modeling_mamba.is_fast_path_available)


def pooled_hidden(hidden, mask, pooling):
    if pooling == "first":
        return hidden[:, 0]
    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
    # Robust even if a caller changes padding side. Mamba itself uses right padding here.
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(mask == 0, -1).max(1).values.clamp_min(0)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class TextClassifier(nn.Module):
    def __init__(self, source, kind, num_labels, pooling=None, dropout=0.1, attention_backend="sdpa"):
        super().__init__()
        self.kind = kind
        self.pooling = pooling or ("last" if kind == "mamba" else "first")
        kwargs = {"local_files_only": True, "torch_dtype": torch.float32}
        if kind == "roberta":
            kwargs["attn_implementation"] = attention_backend
        self.backbone = AutoModel.from_pretrained(source, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_labels)
        if kind == "mamba":
            self.backbone.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if self.kind == "longformer":
            global_mask = torch.zeros_like(attention_mask)
            global_mask[:, 0] = 1
            kwargs["global_attention_mask"] = global_mask
        if self.kind == "mamba":
            kwargs["use_cache"] = False
        hidden = self.backbone(**kwargs).last_hidden_state
        return self.head(self.dropout(pooled_hidden(hidden, attention_mask, self.pooling)))

    def save(self, path, tokenizer):
        from safetensors.torch import save_file
        path = Path(path)
        self.backbone.save_pretrained(path / "backbone", safe_serialization=True)
        tokenizer.save_pretrained(path / "tokenizer")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}, str(path / "head.safetensors"))


def tokenizer_from(source):
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode_rows(rows, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False,
                         return_token_type_ids=False)
    return rows.map(tokenize, batched=True, remove_columns=[c for c in rows.column_names if c != "label"],
                    load_from_cache_file=False, desc="Tokenizing")


class BatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([{k: row[k] for k in ("input_ids", "attention_mask")} for row in rows],
                                   padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        return batch
