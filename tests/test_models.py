import pytest

torch = pytest.importorskip("torch")


def test_padding_aware_pooling():
    from proj6405.models import pooled_hidden
    hidden = torch.tensor([[[1.0], [3.0], [99.0]], [[99.0], [2.0], [4.0]]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert pooled_hidden(hidden, mask, "last").flatten().tolist() == [3.0, 4.0]
    assert pooled_hidden(hidden, mask, "mean").flatten().tolist() == [2.0, 3.0]


@pytest.mark.parametrize("kind", ["roberta", "mamba", "longformer"])
def test_tiny_backbone_backward_and_checkpoint(kind, tmp_path):
    from safetensors.torch import load_file
    from transformers import AutoModel, LongformerConfig, MambaConfig, RobertaConfig, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from proj6405.models import TextClassifier
    base = dict(vocab_size=16, hidden_size=16, num_hidden_layers=1, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    if kind == "mamba":
        config = MambaConfig(**base, state_size=4, expand=2, conv_kernel=4)
    else:
        base.update(num_attention_heads=2, intermediate_size=32, max_position_embeddings=66)
        config = RobertaConfig(**base) if kind == "roberta" else LongformerConfig(**base, attention_window=[4])
    source = tmp_path / "source"
    AutoModel.from_config(config).save_pretrained(source)
    raw = Tokenizer(WordLevel({"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "good": 4}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", bos_token="[CLS]", eos_token="[SEP]", unk_token="[UNK]")
    model = TextClassifier(source, kind, 2)
    ids = torch.tensor([[1, 4, 5, 2, 0, 0, 0, 0], [1, 4, 5, 6, 4, 2, 0, 0]])
    mask = (ids != 0).long()
    logits = model(ids, mask)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    model.eval()
    before = model(ids, mask).detach()
    destination = tmp_path / "checkpoint"
    model.save(destination, tokenizer)
    restored = TextClassifier(destination / "backbone", kind, 2)
    restored.head.load_state_dict(load_file(str(destination / "head.safetensors")))
    restored.eval()
    assert torch.allclose(before, restored(ids, mask), atol=1e-5)
