import pytest

from proj6405.common import score_labels, signature, valid_name
from proj6405.matrix import make_plan
from proj6405.views import common_prefix


class CharTokenizer:
    def __init__(self, width):
        self.width = width

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, return_offsets_mapping=False):
        offsets = [(i, min(i + self.width, len(text))) for i in range(0, len(text), self.width)]
        if truncation:
            offsets = offsets[:max_length - 2]
        return {"input_ids": list(range(len(offsets) + 2)), "offset_mapping": [(0, 0), *offsets, (0, 0)]}


def test_shared_prefix_fits_different_tokenizers():
    tokens = [CharTokenizer(1), CharTokenizer(3)]
    original = "abcdefghijklmnopqrstuvwxyz"
    prefix = common_prefix(original, tokens, 12)
    assert original.startswith(prefix)
    assert all(len(tok(prefix)["input_ids"]) <= 12 for tok in tokens)
    assert prefix == original[:10]


def test_macro_f1_includes_missing_class():
    metrics = score_labels([0, 0], [0, 0], ["a", "b"])
    assert metrics["macro_f1"] == 0.5
    assert metrics["accuracy"] == 1.0


@pytest.mark.parametrize("name", ["../bad", "a/b", "a\\b", "", "..", "/tmp/x"])
def test_run_names_prevent_path_escape(name):
    with pytest.raises(ValueError):
        valid_name(name)


def test_matrix_seeds_and_names():
    plan = make_plan("core", "main", [42, 43, 44], [2048])
    assert len(plan) == 24
    assert len({p["name"] for p in plan}) == len(plan)
    assert len([p for p in plan if p["model"] == "svm"]) == 6
    assert len({p["seed"] for p in plan if p["model"] == "mamba"}) == 3
    assert signature({"a": 1, "b": 2}) == signature({"b": 2, "a": 1})
