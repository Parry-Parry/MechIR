import pytest
from mechir import Cat, Dot
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_dot_instantiation(tokenizer):
    model = Dot("bert-base-uncased", pooling_type="mean", return_cache=False)
    enc = tokenizer("Unit test Dot", return_tensors="pt", padding=True)
    out = model.forward(enc.input_ids, enc.attention_mask)
    assert out.shape[0] == 1


def test_cat_instantiation(tokenizer):
    model = Cat(
        "bert-base-uncased",
        tokenizer=tokenizer,
        softmax_output=True,
        return_cache=False,
    )
    enc = tokenizer("Unit test Cat", return_tensors="pt", padding=True)
    out = model.forward(enc.input_ids, enc.attention_mask)
    assert out.dim() == 1
