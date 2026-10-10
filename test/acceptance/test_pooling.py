"""Pooling must not depend on how far the batch was padded."""

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from mechir import Dot
from mechir.modelling.backbones import POOLING

MODEL = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
TEXTS = [
    "what is a cat",
    "a cat is a small domesticated animal that people keep as a pet",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def encode(tokenizer, max_length):
    return tokenizer(
        TEXTS,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )


# -- pure tensor checks ----------------------------------------------------


def test_mean_pooling_ignores_padding():
    hidden = torch.tensor(
        [[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]]  # third position is padding
    )
    mask = torch.tensor([[1, 1, 0]])

    pooled = POOLING["mean"](hidden, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_mean_pooling_without_a_mask_falls_back_to_plain_mean():
    hidden = torch.tensor([[[1.0], [3.0]]])

    assert torch.allclose(POOLING["mean"](hidden, None), torch.tensor([[2.0]]))


def test_mean_pooling_handles_an_all_padding_row():
    """Must not divide by zero even on a degenerate mask."""
    hidden = torch.ones(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)

    pooled = POOLING["mean"](hidden, mask)

    assert torch.isfinite(pooled).all()


def test_cls_pooling_is_unaffected_by_the_mask():
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])

    with_mask = POOLING["cls"](hidden, torch.tensor([[1, 1, 0]]))
    without = POOLING["cls"](hidden, None)

    assert torch.equal(with_mask, without)


# -- against the real model ------------------------------------------------


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_pooling_is_invariant_to_padding_length(tokenizer, pooling):
    """Regression: mean pooling averaged over padding, so the same text
    scored differently at different batch padding lengths (measured max
    difference 0.0715 between max_length 64 and 128)."""
    model = Dot(MODEL, pooling_type=pooling)

    short = encode(tokenizer, 64)
    long = encode(tokenizer, 128)
    with torch.no_grad():
        a = model.forward(short.input_ids, short.attention_mask)
        b = model.forward(long.input_ids, long.attention_mask)

    assert torch.allclose(a, b, atol=1e-4), (
        f"{pooling} pooling changed by {(a - b).abs().max():.4f} when only "
        "the padding length changed"
    )


def test_mean_pooling_matches_huggingface(tokenizer):
    """Masked mean is what the source bi-encoder actually computes."""
    model = Dot(MODEL, pooling_type="mean")
    hf = AutoModel.from_pretrained(MODEL).eval()

    encoded = encode(tokenizer, 64)
    with torch.no_grad():
        hidden = hf(**encoded).last_hidden_state
    mask = encoded.attention_mask.unsqueeze(-1).float()
    expected = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    with torch.no_grad():
        got = model.forward(encoded.input_ids, encoded.attention_mask)

    assert torch.allclose(
        expected, got, atol=2e-3
    ), f"maxdiff {(expected - got).abs().max():.2e}"


def test_unknown_pooling_type_is_rejected():
    with pytest.raises(AssertionError, match="Unknown pooling_type"):
        Dot(MODEL, pooling_type="maxsim")
