"""MonoT5 coverage.

MonoT5 previously could not be constructed at all, so none of this was
exercised. See test_patch_is_not_yet_supported for the one part that is
still broken.
"""

import pytest
import torch

from mechir import MonoT5

MODEL = "castorini/monot5-small-msmarco-10k"


@pytest.fixture(scope="module")
def monot5():
    return MonoT5(MODEL, return_cache=True)


@pytest.fixture(scope="module")
def batch(monot5):
    enc = monot5.tokenizer(
        "Query: what is a cat Document: a cat is an animal Relevant:",
        return_tensors="pt",
        padding=True,
    )
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


def test_constructs(monot5):
    """Regression: __init__ read self.model_name_or_path before assigning it."""
    assert monot5.model_name_or_path
    assert monot5.tokenizer is not None


def test_relevance_tokens_are_single_ids(monot5):
    """Regression: these were sequences including T5's trailing EOS token."""
    assert isinstance(monot5.pos_token, int)
    assert isinstance(monot5.neg_token, int)
    assert monot5.pos_token != monot5.neg_token


def test_rejects_multi_token_labels():
    with pytest.raises(ValueError, match="single token"):
        MonoT5(MODEL, pos_token="definitely relevant")


def test_score_returns_one_scalar_per_item(monot5, batch):
    """Regression: forward passed `one_hot_attention_mask`, which does not
    exist on HookedEncoderDecoder.forward (it is one_zero_attention_mask)."""
    scores, cache = monot5.score(batch)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch["input_ids"].shape[0],)
    assert torch.isfinite(scores).all()
    assert cache is None


def test_score_with_cache(monot5, batch):
    scores, cache = monot5.score(batch, cache=True)
    assert scores.shape == (batch["input_ids"].shape[0],)
    assert cache is not None


@pytest.mark.parametrize("softmax_output", [False, True])
def test_softmax_output_bounds(batch, softmax_output):
    model = MonoT5(MODEL, softmax_output=softmax_output)
    scores, _ = model.score(batch)

    if softmax_output:
        assert ((scores >= 0) & (scores <= 1)).all()
    else:
        # Raw logits, so not confined to the unit interval.
        assert torch.isfinite(scores).all()


@pytest.mark.parametrize("patch_type", ["block_all", "head_all"])
def test_patch_is_not_yet_supported(monot5, batch, patch_type):
    """MonoT5 patching is broken, and this pins why.

    PatchedMixin builds hook names with `utils.get_act_name`, which yields
    `blocks.{layer}.…`. T5 is an encoder-decoder, so its hooks are named
    `encoder.{layer}.…` and `decoder.{layer}.…` and the lookup misses:

        KeyError: '_model.blocks.0.hook_resid_pre'

    Fixing this needs the hook-naming scheme to become a property of the
    architecture rather than a hardcoded string in the patching loops.
    """
    with pytest.raises(KeyError, match=r"blocks\.0"):
        monot5.patch(batch, batch, patch_type=patch_type)
