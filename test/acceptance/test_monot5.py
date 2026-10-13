"""MonoT5 coverage.

MonoT5 previously could not be constructed at all, so none of this was
exercised. Patching was broken for a second reason -- the hook names were
built for a single-stack model -- which the tests below now pin as fixed.
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


def test_hook_names_target_the_encoder_stack(monot5):
    """Regression: hook names were built as `blocks.{layer}.…`.

    T5 is an encoder-decoder, so `HookedEncoderDecoder` names its hooks
    `encoder.{layer}.…` and `decoder.{layer}.…`; every patching run died on
    `KeyError: '_model.blocks.0.hook_resid_pre'`. Naming is now a property of
    the architecture, so assert the names MonoT5 asks for really exist.
    """
    for component in ("resid_pre", "attn_out", "mlp_out", "z", "pattern"):
        name = monot5._hook_names(component, 0)
        assert name.startswith("_model.encoder.0."), name
        assert name in monot5.hook_dict, name


@pytest.mark.parametrize(
    "patch_type, shape",
    [
        ("block_all", lambda m, seq: (3, m._model.cfg.n_layers, seq)),
        ("head_all", lambda m, seq: (m._model.cfg.n_layers, m._model.cfg.n_heads)),
        ("head_by_pos", lambda m, seq: (2, 1, seq)),
    ],
)
def test_patch_returns_the_shared_shapes(monot5, batch, patch_type, shape):
    """Patching works, and returns the same shapes as Dot and Cat."""
    seq_len = batch["input_ids"].shape[1]

    result, _ = monot5.patch(
        batch, batch, patch_type=patch_type, layer_head_list=[(0, 0)]
    )

    assert result.shape == shape(monot5, seq_len)
    assert torch.isfinite(result).all()


def test_patch_responds_to_a_perturbation(monot5):
    """A patch map over a real perturbation is not identically zero.

    Shapes alone would also be satisfied by a loop that patched nothing, so
    pin that the hooks actually move the score.
    """
    enc = monot5.tokenizer(
        [
            "Query: what is a cat Document: a cat is an animal Relevant:",
            "Query: what is a cat Document: a cat is an animal cat Relevant:",
        ],
        return_tensors="pt",
        padding=True,
    )
    clean = {"input_ids": enc.input_ids[:1], "attention_mask": enc.attention_mask[:1]}
    perturbed = {
        "input_ids": enc.input_ids[1:],
        "attention_mask": enc.attention_mask[1:],
    }

    result, _ = monot5.patch(clean, perturbed, patch_type="head_all")

    assert torch.isfinite(result).all()
    assert (result != 0).any()


def test_patch_does_not_mutate_the_caller_batch(monot5, batch):
    """The patching loops used to assign device tensors back into the batch."""
    before = batch["input_ids"].clone()

    monot5.patch(batch, batch, patch_type="head_all")

    assert torch.equal(batch["input_ids"], before)
    assert batch["input_ids"].device == before.device
