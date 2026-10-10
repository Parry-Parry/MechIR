"""Regression tests: constructing a model must not disable autograd globally.

`Dot.__init__`, `Cat.__init__` and `MonoT5.__init__` each used to call
`torch.set_grad_enabled(False)`, which is a process-wide flag rather than a
context manager. Building a model therefore switched autograd off for the
whole interpreter, which makes every gradient-based interpretability method
(attribution patching, EAP/EAP-IG, integrated gradients, probe training)
impossible to implement.
"""

import pytest
import torch

from mechir import Cat, Dot

MODEL = "bert-base-uncased"


def _tracks_grad():
    """True if a freshly built tensor participates in autograd."""
    return (torch.ones(2, requires_grad=True) * 2).requires_grad


@pytest.mark.parametrize("ctor", [Dot, Cat], ids=["Dot", "Cat"])
def test_construction_leaves_global_grad_untouched(ctor):
    assert torch.is_grad_enabled(), "precondition: grad enabled before the test"

    ctor(MODEL)

    assert torch.is_grad_enabled(), (
        "constructing a model disabled autograd process-wide; use a no_grad "
        "context on the inference paths instead of torch.set_grad_enabled"
    )
    assert _tracks_grad(), "unrelated tensors stopped tracking gradients"


def test_forward_is_differentiable():
    """`forward` must stay grad-capable; gradient-based methods build on it."""
    model = Dot(MODEL, pooling_type="mean")
    tok = model.tokenizer("gradient test", return_tensors="pt", padding=True)

    embeddings = model.forward(tok.input_ids, tok.attention_mask)

    assert embeddings.requires_grad, "forward() output is detached from autograd"
    assert embeddings.grad_fn is not None


def test_score_and_patch_still_run_without_grad():
    """The diagnostic entry points should not accumulate a graph."""
    model = Dot(MODEL, pooling_type="mean")
    tok = model.tokenizer("no grad here", return_tensors="pt", padding=True)
    batch = {"input_ids": tok.input_ids, "attention_mask": tok.attention_mask}

    scores, _, _, _ = model.score(batch, batch)
    assert not scores.requires_grad

    out, _ = model.patch(batch, batch, batch, patch_type="head_all")
    assert not out.requires_grad
    # Global state must survive the call.
    assert torch.is_grad_enabled()
