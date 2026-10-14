"""SPLADE against real masked-language-model checkpoints.

The fast suite runs on ``bert-base-uncased``, a genuine
``BertForMaskedLM`` whose config declares that architecture -- which is
the weight-conversion path the real SPLADE checkpoints take, and the one
that would silently skip the MLM head if ``Splade`` did not load it
itself.

The naver checkpoint is the real thing but is a ~500MB cold download, so
that test carries the ``integration`` marker explicitly: ``make test``,
the Docker default command and the per-push CI job all select on
``-m "not integration"``.
"""

import pytest
import torch

from mechir.modelling.splade import (
    REFERENCE_MODEL,
    Splade,
    splade_activation,
    splade_pool,
)

MODEL = "bert-base-uncased"

QUERY = "what is a cat"
SAME = QUERY
UNRELATED = "the history of roman aqueduct engineering"


@pytest.fixture(scope="module")
def model():
    return Splade(MODEL, return_cache=True)


def encode(tokenizer, texts, **kwargs):
    kwargs.setdefault("padding", True)
    return tokenizer(texts, return_tensors="pt", **kwargs)


def reference_reps(model, batch):
    """SPLADE's definition, computed straight off the HuggingFace model.

    ``Splade`` keeps the source ``AutoModelForMaskedLM`` alongside its
    hooked copy, so this runs on the same device and the only thing the
    comparison can disagree about is whether the hooked encoder plus the
    MLM head reproduce the source model.
    """
    inputs = {
        key: value.to(model._device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    with torch.no_grad():
        logits = model._hf_model(**inputs).logits
    return splade_pool(splade_activation(logits), inputs["attention_mask"])


def test_forward_reproduces_the_definition(model):
    batch = encode(model.tokenizer, [QUERY, UNRELATED])

    reps = model.forward(batch["input_ids"], batch["attention_mask"])
    expected = reference_reps(model, batch)

    assert reps.shape == (2, model.d_vocab)
    torch.testing.assert_close(reps, expected, rtol=1e-3, atol=1e-3)


def test_the_representation_is_sparse(model):
    batch = encode(model.tokenizer, [QUERY])

    reps = model.forward(batch["input_ids"], batch["attention_mask"])

    density = (reps != 0).float().mean().item()
    assert 0.0 < density < 0.9, density
    assert (reps >= 0).all()


def test_scores_are_the_dot_product_of_the_two_sparse_vectors(model):
    queries = encode(model.tokenizer, [QUERY, QUERY])
    documents = encode(model.tokenizer, [SAME, UNRELATED])

    scores, reps_q, reps_d, cache = model.score(queries, documents)
    expected = (reference_reps(model, queries) * reference_reps(model, documents)).sum(
        dim=-1
    )

    assert cache is None
    torch.testing.assert_close(scores, expected, rtol=1e-2, atol=1e-2)


def test_an_identical_document_outscores_an_unrelated_one(model):
    queries = encode(model.tokenizer, [QUERY, QUERY])
    documents = encode(model.tokenizer, [SAME, UNRELATED])

    scores, _, _, _ = model.score(queries, documents)

    assert scores[0] > scores[1]


@pytest.mark.parametrize("max_length", [16, 48])
def test_padding_does_not_change_the_score(model, max_length):
    """A padded batch must score exactly like a tightly packed one."""
    kwargs = {"padding": "max_length", "max_length": max_length, "truncation": True}
    queries = encode(model.tokenizer, [QUERY], **kwargs)
    documents = encode(model.tokenizer, [UNRELATED], **kwargs)

    padded, _, _, _ = model.score(queries, documents)

    tight_q = encode(model.tokenizer, [QUERY])
    tight_d = encode(model.tokenizer, [UNRELATED])
    tight, _, _, _ = model.score(tight_q, tight_d)

    torch.testing.assert_close(padded, tight, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("patch_type", ["block_all", "head_all", "head_by_pos"])
def test_patching_drivers_produce_the_documented_shapes(model, patch_type):
    documents = encode(model.tokenizer, ["a cat is a pet", "a dog is a pet"])
    queries = encode(model.tokenizer, [QUERY])
    clean = {key: value[:1] for key, value in documents.items()}
    corrupted = {key: value[1:] for key, value in documents.items()}
    seq = clean["input_ids"].shape[1]
    layers = model._model.cfg.n_layers
    heads = model._model.cfg.n_heads

    kwargs = {"layer_head_list": [(0, 0)]} if patch_type == "head_by_pos" else {}
    out, cache = model.patch(queries, clean, corrupted, patch_type=patch_type, **kwargs)

    expected = {
        "block_all": (3, layers, seq),
        "head_all": (layers, heads),
        "head_by_pos": (2, 1, seq),
    }[patch_type]
    assert tuple(out.shape) == expected
    assert torch.isfinite(out).all()
    assert cache is not None  # return_cache=True


# -- the real checkpoint ---------------------------------------------------


@pytest.mark.integration
def test_the_reference_checkpoint_is_sparse_and_ranks_sensibly():
    """A trained SPLADE head is far sparser than a plain MLM head."""
    model = Splade(REFERENCE_MODEL)
    queries = encode(model.tokenizer, [QUERY, QUERY])
    documents = encode(
        model.tokenizer,
        ["a cat is a small domesticated animal kept as a pet", UNRELATED],
    )

    scores, reps_q, _, _ = model.score(queries, documents)

    density = (reps_q != 0).float().mean().item()
    assert density < 0.05, density
    assert scores[0] > scores[1]
