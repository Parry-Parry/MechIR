"""`LateInteraction` end to end on a real encoder.

The MaxSim maths is pinned without a model in test/unit/test_late_interaction.py;
what is checked here is that a loaded ranker behaves like one -- token
embeddings rather than a pooled vector out of `forward`, scores that do not
grow with the document length, and the three patching drivers returning
correctly shaped maps.
"""

import pytest
import torch
from transformers import AutoTokenizer
from transformer_lens.ActivationCache import ActivationCache

from mechir.modelling.late_interaction import LateInteraction, maxsim

MODEL = "prajjwal1/bert-tiny"
QUERY = "what is a cat"
DOCUMENT = "a cat is a small domesticated animal"

#: Padding lengths the same document is scored at. The old reduction scored
#: 18.8 / 54.7 / 208.5 for document lengths 7 / 20 / 100.
PAD_LENGTHS = (16, 48, 128)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def model():
    return LateInteraction(MODEL, return_cache=True)


def encode(tokenizer, texts, max_length):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


# -- a late-interaction ranker keeps its tokens ---------------------------


def test_forward_returns_token_embeddings(model, tokenizer):
    """Not pooled: one vector per position, unlike `Dot.forward`."""
    batch = encode(tokenizer, [QUERY], max_length=16)

    out = model.forward(batch["input_ids"], batch["attention_mask"])

    assert out.dim() == 3  # (batch, seq, d_model)
    assert out.shape[:2] == batch["input_ids"].shape


def test_normalised_token_embeddings_are_unit_length(model, tokenizer):
    batch = encode(tokenizer, [QUERY], max_length=16)

    out = model.forward(batch["input_ids"], batch["attention_mask"])

    assert torch.allclose(
        out.norm(dim=-1), torch.ones_like(out.norm(dim=-1)), atol=1e-5
    )


def test_normalisation_can_be_turned_off(tokenizer):
    plain = LateInteraction(MODEL, normalize=False)
    batch = encode(tokenizer, [QUERY], max_length=16)

    out = plain.forward(batch["input_ids"], batch["attention_mask"])

    assert not torch.allclose(out.norm(dim=-1), torch.ones_like(out.norm(dim=-1)))


# -- the regression, on real activations ----------------------------------


def test_score_does_not_grow_with_padding_length(model, tokenizer):
    """The same query and document, padded to three lengths, score the same.

    A real encoder emits ordinary activations at padding positions, so
    without the document mask those positions compete for the max and the
    score drifts with `max_length`. Under the transposed reduction the score
    grows with the padded length outright, because every padding position
    adds a summand.
    """
    scores = []
    for max_length in PAD_LENGTHS:
        queries = encode(tokenizer, [QUERY], max_length=16)
        documents = encode(tokenizer, [DOCUMENT], max_length=max_length)
        score, _, _, _ = model.score(queries, documents)
        scores.append(float(score))

    assert scores == pytest.approx([scores[0]] * len(PAD_LENGTHS), abs=1e-3)


def test_score_is_bounded_by_the_query_length(model, tokenizer):
    """Normalised MaxSim sums one cosine per query token, so it cannot exceed them.

    This is the structural reason the score cannot grow with the document:
    the number of summands is set by the query. The transposed reduction has
    no such bound.
    """
    queries = encode(tokenizer, [QUERY], max_length=16)
    documents = encode(tokenizer, [DOCUMENT], max_length=64)

    score, reps_q, reps_d, _ = model.score(queries, documents)
    n_query_tokens = int(queries["attention_mask"].sum())

    assert float(score) <= n_query_tokens + 1e-4
    # And `score` really is MaxSim over the representations it returns.
    recomputed = maxsim(
        reps_q, reps_d, queries["attention_mask"], documents["attention_mask"]
    )
    assert float(score) == pytest.approx(float(recomputed), abs=1e-4)


def test_repeating_a_document_does_not_inflate_its_score(model, tokenizer):
    """Concatenating a document to itself adds no new content, so no score."""
    queries = encode(tokenizer, [QUERY], max_length=16)
    once = encode(tokenizer, [DOCUMENT], max_length=128)
    twice = encode(tokenizer, [f"{DOCUMENT} {DOCUMENT} {DOCUMENT}"], max_length=128)

    short, _, _, _ = model.score(queries, once)
    long, _, _, _ = model.score(queries, twice)

    # Not exactly equal -- the repeats shift the real tokens' contexts -- but
    # nowhere near the 3x the transposed reduction would produce.
    assert float(long) < float(short) * 1.5


# -- scoring and caching ---------------------------------------------------


@pytest.mark.parametrize("cache_flag", [False, True])
def test_score_cache_flags(model, tokenizer, cache_flag):
    queries = encode(tokenizer, [QUERY], max_length=16)
    documents = encode(tokenizer, [DOCUMENT], max_length=32)

    scores, reps_q, reps_d, cache = model.score(queries, documents, cache=cache_flag)

    assert scores.shape == (1,)
    assert reps_q.dim() == 3 and reps_d.dim() == 3
    if cache_flag:
        assert isinstance(cache, ActivationCache)
    else:
        assert cache is None


def test_score_batches_independently(model, tokenizer):
    queries = encode(tokenizer, [QUERY, QUERY], max_length=16)
    documents = encode(tokenizer, [DOCUMENT, "an unrelated string"], max_length=32)

    scores, _, _, _ = model.score(queries, documents)

    assert scores.shape == (2,)
    assert float(scores[0]) != float(scores[1])


# -- patching --------------------------------------------------------------


@pytest.mark.parametrize("patch_type", ["block_all", "head_all", "head_by_pos"])
def test_patch_methods_shapes(model, tokenizer, patch_type):
    queries = encode(tokenizer, [QUERY], max_length=8)
    documents = encode(tokenizer, [DOCUMENT], max_length=8)
    documents_p = encode(tokenizer, [f"{DOCUMENT} MECHIR"], max_length=8)
    n_layers = model._model.cfg.n_layers

    if patch_type == "head_by_pos":
        out, cache = model.patch(
            queries,
            documents,
            documents_p,
            patch_type=patch_type,
            layer_head_list=[(0, 0)],
        )
        assert out.shape == (2, 1, 8)
    elif patch_type == "head_all":
        out, cache = model.patch(queries, documents, documents_p, patch_type=patch_type)
        assert out.shape == (n_layers, model._model.cfg.n_heads)
    else:
        out, cache = model.patch(queries, documents, documents_p, patch_type=patch_type)
        assert out.shape == (3, n_layers, 8)

    assert isinstance(cache, ActivationCache)  # return_cache=True
    assert torch.isfinite(out).all()


def test_patch_accepts_the_collator_keyword(model, tokenizer):
    """`DotDataCollator` emits "perturbed_documents"; `model.patch(**batch)` must work."""
    queries = encode(tokenizer, [QUERY], max_length=8)
    documents = encode(tokenizer, [DOCUMENT], max_length=8)

    out, _ = model.patch(
        queries, documents, perturbed_documents=documents, patch_type="head_all"
    )

    assert out.shape == (model._model.cfg.n_layers, model._model.cfg.n_heads)


def test_patch_without_perturbed_documents_fails_clearly(model, tokenizer):
    batch = encode(tokenizer, [QUERY], max_length=8)

    with pytest.raises(TypeError, match="documents_p"):
        model.patch(batch, batch)
