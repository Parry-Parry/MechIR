"""SPLADE semantics: the activation, the pooling, and the scoring seam.

Each test here pins one claim about what SPLADE *is*, rather than that a
tensor came out the right shape:

* ``log(1 + relu(x))`` is exactly zero on non-positive logits, so the
  representation is genuinely sparse;
* the sequence reduction is a max, not a mean or a sum;
* padding is masked out, so a padded batch scores like an unpadded one;
* ``_patch_score`` contracts the patched document vector against the query.

The tensor-level checks need no model at all. The model-level ones use
``prajjwal1/bert-tiny`` (4.4M parameters, 2 layers, 2 heads); the real
checkpoints live in test/acceptance/test_splade_model.py.
"""

import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mechir.modelling import BLOCK_HOOK_NAMES, PatchedMixin
from mechir.modelling.splade import (
    REFERENCE_MODEL,
    Splade,
    splade_activation,
    splade_pool,
)

MODEL = "prajjwal1/bert-tiny"

QUERY = "what is a cat"
SAME = QUERY
UNRELATED = "roman aqueduct engineering history"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def model():
    return Splade(MODEL, return_cache=False)


def encode(tokenizer, texts, **kwargs):
    kwargs.setdefault("padding", True)
    return tokenizer(texts, return_tensors="pt", **kwargs)


# -- the activation: log(1 + relu(x)) --------------------------------------


def test_activation_is_exactly_zero_on_non_positive_logits():
    """relu zeroes them and log1p(0) == 0, so the zeros are exact."""
    logits = torch.tensor([[[-100.0, -1e-7, 0.0, 1.0]]])

    weights = splade_activation(logits)

    assert (weights[..., :3] == 0).all()
    assert weights[..., 3] > 0


def test_activation_is_log1p_of_the_positive_logits():
    logits = torch.tensor([[[0.5, 2.0, 10.0]]])

    assert torch.allclose(splade_activation(logits), torch.log1p(logits))


def test_activation_is_sparse_not_merely_small():
    """Sparsity is the point: count the exact zeros, do not check shapes.

    Roughly half of any zero-centred logit tensor is non-positive, so a
    dense ``[.., d_vocab]`` result whose zeros were only *nearly* zero
    would sail through a shape assertion and still not be SPLADE.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 512)

    weights = splade_activation(logits)

    non_zero = (weights != 0).sum().item()
    assert non_zero == (logits > 0).sum().item()
    assert 0 < non_zero < weights.numel()
    assert (weights >= 0).all()


# -- the pooling: max over the sequence, padding excluded -------------------


def test_pool_takes_the_max_not_the_mean_or_the_sum():
    """Built so max, mean and sum all differ: 6.0 vs 3.0 vs 9.0."""
    activations = torch.tensor([[[1.0], [2.0], [6.0]]])

    pooled = splade_pool(activations, torch.ones(1, 3))

    assert torch.allclose(pooled, torch.tensor([[6.0]]))
    assert not torch.allclose(pooled, activations.mean(dim=1))
    assert not torch.allclose(pooled, activations.sum(dim=1))


def test_pool_ignores_padding_positions():
    """Padding carries real MLM weights and would otherwise win the max."""
    activations = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = splade_pool(activations, mask)

    assert torch.allclose(pooled, torch.tensor([[3.0, 3.0]]))


def test_pool_without_a_mask_uses_every_position():
    activations = torch.tensor([[[1.0], [3.0]]])

    assert torch.allclose(splade_pool(activations, None), torch.tensor([[3.0]]))


def test_pool_of_an_all_padding_row_is_finite_zero():
    """Masking with zero rather than -inf keeps a degenerate row finite."""
    activations = torch.ones(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)

    pooled = splade_pool(activations, mask)

    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled, torch.zeros(1, 2))


def test_pool_accepts_an_integer_mask():
    """Tokenizers hand back int64 masks; the pooler must not promote the
    activations to int or fail on the dtype mismatch."""
    activations = torch.tensor([[[1.5], [2.5], [9.5]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    pooled = splade_pool(activations, mask)

    assert pooled.dtype == activations.dtype
    assert torch.allclose(pooled, torch.tensor([[2.5]]))


# -- the scoring seam ------------------------------------------------------


def test_patch_score_contracts_against_the_query():
    reps_q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reps_d = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    scored = Splade._patch_score(object(), reps_d, reps_q=reps_q)

    assert torch.allclose(scored, torch.tensor([17.0, 53.0]))


def test_patch_score_tolerates_surplus_context():
    """`patch` passes layer_head_list through to every driver."""
    scored = Splade._patch_score(
        object(), torch.ones(1, 2), reps_q=torch.ones(1, 2), layer_head_list=[]
    )

    assert torch.allclose(scored, torch.tensor([2.0]))


def test_patch_score_without_reps_q_fails_clearly():
    with pytest.raises(TypeError, match="reps_q"):
        Splade._patch_score(object(), torch.ones(1, 2))


def test_splade_uses_single_stack_hook_names():
    """One encoder stack, so the default blocks.{layer}.… naming is right."""
    assert Splade._hook_names.fget(object()) is BLOCK_HOOK_NAMES


def test_splade_inherits_the_patching_drivers():
    assert issubclass(Splade, PatchedMixin)
    for driver in (
        "get_act_patch_block_every",
        "get_act_patch_attn_head_out_all_pos",
        "get_act_patch_attn_head_by_pos",
    ):
        assert getattr(Splade, driver) is getattr(PatchedMixin, driver)


def test_reference_model_is_the_naver_checkpoint():
    assert REFERENCE_MODEL == "naver/splade-cocondenser-ensembledistil"


# -- the MLM head, which is what SPLADE scores through ---------------------


def test_the_mlm_head_is_loaded_and_transposed(model):
    """The hooked BERT path leaves BertMLMHead uninitialised.

    ``Splade`` copies the head across itself, and has to transpose:
    ``BertMLMHead`` computes ``resid @ W`` where ``nn.Linear`` computes
    ``x @ weight.T``. The weight is square, so loading it untransposed is
    silent -- hence this test rather than a shape check.
    """
    head = model._hf_model.cls.predictions

    assert torch.allclose(
        model._model.mlm_head.W.cpu(), head.transform.dense.weight.T.cpu(), atol=1e-6
    )
    assert torch.allclose(model._model.unembed.b_U.cpu(), head.bias.cpu(), atol=1e-6)


def test_hooked_logits_match_huggingface(model, tokenizer):
    """End-to-end check that the whole MLM path is wired up."""
    enc = encode(tokenizer, ["the cat sat on the mat", "a dog"])
    reference = AutoModelForMaskedLM.from_pretrained(MODEL).eval()

    with torch.no_grad():
        hooked = model._model(
            input=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc["token_type_ids"],
            return_type="logits",
        )
        expected = reference(**enc).logits

    assert torch.allclose(hooked.cpu(), expected, atol=1e-3)


def test_splade_rejects_a_model_without_an_mlm_head():
    from mechir.modelling.splade import _mlm_head_state

    with pytest.raises(ValueError, match="cls.predictions"):
        _mlm_head_state(object())


# -- the model's representation and scores ---------------------------------


def test_forward_is_one_sparse_vector_per_text(model, tokenizer):
    enc = encode(tokenizer, ["what is a cat", "a short document"])

    reps = model.forward(enc["input_ids"], enc["attention_mask"])

    assert reps.shape == (2, model.d_vocab)
    assert (reps >= 0).all()
    # Sparse, and sparse because of the activation rather than by accident:
    # every zero entry is a vocabulary term the head argued against at
    # every position.
    assert 0 < (reps != 0).float().mean().item() < 1.0


def test_scores_do_not_depend_on_how_far_the_batch_was_padded(model, tokenizer):
    """Regression: an unmasked max-pool reads padding positions.

    Padding carries MLM logits of its own -- the encoder masks it out of
    attention, not out of its output -- so the same pair would score
    differently at different ``max_length`` values. See ``_pool_mean`` in
    mechir.modelling.backbones for the same bug in the dense poolers.
    """

    def scored(max_length):
        kwargs = {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
        }
        queries = encode(tokenizer, [QUERY], **kwargs)
        documents = encode(tokenizer, [UNRELATED], **kwargs)
        return model.score(queries, documents)[0]

    tight, padded = scored(12), scored(40)

    assert torch.allclose(tight, padded, atol=1e-4), f"{tight} != {padded}"


def test_unmasked_pooling_would_have_differed(model, tokenizer):
    """The masking in the test above is load-bearing, not decorative."""

    def unmasked(max_length):
        enc = encode(
            tokenizer,
            [UNRELATED],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        with torch.no_grad():
            logits = model._model(
                input=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_type="logits",
            )
        return splade_pool(splade_activation(logits), None)

    assert not torch.allclose(unmasked(12), unmasked(40), atol=1e-4)


def test_an_identical_document_outscores_an_unrelated_one(model, tokenizer):
    queries = encode(tokenizer, [QUERY, QUERY])
    documents = encode(tokenizer, [SAME, UNRELATED])

    scores, _, _, _ = model.score(queries, documents)

    assert scores[0] > scores[1]


def test_score_returns_the_cache_only_when_asked(model, tokenizer):
    from transformer_lens.ActivationCache import ActivationCache

    enc = encode(tokenizer, [QUERY])
    batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    assert model.score(batch, batch, cache=False)[3] is None
    assert isinstance(model.score(batch, batch, cache=True)[3], ActivationCache)


@pytest.mark.parametrize("patch_type", ["block_all", "head_all", "head_by_pos"])
def test_patching_drivers_return_finite_results(model, tokenizer, patch_type):
    """Both document batches are tokenized together so they share a length.

    The patching hooks write cached activations into the corrupted run
    positionally, so clean and corrupted documents must agree on ``seq``.
    """
    queries = encode(tokenizer, [QUERY])
    documents = encode(tokenizer, ["a cat is a pet", "a dog is a pet"])
    clean = {k: v[:1] for k, v in documents.items()}
    corrupted = {k: v[1:] for k, v in documents.items()}

    kwargs = {"layer_head_list": [(0, 0)]} if patch_type == "head_by_pos" else {}
    out, cache = model.patch(queries, clean, corrupted, patch_type=patch_type, **kwargs)

    assert torch.isfinite(out).all()
    assert cache is None  # return_cache=False


def test_patch_without_documents_p_fails_clearly(model, tokenizer):
    enc = encode(tokenizer, [QUERY])
    batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    with pytest.raises(TypeError, match="documents_p"):
        model.patch(batch, batch)


def test_patch_accepts_the_collator_keyword(model, tokenizer):
    """DotDataCollator emits `perturbed_documents`, not `documents_p`."""
    documents = encode(tokenizer, ["a cat is a pet", "a dog is a pet"])
    queries = encode(tokenizer, [QUERY])
    clean = {k: v[:1] for k, v in documents.items()}
    corrupted = {k: v[1:] for k, v in documents.items()}

    out, _ = model.patch(
        queries, clean, perturbed_documents=corrupted, patch_type="block_all"
    )

    assert torch.isfinite(out).all()
