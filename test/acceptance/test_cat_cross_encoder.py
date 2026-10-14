"""Cat against a real cross-encoder, checked against HuggingFace.

Before these fixes Cat could not load a real cross-encoder at all, and on an
untrained head it silently produced all-NaN patch maps. The existing
test_cat.py only asserts output shapes on bert-base-uncased, so none of this
was covered.
"""

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mechir import Cat

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERIES = ["what is a cat", "what is a cat", "capital of france"]
DOCS = [
    "a cat is a small domesticated animal",
    "the stock market closed lower today",
    "paris is the capital of france",
]
PERTURBED = [
    "a cat is a small domesticated animal cat",
    "the stock market closed cat today",
    "paris is the capital of france paris",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def cat_model():
    return Cat(MODEL, softmax_output=False)


def encode(tokenizer, docs):
    enc = tokenizer(
        QUERIES,
        docs,
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True,
    )
    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "token_type_ids": enc.token_type_ids,
    }


@pytest.fixture(scope="module")
def hf_scores(tokenizer):
    hf = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    enc = tokenizer(
        QUERIES,
        DOCS,
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True,
    )
    with torch.no_grad():
        return hf(**enc).logits[:, 0]


def test_loads_real_cross_encoder(cat_model):
    """Regression: classifier.W was transposed, so loading raised
    `size mismatch for classifier.W: copying a param with shape [1, 384] ...
    the shape in current model is [384, 1]`."""
    assert cat_model._model.classifier.W.shape[0] == cat_model._model.cfg.d_model


def test_pooler_weights_are_loaded(cat_model):
    """Regression: the pooler was instantiated but never converted."""
    assert cat_model._model.pooler.W.abs().sum() > 0


def test_scores_match_huggingface(cat_model, tokenizer, hf_scores):
    """The whole point: identical scores to the source model.

    Three separate defects each broke this -- the transposed classifier, the
    skipped pooler, and score() dropping token_type_ids.
    """
    scores, _ = cat_model.score(encode(tokenizer, DOCS))

    # The reference is computed on the CPU while mechir scores on cfg.device,
    # so compare on the CPU: otherwise this fails on device, not on value,
    # anywhere an accelerator is present (MPS, CUDA).
    scores = scores.cpu()

    assert torch.allclose(hf_scores, scores, atol=1e-3), (
        f"mechir {scores.tolist()} != huggingface {hf_scores.tolist()}"
    )


def test_token_type_ids_change_the_score(cat_model, tokenizer):
    """Guards the specific regression: token_type_ids must reach forward."""
    batch = encode(tokenizer, DOCS)
    with_types, _ = cat_model.score(batch)
    without = {k: v for k, v in batch.items() if k != "token_type_ids"}
    without_types, _ = cat_model.score(without)

    assert not torch.allclose(with_types, without_types), (
        "token_type_ids had no effect, so it is being dropped somewhere"
    )


def test_scores_rank_relevant_above_irrelevant(cat_model, tokenizer):
    scores, _ = cat_model.score(encode(tokenizer, DOCS))
    # index 1 is the off-topic document
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


@pytest.mark.parametrize("patch_type", ["block_all", "head_all", "head_by_pos"])
def test_patch_outputs_are_finite(cat_model, tokenizer, patch_type):
    """Regression: every patch output used to be NaN.

    An untrained width-1 head made log_softmax identically 0, so scores and
    scores_p were both 0 and linear_rank_function divided 0 by 0.
    """
    kwargs = (
        {"layer_head_list": [(0, 0), (1, 2)]} if patch_type == "head_by_pos" else {}
    )
    out, _ = cat_model.patch(
        encode(tokenizer, DOCS),
        encode(tokenizer, PERTURBED),
        patch_type=patch_type,
        **kwargs,
    )

    assert torch.isfinite(out).all(), f"{torch.isnan(out).sum()} NaN entries"
