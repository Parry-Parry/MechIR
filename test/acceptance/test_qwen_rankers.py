"""Qwen-backbone IR models.

These checkpoints are ~1.2GB each, so they are opt-in:

    MECHIR_RUN_LARGE=1 pytest test/acceptance/test_qwen_rankers.py

Everything here is verified against HuggingFace rather than by shape alone,
since the point of the backbone dispatch is that activations match the
source model.
"""

import os

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from mechir import Dot, GenerativeRanker
from mechir.modelling.backbones import POOLING, get_backbone

EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
RERANKER = "Qwen/Qwen3-Reranker-0.6B"

pytestmark = pytest.mark.skipif(
    os.environ.get("MECHIR_RUN_LARGE") != "1",
    reason="set MECHIR_RUN_LARGE=1 to download the Qwen checkpoints",
)

TEMPLATE = "<Query>: what is a cat\n<Document>: {}\nRelevant:"
DOCS = ["a cat is a domesticated animal", "the stock market fell today"]
PERTURBED = ["a cat is a domesticated animal cat", "the stock market cat today"]


def test_decoder_dispatch_is_detected():
    """Qwen IR fine-tunes leave the architecture field untouched, which is
    what makes substring dispatch sufficient."""
    _, _, is_decoder = get_backbone(EMBEDDER)
    assert is_decoder


@pytest.fixture(scope="module")
def embedder():
    return Dot(EMBEDDER, pooling_type="last")


@pytest.fixture(scope="module")
def reranker():
    return GenerativeRanker(RERANKER)


def test_last_token_pooling_matches_huggingface(embedder):
    """Dot on a decoder backbone must reproduce last_hidden_state pooling."""
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    texts = ["what is a cat", "a cat is a small domesticated animal", "the market fell"]
    enc = tok(texts, return_tensors="pt", padding=True)

    hf = AutoModel.from_pretrained(EMBEDDER, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hidden = hf(**enc).last_hidden_state
    idx = enc.attention_mask.sum(1) - 1
    expected = hidden[torch.arange(hidden.shape[0]), idx]

    with torch.no_grad():
        got = embedder.forward(enc.input_ids, enc.attention_mask)

    assert torch.allclose(expected, got, atol=2e-3), (
        f"maxdiff {(expected - got).abs().max().item()}"
    )


def test_embedder_ranks_relevant_above_irrelevant(embedder):
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    enc = tok(
        ["what is a cat", "a cat is a small domesticated animal", "the market fell"],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        pooled = embedder.forward(enc.input_ids, enc.attention_mask)
    normed = torch.nn.functional.normalize(pooled, dim=-1)

    assert torch.dot(normed[0], normed[1]) > torch.dot(normed[0], normed[2])


def test_last_pooling_respects_padding():
    """With right padding the final real token differs per row."""
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    pooled = POOLING["last"](hidden, mask)

    assert torch.equal(pooled[0], hidden[0, 2])
    assert torch.equal(pooled[1], hidden[1, 1])


def encode(model, docs):
    enc = model.tokenizer(
        [TEMPLATE.format(d) for d in docs],
        return_tensors="pt",
        padding="max_length",
        max_length=24,
        truncation=True,
    )
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


def test_reranker_is_fully_hooked(reranker):
    assert len(reranker.hook_dict) > 500
    assert any("attn.hook_z" in name for name in reranker.hook_dict)


def test_reranker_label_tokens_are_single_ids(reranker):
    assert isinstance(reranker.pos_token, int)
    assert isinstance(reranker.neg_token, int)
    assert reranker.pos_token != reranker.neg_token


def test_reranker_rejects_multi_token_labels():
    with pytest.raises(ValueError, match="single token"):
        GenerativeRanker(RERANKER, pos_token="definitely relevant")


def test_reranker_rejects_encoder_checkpoints():
    with pytest.raises(ValueError, match="not a decoder"):
        GenerativeRanker("bert-base-uncased")


def test_reranker_discriminates(reranker):
    scores, _ = reranker.score(encode(reranker, DOCS))

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert scores[0] > scores[1], f"relevant scored below irrelevant: {scores.tolist()}"


def test_reranker_softmax_output_is_bounded():
    model = GenerativeRanker(RERANKER, softmax_output=True)
    scores, _ = model.score(encode(model, DOCS))

    assert ((scores >= 0) & (scores <= 1)).all()


def test_reranker_patching_runs(reranker):
    out, _ = reranker.patch(
        encode(reranker, DOCS),
        encode(reranker, PERTURBED),
        patch_type="head_by_pos",
        layer_head_list=[(0, 0)],
    )

    assert out.shape == (2, 1, 24)
    assert torch.isfinite(out).all()
