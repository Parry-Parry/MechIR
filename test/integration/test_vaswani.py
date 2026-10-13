"""End-to-end integration over the Vaswani collection.

Exercises the whole stack on real IR data -- MechIRDataset, the collators,
the perturbation decorator, scoring and patching -- for every model class.
Nothing here mocks anything.

Vaswani is 11,429 documents of 1970s electrical-engineering abstracts with
uppercase queries, so it is badly out of domain for MS MARCO-trained
rankers. Assertions are therefore directional rather than absolute: a
judged-relevant document should outscore a random one on average, and for
most individual pairs. Measured separation at the time of writing:

    TAS-B                 mean_rel=+95.06  mean_rand=+89.23  wins 11/12
    MiniLM cross-encoder  mean_rel=-4.38   mean_rand=-10.45  wins 12/12
    MonoT5-small          mean_rel=-9.30   mean_rand=-11.08  wins  9/12
    Qwen3-Reranker-0.6B   mean_rel=-0.61   mean_rand=-3.11   wins  7/8

No nDCG targets: they would be meaningless on this collection for these
models and would bake in a misleading baseline.
"""

import os
import random

import pandas as pd
import pytest
import torch

from mechir import Cat, Dot, MonoT5
from mechir.data import CatDataCollator, DotDataCollator, MechIRDataset
from mechir.perturb import perturbation

DATASET = "vaswani"
BI_ENCODER = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MONOT5 = "castorini/monot5-small-msmarco-10k"

N_QUERIES = 8
SEED = 0
# Fraction of pairs on which the relevant document must win. Set below 1.0
# because of the domain shift described above, above 0.5 so a coin flip fails.
MIN_PAIRWISE_WIN_RATE = 2 / 3

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def collection():
    """Judged-relevant and random (query, doc) pairs from Vaswani."""
    import ir_datasets

    dataset = ir_datasets.load(DATASET)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    docs = {d.doc_id: d.text for d in dataset.docs_iter()}

    relevant = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            relevant.setdefault(qrel.query_id, []).append(qrel.doc_id)

    query_ids = sorted(relevant)[:N_QUERIES]
    rng = random.Random(SEED)
    doc_ids = sorted(docs)

    positive, negative = [], []
    for query_id in query_ids:
        positive.append((queries[query_id], docs[relevant[query_id][0]]))
        while True:
            candidate = rng.choice(doc_ids)
            if candidate not in relevant[query_id]:
                negative.append((queries[query_id], docs[candidate]))
                break

    return {
        "positive": positive,
        "negative": negative,
        "query_ids": query_ids,
        "relevant": relevant,
    }


def assert_separates(positive_scores, negative_scores):
    """Relevant documents must outscore random ones."""
    positive = torch.as_tensor(positive_scores, dtype=torch.float32)
    negative = torch.as_tensor(negative_scores, dtype=torch.float32)

    assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
    assert positive.mean() > negative.mean(), (
        f"mean relevant {positive.mean():.4f} did not beat "
        f"mean random {negative.mean():.4f}"
    )

    win_rate = (positive > negative).float().mean().item()
    assert win_rate >= MIN_PAIRWISE_WIN_RATE, (
        f"relevant won only {win_rate:.0%} of pairs"
    )


# --------------------------------------------------------------------------
# Dot / bi-encoder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bi_encoder():
    return Dot(BI_ENCODER, pooling_type="cls")


def score_bi_encoder(model, pairs, doc_length=180):
    queries = model.tokenizer(
        [q for q, _ in pairs],
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    docs = model.tokenizer(
        [d for _, d in pairs],
        return_tensors="pt",
        padding="max_length",
        max_length=doc_length,
        truncation=True,
    )
    scores, _, _, _ = model.score(
        {"input_ids": queries.input_ids, "attention_mask": queries.attention_mask},
        {"input_ids": docs.input_ids, "attention_mask": docs.attention_mask},
    )
    return scores


def test_bi_encoder_separates_relevant_from_random(bi_encoder, collection):
    assert_separates(
        score_bi_encoder(bi_encoder, collection["positive"]),
        score_bi_encoder(bi_encoder, collection["negative"]),
    )


def test_bi_encoder_score_is_stable_across_padding(bi_encoder, collection):
    """Padding length must not change the ranking.

    Holds for both pooling modes now that mean pooling masks padding;
    test_pooling.py checks the invariance directly for each.
    """
    short = score_bi_encoder(bi_encoder, collection["positive"], doc_length=180)
    long = score_bi_encoder(bi_encoder, collection["positive"], doc_length=220)

    assert torch.equal(short.argsort(), long.argsort())


@pytest.mark.parametrize("patch_type", ["head_all", "head_by_pos"])
def test_bi_encoder_patching_runs_on_real_data(bi_encoder, collection, patch_type):
    pairs = collection["positive"][:2]
    queries = bi_encoder.tokenizer(
        [q for q, _ in pairs],
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True,
    )
    docs = bi_encoder.tokenizer(
        [d for _, d in pairs],
        return_tensors="pt",
        padding="max_length",
        max_length=24,
        truncation=True,
    )
    perturbed = bi_encoder.tokenizer(
        [f"{d} {q}" for q, d in pairs],
        return_tensors="pt",
        padding="max_length",
        max_length=24,
        truncation=True,
    )
    as_batch = lambda e: {  # noqa: E731
        "input_ids": e.input_ids,
        "attention_mask": e.attention_mask,
    }
    kwargs = {"layer_head_list": [(0, 0)]} if patch_type == "head_by_pos" else {}

    out, _ = bi_encoder.patch(
        as_batch(queries),
        as_batch(docs),
        as_batch(perturbed),
        patch_type=patch_type,
        **kwargs,
    )

    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# Cat / cross-encoder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cross_encoder():
    return Cat(CROSS_ENCODER)


def score_cross_encoder(model, pairs, max_length=200):
    enc = model.tokenizer(
        [q for q, _ in pairs],
        [d for _, d in pairs],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    scores, _ = model.score(
        {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "token_type_ids": enc.token_type_ids,
        }
    )
    return scores


def test_cross_encoder_separates_relevant_from_random(cross_encoder, collection):
    assert_separates(
        score_cross_encoder(cross_encoder, collection["positive"]),
        score_cross_encoder(cross_encoder, collection["negative"]),
    )


def test_cross_encoder_patching_runs_on_real_data(cross_encoder, collection):
    pairs = collection["positive"][:2]

    def batch(docs):
        enc = cross_encoder.tokenizer(
            [q for q, _ in pairs],
            docs,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "token_type_ids": enc.token_type_ids,
        }

    out, _ = cross_encoder.patch(
        batch([d for _, d in pairs]),
        batch([f"{d} extra" for _, d in pairs]),
        patch_type="head_all",
    )

    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# MonoT5
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def monot5():
    return MonoT5(MONOT5)


def score_monot5(model, pairs):
    texts = [f"Query: {q} Document: {d} Relevant:" for q, d in pairs]
    enc = model.tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=200,
        truncation=True,
    )
    scores, _ = model.score(
        {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
    )
    return scores


def test_monot5_separates_relevant_from_random(monot5, collection):
    assert_separates(
        score_monot5(monot5, collection["positive"]),
        score_monot5(monot5, collection["negative"]),
    )


def test_monot5_patching_works_on_real_data(monot5, collection):
    """Regression: the encoder-decoder hook-naming gap, on real data.

    `PatchedMixin` built `blocks.{layer}.…` names while `HookedEncoderDecoder`
    uses `encoder.{layer}.…` / `decoder.{layer}.…`, so every patching run
    raised `KeyError: '_model.blocks.0.hook_resid_pre'`. Hook naming is now an
    architecture-level strategy and MonoT5 selects the encoder stack.
    """
    query, document = collection["positive"][0]

    def encode(text):
        # Vaswani queries alone fill ~30 tokens, so a shorter window truncates
        # the document away entirely and the perturbation becomes a no-op.
        enc = monot5.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

    # Inject the query at the head of the document, where it survives
    # truncation and moves the score.
    clean = encode(f"Query: {query} Document: {document} Relevant:")
    perturbed = encode(f"Query: {query} Document: {query} {document} Relevant:")
    assert not torch.allclose(monot5.score(clean)[0], monot5.score(perturbed)[0]), (
        "perturbation did not move the score, so there is nothing to patch"
    )

    result = monot5.patch(clean, perturbed, patch_type="head_all")

    assert result.shape == (monot5._model.cfg.n_layers, monot5._model.cfg.n_heads)
    assert torch.isfinite(result).all()
    # Shapes alone would also be satisfied by a loop that patched nothing.
    assert (result != 0).any()


# --------------------------------------------------------------------------
# Data layer and perturbations
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mechir_dataset(collection):
    import ir_datasets

    dataset = ir_datasets.load(DATASET)
    qrels = pd.DataFrame(dataset.qrels_iter())
    pairs = qrels[qrels["query_id"].isin(collection["query_ids"])].head(8)
    return MechIRDataset(DATASET, pairs=pairs)


def test_dataset_yields_real_query_document_pairs(mechir_dataset):
    assert len(mechir_dataset) > 0

    query, document = mechir_dataset[0]
    assert isinstance(query, str) and query.strip()
    assert isinstance(document, str) and document.strip()


def test_dot_collator_produces_model_ready_batches(mechir_dataset, bi_encoder):
    @perturbation
    def append_marker(document, query=None):
        return f"{document} MECHIR"

    collate = DotDataCollator(
        bi_encoder.tokenizer,
        transformation_func=append_marker,
        q_max_length=32,
        d_max_length=180,
    )
    batch = collate([mechir_dataset[i] for i in range(4)])

    assert set(batch) >= {"queries", "documents", "perturbed_documents"}
    for key in "queries", "documents", "perturbed_documents":
        assert batch[key]["input_ids"].shape[0] == 4

    scores, _, _, _ = bi_encoder.score(batch["queries"], batch["documents"])
    perturbed, _, _, _ = bi_encoder.score(
        batch["queries"], batch["perturbed_documents"]
    )
    assert torch.isfinite(scores).all()
    assert torch.isfinite(perturbed).all()


def test_cat_collator_emits_token_type_ids(mechir_dataset, cross_encoder):
    """token_type_ids must reach the model, or cross-encoder scores are wrong."""

    @perturbation
    def append_marker(document, query=None):
        return f"{document} MECHIR"

    collate = CatDataCollator(
        cross_encoder.tokenizer,
        transformation_func=append_marker,
        q_max_length=32,
        d_max_length=160,
    )
    batch = collate([mechir_dataset[i] for i in range(4)])

    assert "token_type_ids" in batch["sequences"]

    scores, _ = cross_encoder.score(batch["sequences"])
    assert torch.isfinite(scores).all()


def test_injected_query_terms_raise_cross_encoder_scores(cross_encoder, collection):
    """A crude axiomatic check: repeating query terms in a random document
    should move a lexical cross-encoder's score upward."""
    negative = collection["negative"]
    stuffed = [(q, f"{d} {q} {q}") for q, d in negative]

    baseline = score_cross_encoder(cross_encoder, negative)
    boosted = score_cross_encoder(cross_encoder, stuffed)

    assert boosted.mean() > baseline.mean(), (
        f"term injection lowered the mean score: "
        f"{baseline.mean():.4f} -> {boosted.mean():.4f}"
    )


# --------------------------------------------------------------------------
# Qwen (opt-in: ~1.2GB checkpoints)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("MECHIR_RUN_LARGE") != "1",
    reason="set MECHIR_RUN_LARGE=1 to download the Qwen checkpoints",
)
def test_qwen_reranker_separates_relevant_from_random(collection):
    from mechir import GenerativeRanker

    model = GenerativeRanker("Qwen/Qwen3-Reranker-0.6B")

    def score(pairs):
        texts = [f"<Query>: {q}\n<Document>: {d}\nRelevant:" for q, d in pairs]
        enc = model.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        )
        scores, _ = model.score(
            {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        )
        return scores

    assert_separates(score(collection["positive"]), score(collection["negative"]))
