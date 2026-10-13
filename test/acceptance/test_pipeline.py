"""Pipeline with a hooked ranker and a hooked reader.

Uses a small real cross-encoder and pythia-70m so the whole path is
exercised without a large download.
"""

import pytest
import torch
from transformer_lens import HookedTransformer

from mechir import Cat
from mechir.pipeline import (
    ContextBuilder,
    HardPermutation,
    Pipeline,
    answer_logprob,
    attention_to_documents,
    groundedness,
    swap,
)

RANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
READER = "EleutherAI/pythia-70m"

QUERY = "what is the capital of france"
DOCS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany and a major European city.",
    "The Eiffel Tower is a landmark located in Paris, France.",
    "Bananas are a popular tropical fruit grown in warm climates.",
]
# Near-tied scores, so a single-head intervention can actually flip a rank.
TIGHT_DOCS = [
    "Paris is a large city in northern France.",
    "Paris, France is a city with many residents.",
    "The city of Paris lies in the north of France.",
    "France has a city called Paris in its north.",
]
ANSWER = "Paris"


@pytest.fixture(scope="module")
def reader():
    return HookedTransformer.from_pretrained(READER, device="cpu")


@pytest.fixture(scope="module")
def pipeline(reader):
    return Pipeline(Cat(RANKER), reader, boundary=HardPermutation())


# -- context assembly ------------------------------------------------------


def test_context_spans_cover_each_document(pipeline):
    context = pipeline.context_builder.build(QUERY, DOCS, [0, 1, 2, 3], answer=ANSWER)

    assert len(context.spans) == len(DOCS)
    for rank, span in enumerate(context.spans):
        assert span.rank == rank
        assert len(span) > 0
    # Spans must be contiguous and ordered.
    for earlier, later in zip(context.spans, context.spans[1:]):
        assert earlier.end == later.start


def test_context_spans_follow_the_order_not_the_candidate_index(pipeline):
    context = pipeline.context_builder.build(QUERY, DOCS, [3, 0], answer=ANSWER)

    assert [span.candidate for span in context.spans] == [3, 0]
    assert context.span_for_candidate(3).rank == 0
    assert context.span_for_rank(1).candidate == 0


def test_answer_tokens_sit_after_the_question(pipeline):
    context = pipeline.context_builder.build(QUERY, DOCS, [0], answer=ANSWER)

    assert context.answer_start is not None
    assert context.answer_start < context.length
    assert context.spans[-1].end <= context.answer_start


def test_document_truncation_is_applied(pipeline):
    builder = ContextBuilder(pipeline.context_builder.tokenizer, max_document_tokens=5)
    long_document = " ".join(["token"] * 200)

    context = builder.build(QUERY, [long_document], [0], answer=ANSWER)

    assert len(context.spans[0]) < 30


# -- metrics ---------------------------------------------------------------


def test_answer_logprob_is_negative_and_finite(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0, 1], answer=ANSWER)

    value = answer_logprob(reader, context)

    assert torch.isfinite(value)
    assert value < 0


def test_answer_logprob_requires_an_answer(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0], answer=None)

    with pytest.raises(ValueError, match="without an answer"):
        answer_logprob(reader, context)


def test_mean_reduction_is_greater_than_sum_for_multitoken_answers(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0], answer="Paris in France")

    total = answer_logprob(reader, context, reduction="sum")
    per_token = answer_logprob(reader, context, reduction="mean")

    assert per_token > total  # both negative, so the mean is less negative


def test_attention_to_documents_covers_every_rank(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0, 1, 2], answer=ANSWER)

    attention = attention_to_documents(reader, context)

    assert set(attention) == {0, 1, 2}
    assert all(value >= 0 for value in attention.values())


def test_attention_normalisation_modes_differ(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0, 1], answer=ANSWER)

    summed = attention_to_documents(reader, context, normalize="sum")
    averaged = attention_to_documents(reader, context, normalize="mean")

    # Per-token attention must be smaller than total mass over many tokens.
    assert averaged[0] < summed[0]


def test_attention_rejects_unknown_normalisation(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0], answer=ANSWER)

    with pytest.raises(ValueError, match="normalize must be"):
        attention_to_documents(reader, context, normalize="l2")


def test_groundedness_is_a_bounded_share(pipeline, reader):
    context = pipeline.context_builder.build(QUERY, DOCS, [0, 1], answer=ANSWER)

    value = groundedness(reader, context)

    assert 0.0 <= value <= 1.0 + 1e-6


# -- running and boundary interventions -----------------------------------


def test_run_ranks_the_relevant_document_first(pipeline):
    run = pipeline.run(QUERY, DOCS, ANSWER)

    assert run.order[0] == 0
    assert torch.isfinite(torch.tensor(run.metric))


def test_permutation_changes_the_metric(pipeline):
    """With the candidate set fixed, any difference is due to order alone."""
    baseline = pipeline.run(QUERY, DOCS, ANSWER)
    permuted = pipeline.run(QUERY, DOCS, ANSWER, op=swap(0, 2))

    assert permuted.order != baseline.order
    assert permuted.metric != baseline.metric


def test_position_sensitivity_reports_every_op(pipeline):
    results = pipeline.position_sensitivity(QUERY, DOCS, ANSWER)

    assert "identity" in results
    assert len(results) > 3
    assert all(isinstance(v, float) for v in results.values())
    # Order alone must move the metric.
    assert len(set(results.values())) > 1


def test_rank_vs_attention_reports_both_sides(pipeline):
    rows = pipeline.rank_vs_attention(QUERY, DOCS, ANSWER)

    assert len(rows) == len(DOCS)
    scores = [row["ranker_score"] for row in rows]
    assert scores == sorted(scores, reverse=True)
    assert all(row["reader_attention"] >= 0 for row in rows)


def test_position_controlled_attention_measures_at_one_position(pipeline):
    """Every candidate is promoted to rank 0, so recency is held constant."""
    rows = pipeline.position_controlled_attention(QUERY, DOCS, ANSWER)

    assert len(rows) == len(DOCS)
    assert sorted(row["candidate"] for row in rows) == list(range(len(DOCS)))
    assert all(row["attention_at_rank0"] >= 0 for row in rows)


# -- cross-boundary patching ----------------------------------------------


def test_patching_is_piecewise_constant_on_a_wide_candidate_set(pipeline):
    """The defining property of a hard boundary.

    Scores move under patching, but if no rank flips the reader sees an
    identical context and the metric delta is *exactly* zero. Measured here:
    all heads shift scores, none reorder, so every delta is 0.
    """
    result = pipeline.patch_ranker_heads(
        QUERY,
        DOCS,
        ANSWER,
        documents_p=[f"{d} capital" for d in DOCS],
        layers=[0, 1],
        heads=[0, 1, 2, 3],
        pad_to=48,
    )

    assert result.score_delta.max() > 0, "patching did not change any score"
    assert not result.rank_flipped.any()
    assert torch.equal(result.metric_delta, torch.zeros_like(result.metric_delta))
    assert result.flip_rate == 0.0


def test_patching_registers_when_ranks_can_flip(pipeline):
    """On a near-tied set, interventions do reorder and the metric moves."""
    result = pipeline.patch_ranker_heads(
        QUERY,
        TIGHT_DOCS,
        ANSWER,
        documents_p=[f"{d} capital" for d in TIGHT_DOCS],
        layers=[0, 1, 2],
        heads=list(range(12)),
        pad_to=48,
    )

    assert result.rank_flipped.any(), "no rank flipped even on a tied set"
    assert result.flip_rate > 0
    assert result.metric_delta.abs().max() > 0
    # A head that flipped nothing cannot have moved the metric.
    assert torch.equal(
        result.metric_delta[~result.rank_flipped],
        torch.zeros_like(result.metric_delta[~result.rank_flipped]),
    )


def test_patch_result_shapes_and_summary(pipeline):
    result = pipeline.patch_ranker_heads(
        QUERY, DOCS, ANSWER, layers=[0], heads=[0, 1], pad_to=48
    )

    assert result.metric_delta.shape == (1, 2)
    assert result.rank_flipped.shape == (1, 2)
    assert result.score_delta.shape == (1, 2)
    assert "flipped a rank" in result.summary()
