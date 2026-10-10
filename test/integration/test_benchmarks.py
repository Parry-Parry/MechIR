"""Benchmark loaders against the real datasets.

TREC DL pulls the MS MARCO passage corpus (8.8M passages) on first use, so
those tests are opt-in.
"""

import os

import pytest

from mechir.data import (
    RagExample,
    iter_examples,
    load_hotpotqa,
    load_trec_dl,
    sample_filler,
)

pytestmark = pytest.mark.integration

needs_msmarco = pytest.mark.skipif(
    os.environ.get("MECHIR_RUN_LARGE") != "1",
    reason="set MECHIR_RUN_LARGE=1 to download the MS MARCO passage corpus",
)


# -- RagExample ------------------------------------------------------------


def test_relevance_labels_must_match_documents():
    with pytest.raises(ValueError, match="2 documents but 1 relevance"):
        RagExample(query_id="q", query="q", documents=["a", "b"], relevance=[1])


def test_relevant_and_irrelevant_partition_the_candidates():
    example = RagExample(
        query_id="q",
        query="q",
        documents=["a", "b", "c", "d"],
        relevance=[3, 0, 1, 0],
    )

    assert example.relevant == [0, 2]
    assert example.irrelevant == [1, 3]


def test_iter_examples_skips_answerless_examples():
    with_answer = RagExample("1", "q", ["a"], [1], answer="yes")
    without = RagExample("2", "q", ["a"], [1])

    assert list(iter_examples([with_answer, without])) == [with_answer]


def test_sample_filler_draws_from_other_queries():
    examples = [
        RagExample("1", "q1", ["only-in-one"], [1]),
        RagExample("2", "q2", ["only-in-two"], [1]),
    ]

    filler = sample_filler(examples, exclude=0, count=3)

    assert filler == ["only-in-two"] * 3


def test_sample_filler_needs_more_than_one_example():
    with pytest.raises(ValueError, match="at least two examples"):
        sample_filler([RagExample("1", "q", ["a"], [1])], exclude=0, count=1)


# -- HotpotQA --------------------------------------------------------------


@pytest.fixture(scope="module")
def hotpot():
    return load_hotpotqa(split="validation", max_queries=5)


def test_hotpotqa_examples_carry_answers(hotpot):
    """The reason this uses the HF distractor config: beir/hotpotqa has no
    answer field, so reader metrics cannot be computed from it."""
    assert len(hotpot) == 5
    for example in hotpot:
        assert example.answer
        assert isinstance(example.answer, str)


def test_hotpotqa_gives_ten_candidates_with_gold_labels(hotpot):
    for example in hotpot:
        assert len(example.documents) == 10
        assert set(example.relevance) <= {0, 1}
        # Two supporting paragraphs is the HotpotQA design.
        assert 1 <= len(example.relevant) <= 3
        assert len(example.irrelevant) >= 7


def test_hotpotqa_documents_are_title_prefixed_text(hotpot):
    example = hotpot[0]
    gold = example.documents[example.relevant[0]]

    assert example.doc_ids[example.relevant[0]] in gold
    assert len(gold) > 50


def test_hotpotqa_max_queries_is_respected():
    assert len(load_hotpotqa(split="validation", max_queries=2)) == 2


# -- TREC DL ---------------------------------------------------------------


@needs_msmarco
@pytest.mark.parametrize("year", [2019, 2020])
def test_trec_dl_has_graded_labels_and_no_answers(year):
    examples = load_trec_dl(year=year, max_queries=3, max_documents=8)

    assert len(examples) == 3
    for example in examples:
        assert example.answer is None, "TREC DL has no answers"
        assert len(example.documents) <= 8
        assert max(example.relevance) > 0, "no relevant document in the pool"
        # Graded judgements, not binary.
        assert all(0 <= label <= 3 for label in example.relevance)


@needs_msmarco
def test_trec_dl_documents_are_real_passage_text():
    example = load_trec_dl(year=2019, max_queries=1, max_documents=4)[0]

    assert example.query.strip()
    for document in example.documents:
        assert len(document) > 20


@needs_msmarco
def test_trec_dl_rejects_other_years():
    with pytest.raises(ValueError, match="2019 or 2020"):
        load_trec_dl(year=2021)


@needs_msmarco
def test_trec_dl_is_deterministic_for_a_seed():
    first = load_trec_dl(year=2019, max_queries=2, max_documents=6, seed=7)
    second = load_trec_dl(year=2019, max_queries=2, max_documents=6, seed=7)

    assert [e.doc_ids for e in first] == [e.doc_ids for e in second]
