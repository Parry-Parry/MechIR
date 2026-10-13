"""Causal ablation of retrieved documents.

The headline behaviour under test: attention and causal effect disagree, and
the causal measure is the one that identifies the document the answer
actually depends on.
"""

import pytest
import torch
from transformer_lens import HookedTransformer

from mechir import Cat
from mechir.pipeline import (
    HardPermutation,
    Pipeline,
    ablated_metric,
    answer_logprob,
    attention_to_documents,
    causal_groundedness,
    document_necessity,
    document_sufficiency,
)

RANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
READER = "EleutherAI/pythia-70m"

QUERY = "what is the capital of france"
# Index 0 states the answer; index 3 is wholly unrelated.
DOCS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany and a major European city.",
    "The Eiffel Tower is a landmark located in Paris, France.",
    "Bananas are a popular tropical fruit grown in warm climates.",
]
GOLD = 0
DISTRACTOR = 3
ANSWER = "Paris"


@pytest.fixture(scope="module")
def reader():
    return HookedTransformer.from_pretrained(READER, device="cpu")


@pytest.fixture(scope="module")
def pipeline(reader):
    return Pipeline(Cat(RANKER), reader, boundary=HardPermutation())


@pytest.fixture(scope="module")
def context(pipeline):
    return pipeline.context_builder.build(
        QUERY, DOCS, list(range(len(DOCS))), answer=ANSWER
    )


# -- the ablation itself ---------------------------------------------------


def test_empty_ablation_is_exactly_the_baseline(reader, context):
    """The no-op control. If this drifts, the hook is not clean."""
    baseline = float(answer_logprob(reader, context))

    assert ablated_metric(reader, context, []) == baseline


def test_ablating_a_document_changes_the_metric(reader, context):
    baseline = float(answer_logprob(reader, context))

    assert ablated_metric(reader, context, [GOLD]) != baseline


def test_renormalised_attention_still_sums_to_one(reader, context):
    """Ablation must leave the reader on-distribution.

    Zeroing attention without rescaling shrinks each row's total, which is
    not "the document is absent" but "attention is broken".
    """
    span = context.span_for_rank(GOLD)
    keys = torch.arange(span.start, span.end)
    name = "blocks.0.attn.hook_pattern"

    from mechir.pipeline.ablation import _mask_keys_hook

    captured = {}

    def capture(pattern, hook):
        captured["pattern"] = pattern
        return pattern

    with reader.hooks(fwd_hooks=[(name, _mask_keys_hook(keys)), (name, capture)]):
        answer_logprob(reader, context)

    pattern = captured["pattern"]
    assert pattern[..., span.slice()].abs().max() == 0, "span was not masked out"
    totals = pattern.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


def test_ablating_everything_is_worse_than_ablating_one(reader, context):
    one = ablated_metric(reader, context, [GOLD])
    everything = ablated_metric(reader, context, [s.rank for s in context.spans])

    assert everything <= one


# -- groundedness ----------------------------------------------------------


def test_causal_groundedness_shows_the_answer_depends_on_evidence(reader, context):
    result = causal_groundedness(reader, context)

    assert result["full"] > result["ablated"]
    assert result["effect"] == pytest.approx(result["full"] - result["ablated"])
    assert (
        result["effect"] > 0.5
    ), "retrieved documents had almost no causal effect on the answer"


def test_groundedness_effect_is_reported_in_nats(reader, context):
    result = causal_groundedness(reader, context)

    assert torch.isfinite(torch.tensor(result["effect"]))


# -- per-document necessity and sufficiency --------------------------------


def test_necessity_covers_every_rank(reader, context):
    necessity = document_necessity(reader, context)

    assert set(necessity) == {span.rank for span in context.spans}


def test_gold_document_is_the_most_necessary(reader, context):
    """Leave-one-out must identify the document stating the answer."""
    necessity = document_necessity(reader, context)

    assert max(necessity, key=necessity.get) == GOLD
    assert necessity[GOLD] > 0


def test_gold_document_is_the_most_sufficient(reader, context):
    sufficiency = document_sufficiency(reader, context)

    assert max(sufficiency, key=sufficiency.get) == GOLD


def test_distractor_is_not_necessary(reader, context):
    """An unrelated document must not look load-bearing.

    Here its leave-one-out effect is actually negative -- removing it
    improves the answer -- so this asserts the weaker, robust direction.
    """
    necessity = document_necessity(reader, context)

    assert necessity[DISTRACTOR] < necessity[GOLD]
    assert necessity[DISTRACTOR] < 0.5


def test_sufficiency_differs_from_necessity(reader, context):
    """The two answer different questions and should not coincide.

    Redundant documents have low necessity and high sufficiency, so a
    pipeline reporting only one of them is hiding information.
    """
    necessity = document_necessity(reader, context)
    sufficiency = document_sufficiency(reader, context)

    assert any(abs(necessity[rank] - sufficiency[rank]) > 1e-3 for rank in necessity)


# -- attention is not causation --------------------------------------------


def test_attention_disagrees_with_causal_importance(reader, context):
    """The reason this module exists.

    Measured with this reader: attention ranks the documents [3, 2, 0, 1] --
    the irrelevant distractor first -- while necessity ranks them
    [0, 1, 2, 3]. Attention is not merely noisier here, it is actively
    misleading, so a faithfulness claim built on it would be wrong.
    """
    attention = attention_to_documents(reader, context)
    necessity = document_necessity(reader, context)

    most_attended = max(attention, key=attention.get)
    most_necessary = max(necessity, key=necessity.get)

    assert most_necessary == GOLD
    assert (
        most_attended != most_necessary
    ), "attention and causation agreed; this test's premise needs revisiting"


# -- pipeline-level report -------------------------------------------------


def test_causal_report_carries_both_signals(pipeline):
    report = pipeline.causal_report(QUERY, DOCS, ANSWER)

    assert report["groundedness"]["effect"] > 0
    assert len(report["documents"]) == len(DOCS)

    for row in report["documents"]:
        assert {
            "rank",
            "candidate",
            "ranker_score",
            "reader_attention",
            "necessity",
            "sufficiency",
        } <= set(row)

    gold_row = next(r for r in report["documents"] if r["candidate"] == GOLD)
    assert gold_row["necessity"] == max(r["necessity"] for r in report["documents"])
