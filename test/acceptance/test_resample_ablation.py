"""Resample ablation: replace a document rather than remove it.

Masking answers "did the answer need this document". Resampling answers
"did the answer need *this* document, or just something in that slot".
"""

import pytest
import torch
from transformer_lens import HookedTransformer

from mechir import Cat
from mechir.pipeline import (
    HardPermutation,
    Pipeline,
    document_necessity,
    resample_metric,
    resample_necessity,
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
GOLD = 0
ORDER = [0, 1, 2, 3]
ANSWER = "Paris"
# Neutral filler, unrelated to the query and to every candidate, so it adds
# no evidence of its own.
FILLER = [
    "Sedimentary rock forms from accumulated mineral particles over time.",
    "The violin is a wooden string instrument played with a bow.",
    "Crop rotation helps maintain soil nutrient balance across seasons.",
    "Glass is produced by rapidly cooling molten silica to prevent crystals.",
]


@pytest.fixture(scope="module")
def reader():
    return HookedTransformer.from_pretrained(READER, device="cpu")


@pytest.fixture(scope="module")
def pipeline(reader):
    return Pipeline(Cat(RANKER), reader, boundary=HardPermutation())


@pytest.fixture(scope="module")
def builder(pipeline):
    return pipeline.context_builder


# -- the swap itself -------------------------------------------------------


def test_replacing_a_document_changes_the_metric(reader, builder):
    result = resample_metric(
        reader, builder, QUERY, DOCS, ORDER, GOLD, FILLER[0], ANSWER
    )

    assert result["full"] != result["resampled"]
    assert result["effect"] == pytest.approx(result["full"] - result["resampled"])


def test_replacing_a_document_with_itself_has_no_effect(reader, builder):
    """The no-op control: swapping in identical text must change nothing."""
    result = resample_metric(
        reader, builder, QUERY, DOCS, ORDER, GOLD, DOCS[GOLD], ANSWER
    )

    assert result["effect"] == pytest.approx(0.0, abs=1e-6)
    assert result["length_delta"] == 0


def test_length_matching_keeps_positions_almost_fixed(reader, builder):
    """A long replacement must not push later documents down the context.

    Exact equality is not achievable: truncating ids, decoding and
    re-encoding need not round-trip, so a token or two of drift is normal
    and is reported rather than hidden.
    """
    overlong = " ".join(["unrelated filler text"] * 60)

    result = resample_metric(
        reader, builder, QUERY, DOCS, ORDER, GOLD, overlong, ANSWER
    )

    assert abs(result["length_delta"]) <= 3


def test_length_matching_can_be_disabled(reader, builder):
    overlong = " ".join(["unrelated filler text"] * 60)

    matched = resample_metric(
        reader, builder, QUERY, DOCS, ORDER, GOLD, overlong, ANSWER
    )
    unmatched = resample_metric(
        reader,
        builder,
        QUERY,
        DOCS,
        ORDER,
        GOLD,
        overlong,
        ANSWER,
        match_length=False,
    )

    assert abs(unmatched["length_delta"]) > abs(matched["length_delta"])


# -- per-document effects --------------------------------------------------


def test_resample_necessity_covers_every_rank(reader, builder):
    results = resample_necessity(reader, builder, QUERY, DOCS, ORDER, ANSWER)

    assert set(results) == set(ORDER)
    for value in results.values():
        assert {"full", "resampled", "effect", "length_delta"} <= set(value)


def test_requires_one_replacement_per_rank(reader, builder):
    with pytest.raises(ValueError, match="expected 4 replacements"):
        resample_necessity(
            reader, builder, QUERY, DOCS, ORDER, ANSWER, replacements=["only one"]
        )


def test_gold_document_effect_survives_replacement(reader, builder):
    """The robustness check that motivates this module.

    If the gold document's effect only showed up under masking, it would be
    about the slot being filled rather than about the content. Replacing it
    with neutral filler must still cost the answer.
    """
    results = resample_necessity(
        reader, builder, QUERY, DOCS, ORDER, ANSWER, replacements=FILLER
    )

    assert results[GOLD]["effect"] > 0
    assert results[GOLD]["effect"] == max(r["effect"] for r in results.values())


def test_resample_agrees_with_masking_on_the_gold_document(reader, builder):
    context = builder.build(QUERY, DOCS, ORDER, answer=ANSWER)
    masked = document_necessity(reader, context)
    resampled = resample_necessity(
        reader, builder, QUERY, DOCS, ORDER, ANSWER, replacements=FILLER
    )

    assert max(masked, key=masked.get) == GOLD
    assert max(resampled, key=lambda r: resampled[r]["effect"]) == GOLD


def test_rotation_default_is_confounded_by_relevant_replacements(reader, builder):
    """Pins the documented confound in the zero-config default.

    Rotation replaces each document with another candidate, so when the pool
    holds relevant documents it conflates removing one with duplicating
    another. Neutral filler and rotation therefore disagree, and the
    docstring says to prefer corpus-sampled replacements for real results.
    """
    rotated = resample_necessity(reader, builder, QUERY, DOCS, ORDER, ANSWER)
    neutral = resample_necessity(
        reader, builder, QUERY, DOCS, ORDER, ANSWER, replacements=FILLER
    )

    distractor = 3
    assert rotated[distractor]["effect"] != pytest.approx(
        neutral[distractor]["effect"], abs=0.1
    )


# -- pipeline report -------------------------------------------------------


def test_causal_report_includes_resample_columns(pipeline):
    report = pipeline.causal_report(QUERY, DOCS, ANSWER)

    for row in report["documents"]:
        assert "resample_effect" in row
        assert "resample_length_delta" in row
        assert torch.isfinite(torch.tensor(row["resample_effect"]))
