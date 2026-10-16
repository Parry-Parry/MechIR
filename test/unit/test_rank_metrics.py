"""Permutation-valued patching metrics and the listwise output parser.

The degenerate-case tests here are the point of the file. ``linear_rank_function``
once returned NaN on a no-op perturbation and the NaN spread across a whole
batch through the ``.mean()`` in the patching loops (see
``test/unit/test_rank_metric.py``), and ``docs/research-notes.md`` 5 measures
~40% of real TREC DL queries producing no rank movement at all. Both failure
modes are reachable from every metric in :mod:`mechir.rank_metrics`, so each
is pinned.
"""

import math

import pytest
import torch

from mechir.rank_metrics import (
    RANK_METRICS,
    PermutationParseError,
    batched_rank_metric,
    first_position_flip,
    get_rank_metric,
    is_degenerate_ordering,
    kendall_tau,
    kendall_tau_distance,
    parse_permutation,
    spearman_distance,
    spearman_rho,
    top_k_jaccard,
    top_k_set_distance,
)

#: Every metric with its parameters bound, for the shared-contract tests.
#:
#: ``top_k`` is bound to ``k=2`` rather than left at its default: the default
#: ``k=10`` clamps to the ordering length on the small pools used here, and a
#: top-``k`` cutoff at or beyond the pool size cannot see any reordering at
#: all. That clamping is itself pinned by
#: ``test_top_k_larger_than_the_pool_compares_the_whole_pool``.
ALL_METRICS = (
    pytest.param(kendall_tau_distance, id="kendall"),
    pytest.param(spearman_distance, id="spearman"),
    pytest.param(get_rank_metric("top_k", k=2), id="top_k"),
    pytest.param(first_position_flip, id="first_flip"),
)


# -- shared contract: direction, range, exactness --------------------------


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_no_change_is_exactly_zero(metric):
    """The module's headline guarantee: identical orderings mean zero change."""
    ordering = [3, 0, 1, 2]
    assert metric(ordering, list(ordering)) == 0.0


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_reversal_is_maximal_change(metric):
    """Every metric agrees on the other end of the scale."""
    assert metric([0, 1, 2, 3], [3, 2, 1, 0]) == 1.0


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_every_metric_is_bounded_in_the_unit_interval(metric):
    """No metric may hand the patching loop a value outside [0, 1]."""
    clean = [0, 1, 2, 3, 4]
    for patched in (
        [0, 1, 2, 3, 4],
        [1, 0, 2, 3, 4],
        [0, 1, 2, 4, 3],
        [4, 3, 2, 1, 0],
        [2, 4, 0, 3, 1],
    ):
        value = metric(clean, patched)
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metrics_report_change_not_similarity(metric):
    """A sign flip between metrics would silently invert every conclusion."""
    unchanged = metric([0, 1, 2, 3], [0, 1, 2, 3])
    reordered = metric([0, 1, 2, 3], [3, 2, 1, 0])
    assert unchanged < reordered


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metrics_are_symmetric(metric):
    """Change is a distance, so argument order must not matter."""
    clean, patched = [0, 1, 2, 3], [2, 0, 3, 1]
    assert metric(clean, patched) == metric(patched, clean)


# -- shared contract: degenerate cases -------------------------------------


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_single_item_ordering_is_zero_not_nan(metric):
    """One candidate cannot be reordered, so no component can have moved it."""
    value = metric([0], [0])
    assert value == 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_empty_ordering_is_zero_not_nan(metric):
    value = metric([], [])
    assert value == 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_constant_ordering_is_zero_not_nan(metric):
    """A zero-filled batch row carries no ranking; it must not raise or NaN.

    Kendall tau is undefined here -- every pair is tied, so its denominator
    is zero. The documented convention is 0.0 change.
    """
    value = metric([0, 0, 0, 0], [0, 0, 0, 0])
    assert value == 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_degenerate_against_live_ordering_is_zero_not_raise(metric):
    """A dead row must not abort a sweep by disagreeing on the item set."""
    value = metric([0, 0, 0], [0, 1, 2])
    assert value == 0.0


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_nan_in_an_ordering_does_not_produce_nan(metric):
    """A failed generation yields NaN; it must be absorbed, not propagated."""
    value = metric([0.0, 1.0, float("nan")], [0.0, 1.0, 2.0])
    assert value == 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_length_mismatch_raises(metric):
    with pytest.raises(ValueError, match="differ in length"):
        metric([0, 1, 2], [0, 1])


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_different_candidate_sets_raise(metric):
    """A caller bug, unlike a dead batch row, so it must be loud."""
    with pytest.raises(ValueError, match="different candidates"):
        metric([0, 1, 2], [0, 1, 7])


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metrics_accept_tensors(metric):
    clean = torch.tensor([0, 1, 2, 3])
    patched = torch.tensor([3, 2, 1, 0])
    assert metric(clean, patched) == 1.0


def test_is_degenerate_ordering():
    assert is_degenerate_ordering([]) is True
    assert is_degenerate_ordering([4]) is True
    assert is_degenerate_ordering([1, 1, 2]) is True
    assert is_degenerate_ordering([0.0, float("inf")]) is True
    assert is_degenerate_ordering([0, 1]) is False
    assert is_degenerate_ordering(torch.tensor([2, 0, 1])) is False


# -- metric 1: rank correlation --------------------------------------------


def test_kendall_distance_of_one_adjacent_swap():
    """1 of 3 pairs discordant, so tau = 1/3 and the distance is 1/3."""
    assert kendall_tau_distance([0, 1, 2], [1, 0, 2]) == pytest.approx(1 / 3)


def test_spearman_distance_of_one_adjacent_swap():
    """Sum of squared rank displacement is 2, so rho = 0.5."""
    assert spearman_distance([0, 1, 2], [1, 0, 2]) == pytest.approx(0.25)


def test_correlations_disagree_on_local_versus_distant_movement():
    """The reason to carry both: they weight displacement differently.

    Promoting one document three ranks crosses exactly as many pairs as three
    disjoint adjacent swaps, so Kendall scores the two identically. Spearman
    does not, because one document travelled three ranks on its own.
    """
    clean = [0, 1, 2, 3, 4, 5]
    far = [3, 0, 1, 2, 4, 5]
    local = [1, 0, 3, 2, 5, 4]

    assert kendall_tau_distance(clean, far) == pytest.approx(
        kendall_tau_distance(clean, local)
    )
    assert spearman_distance(clean, far) > spearman_distance(clean, local)


def test_correlation_primitives_are_similarity_valued():
    """The primitives keep the textbook [-1, 1] range, unlike the metrics."""
    assert kendall_tau([0, 1, 2], [0, 1, 2]) == 1.0
    assert kendall_tau([0, 1, 2], [2, 1, 0]) == -1.0
    assert spearman_rho([0, 1, 2], [0, 1, 2]) == 1.0
    assert spearman_rho([0, 1, 2], [2, 1, 0]) == -1.0


def test_kendall_tau_is_zero_on_zero_variance_not_nan():
    """The undefined case: a constant input zeroes tau-b's denominator."""
    assert kendall_tau([1, 1, 1], [0, 1, 2]) == 0.0
    assert math.isfinite(kendall_tau([1, 1, 1], [0, 1, 2]))


def test_spearman_rho_is_zero_on_zero_variance_not_nan():
    assert spearman_rho([1, 1, 1], [0, 1, 2]) == 0.0
    assert math.isfinite(spearman_rho([1, 1, 1], [0, 1, 2]))


def test_correlation_primitives_handle_ties():
    """Tie correction, so the primitives accept score vectors not just ranks."""
    assert kendall_tau([0, 1, 1, 2], [0, 1, 1, 2]) == pytest.approx(1.0)
    assert spearman_rho([0, 1, 1, 2], [0, 1, 1, 2]) == pytest.approx(1.0)
    assert kendall_tau([0, 1, 1, 2], [2, 1, 1, 0]) == pytest.approx(-1.0)


def test_correlation_primitives_reject_unpairable_input():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau([0, 1], [0, 1, 2])
    with pytest.raises(ValueError, match="same length"):
        spearman_rho([0, 1], [0, 1, 2])


def test_correlation_distance_of_an_uncorrelated_pair_is_a_half():
    """The midpoint of the range, which is what "no association" maps to.

    Promoting the last of four documents to the front flips 3 of the 6 pairs,
    so tau is exactly 0.
    """
    assert kendall_tau_distance([0, 1, 2, 3], [3, 0, 1, 2]) == pytest.approx(0.5)


# -- metric 2: top-k set overlap -------------------------------------------


def test_top_k_ignores_reordering_within_the_top_k():
    """Documented blindness number one."""
    assert top_k_set_distance([0, 1, 2, 3], [1, 0, 2, 3], k=2) == 0.0


def test_top_k_ignores_movement_below_the_top_k():
    """Documented blindness number two, and why the correlations exist."""
    assert top_k_set_distance([0, 1, 2, 3], [0, 1, 3, 2], k=2) == 0.0
    assert kendall_tau_distance([0, 1, 2, 3], [0, 1, 3, 2]) > 0.0


def test_top_k_sees_an_entrant():
    """One of two swapped out: intersection 1, union 3, so 1 - 1/3."""
    assert top_k_set_distance([0, 1, 2, 3], [0, 2, 1, 3], k=2) == pytest.approx(2 / 3)


def test_top_k_is_one_when_the_top_k_sets_are_disjoint():
    assert top_k_set_distance([0, 1, 2, 3], [2, 3, 0, 1], k=2) == 1.0


def test_top_k_larger_than_the_pool_compares_the_whole_pool():
    """Clamped to the ordering length, so it can only return 0.0."""
    assert top_k_set_distance([0, 1, 2], [2, 1, 0], k=99) == 0.0


def test_top_k_rejects_a_nonpositive_cutoff():
    with pytest.raises(ValueError, match="at least 1"):
        top_k_set_distance([0, 1], [1, 0], k=0)
    with pytest.raises(ValueError, match="at least 1"):
        top_k_jaccard([0, 1], [1, 0], k=-3)


def test_top_k_jaccard_is_the_similarity_counterpart():
    clean, patched = [0, 1, 2, 3], [0, 2, 1, 3]
    assert top_k_jaccard(clean, patched, k=2) + top_k_set_distance(
        clean, patched, k=2
    ) == pytest.approx(1.0)


# -- metric 3: first-position flip -----------------------------------------


def test_first_position_flip_detects_a_new_winner():
    assert first_position_flip([0, 1, 2], [1, 0, 2]) == 1.0


def test_first_position_flip_ignores_everything_below_rank_one():
    """The documented sparsity: a reversal of ranks 2 and 3 scores nothing."""
    assert first_position_flip([0, 1, 2], [0, 2, 1]) == 0.0


def test_first_position_flip_is_binary():
    """Two different winners are not more different than one."""
    assert first_position_flip([0, 1, 2, 3], [3, 1, 2, 0]) == 1.0
    assert first_position_flip([0, 1, 2, 3], [3, 2, 1, 0]) == 1.0


# -- selection -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RANK_METRICS))
def test_every_registered_metric_is_selectable(name):
    """Selection by name yields something honouring the module's contract.

    The reversal here is only asserted to be in range, not to be 1.0:
    ``top_k`` at its default cutoff of 10 clamps to a 3-document pool and so
    returns 0.0 by design.
    """
    metric = get_rank_metric(name)
    assert metric([0, 1, 2], [0, 1, 2]) == 0.0

    reordered = metric([0, 1, 2], [2, 1, 0])
    assert 0.0 <= reordered <= 1.0
    assert math.isfinite(reordered)


def test_get_rank_metric_binds_parameters():
    metric = get_rank_metric("top_k", k=2)
    assert metric([0, 1, 2, 3], [0, 1, 3, 2]) == 0.0
    assert metric([0, 1, 2, 3], [2, 3, 0, 1]) == 1.0


def test_get_rank_metric_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="Unknown rank metric"):
        get_rank_metric("ndcg")


def test_get_rank_metric_rejects_an_unknown_parameter():
    """Caught at selection, not deep inside a patching sweep."""
    with pytest.raises(TypeError, match="no parameter 'k'"):
        get_rank_metric("first_flip", k=5)


def test_registry_covers_all_three_candidate_metrics():
    assert set(RANK_METRICS) == {"kendall", "spearman", "top_k", "first_flip"}


# -- the patching-loop adapter ---------------------------------------------


def test_batched_metric_matches_the_patching_call_convention():
    """``patching_metric(score, scores, scores_p)``, as PatchedMixin calls it."""
    metric = batched_rank_metric(first_position_flip)
    patched = [[1, 0, 2], [0, 1, 2]]
    clean = [[0, 1, 2], [0, 1, 2]]
    perturbed = [[2, 1, 0], [2, 1, 0]]

    out = metric(patched, clean, perturbed)

    assert out.shape == (2,)
    assert out.dtype is torch.float32
    assert out.tolist() == [1.0, 0.0]
    assert out.mean().item() == pytest.approx(0.5)


def test_batched_metric_swaps_into_clean_patched_order():
    """The adapter's one job beyond batching, and a silent-inversion risk."""
    calls = []

    def recording_metric(clean, patched):
        calls.append((list(clean), list(patched)))
        return 0.0

    batched_rank_metric(recording_metric)([[1, 0]], [[0, 1]])
    assert calls == [([0, 1], [1, 0])]


def test_batched_metric_ignores_the_perturbed_argument():
    metric = batched_rank_metric(kendall_tau_distance)
    patched, clean = [[1, 0]], [[0, 1]]
    assert metric(patched, clean).tolist() == metric(patched, clean, [[0, 1]]).tolist()


def test_batched_metric_accepts_a_two_dimensional_tensor():
    metric = batched_rank_metric(first_position_flip)
    out = metric(torch.tensor([[1, 0], [0, 1]]), torch.tensor([[0, 1], [0, 1]]))
    assert out.tolist() == [1.0, 0.0]


def test_batched_metric_accepts_an_unbatched_ordering():
    metric = batched_rank_metric(first_position_flip)
    assert metric([1, 0, 2], [0, 1, 2]).tolist() == [1.0]
    assert metric(torch.tensor([1, 0, 2]), torch.tensor([0, 1, 2])).tolist() == [1.0]


def test_batched_metric_rejects_a_batch_size_mismatch():
    metric = batched_rank_metric(first_position_flip)
    with pytest.raises(ValueError, match="batch size mismatch"):
        metric([[0, 1], [1, 0]], [[0, 1]])


def test_batched_metric_rejects_an_empty_batch():
    """An empty result would reduce to NaN under the loop's .mean()."""
    metric = batched_rank_metric(first_position_flip)
    with pytest.raises(ValueError, match="nothing to score"):
        metric([], [])


def test_one_degenerate_item_does_not_poison_the_batch():
    """Regression, ported from test_rank_metric.py's batch test.

    NaN propagates through the ``.mean()`` in the patching loops, so one dead
    batch item used to turn every entry of the patch map into NaN. The middle
    item here is a zero-filled row, the realistic source of the problem.
    """
    metric = batched_rank_metric(kendall_tau_distance)
    patched = [[1, 0, 2], [0, 0, 0], [2, 1, 0]]
    clean = [[0, 1, 2], [0, 0, 0], [0, 1, 2]]

    out = metric(patched, clean)

    assert torch.isfinite(out).all()
    assert torch.isfinite(out.mean())
    assert out[0].item() == pytest.approx(1 / 3)
    assert out[1].item() == 0.0
    assert out[2].item() == 1.0


def test_a_nan_returning_metric_cannot_poison_the_batch():
    """The adapter is a firewall, so a custom metric cannot break a sweep."""

    def broken_metric(clean, patched):
        return float("nan") if clean[0] == 1 else 0.5

    out = batched_rank_metric(broken_metric)([[0, 1], [0, 1]], [[0, 1], [1, 0]])

    assert torch.isfinite(out).all()
    assert out.tolist() == [0.5, 0.0]
    assert torch.isfinite(out.mean())


def test_batched_metric_names_itself_after_the_metric():
    assert batched_rank_metric(spearman_distance).__name__ == (
        "batched_spearman_distance"
    )


# -- permutation parsing ---------------------------------------------------


def test_parses_the_canonical_format():
    """One-indexed in, zero-indexed out."""
    assert parse_permutation("[3] > [1] > [2]", n=3) == [2, 0, 1]


def test_parses_zero_indexed_output_when_asked():
    assert parse_permutation("[2] > [0] > [1]", n=3, one_indexed=False) == [2, 0, 1]


def test_tolerates_whitespace_and_separators():
    assert parse_permutation("[ 2 ]>[1]  >  [3]", n=3) == [1, 0, 2]
    assert parse_permutation("[2], [1], [3]", n=3) == [1, 0, 2]


def test_ignores_prose_around_the_list():
    text = "Sure! Here is the ranking: [2] > [1] > [3]. Let me know if that helps."
    assert parse_permutation(text, n=3) == [1, 0, 2]


def test_ignores_unbracketed_digits_in_prose():
    """The reason brackets are required rather than merely preferred."""
    text = "Ranking the top 3 of 100 documents: [2] > [1] > [3]"
    assert parse_permutation(text, n=3) == [1, 0, 2]


def test_unbracketed_output_raises_rather_than_guessing():
    with pytest.raises(PermutationParseError, match="no bracketed index"):
        parse_permutation("2 > 1 > 3", n=3)


def test_duplicates_keep_the_first_occurrence():
    assert parse_permutation("[2] > [1] > [2] > [3]", n=3) == [1, 0, 2]


def test_omitted_items_go_to_the_tail_in_their_original_order():
    """Policy: the model declined to rank them, so retrieval order stands."""
    assert parse_permutation("[4] > [2]", n=5) == [3, 1, 0, 2, 4]


def test_out_of_range_indices_are_dropped():
    """A routine hallucination; the rest of the ordering is still usable."""
    assert parse_permutation("[2] > [9] > [1] > [0]", n=3) == [1, 0, 2]


def test_a_single_valid_index_still_parses():
    assert parse_permutation("[3]", n=4) == [2, 0, 1, 3]


def test_no_index_in_range_raises():
    """Would otherwise silently become the identity ordering."""
    with pytest.raises(PermutationParseError, match="no index in range"):
        parse_permutation("[9] > [10]", n=3)


def test_empty_text_raises():
    with pytest.raises(PermutationParseError, match="no bracketed index"):
        parse_permutation("", n=3)


def test_parse_always_returns_a_full_permutation():
    for text in ("[1]", "[3] > [3] > [99]", "[2] > [1] > [4] > [3]"):
        out = parse_permutation(text, n=4)
        assert sorted(out) == [0, 1, 2, 3]


def test_parse_rejects_a_nonpositive_candidate_count():
    with pytest.raises(ValueError, match="n must be at least 1"):
        parse_permutation("[1]", n=0)


@pytest.mark.parametrize(
    "text, reason",
    [
        ("[2] > [1] > [2] > [3]", "duplicates"),
        ("[2] > [9] > [1] > [3]", "out_of_range"),
        ("[2] > [1]", "missing"),
    ],
)
def test_strict_mode_raises_on_each_repaired_malformation(text, reason):
    with pytest.raises(PermutationParseError, match=reason):
        parse_permutation(text, n=3, strict=True)


def test_strict_mode_accepts_a_clean_permutation():
    assert parse_permutation("[3] > [1] > [2]", n=3, strict=True) == [2, 0, 1]


def test_parsed_output_feeds_the_metrics_directly():
    """The two halves of the module have to compose without adaptation."""
    clean = parse_permutation("[1] > [2] > [3]", n=3)
    patched = parse_permutation("[3] > [2] > [1]", n=3)
    assert kendall_tau_distance(clean, patched) == 1.0
    assert first_position_flip(clean, patched) == 1.0
