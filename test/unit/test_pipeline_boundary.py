"""Boundary permutations and context assembly. No models needed."""

import pytest
import torch

from mechir.pipeline import (
    HardPermutation,
    demote,
    identity,
    promote,
    reverse,
    shuffle,
    swap,
)

SCORES = torch.tensor([0.1, 0.9, 0.5, 0.3])
# descending order of the above
RANKED = [1, 2, 3, 0]


def test_select_orders_by_score_descending():
    selection = HardPermutation().select(SCORES)

    assert selection.order == RANKED
    assert selection.op_name == "identity"


def test_select_truncates_to_k():
    assert HardPermutation(k=2).select(SCORES).order == RANKED[:2]


def test_select_rejects_non_vector_scores():
    with pytest.raises(ValueError, match="one score per candidate"):
        HardPermutation().select(SCORES.unsqueeze(0))


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        HardPermutation(k=0)


def test_ranks_maps_candidate_to_position():
    assert HardPermutation().select(SCORES).ranks() == {1: 0, 2: 1, 3: 2, 0: 3}


@pytest.mark.parametrize(
    "op,expected",
    [
        (identity(), [1, 2, 3, 0]),
        (swap(0, 1), [2, 1, 3, 0]),
        (swap(0, 3), [0, 2, 3, 1]),
        (reverse(), [0, 3, 2, 1]),
        (demote(0), [2, 3, 0, 1]),
        (promote(2), [3, 1, 2, 0]),
    ],
)
def test_permutation_ops(op, expected):
    assert HardPermutation().select(SCORES, op=op).order == expected


def test_ops_are_permutations_not_rewrites():
    """Every op must preserve the candidate set exactly."""
    for op in (identity(), swap(1, 2), reverse(), demote(0), promote(3), shuffle(7)):
        order = HardPermutation().select(SCORES, op=op).order
        assert sorted(order) == sorted(RANKED), op.name


def test_out_of_range_swap_is_a_noop():
    assert HardPermutation().select(SCORES, op=swap(0, 99)).order == RANKED


def test_shuffle_is_deterministic():
    first = HardPermutation().select(SCORES, op=shuffle(3)).order
    second = HardPermutation().select(SCORES, op=shuffle(3)).order

    assert first == second


def test_permutation_applies_after_truncation():
    """An op must only reorder what the reader would have seen.

    If it applied before truncation, swapping in a document from outside the
    top-k would conflate reordering with changing the candidate set.
    """
    order = HardPermutation(k=2).select(SCORES, op=swap(0, 1)).order

    assert sorted(order) == sorted(RANKED[:2])


def test_sweep_pairs_every_rank_with_the_top():
    names = [op.name for op in HardPermutation().sweep(4)]

    assert names[0] == "identity"
    assert "swap(0,1)" in names
    assert "swap(0,3)" in names
    assert "reverse" in names


def test_sweep_respects_k():
    names = [op.name for op in HardPermutation(k=2).sweep(4)]

    assert "swap(0,1)" in names
    assert "swap(0,2)" not in names
