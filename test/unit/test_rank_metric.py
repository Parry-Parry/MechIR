"""linear_rank_function must not emit NaN when a perturbation is a no-op."""

import torch

from mechir.util import linear_rank_function


def test_recovers_nothing_is_zero():
    score = torch.tensor([2.0])
    score_p = torch.tensor([4.0])
    assert linear_rank_function(score, score, score_p).item() == 0.0


def test_recovers_everything_is_one():
    score = torch.tensor([2.0])
    score_p = torch.tensor([4.0])
    assert linear_rank_function(score_p, score, score_p).item() == 1.0


def test_halfway_recovery():
    out = linear_rank_function(
        torch.tensor([3.0]), torch.tensor([2.0]), torch.tensor([4.0])
    )
    assert out.item() == 0.5


def test_negative_direction_is_preserved():
    """A perturbation that lowers the score still normalises to 1."""
    score = torch.tensor([4.0])
    score_p = torch.tensor([2.0])
    assert linear_rank_function(score_p, score, score_p).item() == 1.0


def test_no_change_gives_zero_not_nan():
    """Regression: identical scores divided 0 by 0 and returned NaN."""
    score = torch.tensor([3.0])
    out = linear_rank_function(torch.tensor([3.5]), score, score)

    assert torch.isfinite(out).all()
    assert out.item() == 0.0


def test_nan_does_not_spread_across_a_batch():
    """Regression: one dead item used to poison the whole patch map.

    NaN propagates through the .mean() in the patching loops, so a single
    unaffected batch item turned every entry of the result into NaN.
    """
    patch_score = torch.tensor([2.5, 9.9, 3.5])
    score = torch.tensor([2.0, 5.0, 3.0])
    score_p = torch.tensor([4.0, 5.0, 7.0])  # middle item unchanged

    out = linear_rank_function(patch_score, score, score_p)

    assert torch.isfinite(out).all()
    assert torch.isfinite(out.mean())
    assert out[0].item() == 0.25
    assert out[1].item() == 0.0
    assert out[2].item() == 0.125
