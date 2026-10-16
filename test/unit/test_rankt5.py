"""RankT5's scoring rule, and what it inherits instead of copying.

RankT5 and MonoT5 are the same architecture -- an encoder-decoder run for a
single decoder step -- and differ only in how that step becomes a score.
MonoT5 contrasts the `true`/`false` vocabulary logits; RankT5 reads one
regression-style relevance logit. These tests need no checkpoint: the
scoring rule is pinned on synthetic decoder logits built so that the two
rules rank the batch in opposite orders, and the inherited seams are pinned
by identity. Real-weight coverage lives in
test/acceptance/test_rankt5_model.py, which carries the `_model` suffix
because pytest needs unique test-module basenames.
"""

from types import SimpleNamespace

import pytest
import torch

from mechir.modelling import ENCODER_HOOK_NAMES, MonoT5, PatchedMixin
from mechir.modelling.rankt5 import RELEVANCE_TOKEN, RankT5

PATCH_METHODS = (
    "get_act_patch_block_every",
    "get_act_patch_attn_head_out_all_pos",
    "get_act_patch_attn_head_by_pos",
)

#: Vocabulary ids used by the synthetic logits below.
POS_ID, NEG_ID, REL_ID = 3, 4, 7
VOCAB = 16


def synthetic_logits() -> torch.Tensor:
    """Decoder logits on which the two scoring rules disagree.

    Row 0 looks relevant to MonoT5 (`true` beats `false`) and irrelevant to
    RankT5; row 1 is the reverse, so the two rules rank the batch in
    opposite orders and a test cannot pass under both. Decoder position 1
    holds decoys, to pin that only position 0 is read.
    """
    logits = torch.zeros(2, 2, VOCAB)
    logits[0, 0, POS_ID], logits[0, 0, NEG_ID], logits[0, 0, REL_ID] = 5.0, 1.0, -2.0
    logits[1, 0, POS_ID], logits[1, 0, NEG_ID], logits[1, 0, REL_ID] = -4.0, 0.0, 9.0
    logits[:, 1, :] = 100.0
    return logits


def monot5_rule(logits: torch.Tensor) -> torch.Tensor:
    """MonoT5's rule, as `MonoT5.forward` applies it: the `true` logit."""
    return logits[:, 0, (POS_ID, NEG_ID)][:, 0]


class FakeBackbone:
    """Stands in for the hooked encoder-decoder: fixed logits, recorded calls."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.cfg = SimpleNamespace(decoder_start_token_id=0)
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return self.logits


def fake_ranker(logits: torch.Tensor) -> RankT5:
    """A RankT5 whose backbone returns `logits`, built without a checkpoint."""
    ranker = RankT5.__new__(RankT5)
    ranker._model = FakeBackbone(logits)
    ranker.relevance_token = REL_ID
    return ranker


# -- the scoring rule ------------------------------------------------------


def test_score_is_the_relevance_logit_not_a_token_contrast():
    """The point of the family: no contrast of two vocabulary tokens."""
    logits = synthetic_logits()

    scores = RankT5._relevance(fake_ranker(logits), logits)

    assert torch.allclose(scores, torch.tensor([-2.0, 9.0]))
    assert not torch.allclose(scores, monot5_rule(logits))


def test_the_two_rules_rank_the_batch_oppositely():
    """So no implementation can satisfy both; assert RankT5's order."""
    logits = synthetic_logits()
    rankt5 = RankT5._relevance(fake_ranker(logits), logits)
    monot5 = monot5_rule(logits)

    assert rankt5[1] > rankt5[0]
    assert monot5[1] < monot5[0]


def test_score_is_unnormalised():
    """RankT5 is a regression output: no softmax, so no [0, 1] bound."""
    logits = synthetic_logits()

    scores = RankT5._relevance(fake_ranker(logits), logits)

    assert (scores.abs() > 1).all()


def test_only_the_first_decoder_position_is_read():
    """Position 1 of `synthetic_logits` holds 100.0 decoys."""
    logits = synthetic_logits()

    scores = RankT5._relevance(fake_ranker(logits), logits)

    assert (scores != 100.0).all()


def test_relevance_token_is_configurable():
    logits = synthetic_logits()
    ranker = fake_ranker(logits)
    ranker.relevance_token = POS_ID

    assert torch.allclose(RankT5._relevance(ranker, logits), monot5_rule(logits))


def test_default_relevance_token_is_the_trained_one():
    """RankT5 is trained on the logit of T5's unused `<extra_id_10>`."""
    assert RELEVANCE_TOKEN == "<extra_id_10>"


# -- forward: the rule applied over the shared T5 plumbing -----------------


def test_forward_reduces_to_one_score_per_item():
    logits = synthetic_logits()
    ranker = fake_ranker(logits)

    scores = RankT5.forward(
        ranker, torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.long)
    )

    assert scores.shape == (2,)
    assert torch.allclose(scores, torch.tensor([-2.0, 9.0]))


def test_forward_runs_one_decoder_step_with_the_mask_keyword():
    """Regression guard inherited from MonoT5: the keyword is
    `one_zero_attention_mask`, and the decoder is run for a single step."""
    ranker = fake_ranker(synthetic_logits())

    RankT5.forward(
        ranker, torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.long)
    )

    (call,) = ranker._model.calls
    assert "one_zero_attention_mask" in call
    assert call["decoder_input"].shape == (2, 1)
    assert (call["decoder_input"] == 0).all()
    assert call["return_type"] == "logits"


# -- the seams, and what is not copied -------------------------------------


def test_hook_naming_is_inherited_from_the_shared_architecture():
    """Same architecture as MonoT5, so the same encoder-stack naming."""
    assert "_hook_names" not in vars(RankT5)
    assert RankT5._hook_names.fget(object()) is ENCODER_HOOK_NAMES


def test_patch_score_stays_the_identity():
    """`forward` already returns a score, so reducing again would be wrong."""
    assert "_patch_score" not in vars(RankT5)
    assert RankT5._patch_score is PatchedMixin._patch_score


@pytest.mark.parametrize("method", PATCH_METHODS)
def test_patching_drivers_are_inherited(method):
    """A new family costs a scoring rule, not a copy of the patching loops."""
    assert method not in vars(RankT5), (
        f"RankT5.{method} shadows the shared driver on PatchedMixin"
    )
    assert getattr(RankT5, method) is getattr(PatchedMixin, method)


@pytest.mark.parametrize("method", ["score", "patch", "_decoder_start"])
def test_t5_plumbing_is_inherited(method):
    """Loading, scoring entry points and the one-step decoder are shared."""
    assert method not in vars(RankT5)
    assert getattr(RankT5, method) is getattr(MonoT5, method)


def test_patch_type_names_are_stable():
    assert set(PatchedMixin._patch_funcs.fget(RankT5)) == {
        "block_all",
        "head_all",
        "head_by_pos",
    }
