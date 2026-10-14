"""The two seams the patching refactor rests on.

The patching loops live once, on `PatchedMixin`. Everything a ranker family
or an architecture needs to vary goes through exactly two seams:

* `_hook_names` -- `(component, layer)` to hook name, per architecture;
* `_patch_score` -- patched forward output to a score, per ranker family.

These tests need no model, so they pin the seams cheaply. The behaviour they
guard end to end lives in test/acceptance/test_monot5.py and test_dot.py.
"""

import inspect

import pytest
import torch

from mechir.modelling import (
    BLOCK_HOOK_NAMES,
    DECODER_HOOK_NAMES,
    ENCODER_HOOK_NAMES,
    Cat,
    Dot,
    GenerativeRanker,
    MonoT5,
    PatchedMixin,
    StackHookNames,
)

PATCH_METHODS = (
    "get_act_patch_block_every",
    "get_act_patch_attn_head_out_all_pos",
    "get_act_patch_attn_head_by_pos",
)

RANKERS = (Dot, Cat, MonoT5, GenerativeRanker)


# -- seam 1: hook naming ---------------------------------------------------


@pytest.mark.parametrize(
    "component, expected",
    [
        ("resid_pre", "_model.blocks.3.hook_resid_pre"),
        ("attn_out", "_model.blocks.3.hook_attn_out"),
        ("mlp_out", "_model.blocks.3.hook_mlp_out"),
        ("z", "_model.blocks.3.attn.hook_z"),
        ("pattern", "_model.blocks.3.attn.hook_pattern"),
    ],
)
def test_single_stack_names(component, expected):
    assert BLOCK_HOOK_NAMES(component, 3) == expected


@pytest.mark.parametrize(
    "strategy, stack",
    [(ENCODER_HOOK_NAMES, "encoder"), (DECODER_HOOK_NAMES, "decoder")],
)
@pytest.mark.parametrize("component", ["resid_pre", "attn_out", "mlp_out", "z"])
def test_encoder_decoder_names_rewrite_the_stack(strategy, stack, component):
    """Regression: encoder-decoder hooks are not named `blocks.{layer}.…`."""
    name = strategy(component, 3)

    assert name.startswith(f"_model.{stack}.3.")
    assert "blocks" not in name


def test_only_the_stack_segment_is_rewritten():
    """A naive `str.replace` would also hit a component containing the name."""
    assert StackHookNames("encoder")("resid_pre", 0) == (
        "_model.encoder.0.hook_resid_pre"
    )


def test_prefix_is_configurable():
    assert StackHookNames("blocks", prefix="")("resid_pre", 1) == (
        "blocks.1.hook_resid_pre"
    )


def test_default_strategy_is_single_stack():
    assert PatchedMixin._hook_names.fget(object()) is BLOCK_HOOK_NAMES


def test_monot5_overrides_hook_naming():
    """MonoT5 is the only ranker whose architecture needs a different stack."""
    assert MonoT5._hook_names.fget(object()) is ENCODER_HOOK_NAMES


# -- seam 2: scoring -------------------------------------------------------


def test_default_scorer_is_the_identity():
    """Rankers whose `forward` already returns a score need no override."""
    output = torch.tensor([1.0, 2.0])

    assert PatchedMixin._patch_score(object(), output) is output


def test_dot_scorer_contracts_against_the_query():
    reps_q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reps_d = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    scored = Dot._patch_score(object(), reps_d, reps_q=reps_q)

    assert torch.allclose(scored, torch.tensor([17.0, 53.0]))


def test_dot_scorer_tolerates_surplus_context():
    """`patch` passes `layer_head_list` through to every driver."""
    reps_q = torch.ones(1, 2)

    scored = Dot._patch_score(object(), torch.ones(1, 2), reps_q=reps_q, other=1)

    assert torch.allclose(scored, torch.tensor([2.0]))


def test_dot_scorer_without_reps_q_fails_clearly():
    with pytest.raises(TypeError, match="reps_q"):
        Dot._patch_score(object(), torch.ones(1, 2))


@pytest.mark.parametrize("ranker", [Cat, MonoT5, GenerativeRanker])
def test_families_scoring_from_forward_do_not_override(ranker):
    assert "_patch_score" not in vars(ranker)


# -- the refactor's invariant ---------------------------------------------


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.parametrize("method", PATCH_METHODS)
def test_patching_loops_are_not_duplicated_per_ranker(ranker, method):
    """The payoff: one driver each, inherited by every ranker family.

    Re-adding a per-class copy would silently reintroduce the ~270 lines of
    near-identical loops this replaced, so pin that they are inherited.
    """
    assert method not in vars(ranker), (
        f"{ranker.__name__}.{method} shadows the shared driver on PatchedMixin"
    )
    assert getattr(ranker, method) is getattr(PatchedMixin, method)


@pytest.mark.parametrize("method", PATCH_METHODS)
def test_drivers_share_one_keyword_contract(method):
    """Every driver takes the same leading parameters, so `patch` can be generic."""
    params = list(inspect.signature(getattr(PatchedMixin, method)).parameters)

    assert params[:3] == ["self", "corrupted_tokens", "clean_cache"]
    assert params[-1] == "kwargs"
    for required in ("patching_metric", "scores", "scores_p"):
        assert required in params


@pytest.mark.parametrize("ranker", RANKERS)
def test_patch_type_names_are_stable(ranker):
    funcs = PatchedMixin._patch_funcs.fget(ranker)

    assert set(funcs) == {"block_all", "head_all", "head_by_pos"}
