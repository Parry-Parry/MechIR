"""Unit tests for :mod:`mechir.modelling.steering`.

The sign convention is the defect most likely to regress silently, so most of
these tests pin it directly: ``increase`` must raise the component along the
steering direction and ``decrease`` must lower it.
"""

from typing import Dict, Tuple

import pytest
import torch

from mechir.modelling.steering import (
    EmbeddingSteer,
    EmbeddingSteerConfig,
    EmbeddingSteeringContext,
    EmbeddingSteerWrapper,
)

HOOK_POINT = "model.embedding.W"
D_MODEL = 6
N_ROWS = 4


def make_steer(mode: str, idx=(1,), scale: float = 10.0) -> EmbeddingSteer:
    """Build a steer over a random square weight matrix."""
    model: Dict[str, torch.Tensor] = {HOOK_POINT: torch.randn(D_MODEL, D_MODEL)}
    config = EmbeddingSteerConfig(
        idx=list(idx), mode=mode, scale=scale, out_hook_point=HOOK_POINT
    )
    return EmbeddingSteer(model, config)


def project(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Component of each row of ``x`` along ``direction``."""
    return x @ direction


def steer_pair(idx=(1,), scale: float = 10.0) -> Tuple[torch.Tensor, ...]:
    """Return ``(x, steered_up, steered_down, direction)`` for one activation.

    Both steers are built from the same weight matrix, so they share a
    steering direction and differ only in ``mode``.
    """
    model: Dict[str, torch.Tensor] = {HOOK_POINT: torch.randn(D_MODEL, D_MODEL)}

    def build(mode: str) -> EmbeddingSteer:
        config = EmbeddingSteerConfig(
            idx=list(idx), mode=mode, scale=scale, out_hook_point=HOOK_POINT
        )
        return EmbeddingSteer(model, config)

    up, down = build("increase"), build("decrease")
    assert torch.allclose(up.V_0, down.V_0)
    x = torch.randn(2, N_ROWS, D_MODEL)
    return x, up(x), down(x), up.V_0


def test_v_0_is_initialised():
    """Regression test: ``V_0`` used to be set in a never-called __post_init__."""
    steer = make_steer("increase")
    assert isinstance(steer.V_0, torch.Tensor)
    assert steer.V_0.shape == (D_MODEL,)
    assert torch.allclose(steer.V_0.norm(), torch.tensor(1.0), atol=1e-5)


def test_direction_is_d_model_length_for_a_vocab_sized_weight():
    """The direction is the top *right* singular vector, not the left one.

    The other tests use a square weight, where ``U[:, 0]`` and ``Vh[0]`` have
    the same length and the distinction is invisible. A real embedding is
    ``[d_vocab, d_model]``: there ``U[:, 0]`` is vocab-length and cannot be
    added to a ``d_model`` activation at all.
    """
    d_vocab = 20
    model = {HOOK_POINT: torch.randn(d_vocab, D_MODEL)}
    config = EmbeddingSteerConfig(idx=[1], out_hook_point=HOOK_POINT)
    steer = EmbeddingSteer(model, config)

    assert steer.V_0.shape == (D_MODEL,)
    x = torch.randn(2, N_ROWS, D_MODEL)
    assert steer(x).shape == x.shape


def test_forward_returns_a_tensor():
    """Regression test: ``forward`` used to return ``None``."""
    steer = make_steer("increase")
    x = torch.randn(2, N_ROWS, D_MODEL)
    out = steer(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


def test_increase_raises_the_component_along_the_direction():
    """SIGN CONVENTION: ``increase`` moves *along* the steering direction."""
    x, up, _, direction = steer_pair(idx=(1,))
    before = project(x[:, 1, :], direction)
    after = project(up[:, 1, :], direction)
    assert torch.all(after >= before)
    assert torch.all(after > before)


def test_decrease_lowers_the_component_along_the_direction():
    """SIGN CONVENTION: ``decrease`` moves *against* the steering direction."""
    x, _, down, direction = steer_pair(idx=(1,))
    before = project(x[:, 1, :], direction)
    after = project(down[:, 1, :], direction)
    assert torch.all(after <= before)
    assert torch.all(after < before)


def test_increase_and_decrease_are_mirror_images():
    """SIGN CONVENTION: the two modes differ only in the sign of the step."""
    x, up, down, _ = steer_pair(idx=(1,))
    assert torch.allclose(up - x, -(down - x), atol=1e-6)


def test_step_magnitude_follows_scale():
    """The step is ``scale`` times the unit-norm direction."""
    x, up, _, direction = steer_pair(idx=(1,), scale=3.0)
    delta = up[:, 1, :] - x[:, 1, :]
    assert torch.allclose(delta, direction.expand_as(delta) * 3.0, atol=1e-6)


def test_only_selected_rows_are_steered():
    x, up, _, _ = steer_pair(idx=(1,))
    untouched = [0, 2, 3]
    assert torch.allclose(up[:, untouched, :], x[:, untouched, :])


def test_input_is_not_mutated():
    """Regression test: ``forward`` used to mutate state in place."""
    steer = make_steer("increase")
    x = torch.randn(2, N_ROWS, D_MODEL)
    original = x.clone()
    steer(x)
    assert torch.allclose(x, original)


def test_repeated_calls_are_idempotent():
    """A second call must not accumulate -- the direction is not mutated."""
    steer = make_steer("increase")
    x = torch.randn(2, N_ROWS, D_MODEL)
    assert torch.allclose(steer(x), steer(x))


def test_unbatched_activation_is_supported():
    steer = make_steer("increase", idx=(0,))
    x = torch.randn(N_ROWS, D_MODEL)
    out = steer(x)
    assert out.shape == x.shape
    assert torch.allclose(out[1:], x[1:])


def test_invalid_mode_is_rejected():
    model = {HOOK_POINT: torch.randn(D_MODEL, D_MODEL)}
    config = EmbeddingSteerConfig(idx=[0], mode="sideways", out_hook_point=HOOK_POINT)
    with pytest.raises(ValueError, match="Invalid mode sideways"):
        EmbeddingSteer(model, config)


def test_dimension_mismatch_is_reported():
    model = {HOOK_POINT: torch.randn(D_MODEL, D_MODEL)}
    config = EmbeddingSteerConfig(idx=[0], out_hook_point=HOOK_POINT)
    steer = EmbeddingSteer(model, config)
    with pytest.raises(ValueError, match="Steering direction has length"):
        steer(torch.randn(2, N_ROWS, D_MODEL + 1))


def test_wrapper_delegates_to_the_steer():
    steer = make_steer("increase")
    wrapper = EmbeddingSteerWrapper(steer)
    x = torch.randn(2, N_ROWS, D_MODEL)
    assert torch.allclose(wrapper(x), steer(x))


def test_context_restores_the_original_entry():
    weight = torch.randn(D_MODEL, D_MODEL)
    model: Dict[str, object] = {HOOK_POINT: weight}
    config = EmbeddingSteerConfig(idx=[1], out_hook_point=HOOK_POINT)
    steer = EmbeddingSteer(model, config)
    with EmbeddingSteeringContext(model, steer) as installed:
        assert model[HOOK_POINT] is steer
        assert installed is steer
    assert model[HOOK_POINT] is weight
