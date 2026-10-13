"""Embedding steering via the top singular direction.

The steering direction is the top *right* singular vector ``Vh[0]`` of the
weight matrix found at ``EmbeddingSteerConfig.out_hook_point`` -- the dominant
direction of that matrix in model space. :class:`EmbeddingSteer` adds a scaled
copy of that direction to selected rows of a representation.

The right singular vector, not the left: for an embedding stored as
``[d_vocab, d_model]`` -- the usual layout -- ``U[:, 0]`` is vocab-length and
cannot be added to a ``d_model`` activation at all, whereas ``Vh[0]`` is
``d_model``-length by construction. Note ``torch.linalg.svd`` returns ``Vh``,
already transposed, so the top right singular vector is its first *row*.

Conventions (pinned by ``test/unit/test_steering.py``):

  * ``mode="increase"`` adds ``+scale * V_0``, so the component of the steered
    representation along ``V_0`` *grows*; ``mode="decrease"`` adds
    ``-scale * V_0``, so it *shrinks*. The two modes are exact mirror images:
    ``increase(x) - x == -(decrease(x) - x)``.
  * ``config.idx`` selects rows along the second-to-last axis (positions, for
    an ``[batch, seq, d_model]`` activation). Every other row is returned
    untouched and the input tensor is never mutated in place.
  * ``V_0`` must be the same length as the final axis of the steered
    activation, so ``out_hook_point`` has to name a weight matrix whose
    *columns* are indexed by that axis (i.e. ``[..., d_model]``). ``forward``
    raises :class:`ValueError` on a mismatch rather than broadcasting
    silently.

This module is intentionally not exported from ``mechir.modelling``: nothing in
the package wires it up yet.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, MutableMapping

import torch
from torch import nn

from transformer_lens.hook_points import HookedRootModule


@dataclass
class EmbeddingSteerConfig:
    """Configuration for :class:`EmbeddingSteer`.

    Args:
        idx: Rows of the representation to steer, indexed along its
            second-to-last axis.
        mode: ``"increase"`` to move along the steering direction,
            ``"decrease"`` to move against it.
        scale: Magnitude of the steering step applied to the unit-norm
            steering direction.
        out_hook_point: Key of the weight matrix whose top singular direction
            is used for steering.
    """

    idx: List[int]
    mode: str = "increase"
    scale: float = 10.0
    out_hook_point: str = "model.embedding.W"


class EmbeddingSteerWrapper(nn.Module):
    """Adapts an :class:`EmbeddingSteer` to a plain ``nn.Module`` call."""

    def __init__(self, steer: "EmbeddingSteer") -> None:
        super().__init__()
        self.steer = steer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.steer(x)


class EmbeddingSteer(HookedRootModule):
    """Steers a representation along the top singular direction of a weight.

    Args:
        model: Anything indexable by ``config.out_hook_point``, returning the
            2D weight matrix to decompose.
        config: The steering configuration.

    Raises:
        ValueError: If ``config.mode`` is neither ``"increase"`` nor
            ``"decrease"``.
    """

    def __init__(self, model: Mapping[str, Any], config: EmbeddingSteerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = model[config.out_hook_point]
        self.scale = config.scale
        self.idx = config.idx

        self.scaling: Callable[[torch.Tensor], torch.Tensor]
        if config.mode == "increase":
            self.scaling = self.increase
        elif config.mode == "decrease":
            self.scaling = self.decrease
        else:
            raise ValueError(f"Invalid mode {config.mode}")

        # Done here rather than in a `__post_init__`: this is an nn.Module, not
        # a dataclass, so `__post_init__` would never be called and `V_0` would
        # never exist.
        #
        # `torch.linalg.svd` returns Vh (already transposed), so the top right
        # singular vector is row 0, and it is d_model-length for a weight
        # stored [d_vocab, d_model]. See the module docstring.
        U, S, Vh = torch.linalg.svd(self.layer, full_matrices=False)
        self.U, self.S, self.Vh = U, S, Vh
        self.register_buffer("V_0", Vh[0].clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` with the steering step added to the rows in ``idx``.

        Args:
            x: Representation of shape ``[..., n_rows, d]``, where ``d`` is the
                length of the steering direction.

        Returns:
            A new tensor; ``x`` itself is left unchanged.

        Raises:
            ValueError: If ``x``'s final axis does not match the steering
                direction.
        """
        direction = self.V_0.to(device=x.device, dtype=x.dtype)
        if x.shape[-1] != direction.shape[0]:
            raise ValueError(
                f"Steering direction has length {direction.shape[0]} but the "
                f"activation's final axis is {x.shape[-1]}. "
                f"{self.config.out_hook_point} must be a weight matrix whose "
                "columns are indexed by that axis."
            )

        steered = x.clone()
        steered[..., self.idx, :] = steered[..., self.idx, :] + self.scaling(direction)
        return steered

    def increase(self, direction: torch.Tensor) -> torch.Tensor:
        """Scale ``direction`` so steering moves *along* it."""
        return direction * self.scale

    def decrease(self, direction: torch.Tensor) -> torch.Tensor:
        """Scale ``direction`` so steering moves *against* it."""
        return direction * -self.scale


class EmbeddingSteeringContext:
    """Swaps a model's embedding for a steered one for the duration of a block.

    Args:
        model: Anything indexable by the steer's ``out_hook_point``.
        steer: The steering module to install.
    """

    def __init__(self, model: MutableMapping[str, Any], steer: EmbeddingSteer) -> None:
        self.embedding = steer.config.out_hook_point
        self.original = model[self.embedding]
        self.steer = steer
        self.model = model

    def __enter__(self) -> EmbeddingSteer:
        self.model[self.embedding] = self.steer
        return self.steer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.model[self.embedding] = self.original
