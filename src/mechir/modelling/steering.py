"""Embedding steering via the top singular direction.

TODO: this module is not exported from ``mechir.modelling`` and has never
been executed. Importing it now works, but three defects remain before it
can be wired up:

  * ``EmbeddingSteer.increase`` returns ``x * -self.scale`` while
    ``decrease`` returns ``x * self.scale``, so the two modes are negated
    with respect to their names.
  * ``__post_init__`` is defined on a plain class, not a dataclass, so it is
    never called and ``self.U_0`` is never assigned. ``forward`` therefore
    raises ``AttributeError``.
  * ``forward`` mutates ``self.U_0`` in place and returns ``None``.

Fixing these needs a decision about the intended steering semantics, so they
are left documented rather than guessed at.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, List

from transformer_lens.hook_points import HookedRootModule


@dataclass
class EmbeddingSteerConfig:
    idx: List[int]
    mode: Optional[str] = "increase"
    scale: Optional[int] = 10
    out_hook_point: Optional[str] = "model.embedding.W"


class EmbeddingSteerWrapper(nn.Module):
    def __init__(self, steer):
        super().__init__()
        self.steer = steer

    def forward(self, x):
        return self.steer(x)


class EmbeddingSteer(HookedRootModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.layer = model[self.config.out_hook_point]
        self.scale = self.config.scale
        self.idx = self.config.idx
        if self.config.mode == "increase":
            self.scaling = self.increase
        elif self.config.mode == "decrease":
            self.scaling = self.decrease
        else:
            raise ValueError(f"Invalid mode {self.config.mode}")

    def increase(self, x):
        return x * -self.scale

    def decrease(self, x):
        return x * self.scale

    def __post_init__(self):
        U, S, V = torch.linalg.svd(self.layer, full_matrices=False)
        self.U, self.S, self.V = U, S, V

        self.U_0 = U[:, 0].clone().detach()

    def forward(self, x):
        self.U_0[x] = self.scaling(self.U_0[x])


class EmbeddingSteeringContext:
    def __init__(self, model, steer):
        self.embedding = steer.config.out_hook_point
        self.original = model[self.embedding]
        self.steer = steer
        self.model = model

    def __enter__(self):
        self.model[self.embedding] = self.steer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model[self.embedding] = self.original
