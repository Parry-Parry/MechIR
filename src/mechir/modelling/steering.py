import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, List


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
