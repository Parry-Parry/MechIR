"""Hooked Transformer Linear Component.

This module contains all the component :class:`Linear`.
"""

from typing import Dict, Union
import torch
import torch.nn as nn
from jaxtyping import Float
from transformer_lens.utilities.addmm import batch_addmm
from .HookedTransformerConfig import HookedTransformerConfig


class ClassificationHead(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.num_labels, dtype=self.cfg.dtype)
        )
        self.b = nn.Parameter(torch.zeros(self.cfg.num_labels, dtype=self.cfg.dtype))

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos num_labels"]:
        return batch_addmm(self.b, self.W, x)


class HiddenLinear(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        self.b = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        return batch_addmm(self.b, self.W.T, x)
