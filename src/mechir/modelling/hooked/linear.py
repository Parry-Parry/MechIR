"""Hooked Transformer Linear Component.

This module contains all the component :class:`Linear`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from .HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.addmm import batch_addmm

class ClassificationHead(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.num_labels, dtype=self.cfg.dtype))
        self.b_in = nn.Parameter(torch.zeros(self.num_labels, dtype=self.cfg.dtype))

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos num_labels"]:
        
        return self.hook_pre(batch_addmm(self.b_in, self.W_in, x))  # [batch, pos, d_mlp]
    
class HiddenLinear(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_model, dtype=self.cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        return self.hook_pre(batch_addmm(self.b, self.W, x))  # [batch, pos, d_mlp]