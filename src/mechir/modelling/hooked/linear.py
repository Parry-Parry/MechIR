"""Hooked Transformer Linear Component.

This module contains all the component :class:`Linear`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from fancy_einsum import einsum

'''
TODO:
- Ensure naming convention matches cfg
- Check if additional naming is required in overall cfg
'''

class Linear(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.num_labels, dtype=self.cfg.dtype))
        self.b_in = nn.Parameter(torch.zeros(self.num_labels, dtype=self.cfg.dtype))

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos num_labels"]:
        
        return (
            einsum(
                "batch pos d_model_in, num_labels d_model_in -> batch pos num_labels",
                x,
                self.W,
            )
            + self.b
        )
