"""Hooked Transformer Linear Component.

This module contains all the component :class:`Linear`.
"""

from typing import Dict, Union
import torch
import torch.nn as nn
from jaxtyping import Float
from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens.hook_points import HookPoint
from transformer_lens.factories.activation_function_factory import (
    ActivationFunctionFactory,
)
from mechir.modelling.hooked.config import HookedTransformerConfig


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


class MLPClassificationHead(nn.Module):
    """
    Transforms ELECTRA embeddings into logits. The purpose of this module is to predict masked tokens in a sentence.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.dense = HiddenLinear(cfg)
        self.out_proj = ClassificationHead(cfg)
        self.activation = ActivationFunctionFactory.pick_activation_function(self.cfg)

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def forward(self, resid: Float[torch.Tensor, "batch d_model"]) -> torch.Tensor:
        pre_act = self.hook_pre(self.dense(resid))
        post_act = self.hook_post(self.activation(pre_act))
        return self.out_proj(post_act)
