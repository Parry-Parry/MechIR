"""Hooked ELECTRA.

Contains a ELECTRA style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import torch
from jaxtyping import Float, Int
from torch import nn
from transformer_lens.hook_points import HookPoint
from mechir.modelling.hooked.linear import ClassificationHead, HiddenLinear
from mechir.modelling.architectures.base import HookedEncoder
from mechir.modelling.hooked.config import HookedTransformerConfig


class ElectraClassificationHead(nn.Module):
    """
    Transforms ELECTRA embeddings into logits. The purpose of this module is to predict masked tokens in a sentence.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.dense = HiddenLinear(cfg)
        self.out_proj = ClassificationHead(cfg)
        self.activation = nn.GELU()

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def forward(self, resid: Float[torch.Tensor, "batch d_model"]) -> torch.Tensor:
        pre_act = self.hook_pre(self.dense(resid))
        post_act = self.hook_post(self.activation(pre_act))
        return self.out_proj(post_act)

HookedElectra = HookedEncoder

class HookedElectraForSequenceClassification(HookedEncoder):
    """
    This class implements a ELECTRA-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.


    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__(cfg, tokenizer, move_to_device, **kwargs)
        self.classifier = ElectraClassificationHead(cfg)
        self.setup()

    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Optional[str] = "embeddings",
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos d_vocab"]]:
        """Input must be a batch of tokens. Strings and lists of strings are not yet supported.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), or 'logits' (return logits).

        token_type_ids Optional[torch.Tensor]: Binary ids indicating whether a token belongs to sequence A or B. For example, for two sentences: "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A, `1` from Sentence B. If not provided, BERT assumes a single sequence input. Typically, shape is (batch_size, sequence_length).

        attention_mask: Optional[torch.Tensor]: A binary mask which indicates which tokens should be attended to (1) and which should be ignored (0). Primarily used for padding variable-length sentences in a batch. For instance, in a batch with sentences of differing lengths, shorter sentences are padded with 0s on the right. If not provided, the model assumes all tokens should be attended to.
        """

        hidden = super().forward(
            input,
            token_type_ids=token_type_ids,
            start_at_layer=start_at_layer,
            stop_at_layer=stop_at_layer,
            return_type="embeddings",
            attention_mask=attention_mask,
        )
        if return_type == "embeddings" or stop_at_layer is not None:
            return hidden
        logits = self.classifier(hidden[:, 0, :])

        if return_type is None:
            return None
        return logits


__all__ = ["HookedElectra", "HookedElectraForSequenceClassification"]
