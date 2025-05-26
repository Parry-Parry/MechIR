"""Hooked Encoder.

Contains a BERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

from typing import Dict, Optional, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import Literal

from transformer_lens.components import BertBlock, BertMLMHead, Unembed
from transformer_lens.hook_points import HookPoint
from mechir.modelling.hooked.components import BertEmbed
from mechir.modelling.hooked.linear import MLPClassificationHead
from mechir.modelling.architectures.base import HookedEncoder
from mechir.modelling.hooked.config import HookedTransformerConfig


HookedDistilBert = HookedEncoder

class HookedDistilBertForSequenceClassification(HookedDistilBert):
    """
    This class implements a BERT-style encoder for sequence classification using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedDistilBert.

    Limitations:
    - The current MVP implementation supports only the masked language modelling (MLM) task. Next sentence prediction (NSP), causal language modelling, and other tasks are not yet supported.
    - Also note that model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__(cfg, tokenizer, move_to_device=move_to_device, **kwargs)
        self.classifier = MLPClassificationHead(self.cfg)
        self.setup()

    @overload
    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Literal["logits"],
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_vocab"]: ...

    @overload
    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Literal[None],
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos d_vocab"]]: ...

    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Optional[str] = "logits",
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos d_vocab"]]:
        """Input must be a batch of tokens. Strings and lists of strings are not yet supported.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), or 'logits' (return logits).

        token_type_ids Optional[torch.Tensor]: Binary ids indicating whether a token belongs to sequence A or B. For example, for two sentences: "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A, `1` from Sentence B. If not provided, BERT assumes a single sequence input. Typically, shape is (batch_size, sequence_length).

        attention_mask: Optional[torch.Tensor]: A binary mask which indicates which tokens should be attended to (1) and which should be ignored (0). Primarily used for padding variable-length sentences in a batch. For instance, in a batch with sentences of differing lengths, shorter sentences are padded with 0s on the right. If not provided, the model assumes all tokens should be attended to.
        """

        tokens = input

        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.cfg.device)

        resid = self.hook_full_embed(self.embed(tokens))

        large_negative_number = -torch.inf
        mask = (
            repeat(1 - attention_mask, "batch pos -> batch 1 1 pos")
            if attention_mask is not None
            else None
        )
        additive_attention_mask = (
            torch.where(mask == 1, large_negative_number, 0)
            if mask is not None
            else None
        )

        for block in self.blocks:
            resid = block(resid, additive_attention_mask)

        if return_type == "embeddings":
            return resid

        if return_type is None:
            return

        logits = self.mlp(resid[:, 0, :])
        logits = self.out_proj(logits)
        return logits


__all__ = ["HookedDistilBert", "HookedDistilBertForSequenceClassification"]
