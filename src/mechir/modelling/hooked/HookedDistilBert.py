"""Hooked Encoder.

Contains a BERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import Literal

from transformer_lens import HookedTransformerConfig
from transformer_lens.components import BertBlock, BertMLMHead, Unembed
from transformer_lens.hook_points import HookPoint
from .hooked_components import BertEmbed
from .HookedEncoder import HookedEncoder


class HookedDistilBert(HookedEncoder):
    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        assert (
            self.cfg.n_devices == 1
        ), "Multiple devices not supported for HookedEncoder"
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.cfg.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        else:
            self.tokenizer = None

        if self.cfg.d_vocab == -1:
            # If we have a tokenizer, vocab size can be inferred from it.
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

        self.embed = BertEmbed(self.cfg)
        self.blocks = nn.ModuleList(
            [BertBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.mlm_head = BertMLMHead(cfg)
        self.unembed = Unembed(self.cfg)

        self.hook_full_embed = HookPoint()

        if move_to_device:
            self.to(self.cfg.device)

        self.setup()


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
        self.classifier = nn.Linear(cfg.d_model, cfg.n_labels)
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

        logits = self.classifier(resid[:, 0, :])
        return logits
