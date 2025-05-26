"""Hooked Encoder.

Contains a BERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import Literal

from mechir.modelling.hooked import loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    BertBlock,
    BertMLMHead,
    Unembed,
    BertNSPHead,
    BertPooler,
)
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utilities import devices

from mechir.modelling.hooked.config import HookedTransformerConfig
from mechir.modelling.hooked.components import BertEmbed
from mechir.modelling.hooked.linear import ClassificationHead, MLPClassificationHead


class HookedEncoder(HookedRootModule):
    """
    This class implements a BERT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__()
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
            huggingface_token = os.environ.get("HF_TOKEN", "")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            )
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
        self.mlm_head = BertMLMHead(self.cfg)
        self.unembed = Unembed(self.cfg)
        self.nsp_head = BertNSPHead(self.cfg)
        self.pooler = BertPooler(self.cfg)

        self.hook_full_embed = HookPoint()

        self.use_token_type_ids = self.cfg.use_token_type_ids

        if move_to_device:
            self.to(self.cfg.device)

        self.setup()

    def to_tokens(
        self,
        input: Union[str, List[str]],
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Tuple[
        Int[torch.Tensor, "batch pos"],
        Int[torch.Tensor, "batch pos"],
        Int[torch.Tensor, "batch pos"],
    ]:
        """Converts a string to a tensor of tokens.
        Taken mostly from the HookedTransformer implementation, but does not support default padding
        sides or prepend_bos.
        Args:
            input (Union[str, List[str]]): The input to tokenize.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the model lives on. Defaults to True
            truncate (bool): If the output tokens are too long, whether to truncate the output
            tokens to the model's max context window. Does nothing for shorter inputs. Defaults to
            True.
        """

        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"

        encodings = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )

        tokens = encodings.input_ids

        if move_to_device:
            tokens = tokens.to(self.cfg.device)
            token_type_ids = (
                encodings.token_type_ids.to(self.cfg.device)
                if self.use_token_type_ids
                else None
            )
            attention_mask = encodings.attention_mask.to(self.cfg.device)

        return tokens, token_type_ids, attention_mask

    def encoder_output(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_vocab"]:
        """Processes input through the encoder layers and returns the resulting residual stream.

        Args:
            input: Input tokens as integers with shape (batch, position)
            token_type_ids: Optional binary ids indicating segment membership.
                Shape (batch_size, sequence_length). For example, with input
                "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be
                [0, 0, ..., 0, 1, ..., 1, 1] where 0 marks tokens from sentence A
                and 1 marks tokens from sentence B.
            one_zero_attention_mask: Optional binary mask of shape (batch_size, sequence_length)
                where 1 indicates tokens to attend to and 0 indicates tokens to ignore.
                Used primarily for handling padding in batched inputs.

        Returns:
            resid: Final residual stream tensor of shape (batch, position, d_model)

        Raises:
            AssertionError: If using string input without a tokenizer
        """

        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
            if one_zero_attention_mask is not None:
                one_zero_attention_mask = one_zero_attention_mask.to(self.cfg.device)

        resid = self.hook_full_embed(self.embed(tokens, token_type_ids))

        large_negative_number = -torch.inf
        mask = (
            repeat(1 - one_zero_attention_mask, "batch pos -> batch 1 1 pos")
            if one_zero_attention_mask is not None
            else None
        )
        additive_attention_mask = (
            torch.where(mask == 1, large_negative_number, 0)
            if mask is not None
            else None
        )

        if start_at_layer is None:
            start_at_layer = 0

        idx_and_block = list(zip(range(self.cfg.n_layers), self.blocks))

        for _, block in idx_and_block[start_at_layer:stop_at_layer]:
            resid = block(resid, additive_attention_mask)

        return resid

    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Optional[str] = "embeddings",
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
    ) -> Union[Float[torch.Tensor, "batch pos d_vocab"], None]:
        """Input must be a batch of tokens. Strings and lists of strings are not yet supported.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), or 'logits' (return logits).

        token_type_ids Optional[torch.Tensor]: Binary ids indicating whether a token belongs to sequence A or B. For example, for two sentences: "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A, `1` from Sentence B. If not provided, BERT assumes a single sequence input. Typically, shape is (batch_size, sequence_length).

        attention_mask: Optional[torch.Tensor]: A binary mask which indicates which tokens should be attended to (1) and which should be ignored (0). Primarily used for padding variable-length sentences in a batch. For instance, in a batch with sentences of differing lengths, shorter sentences are padded with 0s on the right. If not provided, the model assumes all tokens should be attended to.
        """

        if start_at_layer is None:
            if isinstance(input, str) or isinstance(input, list):
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if input is a string"
                input, token_type_ids_from_tokenizer, attention_mask = self.to_tokens(
                    input
                )

                # If token_type_ids or attention mask are not provided, use the ones from the tokenizer
                token_type_ids = (
                    token_type_ids_from_tokenizer
                    if token_type_ids is None
                    else token_type_ids
                )
        else:
            assert type(input) is torch.Tensor
        residual = input

        if residual.device.type != self.cfg.device:
            residual = residual.to(self.cfg.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.cfg.device)
        if start_at_layer is None:
            start_at_layer = 0

        resid = self.encoder_output(
            residual,
            token_type_ids=token_type_ids,
            start_at_layer=start_at_layer,
            stop_at_layer=stop_at_layer,
            one_zero_attention_mask=attention_mask,
        )

        if stop_at_layer is not None or return_type == "embeddings":
            return resid

        resid = self.mlm_head(resid)
        logits = self.unembed(resid)

        if return_type == "predictions":
            # Get predictions for masked tokens
            logprobs = logits[logits == self.tokenizer.mask_token_id].log_softmax(
                dim=-1
            )
            predictions = self.tokenizer.decode(logprobs.argmax(dim=-1))

            # If input was a list of strings, split predictions into a list
            if " " in predictions:
                # Split along space
                predictions = predictions.split(" ")
                predictions = [
                    f"Prediction {i}: {p}" for i, p in enumerate(predictions)
                ]
            return predictions

        elif return_type is None:
            return None

        return logits

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        ActivationCache,
    ]: ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Dict[str, torch.Tensor],
    ]: ...

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        cache_as_dict: bool = False,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            if not cache_as_dict:
                cache = ActivationCache(
                    cache_dict, self, has_batch_dim=not remove_batch_dim
                )
            return out, cache
        else:
            return out, None

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    def mps(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("mps")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model=None,
        device: Optional[str] = None,
        tokenizer=None,
        move_to_device=True,
        dtype=torch.float32,
        **from_pretrained_kwargs,
    ) -> HookedEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for BERT in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using BERT for interpretability research, keep in mind that BERT has some significant architectural "
            "differences to GPT. For example, LayerNorms are applied *after* the attention and MLP components, meaning "
            "that the last LayerNorm in a block cannot be folded."
        )

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]

        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            **from_pretrained_kwargs,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        model = cls(cfg, tokenizer, move_to_device=False)

        model.load_state_dict(state_dict, strict=False)

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedEncoder")

        return model

    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        """
        Convenience to get the unembedding bias
        """
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.embed.W_E

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.embed.pos_embed.W_pos

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_K for block in self.blocks], dim=0
        )

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_Q for block in self.blocks], dim=0
        )

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_V for block in self.blocks], dim=0
        )

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_O for block in self.blocks], dim=0
        )

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.W_in for block in self.blocks], dim=0
        )

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.W_out for block in self.blocks], dim=0
        )

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_K for block in self.blocks], dim=0
        )

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_Q for block in self.blocks], dim=0
        )

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_V for block in self.blocks], dim=0
        )

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_O for block in self.blocks], dim=0
        )

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.b_in for block in self.blocks], dim=0
        )

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.b_out for block in self.blocks], dim=0
        )

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
        Useful for visualizing attention patterns."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> List[str]:
        """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
        return [
            f"L{l}H{h}"
            for l in range(self.cfg.n_layers)
            for h in range(self.cfg.n_heads)
        ]


class HookedEncoderForSequenceClassification(HookedEncoder):
    """
    This class implements a BERT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The current MVP implementation supports only the masked language modelling (MLM) task. Next sentence prediction (NSP), causal language modelling, and other tasks are not yet supported.
    - Also note that model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__(cfg, tokenizer, move_to_device, **kwargs)
        self.classifier = (
            ClassificationHead(cfg)
            if not self.cfg.use_mlp_head
            else MLPClassificationHead(cfg)
        )
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


__all__ = ["HookedEncoder", "HookedEncoderForSequenceClassification"]
