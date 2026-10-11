"""Generative relevance scoring.

A generative reranker scores a query/document pair by contrasting the logits
of two label tokens at one position. Qwen3-Reranker contrasts "yes" against
"no" at the final position; MonoT5 contrasts "true" against "false" at the
first decoder position. Same scorer, different backbone and position.
"""

from typing import Callable, Optional

import logging
import torch
from jaxtyping import Float
from transformers import AutoTokenizer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookedRootModule
import transformer_lens.utils as utils

from mechir.modelling.backbones import load_backbone
from mechir.modelling.hooked.loading_from_pretrained import get_official_model_name
from mechir.modelling.patched import PatchedMixin
from mechir.modelling.sae import SAEMixin
from mechir.util import linear_rank_function

logger = logging.getLogger(__name__)


class GenerativeRanker(HookedRootModule, PatchedMixin, SAEMixin):
    """Decoder-backbone reranker scored by a two-token logit contrast.

    Parameters
    ----------
    model_name_or_path: str
        A decoder checkpoint, e.g. ``Qwen/Qwen3-Reranker-0.6B``.
    pos_token, neg_token: str
        Label tokens whose logits are contrasted. Each must encode to exactly
        one vocabulary id.
    softmax_output: bool
        Return ``softmax([pos, neg])[0]`` in ``[0, 1]`` instead of the raw
        logit difference.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pos_token: str = "yes",
        neg_token: str = "no",
        tokenizer=None,
        special_token: str = "a",
        softmax_output: bool = False,
        return_cache: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = get_official_model_name(model_name_or_path)
        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name_or_path)
            if tokenizer is None
            else tokenizer
        )
        self.special_token = special_token
        self.pos_token = self._single_token_id(pos_token)
        self.neg_token = self._single_token_id(neg_token)

        self._device = utils.get_device()
        self._model, self.__hf_model, is_decoder = load_backbone(
            model_name_or_path, self.model_name_or_path, self._device
        )
        if not is_decoder:
            raise ValueError(
                f"{model_name_or_path} is not a decoder checkpoint. Use Cat for "
                "encoder cross-encoders or MonoT5 for encoder-decoders."
            )

        self.softmax_output = softmax_output
        self._return_cache = return_cache

        self.setup()

    def _single_token_id(self, token: str) -> int:
        """Resolve ``token`` to one vocabulary id."""
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Relevance label {token!r} maps to {len(ids)} tokens ({ids}); "
                "generative scoring needs labels that are a single token."
            )
        return ids[0]

    def _contrast(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        attention_mask: Optional[Float[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch"]:
        """Reduce full logits to one relevance score per item.

        The label is predicted at the last non-padding position, which with
        right padding differs per row.
        """
        if attention_mask is None:
            last = torch.full(
                (logits.shape[0],),
                logits.shape[1] - 1,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            last = attention_mask.sum(dim=1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        pair = logits[rows, last][:, (self.pos_token, self.neg_token)]

        if self.softmax_output:
            return pair.softmax(dim=-1)[:, 0]
        return pair[:, 0] - pair[:, 1]

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
        token_type_ids: Float[torch.Tensor, "batch seq"] = None,
    ):
        logits = self._model(
            input_ids, attention_mask=attention_mask, return_type="logits"
        )
        return self._contrast(logits, attention_mask)

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        cache_as_dict: bool = False,
        remove_batch_dim: bool = False,
        **kwargs,
    ):
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object and not cache_as_dict:
            return out, ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
        return out, cache_dict if return_cache_object else None

    @torch.no_grad()
    def score(self, sequences: dict, cache=False, cache_as_dict=False):
        if cache:
            scores, cache = self.run_with_cache(
                sequences["input_ids"],
                sequences["attention_mask"],
                cache_as_dict=cache_as_dict,
            )
            return scores, cache

        scores = self.forward(sequences["input_ids"], sequences["attention_mask"])
        return scores, None

    def get_act_patch_block_every(
        self,
        corrupted_tokens,
        clean_cache: ActivationCache,
        patching_metric: Callable,
        scores,
        scores_p,
        **kwargs,
    ):
        _, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(
            3,
            self._model.cfg.n_layers,
            seq_len,
            device=self._device,
            dtype=torch.float32,
        )
        for index, output in self._get_act_patch_block_every(
            corrupted_tokens=corrupted_tokens, clean_cache=clean_cache
        ):
            results[index] = patching_metric(output, scores, scores_p).mean()
        return results

    def get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens,
        clean_cache: ActivationCache,
        patching_metric: Callable,
        scores,
        scores_p,
        **kwargs,
    ):
        results = torch.zeros(
            self._model.cfg.n_layers,
            self._model.cfg.n_heads,
            device=self._device,
            dtype=torch.float32,
        )
        for index, output in self._get_act_patch_attn_head_out_all_pos(
            corrupted_tokens=corrupted_tokens, clean_cache=clean_cache
        ):
            results[index] = patching_metric(output, scores, scores_p).mean()
        return results

    def get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens,
        clean_cache: ActivationCache,
        layer_head_list,
        patching_metric: Callable,
        scores,
        scores_p,
        **kwargs,
    ):
        _, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(
            2, len(layer_head_list), seq_len, device=self._device, dtype=torch.float32
        )
        for index, output in self._get_act_patch_attn_head_by_pos(
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            layer_head_list=layer_head_list,
        ):
            results[index] = patching_metric(output, scores, scores_p).mean()
        return results

    @torch.no_grad()
    def patch(
        self,
        sequences: dict,
        sequences_p: dict,
        patch_type: str = "block_all",
        layer_head_list: list = [],
        patching_metric: Callable = linear_rank_function,
    ):
        assert patch_type in self._patch_funcs, (
            f"Patch type {patch_type} not recognized. "
            f"Choose from {self._patch_funcs.keys()}"
        )
        scores, _ = self.score(sequences)
        scores_p, cache = self.score(sequences_p, cache=True)

        patched_output = self._patch_funcs[patch_type](
            corrupted_tokens=sequences,
            clean_cache=cache,
            patching_metric=patching_metric,
            layer_head_list=layer_head_list,
            scores=scores,
            scores_p=scores_p,
        )
        if self._return_cache:
            return patched_output, cache
        return patched_output, None


__all__ = ["GenerativeRanker"]
