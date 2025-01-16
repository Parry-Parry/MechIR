from typing import Callable
import logging
import torch
from jaxtyping import Float
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformer_lens import HookedEncoderDecoder
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookedRootModule
import transformer_lens.utils as utils
from .patched import PatchedMixin
from .sae import SAEMixin
from ..util import linear_rank_function
from .hooked.loading_from_pretrained import get_official_model_name

logger = logging.getLogger(__name__)

"""
monoT5
"""


class MonoT5(HookedRootModule, PatchedMixin, SAEMixin):
    def __init__(
        self,
        model_name_or_path: str,
        pos_token: str = "true",
        neg_token: str = "false",
        special_token: str = "X",
        softmax_output: bool = False,
        return_cache: bool = False,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.special_token = special_token
        self.pos_token = self.tokenizer.encode(pos_token, return_tensors="pt")[0]
        self.neg_token = self.tokenizer.encode(neg_token, return_tensors="pt")[0]

        torch.set_grad_enabled(False)
        self._device = utils.get_device()
        self.model_name_or_path = get_official_model_name(model_name_or_path)
        self.__hf_model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            .eval()
            .to(self._device)
        )
        self._model = HookedEncoderDecoder.from_pretrained(
            self.model_name_or_path, device=self._device, hf_model=self.__hf_model
        )

        self.softmax_output = softmax_output
        self._return_cache = return_cache

        self.setup()

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
    ):
        model_output = self._model(input=input_ids, one_hot_attention_mask=attention_mask, return_type="logits")
        model_output = (
            model_output[:, 0, (self.pos_token, self.neg_token)].softmax(dim=-1)[:, 0]
            if self.softmax_output
            else model_output[:, 0, (self.pos_token, self.neg_token)][:, 0]
        )
        return model_output

    def get_act_patch_block_every(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
        scores: Float[torch.Tensor, "batch pos"],
        scores_p: Float[torch.Tensor, "batch pos"],
        **kwargs,
    ) -> Float[torch.Tensor, "3 layer pos"]:
        """
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        """

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
            output = (
                output[:, 0, (self.pos_token, self.neg_token)].softmax(dim=-1)[:, 0]
                if self.softmax_output
                else output[:, 0, (self.pos_token, self.neg_token)][:, 0]
            )
            results[index] = patching_metric(output, scores, scores_p)

        return results

    def get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        patching_metric: Callable,
        scores: Float[torch.Tensor, "batch pos"],
        scores_p: Float[torch.Tensor, "batch pos"],
        **kwargs,
    ) -> Float[torch.Tensor, "layer head"]:
        """
        Returns an array of results of patching at all positions for each head in each
        layer, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's embedding output.
        """

        results = torch.zeros(
            self._model.cfg.n_layers,
            self._model.cfg.n_heads,
            device=self._device,
            dtype=torch.float32,
        )

        for index, output in self._get_act_patch_attn_head_out_all_pos(
            corrupted_tokens=corrupted_tokens, clean_cache=clean_cache
        ):
            results[index] = patching_metric(output, scores, scores_p)

        return results

    def get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        layer_head_list,
        patching_metric: Callable,
        scores: Float[torch.Tensor, "batch pos"],
        scores_p: Float[torch.Tensor, "batch pos"],
        **kwargs,
    ) -> Float[torch.Tensor, "layer pos head"]:

        _, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(
            2, len(layer_head_list), seq_len, device=self._device, dtype=torch.float32
        )

        for index, output in self._get_act_patch_attn_head_by_pos(
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            layer_head_list=layer_head_list,
        ):
            results[index] = patching_metric(output, scores, scores_p)

        return results

    def score(self, sequences: dict, cache=False):
        if cache:
            logits, cache = self.run_with_cache(
                sequences["input_ids"], sequences["attention_mask"]
            )
            return logits, cache

        logits = self.forward(sequences["input_ids"], sequences["attention_mask"])
        return logits

    def patch(
        self,
        sequences: dict,
        sequences_p: dict,
        patch_type: str = "block_all",
        layer_head_list: list = [],
        patching_metric: Callable = linear_rank_function,
    ):
        assert (
            patch_type in self._patch_funcs
        ), f"Patch type {patch_type} not recognized. Choose from {self._patch_funcs.keys()}"
        scores, _ = self.score(sequences)
        scores_p, cache = self.score(sequences_p, cache=True)

        patching_kwargs = {
            "corrupted_tokens": sequences,
            "clean_cache": cache,
            "patching_metric": patching_metric,
            "layer_head_list": layer_head_list,
            "scores": scores,
            "scores_p": scores_p,
        }
        patched_output = self._patch_funcs[patch_type](**patching_kwargs)
        if self._return_cache:
            return patched_output, cache
        return patched_output
