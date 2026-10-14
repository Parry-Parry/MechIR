from typing import Callable
import logging
import torch
from jaxtyping import Float
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformer_lens import HookedEncoderDecoder
from transformer_lens.hook_points import HookedRootModule
import transformer_lens.utils as utils
from mechir.modelling.hook_names import ENCODER_HOOK_NAMES, HookNameStrategy
from mechir.modelling.patched import PatchedMixin
from mechir.modelling.sae import SAEMixin
from mechir.modelling.hooked.loading_from_pretrained import get_official_model_name
from mechir.util import linear_rank_function

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
        super().__init__()
        self.model_name_or_path = get_official_model_name(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.special_token = special_token
        # Single vocabulary ids. `encode(..., return_tensors="pt")[0]` would
        # return the whole sequence, including T5's trailing EOS.
        self.pos_token = self._single_token_id(pos_token)
        self.neg_token = self._single_token_id(neg_token)

        self._device = utils.get_device()
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

    def _single_token_id(self, token: str) -> int:
        """Resolve ``token`` to one vocabulary id.

        The relevance head reads a single logit per class, so a label that
        does not map to exactly one token cannot be scored.
        """
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Relevance label {token!r} maps to {len(ids)} tokens ({ids}); "
                "MonoT5 scoring needs labels that are a single token."
            )
        return ids[0]

    def _decoder_start(
        self, input_ids: Float[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, "batch 1"]:
        """One decoder step holding just the start token.

        MonoT5 reads the relevance label off the first decoder position, so
        the decoder only ever needs to be run for a single step.
        """
        start_id = self._model.cfg.decoder_start_token_id or 0
        return torch.full(
            (input_ids.shape[0], 1),
            start_id,
            dtype=torch.long,
            device=input_ids.device,
        )

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
    ):
        model_output = self._model(
            input=input_ids,
            one_zero_attention_mask=attention_mask,
            decoder_input=self._decoder_start(input_ids),
            return_type="logits",
        )
        model_output = (
            model_output[:, 0, (self.pos_token, self.neg_token)].softmax(dim=-1)[:, 0]
            if self.softmax_output
            else model_output[:, 0, (self.pos_token, self.neg_token)][:, 0]
        )
        return model_output

    @property
    def _hook_names(self) -> HookNameStrategy:
        """Encoder-stack hook naming.

        The architecture seam of
        :class:`~mechir.modelling.patched.PatchedMixin`. ``HookedEncoderDecoder``
        names its hooks ``encoder.{layer}.…`` and ``decoder.{layer}.…`` rather
        than the single-stack ``blocks.{layer}.…``, so the inherited default
        misses with ``KeyError: '_model.blocks.0.hook_resid_pre'``.

        Patching targets the encoder: the positions being swept are the
        encoder's input tokens, and the decoder is only ever run for the one
        step that carries the relevance label.
        """
        return ENCODER_HOOK_NAMES

    # `forward` already reduces the decoder logits to one score per item, so
    # the default `PatchedMixin._patch_score` (the identity) is correct and
    # the patching methods are inherited unchanged.

    @torch.no_grad()
    def score(self, sequences: dict, cache=False):
        if cache:
            logits, cache = self.run_with_cache(
                sequences["input_ids"], sequences["attention_mask"]
            )
            return logits, cache

        logits = self.forward(sequences["input_ids"], sequences["attention_mask"])
        return logits, None

    @torch.no_grad()
    def patch(
        self,
        sequences: dict,
        sequences_p: dict = None,
        patch_type: str = "block_all",
        layer_head_list: list = [],
        patching_metric: Callable = linear_rank_function,
        perturbed_sequences: dict = None,
    ):
        # MonoT5DataCollator emits "perturbed_sequences", so accept that name too --
        # otherwise the documented `model.patch(**batch)` workflow raises
        # TypeError.
        sequences_p = sequences_p if sequences_p is not None else perturbed_sequences
        if sequences_p is None:
            raise TypeError(
                "patch() needs sequences_p (or perturbed_sequences, the key "
                "MonoT5DataCollator emits)"
            )
        assert patch_type in self._patch_funcs, (
            f"Patch type {patch_type} not recognized. Choose from {self._patch_funcs.keys()}"
        )
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
