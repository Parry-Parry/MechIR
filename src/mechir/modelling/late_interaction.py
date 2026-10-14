"""ColBERT-style late interaction.

:class:`~mechir.Dot` pools each sequence to one vector and scores with a dot
product, so the query/document interaction happens after all the token
detail has been averaged away. Late interaction keeps the token matrices and
scores with MaxSim: every *query* token picks its best-matching *document*
token, and those per-query-token maxima are summed.

That asymmetry is the whole content of the operator, and getting it backwards
is easy because the similarity matrix is square whenever the query and the
document happen to be padded to the same length. See :func:`maxsim`.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import logging

import torch
import torch.nn.functional as F
from jaxtyping import Float
from transformers import AutoTokenizer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookedRootModule
import transformer_lens.utils as utils

from mechir.modelling.backbones import load_backbone
from mechir.modelling.hooked.loading_from_pretrained import get_official_model_name
from mechir.modelling.patched import PatchedMixin
from mechir.util import linear_rank_function

logger = logging.getLogger(__name__)

#: Similarity written over padding document tokens before the max. Finite --
#: ``-inf`` would make an all-padding document's score NaN once multiplied by
#: a zero query mask -- but far below any real cosine or dot product.
MASKED_SIMILARITY = -1e4


def maxsim(
    reps_q: Float[torch.Tensor, "batch q_len d_model"],
    reps_d: Float[torch.Tensor, "batch d_len d_model"],
    mask_q: Optional[Float[torch.Tensor, "batch q_len"]] = None,
    mask_d: Optional[Float[torch.Tensor, "batch d_len"]] = None,
) -> Float[torch.Tensor, "batch"]:  # noqa: F821
    """Score document token matrices against query token matrices by MaxSim.

    The reduction is ``similarity.max(-1).values.sum(-1)``: the max runs over
    the *last* axis of ``(batch, q_len, d_len)``, which is the document
    tokens, and the sum then runs over the query tokens.

    Parameters
    ----------
    reps_q: torch.Tensor
        Query token embeddings, ``(batch, q_len, d_model)``.
    reps_d: torch.Tensor
        Document token embeddings, ``(batch, d_len, d_model)``.
    mask_q: torch.Tensor, optional
        Query attention mask. Padding query tokens would otherwise each
        contribute their own maximum to the sum, making the score depend on
        how far the query batch was padded.
    mask_d: torch.Tensor, optional
        Document attention mask. Padding document tokens would otherwise be
        eligible to win the max, so a padded batch would not score the same
        as an unpadded one.

    Returns
    -------
    torch.Tensor
        One score per batch item, ``(batch,)``.

    Notes
    -----
    Reducing the other way round -- ``max`` over query tokens, ``sum`` over
    document tokens -- is a different operator that grows without bound in
    the document length, because a longer document contributes more summands
    rather than more candidates for the same number of maxima. It is not a
    relevance score: it ranks long documents above short ones regardless of
    content. ``test/unit/test_late_interaction.py`` pins the orientation.

    Both masks are optional only so that unpadded single sequences still
    work. Omitting one when the batch *is* padded makes the score depend on
    the padding length.
    """
    if reps_d.dim() != 3:
        raise ValueError(
            f"reps_d must be (batch, d_len, d_model); got shape {tuple(reps_d.shape)}"
        )
    if reps_q.dim() != 3:
        raise ValueError(
            f"reps_q must be (batch, q_len, d_model); got shape {tuple(reps_q.shape)}"
        )

    reps_q = reps_q.to(reps_d.device)
    similarity = reps_q @ reps_d.transpose(-1, -2)

    if mask_d is not None:
        # Masks index the activations, so they have to share their device:
        # the model moves its own copy of the inputs to cfg.device while the
        # mask reaching here is still the caller's. See backbones._pool_mean.
        keep_d = mask_d.to(similarity.device).bool().unsqueeze(1)
        similarity = similarity.masked_fill(~keep_d, MASKED_SIMILARITY)

    # Max over document tokens, leaving one best match per query token.
    per_query_token = similarity.max(dim=-1).values

    if mask_d is not None:
        # A row that is entirely padding has no token to match against, so
        # it scores 0 rather than q_len * MASKED_SIMILARITY.
        per_query_token = per_query_token.masked_fill(
            ~keep_d.squeeze(1).any(dim=-1, keepdim=True), 0.0
        )

    if mask_q is not None:
        per_query_token = per_query_token * mask_q.to(
            device=per_query_token.device, dtype=per_query_token.dtype
        )

    # Sum over query tokens.
    return per_query_token.sum(dim=-1)


class LateInteraction(HookedRootModule, PatchedMixin):
    """A hooked late-interaction (ColBERT-style) ranker.

    A single-stack encoder whose ``forward`` returns token embeddings rather
    than a pooled vector or a score, scored with :func:`maxsim`.

    Parameters
    ----------
    model_name_or_path: str
        Checkpoint to load. Any encoder-only or decoder-only backbone
        :func:`~mechir.modelling.backbones.load_backbone` resolves.
    tokenizer: optional
        Tokenizer to use; loaded from ``model_name_or_path`` when omitted.
    normalize: bool
        L2-normalise token embeddings before scoring, which makes MaxSim a
        sum of cosine similarities as in ColBERT. Pass ``False`` for an
        unnormalised dot-product MaxSim.
    special_token: str
        Token used by the perturbation helpers, as on the sibling rankers.
    return_cache: bool
        Have :meth:`patch` also return the clean activation cache.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer=None,
        normalize: bool = True,
        special_token: str = "a",
        return_cache: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name_or_path)
            if tokenizer is None
            else tokenizer
        )
        self.special_token = special_token
        self._device = utils.get_device()
        self.model_name_or_path = get_official_model_name(model_name_or_path)

        self._model, self.__hf_model, self._is_decoder = load_backbone(
            model_name_or_path, self.model_name_or_path, self._device
        )

        self._normalize = normalize
        self._return_cache = return_cache

        self.setup()

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
        token_type_ids: Float[torch.Tensor, "batch seq"] = None,
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Token embeddings for one sequence batch -- no pooling.

        Padding positions are left in place; they are excluded at scoring
        time, where the masks are in scope.
        """
        if self._is_decoder:
            # As in Dot: HookedTransformer has no return_type="embeddings",
            # so stop after the last block and apply the final norm by hand
            # to match the source model's last_hidden_state.
            resid = self._model(
                input_ids,
                attention_mask=attention_mask,
                stop_at_layer=self._model.cfg.n_layers,
            )
            model_output = self._model.ln_final(resid)
        else:
            model_output = self._model(
                input=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_type="embeddings",
            )
        if self._normalize:
            return F.normalize(model_output, p=2, dim=-1)
        return model_output

    def _patch_score(
        self,
        patched_output: Float[torch.Tensor, "batch d_len d_model"],
        reps_q: Float[torch.Tensor, "batch q_len d_model"] = None,
        mask_q: Float[torch.Tensor, "batch q_len"] = None,
        mask_d: Float[torch.Tensor, "batch d_len"] = None,
        **context,
    ) -> torch.Tensor:
        """Score a patched document token matrix against the query tokens.

        The per-family scoring seam of
        :class:`~mechir.modelling.patched.PatchedMixin`. ``forward`` returns
        token embeddings, so -- as for :class:`~mechir.Dot` -- the relevance
        score only exists once they meet the query side; unlike ``Dot`` the
        contraction is MaxSim over tokens rather than a dot product over one
        pooled vector.
        """
        if reps_q is None:
            raise TypeError(
                "LateInteraction patching needs reps_q, the query token "
                "embeddings. Call patch(), which supplies them."
            )
        return maxsim(reps_q, patched_output, mask_q, mask_d)

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        cache_as_dict: bool = False,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch seq d_model"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """``HookedRootModule.run_with_cache``, wrapping the cache as needed.

        Mirrors :meth:`mechir.Dot.run_with_cache`: an
        :class:`~transformer_lens.ActivationCache.ActivationCache` by
        default, so the patching drivers can index it by hook name.
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

    @torch.no_grad()
    def score(
        self,
        queries: dict,
        documents: dict,
        reps_q=None,
        cache=False,
        cache_as_dict=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[ActivationCache]]:
        """MaxSim scores for a batch of query/document pairs.

        Returns ``(scores, reps_q, reps_d, cache_d)``, as
        :meth:`mechir.Dot.score` does; ``cache_d`` is ``None`` unless
        ``cache`` is set.
        """
        if reps_q is None:
            reps_q = self.forward(queries["input_ids"], queries["attention_mask"])
        mask_q = queries["attention_mask"]
        mask_d = documents["attention_mask"]

        if cache:
            reps_d, cache_d = self.run_with_cache(
                documents["input_ids"],
                documents["attention_mask"],
                cache_as_dict=cache_as_dict,
            )
            return maxsim(reps_q, reps_d, mask_q, mask_d), reps_q, reps_d, cache_d
        reps_d = self.forward(documents["input_ids"], documents["attention_mask"])
        return maxsim(reps_q, reps_d, mask_q, mask_d), reps_q, reps_d, None

    @torch.no_grad()
    def patch(
        self,
        queries: dict,
        documents: dict,
        documents_p: dict = None,
        patch_type: str = "block_all",
        layer_head_list: list = [],
        patching_metric: Callable = linear_rank_function,
        perturbed_documents: dict = None,
    ):
        """Activation-patch the document side and score every patched run.

        ``DotDataCollator`` emits "perturbed_documents", so that name is
        accepted too -- see :meth:`mechir.Dot.patch`.
        """
        documents_p = documents_p if documents_p is not None else perturbed_documents
        if documents_p is None:
            raise TypeError(
                "patch() needs documents_p (or perturbed_documents, the key "
                "DotDataCollator emits)"
            )
        assert patch_type in self._patch_funcs, (
            f"Patch type {patch_type} not recognized. "
            f"Choose from {self._patch_funcs.keys()}"
        )
        scores, reps_q, _, _ = self.score(queries, documents)
        scores_p, _, _, cache_d = self.score(
            queries, documents_p, cache=True, reps_q=reps_q
        )

        patching_kwargs = {
            "corrupted_tokens": documents,
            "clean_cache": cache_d,
            "reps_q": reps_q,
            # The masks ride along to _patch_score: MaxSim over a padded
            # token matrix needs them, and only `patch` has them in scope.
            "mask_q": queries["attention_mask"],
            "mask_d": documents["attention_mask"],
            "patching_metric": patching_metric,
            "layer_head_list": layer_head_list,
            "scores": scores,
            "scores_p": scores_p,
        }

        patched_output = self._patch_funcs[patch_type](**patching_kwargs)
        if self._return_cache:
            return patched_output, cache_d
        return patched_output, None


__all__ = ["MASKED_SIMILARITY", "LateInteraction", "maxsim"]
