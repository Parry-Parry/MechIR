"""SPLADE: sparse lexical expansion over a masked-language-model head.

SPLADE scores a (query, document) pair with no learned projection and no
pooled embedding. It reads the encoder through its MLM head, turns the
vocabulary logits at every position into a non-negative term weight with
``log(1 + relu(x))``, max-pools those over the sequence, and takes the dot
product of the query's and the document's resulting vectors.

The vectors are held densely, ``[batch, d_vocab]``. Sparsity in SPLADE is a
property of the *representation* -- ``relu`` zeroes every non-positive logit
and ``log1p(0) == 0`` keeps it zero, so most entries are exactly zero -- not
of the storage. A dense tensor is therefore the right in-memory form, and the
usual inverted-index machinery only enters at retrieval time, which is not
what this module is for.

Architecturally this is the :class:`~mechir.Dot` story with a different
contraction space: one encoder tower, two texts through it, a dot product at
the end. The hook names are the single-stack ``blocks.{layer}.…`` default, and
``_patch_score`` contracts the patched document vector against the query's --
over ``d_vocab`` rather than ``d_model``.

Reference checkpoint: :data:`REFERENCE_MODEL`.
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import transformer_lens.utils as utils
from jaxtyping import Float
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mechir.modelling.architectures import HookedEncoder
from mechir.modelling.hooked.loading_from_pretrained import get_official_model_name
from mechir.modelling.patched import PatchedMixin
from mechir.util import batched_dot_product, linear_rank_function

logger = logging.getLogger(__name__)

#: The SPLADE checkpoint this module was developed against.
REFERENCE_MODEL = "naver/splade-cocondenser-ensembledistil"


def splade_activation(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
) -> Float[torch.Tensor, "batch seq d_vocab"]:
    """``log(1 + relu(x))``: SPLADE's sparsifying term-weight activation.

    ``relu`` sends every non-positive logit to zero and ``log1p`` keeps it
    there, so the output is exactly -- not approximately -- sparse. The
    surviving entries are the vocabulary terms the MLM head actually
    predicts for the text, which is what makes this an expansion rather
    than a bag of its own tokens.

    ``log1p`` rather than ``log(1 + x)`` for accuracy on small weights.

    Parameters
    ----------
    logits: torch.Tensor
        MLM head logits, ``[batch, seq, d_vocab]``.
    """
    return torch.log1p(torch.relu(logits))


def splade_pool(
    activations: Float[torch.Tensor, "batch seq d_vocab"],
    attention_mask: Optional[Float[torch.Tensor, "batch seq"]] = None,
) -> Float[torch.Tensor, "batch d_vocab"]:
    """Max over sequence positions, ignoring padding.

    Max-pooling, not mean or sum: a term's weight is its weight at the one
    position that argued for it hardest, so a term predicted strongly once
    is not diluted by the rest of the sequence.

    Padding has to be masked out, for the reason set out in
    :func:`mechir.modelling.backbones._pool_mean`: an unmasked reduction
    makes the score depend on how far the batch happened to be padded, so
    the same document scores differently at ``max_length=180`` and
    ``max_length=220``. Padding positions carry real MLM logits -- the
    encoder masks them out of *attention*, not out of its output -- so
    their term weights are generally non-zero and would win the max.

    Masked positions are zeroed rather than filled with ``-inf`` because
    the activation is non-negative: zero is a true floor, and an
    all-padding row still pools to a finite zero vector rather than
    ``-inf``.

    The mask is optional only so that unpadded single sequences still
    work; omitting it when the batch *is* padded reintroduces the bug.

    Parameters
    ----------
    activations: torch.Tensor
        Term weights from :func:`splade_activation`, ``[batch, seq,
        d_vocab]``.
    attention_mask: torch.Tensor, optional
        ``1`` for real tokens, ``0`` for padding, ``[batch, seq]``.
    """
    if attention_mask is not None:
        # Device as well as dtype: the model moves its own copy of the
        # inputs to cfg.device, but the mask reaching the pooler is the
        # caller's, so on MPS or CUDA the activations are on the
        # accelerator while the mask is still on the CPU.
        mask = attention_mask.unsqueeze(-1).to(
            device=activations.device, dtype=activations.dtype
        )
        activations = activations * mask
    return activations.amax(dim=1)


def _mlm_head_state(hf_model) -> Dict[str, torch.Tensor]:
    """The hooked-encoder state dict entries for a BERT MLM head.

    The hooked BERT path does not bring the MLM head with it. mechir
    resolves encoder checkpoints through ``AutoModel``, and the registered
    weight conversions only read ``cls.predictions`` when the source model
    has it, so ``BertMLMHead``'s parameters are left as
    ``torch.empty``. SPLADE *is* the MLM head, so :class:`Splade` loads it
    itself from the ``AutoModelForMaskedLM`` it already holds.

    Raises
    ------
    ValueError
        If ``hf_model`` has no BERT-style ``cls.predictions`` head.
    """
    predictions = getattr(getattr(hf_model, "cls", None), "predictions", None)
    if predictions is None:
        raise ValueError(
            f"{type(hf_model).__name__} exposes no BERT-style MLM head at "
            "`cls.predictions`. Splade scores through that head, so it needs "
            "a BERT-family *ForMaskedLM checkpoint -- for example "
            f"{REFERENCE_MODEL!r}."
        )
    transform = predictions.transform
    return {
        # BertMLMHead computes `resid @ W` while nn.Linear computes
        # `x @ weight.T`, so the weight transposes on the way in. Loading
        # it untransposed is silent -- the shape is square -- and gives
        # logits tens of units away from the source model's.
        "mlm_head.W": transform.dense.weight.T,
        "mlm_head.b": transform.dense.bias,
        "mlm_head.ln.w": transform.LayerNorm.weight,
        "mlm_head.ln.b": transform.LayerNorm.bias,
        # Likewise Unembed, which holds (d_model, d_vocab). Read off the
        # decoder rather than the word embeddings so that a checkpoint
        # with an untied head still loads correctly.
        "unembed.W_U": predictions.decoder.weight.T,
        "unembed.b_U": predictions.bias,
    }


def _load_mlm_head(hooked: HookedEncoder, hf_model) -> None:
    """Copy ``hf_model``'s MLM head into ``hooked`` in place."""
    state = _mlm_head_state(hf_model)
    _, unexpected = hooked.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(
            f"{type(hooked).__name__} rejected MLM head keys {sorted(unexpected)}; "
            "it does not have the BertMLMHead/Unembed layout Splade expects."
        )


class Splade(HookedRootModule, PatchedMixin):
    """A hooked SPLADE sparse lexical ranker.

    ``forward`` returns one sparse vocabulary vector per text rather than a
    score: :func:`splade_activation` over the MLM head's logits, then
    :func:`splade_pool` over the sequence. A relevance score only exists
    once a query vector and a document vector are contracted, which
    ``score`` and ``_patch_score`` do.

    Parameters
    ----------
    model_name_or_path: str
        A BERT-family ``*ForMaskedLM`` checkpoint, e.g.
        :data:`REFERENCE_MODEL`.
    tokenizer: optional
        Tokenizer to use; loaded from the checkpoint when omitted.
    special_token: str
        Placeholder token used by the perturbation utilities, kept for
        signature parity with the other rankers.
    return_cache: bool
        Whether :meth:`patch` also returns the clean activation cache.

    Examples
    --------
    >>> model = Splade(REFERENCE_MODEL)  # doctest: +SKIP
    >>> scores, reps_q, reps_d, _ = model.score(queries, documents)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer=None,
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

        self._hf_model = (
            AutoModelForMaskedLM.from_pretrained(model_name_or_path)
            .eval()
            .to(self._device)
        )

        # The base encoder, not the full MaskedLM model: the registered
        # weight conversions read the transformer blocks off a BertModel
        # layout, which is what every encoder checkpoint resolves to here.
        # The head that conversion therefore skips is loaded below.
        self._model = HookedEncoder.from_pretrained(
            self.model_name_or_path,
            hf_model=self._hf_model.base_model,
            device=self._device,
        )
        _load_mlm_head(self._model, self._hf_model)

        self._return_cache = return_cache

        self.setup()

    @property
    def d_vocab(self) -> int:
        """Width of the sparse representation."""
        return self._model.cfg.d_vocab_out

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
        token_type_ids: Float[torch.Tensor, "batch seq"] = None,
    ) -> Float[torch.Tensor, "batch d_vocab"]:
        """The sparse vocabulary vector of every sequence in the batch."""
        logits = self._model(
            input=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_type="logits",
        )
        return splade_pool(splade_activation(logits), attention_mask)

    def _patch_score(
        self,
        patched_output: Float[torch.Tensor, "batch d_vocab"],
        reps_q: Float[torch.Tensor, "batch d_vocab"] = None,
        **context,
    ) -> torch.Tensor:
        """Score patched document vectors against the query vectors.

        The per-family scoring seam of
        :class:`~mechir.modelling.patched.PatchedMixin`. ``forward`` ran
        under the patching hooks, so the patched logits have already been
        through :func:`splade_activation` and :func:`splade_pool`; what is
        left is the contraction against the query side, exactly as for
        :class:`~mechir.Dot` but over ``d_vocab``.
        """
        if reps_q is None:
            raise TypeError(
                "Splade patching needs reps_q, the sparse query vectors. "
                "Call patch(), which supplies them."
            )
        return batched_dot_product(reps_q, patched_output)

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        cache_as_dict: bool = False,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """``HookedRootModule.run_with_cache``, optionally wrapping the cache.

        Mirrors :meth:`mechir.Dot.run_with_cache`: with
        ``return_cache_object`` the activations come back as an
        :class:`~transformer_lens.ActivationCache`, which is what the
        patching drivers index.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            if not cache_as_dict:
                cache_dict = ActivationCache(
                    cache_dict, self, has_batch_dim=not remove_batch_dim
                )
            return out, cache_dict
        return out, None

    @torch.no_grad()
    def score(
        self,
        queries: dict,
        documents: dict,
        reps_q: Float[torch.Tensor, "batch d_vocab"] = None,
        cache: bool = False,
        cache_as_dict: bool = False,
    ):
        """Score each query against its paired document.

        Returns ``(scores, reps_q, reps_d, cache_d)``; ``cache_d`` is
        ``None`` unless ``cache`` is set.
        """
        if reps_q is None:
            reps_q = self.forward(queries["input_ids"], queries["attention_mask"])
        if cache:
            reps_d, cache_d = self.run_with_cache(
                documents["input_ids"],
                documents["attention_mask"],
                cache_as_dict=cache_as_dict,
            )
            return batched_dot_product(reps_q, reps_d), reps_q, reps_d, cache_d
        reps_d = self.forward(documents["input_ids"], documents["attention_mask"])
        return batched_dot_product(reps_q, reps_d), reps_q, reps_d, None

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
        """Patch ``documents`` with activations cached from ``documents_p``.

        ``perturbed_documents`` is accepted as an alias for
        ``documents_p``, which is the key ``DotDataCollator`` emits, so the
        documented ``model.patch(**batch)`` workflow works unchanged.
        """
        documents_p = documents_p if documents_p is not None else perturbed_documents
        if documents_p is None:
            raise TypeError(
                "patch() needs documents_p (or perturbed_documents, the key "
                "DotDataCollator emits)"
            )
        assert patch_type in self._patch_funcs, (
            f"Patch type {patch_type} not recognized. "
            f"Choose from {sorted(self._patch_funcs)}"
        )
        scores, reps_q, _, _ = self.score(queries, documents)
        scores_p, _, _, cache_d = self.score(
            queries, documents_p, cache=True, reps_q=reps_q
        )

        patching_kwargs = {
            "corrupted_tokens": documents,
            "clean_cache": cache_d,
            "reps_q": reps_q,
            "patching_metric": patching_metric,
            "layer_head_list": layer_head_list,
            "scores": scores,
            "scores_p": scores_p,
        }

        patched_output = self._patch_funcs[patch_type](**patching_kwargs)
        if self._return_cache:
            return patched_output, cache_d
        return patched_output, None


__all__ = ["REFERENCE_MODEL", "Splade", "splade_activation", "splade_pool"]
