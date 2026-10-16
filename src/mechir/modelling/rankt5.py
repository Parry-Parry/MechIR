"""RankT5: a T5 reranker scored by one relevance logit.

MonoT5 and RankT5 share an architecture -- an encoder-decoder run for a
single decoder step -- and differ only in how that step becomes a score.
MonoT5 contrasts the ``true`` and ``false`` vocabulary logits. RankT5 is
trained with a ranking loss directly on the logit of one otherwise unused
token (``<extra_id_10>`` in the released checkpoints) and reads that logit
straight off the decoder as a regression-style relevance score, with no
contrast and no normalisation.

That scoring rule is the whole of this module. The patching drivers come
from :class:`~mechir.modelling.patched.PatchedMixin`, and the T5 plumbing --
hooked backbone, single-step decoder, encoder-stack hook naming, ``score``
and ``patch`` -- comes from :class:`~mechir.modelling.t5.MonoT5`, whose
architecture RankT5 shares exactly.
"""

import logging

import torch
from jaxtyping import Float

from mechir.modelling.hooked.loading_from_pretrained import add_official_model
from mechir.modelling.t5 import MonoT5

logger = logging.getLogger(__name__)

#: The unused vocabulary token RankT5 is trained to read its score from.
#: T5's extra ids run backwards from the end of the vocabulary, so this is
#: id 32089 for a 32100-token T5 tokenizer.
RELEVANCE_TOKEN = "<extra_id_10>"

#: Released RankT5 checkpoints. These are unmodified T5 architectures, so
#: TransformerLens's existing T5 conversion handles them and only the names
#: need registering -- without that, its own ``from_pretrained`` rejects the
#: name and ``RankT5`` cannot be constructed at all, exactly as was true of
#: the MonoT5 checkpoints registered in
#: :mod:`mechir.modelling.hooked.loading_from_pretrained`.
CHECKPOINTS = (
    "Soyoung97/RankT5-base",
    "Soyoung97/RankT5-large",
    "Soyoung97/RankT5-3b",
)

for _checkpoint in CHECKPOINTS:
    add_official_model(_checkpoint)


class RankT5(MonoT5):
    """Encoder-decoder reranker scored by a single decoder logit.

    Parameters
    ----------
    model_name_or_path: str
        A RankT5 checkpoint, e.g. ``Soyoung97/RankT5-base``.
    relevance_token: str
        The token whose logit is the relevance score. Must encode to exactly
        one vocabulary id. Defaults to :data:`RELEVANCE_TOKEN`.
    special_token: str
        Passed through to :class:`~mechir.modelling.t5.MonoT5`.
    return_cache: bool
        Return the clean activation cache alongside a patching result.

    Notes
    -----
    There is no ``softmax_output``: the score is a regression output, and a
    softmax over one logit is identically 1.
    """

    def __init__(
        self,
        model_name_or_path: str,
        relevance_token: str = RELEVANCE_TOKEN,
        special_token: str = "X",
        return_cache: bool = False,
    ) -> None:
        # MonoT5's label validation is reused by pointing both of its labels
        # at the one relevance token, so `pos_token == neg_token ==
        # relevance_token` rather than leaving MonoT5's `true`/`false` ids
        # lying around unused. RankT5 never contrasts them.
        super().__init__(
            model_name_or_path,
            pos_token=relevance_token,
            neg_token=relevance_token,
            special_token=special_token,
            softmax_output=False,
            return_cache=return_cache,
        )
        self.relevance_token = self.pos_token

    def _relevance(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        # noqa below: flake8 reads a single-token jaxtyping shape as a
        # forward reference and reports F821, as in `generative.py`.
    ) -> Float[torch.Tensor, "batch"]:  # noqa: F821
        """Read one relevance score per item off the first decoder position.

        The per-family scoring rule. Where MonoT5 contrasts two vocabulary
        logits, RankT5 was trained with a ranking loss on this single logit,
        so the score is the logit itself.
        """
        return logits[:, 0, self.relevance_token]

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
        # noqa below: see `_relevance`.
    ) -> Float[torch.Tensor, "batch"]:  # noqa: F821
        """Score a batch of tokenised query/document pairs."""
        logits = self._model(
            input=input_ids,
            one_zero_attention_mask=attention_mask,
            decoder_input=self._decoder_start(input_ids),
            return_type="logits",
        )
        return self._relevance(logits)

    # `forward` already reduces the decoder logits to one score per item, so
    # the inherited `PatchedMixin._patch_score` (the identity) is correct and
    # every patching method is inherited unchanged -- as is `_hook_names`,
    # which MonoT5 already points at the encoder stack of the shared
    # encoder-decoder architecture.


__all__ = ["RELEVANCE_TOKEN", "RankT5"]
