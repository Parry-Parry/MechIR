"""Causal ablation of retrieved documents.

``attention_to_documents`` says where the reader looked. It does not say
whether looking mattered -- attention is correlational, and reading it as
explanation is the standard criticism of attention-based faithfulness
claims. These functions intervene instead: remove a document's causal
influence and measure how far ``answer_logprob`` moves.

The ablation zeroes attention onto a span's key positions and renormalises
the remaining distribution, so the reader computes as though the document
were absent *without* the sequence shifting. Deleting tokens instead would
move every later position and confound the result with position change --
which, given how strongly this reader responds to position, would dominate
the measurement.
"""

from typing import Callable, Dict, List, Optional, Sequence

import torch

from mechir.pipeline.context import Context
from mechir.pipeline.metrics import answer_logprob

EPS = 1e-9


def _pattern_names(reader, layer: Optional[int] = None) -> List[str]:
    if layer is not None:
        return [f"blocks.{layer}.attn.hook_pattern"]
    return [f"blocks.{i}.attn.hook_pattern" for i in range(reader.cfg.n_layers)]


def _mask_keys_hook(key_positions: torch.Tensor, renormalise: bool = True):
    """Hook that removes ``key_positions`` from every attention distribution."""

    def hook(pattern, hook):  # pattern: [batch, head, query, key]
        original = pattern
        pattern = pattern.clone()
        pattern[..., key_positions] = 0.0

        if not renormalise:
            return pattern

        total = pattern.sum(dim=-1, keepdim=True)
        # A row whose entire mass sat on the ablated span cannot be
        # renormalised; leaving it untouched is the conservative choice.
        safe = total > EPS
        return torch.where(safe, pattern / total.clamp(min=EPS), original)

    return hook


@torch.no_grad()
def ablated_metric(
    reader,
    context: Context,
    ranks: Sequence[int],
    metric: Callable[..., torch.Tensor] = answer_logprob,
    layer: Optional[int] = None,
    renormalise: bool = True,
) -> float:
    """``metric`` with the documents at ``ranks`` made invisible.

    Parameters
    ----------
    ranks:
        Which document ranks to ablate. Empty means no ablation, which is a
        useful control: it must reproduce the unablated value exactly.
    layer:
        Ablate in one layer only. ``None`` ablates in every layer, which is
        what "the reader cannot see this document" means.
    renormalise:
        Rescale the surviving attention so each distribution still sums to
        1. Leaving it off shrinks total attention and takes the reader
        off-distribution.
    """
    if not ranks:
        return float(metric(reader, context))

    positions: List[int] = []
    for rank in ranks:
        span = context.span_for_rank(rank)
        positions.extend(range(span.start, span.end))

    key_positions = torch.tensor(
        sorted(set(positions)), dtype=torch.long, device=context.input_ids.device
    )
    hook = _mask_keys_hook(key_positions, renormalise=renormalise)
    names = set(_pattern_names(reader, layer))

    with reader.hooks(fwd_hooks=[(name, hook) for name in names]):
        return float(metric(reader, context))


@torch.no_grad()
def causal_groundedness(
    reader,
    context: Context,
    metric: Callable[..., torch.Tensor] = answer_logprob,
) -> dict:
    """How much the answer actually depends on the retrieved documents.

    Ablates every document at once. ``effect`` is the drop in the metric, in
    nats, so a large positive value means the evidence is load-bearing and a
    value near zero means the answer is coming from the reader's weights
    regardless of what was retrieved.

    This is the causal counterpart of ``metrics.groundedness``, which only
    reports attention mass.
    """
    full = float(metric(reader, context))
    ablated = ablated_metric(
        reader, context, [span.rank for span in context.spans], metric=metric
    )
    return {
        "full": full,
        "ablated": ablated,
        "effect": full - ablated,
    }


@torch.no_grad()
def document_necessity(
    reader,
    context: Context,
    metric: Callable[..., torch.Tensor] = answer_logprob,
) -> Dict[int, float]:
    """Leave-one-out effect per document, keyed by rank.

    Each value is the metric drop caused by removing that document alone.
    Positive means the document was helping. Negative means it was hurting,
    which is how a distractor shows up.
    """
    full = float(metric(reader, context))
    return {
        span.rank: full - ablated_metric(reader, context, [span.rank], metric=metric)
        for span in context.spans
    }


@torch.no_grad()
def document_sufficiency(
    reader,
    context: Context,
    metric: Callable[..., torch.Tensor] = answer_logprob,
) -> Dict[int, float]:
    """Keep-one-only effect per document, keyed by rank.

    Each value is the metric with every *other* document ablated, minus the
    metric with all of them ablated. It asks what one document supplies on
    its own, which can differ sharply from its leave-one-out effect when
    documents are redundant: three documents that each state the answer all
    have near-zero necessity and high sufficiency.
    """
    all_ranks = [span.rank for span in context.spans]
    nothing = ablated_metric(reader, context, all_ranks, metric=metric)

    return {
        rank: ablated_metric(
            reader, context, [r for r in all_ranks if r != rank], metric=metric
        )
        - nothing
        for rank in all_ranks
    }


__all__ = [
    "ablated_metric",
    "causal_groundedness",
    "document_necessity",
    "document_sufficiency",
]
