"""Causal ablation of retrieved documents.

``attention_to_documents`` says where the reader looked. It does not say
whether looking mattered -- attention is correlational, and reading it as
explanation is the standard criticism of attention-based faithfulness
claims. These functions intervene instead: remove a document's causal
influence and measure how far ``answer_logprob`` moves.

Two ablation styles live here, and they are not interchangeable.

*Masking* (``ablated_metric`` and friends) zeroes attention onto a span's
key positions and renormalises the remaining distribution, so the reader
computes as though the document were absent without the sequence shifting.
Deleting tokens instead would move every later position and confound the
result with position change -- which, given how strongly readers respond to
position, would dominate the measurement.

*Resampling* (``resample_metric`` and friends) replaces the document with
other real text of the same token length. It separates "the answer needed
this document" from "the answer needed something in that slot", and keeps
the reader on-distribution rather than showing it a hole.

Which style you choose changes the per-document conclusion, so it is worth
choosing deliberately. Measured on one query with pythia-70m, effects in
nats:

    document              mask     resample(rotation)  resample(filler)
    Paris/France (gold)  +2.7311          +1.6418           +2.9850
    Berlin/Germany       +0.0010          +0.3952           +0.3188
    Eiffel Tower/Paris   -0.0487          -0.1806           +0.5773
    Bananas (unrelated)  -0.7027          -2.2157           +0.0287

The gold document agrees across mask and neutral filler, which is the
robustness check passing. The unrelated document does not: masking made it
look actively harmful (-0.70) and rotation made that worse (-2.22), but
against neutral filler its effect is ~0, which is the honest answer --
both of the others were artifacts of what replaced it. Prefer
neutral, corpus-sampled replacements for anything you intend to report.
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


def _match_token_length(tokenizer, text: str, target_tokens: int) -> str:
    """Truncate ``text`` to ``target_tokens`` tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_tokens:
        return text
    return tokenizer.decode(ids[:target_tokens])


@torch.no_grad()
def resample_metric(
    reader,
    builder,
    query: str,
    documents: Sequence[str],
    order: Sequence[int],
    rank: int,
    replacement: str,
    answer: str,
    metric: Callable[..., torch.Tensor] = answer_logprob,
    match_length: bool = True,
    device=None,
) -> dict:
    """``metric`` with the document at ``rank`` swapped for ``replacement``.

    Masking removes a document; resampling *replaces* it with other real
    text. That distinguishes "the answer needed this document" from "the
    answer needed something in this slot", which masking alone cannot do,
    and keeps the reader on-distribution rather than showing it a hole.

    ``match_length`` truncates the replacement to the original's token
    count so later documents do not shift position. The realised shift is
    returned as ``length_delta`` -- it is rarely exactly zero, because
    decoding a truncated id sequence and re-encoding it need not round-trip,
    and a non-zero value means some of the effect is positional.
    """
    candidate = list(order)[rank]
    original = documents[candidate]

    if match_length:
        target = builder.document_token_length(original)
        replacement = _match_token_length(builder.tokenizer, replacement, target)

    swapped = list(documents)
    swapped[candidate] = replacement

    full_context = builder.build(query, documents, order, answer=answer, device=device)
    resampled_context = builder.build(
        query, swapped, order, answer=answer, device=device
    )

    full = float(metric(reader, full_context))
    resampled = float(metric(reader, resampled_context))

    return {
        "full": full,
        "resampled": resampled,
        "effect": full - resampled,
        "length_delta": resampled_context.length - full_context.length,
    }


@torch.no_grad()
def resample_necessity(
    reader,
    builder,
    query: str,
    documents: Sequence[str],
    order: Sequence[int],
    answer: str,
    replacements: Optional[Sequence[str]] = None,
    metric: Callable[..., torch.Tensor] = answer_logprob,
    match_length: bool = True,
    device=None,
) -> Dict[int, dict]:
    """Resample-ablation effect per document, keyed by rank.

    The robustness counterpart to ``document_necessity``. An effect that
    survives replacement is about the document's content; one that only
    appears under masking may be about the slot being occupied at all.

    Parameters
    ----------
    replacements:
        One substitute per rank. Defaults to rotating the candidate set, so
        each document is replaced by another real document from the same
        pool -- on-distribution and needing no external corpus.

        That default is convenient but **confounded whenever the pool
        contains relevant documents**, because it conflates removing the
        document under test with adding a second copy of whatever replaces
        it. Measured on a four-document example, rotation made an
        irrelevant document look far more harmful than masking did
        (-2.2157 against -0.7027) purely because its replacement was the
        gold document. For results you intend to report, pass replacements
        sampled from the corpus at large, where that cannot happen.
    """
    order = list(order)
    if replacements is None:
        replacements = [
            documents[order[(i + 1) % len(order)]] for i in range(len(order))
        ]
    if len(replacements) != len(order):
        raise ValueError(f"expected {len(order)} replacements, got {len(replacements)}")

    return {
        rank: resample_metric(
            reader,
            builder,
            query,
            documents,
            order,
            rank,
            replacements[rank],
            answer,
            metric=metric,
            match_length=match_length,
            device=device,
        )
        for rank in range(len(order))
    }


__all__ = [
    "ablated_metric",
    "causal_groundedness",
    "document_necessity",
    "document_sufficiency",
    "resample_metric",
    "resample_necessity",
]
