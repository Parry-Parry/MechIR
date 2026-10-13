"""Reader-side metrics.

Patching a ranker only means something if there is a reader-side quantity to
watch. ``linear_rank_function`` cannot serve here: the pipeline's output is a
generation, not a score, so these replace it at the boundary.
"""

from typing import Dict, List, Optional

import torch

from mechir.pipeline.context import Context


@torch.no_grad()
def answer_logprob(
    reader,
    context: Context,
    reduction: str = "sum",
) -> torch.Tensor:
    """Log-probability the reader assigns to the answer tokens.

    The primary pipeline metric: it responds to anything that changes the
    retrieved evidence, and it is comparable across permutations because the
    answer tokens are fixed.

    Parameters
    ----------
    reduction:
        ``"sum"`` for total answer log-probability, ``"mean"`` for
        per-token, which is comparable across answers of different lengths.
    """
    if context.answer_start is None:
        raise ValueError("context was built without an answer")

    logits = reader(context.input_ids, return_type="logits")
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)

    # Position t predicts token t+1, so the logits that score the answer
    # start one before answer_start.
    targets = context.input_ids[0, context.answer_start :]
    predicting = logprobs[context.answer_start - 1 : -1]
    if predicting.shape[0] == 0:
        raise ValueError("context has no answer tokens to score")

    token_logprobs = predicting.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum() if reduction == "sum" else token_logprobs.mean()


@torch.no_grad()
def attention_to_documents(
    reader,
    context: Context,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    from_positions: Optional[List[int]] = None,
    normalize: str = "mean",
) -> Dict[int, float]:
    """Attention from the answer positions onto each document span.

    Keyed by rank. This is the measurement behind ranker/reader
    disagreement: compare it against the ranker's score ordering and see
    where monotonicity breaks.

    Two confounds matter, and ignoring them produces results that look like
    findings but are not:

    * **Span length.** Summed mass grows with the number of tokens, so a
      long document outscores a short one for free. ``normalize="mean"``
      (the default) divides by span length; ``normalize="sum"`` gives raw
      mass when total budget is what you want.
    * **Position.** In a causal reader, later spans sit nearer the answer
      positions and attract more attention regardless of content. There is
      no normalisation that removes this. Control for it instead by holding
      the candidate set fixed and comparing the *same* document across
      ranks, which is what ``HardPermutation`` and
      ``Pipeline.position_sensitivity`` exist to make cheap. Comparing raw
      attention across different ranks in a single ordering mostly measures
      recency.

    Parameters
    ----------
    layer, head:
        Restrict to one layer and/or head. Both ``None`` averages over
        everything, which is the usual starting point.
    from_positions:
        Query positions to read attention from. Defaults to the answer
        positions, or the final position when no answer is present.
    normalize:
        ``"mean"`` for attention per document token, ``"sum"`` for raw mass.
    """
    if normalize not in {"mean", "sum"}:
        raise ValueError(f"normalize must be 'mean' or 'sum', got {normalize!r}")
    names = [f"blocks.{i}.attn.hook_pattern" for i in range(reader.cfg.n_layers)]
    if layer is not None:
        names = [f"blocks.{layer}.attn.hook_pattern"]

    wanted = set(names)
    _, cache = reader.run_with_cache(
        context.input_ids, names_filter=lambda n: n in wanted
    )

    if from_positions is None:
        if context.answer_start is not None:
            from_positions = list(range(context.answer_start - 1, context.length - 1))
        else:
            from_positions = [context.length - 1]

    # Stack to [layer, head, query_pos, key_pos] then reduce.
    patterns = torch.stack([cache[name][0] for name in names])
    if head is not None:
        patterns = patterns[:, head : head + 1]
    attention = patterns[:, :, from_positions, :].mean(dim=(0, 1, 2))

    per_rank = {}
    for span in context.spans:
        mass = attention[span.slice()].sum().item()
        per_rank[span.rank] = mass / max(len(span), 1) if normalize == "mean" else mass
    return per_rank


@torch.no_grad()
def groundedness(
    reader,
    context: Context,
    layer: Optional[int] = None,
) -> float:
    """Share of answer-position attention landing on retrieved documents.

    A coarse causal alternative to the string-overlap and NLI proxies RAG
    evaluation usually relies on. Near 0 means the answer is being produced
    without consulting the evidence.

    This measures attention, not a causal effect: a definitive version
    ablates the document spans and watches ``answer_logprob`` move. Treat
    this as the cheap screen.
    """
    per_rank = attention_to_documents(reader, context, layer=layer, normalize="sum")
    return float(sum(per_rank.values()))


__all__ = ["answer_logprob", "attention_to_documents", "groundedness"]
