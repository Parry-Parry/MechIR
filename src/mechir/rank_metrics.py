"""Patching metrics for permutation-valued ranker outputs.

A listwise ranker (RankZephyr/RankGPT style) does not emit a score, it emits
an *ordering*. :func:`mechir.util.linear_rank_function` therefore has nothing
to divide: there is no scalar to take a difference of. These are its
permutation-valued counterparts, plus a parser that turns the text a listwise
LLM actually produces into an ordering.

Direction convention
--------------------
**Every metric in this module reports amount of change, not similarity.**

* ``0.0`` means "the patch changed nothing about the ordering".
* ``1.0`` means "maximally changed" -- reversed, or disjoint top-k, or a
  different first document.
* Every metric is bounded in ``[0.0, 1.0]``.

This matches :func:`~mechir.util.linear_rank_function`, which reports 0 for a
component that recovers nothing. A patch map of zeros therefore reads the same
way whichever metric produced it, and the three metrics can be compared
head-to-head without a sign flip silently inverting the conclusion.

The underlying correlations (:func:`kendall_tau`, :func:`spearman_rho`) are
*similarity*-valued in ``[-1, 1]``, as their definitions require. They are
exposed because they are useful on their own, but they are deliberately named
differently from the metrics and are **not** the pluggable layer. Only the
``*_distance`` / ``*_flip`` functions are metrics.

Representation
--------------
An ordering is a sequence of candidate identifiers in rank order, so
``[3, 1, 2]`` means "candidate 3 is ranked first". This is the same
convention as :data:`mechir.pipeline.boundary.PermutationOp` ("orders are
lists of indices into the candidate list"). Entries are compared by equality;
in practice they are integer indices.

Plugging into a patching loop
-----------------------------
:class:`~mechir.modelling.patched.PatchedMixin` calls its metric as
``patching_metric(score, scores, scores_p).mean()`` -- that is,
``(patched, clean, perturbed)``, returning one entry per batch item. The
metrics here are two-argument and ``(clean, patched)``-ordered, which reads
better on its own, so :func:`batched_rank_metric` adapts one into that
calling convention, swapping the first two arguments and vectorising over the
batch::

    from mechir.rank_metrics import batched_rank_metric, get_rank_metric

    metric = batched_rank_metric(get_rank_metric("top_k", k=5))
    ranker.get_act_patch_block_every(tokens, cache, metric, clean, perturbed)

The ``perturbed`` argument is accepted and ignored: these metrics compare the
patched ordering against the clean one only. There is no permutation analogue
of ``linear_rank_function``'s recovery fraction here, because a normalised
"fraction of the perturbation's reordering undone" needs a direction on
permutation space that this module does not define.
"""

import math
import re
from collections.abc import Sequence as SequenceABC
from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor

#: A ranking: candidate identifiers in rank order, best first.
Ordering = Union[Sequence[int], Tensor]

#: A batch of orderings: a 2-D tensor, or a sequence of :data:`Ordering`.
OrderingBatch = Union[Sequence[Ordering], Tensor]

#: The pluggable metric interface: ``(clean, patched) -> change in [0, 1]``.
RankMetric = Callable[[Ordering, Ordering], float]

#: The patching-loop interface: ``(patched, clean, perturbed) -> Tensor``.
PatchingMetric = Callable[..., Tensor]

#: Matches a bracketed integer, the only token the parser trusts.
_BRACKETED = re.compile(r"\[\s*(\d+)\s*\]")


class PermutationParseError(ValueError):
    """A listwise ranker's text output could not be read as an ordering.

    Raised only for the genuinely ambiguous cases -- no bracketed indices at
    all, or none of them naming a real candidate -- where guessing would
    silently substitute the identity ordering and corrupt every number
    downstream. Recoverable malformations are repaired instead; see
    :func:`parse_permutation`.
    """


# -- the three metrics -----------------------------------------------------


def kendall_tau_distance(clean: Ordering, patched: Ordering) -> float:
    """Pairwise-discordance distance between two orderings.

    ``(1 - tau) / 2``, so 0.0 for identical orderings, 0.5 for uncorrelated
    and 1.0 for an exact reversal. Dense: every pair of candidates
    contributes, so a single adjacent swap deep in the ranking still moves it.

    Parameters
    ----------
    clean: Ordering
        Ordering from the unpatched run.
    patched: Ordering
        Ordering from the run with a component patched in.

    Returns
    -------
    float
        Change in ``[0.0, 1.0]``. Exactly 0.0 when the orderings are equal
        and when either is degenerate -- see :func:`is_degenerate_ordering`.

    Raises
    ------
    ValueError
        If the orderings differ in length, or rank different candidates.
    """
    return _ordering_distance(clean, patched, _kendall_distance)


def spearman_distance(clean: Ordering, patched: Ordering) -> float:
    """Rank-displacement distance between two orderings.

    ``(1 - rho) / 2``, so 0.0 for identical orderings, 0.5 for uncorrelated
    and 1.0 for an exact reversal. Like :func:`kendall_tau_distance` it is
    dense, but it weights a candidate's *distance* moved rather than the
    number of pairs it crossed, so one document falling a long way registers
    more strongly than several local swaps.

    Parameters
    ----------
    clean: Ordering
        Ordering from the unpatched run.
    patched: Ordering
        Ordering from the run with a component patched in.

    Returns
    -------
    float
        Change in ``[0.0, 1.0]``, with the same exactness and degeneracy
        guarantees as :func:`kendall_tau_distance`.

    Raises
    ------
    ValueError
        If the orderings differ in length, or rank different candidates.
    """
    return _ordering_distance(clean, patched, _spearman_distance)


def top_k_set_distance(clean: Ordering, patched: Ordering, k: int = 10) -> float:
    """Jaccard distance between the top-``k`` sets before and after.

    ``1 - |A n B| / |A u B|`` over the top-``k`` candidates. This is the
    metric that matches how rankers are actually evaluated -- nDCG@10 cannot
    see a document that stays outside the top 10 -- but it is blind in two
    directions: to reordering *within* the top ``k``, and to any movement
    below it. A head that reverses ranks 1 and 2 scores 0.0 here.

    Parameters
    ----------
    clean: Ordering
        Ordering from the unpatched run.
    patched: Ordering
        Ordering from the run with a component patched in.
    k: int
        Cutoff. Clamped to the ordering length, so ``k`` larger than the
        candidate pool compares the whole pool and can only return 0.0.

    Returns
    -------
    float
        Change in ``[0.0, 1.0]``. Exactly 0.0 when the top-``k`` sets match;
        1.0 only when they are disjoint.

    Raises
    ------
    ValueError
        If ``k < 1``, if the orderings differ in length, or if they rank
        different candidates.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return _ordering_distance(clean, patched, partial(_top_k_distance, k=k))


def first_position_flip(clean: Ordering, patched: Ordering) -> float:
    """Whether the top-ranked candidate changed.

    1.0 if the patch put a different candidate first, 0.0 otherwise. The
    cheapest and most interpretable of the three, and the sparsest: it is
    exactly the ``rank_flipped`` signal that ``docs/research-notes.md`` 5
    measured at a mean flip rate of 12-20% over TREC DL, with no signal at
    all on ~40% of queries. Expect a mostly-zero patch map and read it as
    "which components can flip the winner", not as an effect size.

    Parameters
    ----------
    clean: Ordering
        Ordering from the unpatched run.
    patched: Ordering
        Ordering from the run with a component patched in.

    Returns
    -------
    float
        Either 0.0 or 1.0. Degenerate orderings give 0.0.

    Raises
    ------
    ValueError
        If the orderings differ in length, or rank different candidates.
    """
    return _ordering_distance(clean, patched, _first_flip_distance)


# -- selecting one ---------------------------------------------------------

#: The selectable metrics, by the short name a caller or config passes.
RANK_METRICS: Dict[str, Callable[..., float]] = {
    "kendall": kendall_tau_distance,
    "spearman": spearman_distance,
    "top_k": top_k_set_distance,
    "first_flip": first_position_flip,
}


def get_rank_metric(name: str, **kwargs: Any) -> RankMetric:
    """Look up a metric by name, binding any of its parameters.

    Parameters
    ----------
    name: str
        A key of :data:`RANK_METRICS`.
    **kwargs
        Parameters to bind, e.g. ``k=5`` for ``"top_k"``. Validated against
        the metric's signature now rather than failing at call time.

    Returns
    -------
    RankMetric
        A ``(clean, patched) -> float`` callable, ready for
        :func:`batched_rank_metric`.

    Raises
    ------
    KeyError
        If ``name`` is not a known metric.
    TypeError
        If a keyword is not a parameter of the named metric.
    """
    if name not in RANK_METRICS:
        known = ", ".join(sorted(RANK_METRICS))
        raise KeyError(f"Unknown rank metric {name!r}; known metrics: {known}")

    metric = RANK_METRICS[name]
    if not kwargs:
        return metric

    accepted = signature(metric).parameters
    for key in kwargs:
        if key not in accepted:
            raise TypeError(f"{name!r} takes no parameter {key!r}")
    return partial(metric, **kwargs)


def batched_rank_metric(
    metric: RankMetric,
    device: Union[str, torch.device, None] = None,
) -> PatchingMetric:
    """Adapt a :data:`RankMetric` to the patching loop's calling convention.

    The returned callable takes ``(patched, clean, perturbed=None)`` -- the
    argument order :class:`~mechir.modelling.patched.PatchedMixin` uses --
    and returns one float32 entry per batch item, so the loop's ``.mean()``
    reduces it to a scalar. ``perturbed`` is accepted and ignored.

    This is also the batch firewall. Each batch item is scored independently
    and any non-finite result is replaced with 0.0, because NaN propagates
    through the ``.mean()`` in the patching loops and a single dead item
    would otherwise turn every entry of the patch map into NaN -- the exact
    failure ``test/unit/test_rank_metric.py`` exists to prevent.

    Parameters
    ----------
    metric: RankMetric
        A ``(clean, patched) -> float`` callable, e.g. from
        :func:`get_rank_metric`.
    device: str | torch.device | None
        Device for the result. Defaults to CPU; the patching loops assign
        the reduced scalar into their own result tensor regardless.

    Returns
    -------
    PatchingMetric
        A callable matching ``patching_metric`` in
        :mod:`mechir.modelling.patched`.
    """

    def patching_metric(
        patched: OrderingBatch,
        clean: OrderingBatch,
        perturbed: OrderingBatch = None,
        **kwargs: Any,
    ) -> Tensor:
        patched_rows = _as_batch(patched, "patched")
        clean_rows = _as_batch(clean, "clean")
        if len(patched_rows) != len(clean_rows):
            raise ValueError(
                f"batch size mismatch: {len(clean_rows)} clean orderings "
                f"against {len(patched_rows)} patched"
            )

        values = [
            metric(clean_row, patched_row)
            for clean_row, patched_row in zip(clean_rows, patched_rows)
        ]
        out = torch.tensor(values, dtype=torch.float32, device=device)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    name = getattr(metric, "__name__", None) or "rank_metric"
    patching_metric.__name__ = f"batched_{name}"
    patching_metric.__doc__ = (
        f"Batched {name}, as a mechir patching metric. "
        "Called as (patched, clean, perturbed)."
    )
    return patching_metric


# -- correlation primitives ------------------------------------------------


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b between two paired numeric sequences.

    Similarity-valued in ``[-1, 1]``: 1.0 for identical orderings, -1.0 for
    reversed. Tie-corrected, so it accepts score vectors as well as ranks.
    Pairs are taken positionally, so ``x[i]`` and ``y[i]`` must describe the
    same item.

    Parameters
    ----------
    x, y: Sequence[float]
        Equal-length paired observations.

    Returns
    -------
    float
        tau-b, or 0.0 when either input has zero variance -- see Notes.

    Notes
    -----
    tau-b divides by ``sqrt((n0 - n1) * (n0 - n2))``, which is zero whenever
    one input is constant (every value tied) or shorter than two elements.
    The textbook answer is "undefined"; returning NaN here is not an option,
    because NaN survives the ``.mean()`` in the patching loops and poisons a
    whole batch. 0.0 -- no measurable association -- is returned instead.

    Note that 0.0 *correlation* maps to 0.5 *distance*, which is not what a
    degenerate input should contribute to a patch map. The metric layer
    therefore guards degeneracy itself and short-circuits to 0.0 change
    before calling this; see :func:`is_degenerate_ordering`.
    """
    _require_pairable(x, y)
    n = len(x)
    if n < 2:
        return 0.0

    n0 = n * (n - 1) // 2
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = _sign(x[i] - x[j])
            dy = _sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                tied_x += 1
                tied_y += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1

    denominator_squared = (n0 - tied_x) * (n0 - tied_y)
    if denominator_squared <= 0:
        return 0.0
    if tied_x == 0 and tied_y == 0:
        # Exact integer division in the untied case, so an identical pair is
        # exactly 1.0 and a reversed pair exactly -1.0.
        return (concordant - discordant) / n0
    return (concordant - discordant) / math.sqrt(denominator_squared)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation between two paired numeric sequences.

    Similarity-valued in ``[-1, 1]``: 1.0 for identical orderings, -1.0 for
    reversed. Ties are resolved to average ranks. Pairs are taken
    positionally, so ``x[i]`` and ``y[i]`` must describe the same item.

    Parameters
    ----------
    x, y: Sequence[float]
        Equal-length paired observations.

    Returns
    -------
    float
        rho, or 0.0 when either input has zero variance, for the reason
        given in :func:`kendall_tau`.
    """
    _require_pairable(x, y)
    n = len(x)
    if n < 2:
        return 0.0

    ranks_x, tied_x = _average_ranks(x)
    ranks_y, tied_y = _average_ranks(y)
    if not tied_x and not tied_y:
        # The d-squared form is exact integer arithmetic for untied input,
        # so identical ranks give exactly 1.0 and reversed exactly -1.0.
        squared = sum((a - b) ** 2 for a, b in zip(ranks_x, ranks_y))
        return 1.0 - (6 * squared) / (n * (n * n - 1))
    return _pearson(ranks_x, ranks_y)


def top_k_jaccard(clean: Ordering, patched: Ordering, k: int = 10) -> float:
    """Jaccard similarity of the top-``k`` sets, in ``[0, 1]``.

    The similarity counterpart of :func:`top_k_set_distance`: 1.0 when the
    two top-``k`` sets match and 0.0 when they are disjoint. ``k`` is clamped
    to the ordering length. An empty ordering gives 1.0, since two empty
    sets are equal.

    Raises
    ------
    ValueError
        If ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    first = set(_as_ordering(clean, "clean")[:k])
    second = set(_as_ordering(patched, "patched")[:k])
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


# -- parsing a listwise ranker's text output -------------------------------


def parse_permutation(
    text: str,
    n: int,
    one_indexed: bool = True,
    strict: bool = False,
) -> List[int]:
    """Read a listwise ranker's text output as an ordering.

    Listwise LLMs are prompted to emit ``[3] > [1] > [2]`` and mostly do, but
    they also duplicate items, omit items, invent indices past the end of the
    candidate list, and wrap the whole thing in prose. The repairs below are
    chosen so that a malformed generation degrades to a usable ordering
    rather than either crashing a sweep or -- worse -- silently becoming the
    identity ordering.

    Parameters
    ----------
    text: str
        Raw generation.
    n: int
        Number of candidates that were ranked. The result is always a
        permutation of ``range(n)``.
    one_indexed: bool
        Whether the model's indices start at 1, as RankGPT-style prompts
        specify. Default ``True``. The returned ordering is always 0-indexed
        into the candidate list, matching the rest of mechir.
    strict: bool
        Raise :class:`PermutationParseError` on *any* malformation instead of
        repairing it. Use this when measuring how often a model emits a
        clean permutation; leave it off for sweeps.

    Returns
    -------
    List[int]
        A permutation of ``range(n)``, best-ranked candidate first.

    Raises
    ------
    ValueError
        If ``n < 1``.
    PermutationParseError
        If ``text`` contains no bracketed index at all, or none naming a
        candidate in range -- both genuinely ambiguous. Also raised for any
        repaired malformation when ``strict`` is set.

    Notes
    -----
    Malformation policies:

    ``extra prose``
        Ignored. Every ``[<digits>]`` token is extracted in order of
        appearance and everything else is discarded, so
        ``"Sure! The ranking is [2] > [1]. Hope that helps"`` parses.
    ``unbracketed indices``
        Not parsed. ``"2 > 1"`` raises rather than guessing, because prose
        around a list contains digits ("top 3 documents") and a bare-number
        fallback would turn those into ranks. Brackets are what the prompt
        asks for and what the models emit.
    ``duplicates``
        First occurrence wins, later ones are dropped. A model that says
        ``[2] > [1] > [2]`` placed candidate 2 first with more context than
        it had by the time it repeated itself, and the repeat is the
        artefact.
    ``out-of-range indices``
        Dropped. ``[11]`` in a 10-candidate pool names nothing, so there is
        nothing to rank. This is a routine hallucination and raising would
        lose the rest of a perfectly good ordering.
    ``omitted items``
        Appended at the tail in their original (input) relative order, which
        is the least-assuming completion: the model declined to rank them, so
        they keep the order the first-stage retriever gave them.
    ``no usable indices``
        Raises. An empty parse would otherwise become "every item omitted",
        i.e. exactly the input ordering, and a sweep would report a
        confidently wrong zero for every component.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    found = [int(match) for match in _BRACKETED.findall(text)]
    if not found:
        raise PermutationParseError(
            f"no bracketed index in {text!r}; expected output like '[3] > [1] > [2]'"
        )

    offset = 1 if one_indexed else 0
    ordering: List[int] = []
    seen = set()
    duplicates: List[int] = []
    out_of_range: List[int] = []
    for raw in found:
        index = raw - offset
        if not 0 <= index < n:
            out_of_range.append(raw)
            continue
        if index in seen:
            duplicates.append(raw)
            continue
        seen.add(index)
        ordering.append(index)

    if not ordering:
        raise PermutationParseError(
            f"no index in range [{offset}, {n - 1 + offset}] in {text!r}; found {found}"
        )

    missing = [index for index in range(n) if index not in seen]
    if strict and (duplicates or out_of_range or missing):
        raise PermutationParseError(
            f"malformed permutation in {text!r}: duplicates={duplicates}, "
            f"out_of_range={out_of_range}, missing={missing}"
        )

    return ordering + missing


# -- shared machinery ------------------------------------------------------


def is_degenerate_ordering(ordering: Ordering) -> bool:
    """Whether an ordering carries no rank information at all.

    An ordering is degenerate when it is shorter than two entries, repeats an
    entry, or contains a non-finite value. Every metric in this module
    short-circuits to 0.0 change on a degenerate input rather than raising or
    returning NaN.

    Repeats and NaNs are not valid permutations, so this looks like something
    that ought to raise. It is not, because of where they come from: a padded
    or never-filled row of a batched ordering tensor is all zeros, and a
    failed generation yields NaN. Raising there would abort a whole patching
    sweep over one dead batch item; reporting 0.0 reads correctly, since an
    ordering that carries no information cannot be shown to have changed.
    Genuinely inconsistent input -- two orderings over *different* candidate
    sets -- still raises, because that is a caller bug rather than a dead
    batch row.
    """
    values = _as_ordering(ordering, "ordering")
    if len(values) < 2:
        return True
    if any(isinstance(v, float) and not math.isfinite(v) for v in values):
        return True
    return len(set(values)) < len(values)


def _ordering_distance(
    clean: Ordering,
    patched: Ordering,
    distance: Callable[[List[Any], List[Any]], float],
) -> float:
    """Validate, guard the degenerate cases, then apply ``distance``.

    The single place every metric's contract is enforced, so the four of them
    cannot drift apart on exactness, degeneracy or validation.
    """
    first = _as_ordering(clean, "clean")
    second = _as_ordering(patched, "patched")

    if len(first) != len(second):
        raise ValueError(
            f"orderings differ in length: {len(first)} clean against "
            f"{len(second)} patched"
        )
    # Degeneracy is checked before set agreement so that a dead batch row
    # scores 0.0 instead of raising against a live clean ordering.
    if is_degenerate_ordering(first) or is_degenerate_ordering(second):
        return 0.0
    if set(first) != set(second):
        raise ValueError(
            "orderings rank different candidates; they must be permutations "
            "of the same candidate set"
        )
    # Exact zero for no change, rather than a float artefact of the formula.
    if first == second:
        return 0.0
    return _clamp_unit(distance(first, second))


def _kendall_distance(first: List[Any], second: List[Any]) -> float:
    """``(1 - tau) / 2`` over the two orderings' per-candidate ranks."""
    ranks_a, ranks_b = _aligned_ranks(first, second)
    return (1.0 - kendall_tau(ranks_a, ranks_b)) / 2.0


def _spearman_distance(first: List[Any], second: List[Any]) -> float:
    """``(1 - rho) / 2`` over the two orderings' per-candidate ranks."""
    ranks_a, ranks_b = _aligned_ranks(first, second)
    return (1.0 - spearman_rho(ranks_a, ranks_b)) / 2.0


def _top_k_distance(first: List[Any], second: List[Any], k: int) -> float:
    """``1 - Jaccard`` over the two top-``k`` sets."""
    return 1.0 - top_k_jaccard(first, second, k)


def _first_flip_distance(first: List[Any], second: List[Any]) -> float:
    """1.0 if the orderings disagree on the top-ranked candidate."""
    return 0.0 if first[0] == second[0] else 1.0


def _aligned_ranks(first: List[Any], second: List[Any]) -> Tuple[List[int], List[int]]:
    """Per-candidate ranks from two orderings, in a shared candidate order.

    Correlations are positional, so both rank vectors have to be indexed by
    the same candidate. Callers have already established that the two
    orderings are permutations of one candidate set.
    """
    rank_of_second = {item: rank for rank, item in enumerate(second)}
    return (
        list(range(len(first))),
        [rank_of_second[item] for item in first],
    )


def _as_ordering(ordering: Ordering, label: str) -> List[Any]:
    """One ordering as a plain list, without copying tensors to a new device."""
    if isinstance(ordering, Tensor):
        if ordering.ndim != 1:
            raise ValueError(
                f"{label} ordering must be 1-D, got shape {tuple(ordering.shape)}"
            )
        return ordering.detach().cpu().tolist()
    if isinstance(ordering, str) or not isinstance(ordering, SequenceABC):
        raise TypeError(
            f"{label} ordering must be a 1-D tensor or a sequence of "
            f"candidate indices, got {type(ordering).__name__}"
        )
    return list(ordering)


def _as_batch(batch: OrderingBatch, label: str) -> List[Ordering]:
    """A batch of orderings as a list of rows, accepting an unbatched one."""
    if isinstance(batch, Tensor):
        if batch.ndim == 1:
            rows: List[Ordering] = [batch]
        elif batch.ndim == 2:
            rows = list(batch)
        else:
            raise ValueError(
                f"{label} orderings must be 1-D or 2-D, got shape {tuple(batch.shape)}"
            )
    elif isinstance(batch, str) or not isinstance(batch, SequenceABC):
        raise TypeError(
            f"{label} orderings must be a tensor or a sequence, got "
            f"{type(batch).__name__}"
        )
    elif len(batch) == 0 or _is_row(batch[0]):
        # An empty sequence is an empty batch, not a batch of one empty
        # ordering; it falls through to the guard below.
        rows = list(batch)
    else:
        rows = [batch]

    if not rows:
        raise ValueError(
            f"{label} orderings are empty; there is nothing to score, and an "
            "empty result would reduce to NaN under .mean()"
        )
    return rows


def _is_row(item: Any) -> bool:
    """Whether a batch's first element is itself an ordering."""
    return isinstance(item, Tensor) or (
        isinstance(item, SequenceABC) and not isinstance(item, str)
    )


def _average_ranks(values: Sequence[float]) -> Tuple[List[float], bool]:
    """Average ranks of ``values``, and whether any were tied."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tied = False
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        if stop - start > 1:
            tied = True
        shared = (start + stop - 1) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = shared
        start = stop
    return ranks, tied


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation, returning 0.0 rather than NaN on zero variance."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    variance = sum(a * a for a in dx) * sum(b * b for b in dy)
    if variance <= 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(variance)


def _require_pairable(x: Sequence[float], y: Sequence[float]) -> None:
    """Reject inputs that cannot be paired up positionally."""
    if len(x) != len(y):
        raise ValueError(
            f"paired sequences must be the same length, got {len(x)} and {len(y)}"
        )


def _sign(value: float) -> int:
    """-1, 0 or 1."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _clamp_unit(value: float) -> float:
    """Clamp to ``[0, 1]``, mapping a non-finite value to 0.0.

    Floating point can push ``(1 - rho) / 2`` a few ulps outside the unit
    interval on tie-corrected inputs, and the whole point of this module's
    range guarantee is that callers can compare metrics without checking.
    Non-finite values are mapped to 0.0 for the reason given in
    :func:`kendall_tau`.
    """
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


__all__ = [
    "Ordering",
    "OrderingBatch",
    "PatchingMetric",
    "PermutationParseError",
    "RANK_METRICS",
    "RankMetric",
    "batched_rank_metric",
    "first_position_flip",
    "get_rank_metric",
    "is_degenerate_ordering",
    "kendall_tau",
    "kendall_tau_distance",
    "parse_permutation",
    "spearman_distance",
    "spearman_rho",
    "top_k_jaccard",
    "top_k_set_distance",
]
