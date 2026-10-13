"""The ranker/reader boundary.

Between a ranker and a reader sits top-k selection and context assembly. That
step is discrete, so gradients do not cross it: attribution patching, AtP*
and EAP-IG all stop at the boundary. Activation patching does not care,
because it intervenes rather than differentiates.

``HardPermutation`` takes the interventional route. The candidate set is
fixed and only the *order* varies, which keeps every intervention an exact
discrete operation with no relaxation to defend. The cost is one reader
forward per permutation.

A differentiable counterpart (continuous attention gating over document
spans, which lets gradients cross) is the natural sibling of this class; the
interventional results here are what it would be validated against.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

# A permutation operation maps a ranked order to a new order. Orders are
# lists of indices into the candidate list.
PermutationOp = Callable[[List[int]], List[int]]


def identity() -> PermutationOp:
    """Leave the ranker's order untouched."""

    def op(order: List[int]) -> List[int]:
        return list(order)

    op.name = "identity"
    return op


def swap(i: int, j: int) -> PermutationOp:
    """Exchange the documents at ranks ``i`` and ``j``."""

    def op(order: List[int]) -> List[int]:
        out = list(order)
        if i < len(out) and j < len(out):
            out[i], out[j] = out[j], out[i]
        return out

    op.name = f"swap({i},{j})"
    return op


def reverse() -> PermutationOp:
    """Invert the ranking."""

    def op(order: List[int]) -> List[int]:
        return list(reversed(order))

    op.name = "reverse"
    return op


def demote(i: int) -> PermutationOp:
    """Move the document at rank ``i`` to the end of the context."""

    def op(order: List[int]) -> List[int]:
        out = list(order)
        if i < len(out):
            out.append(out.pop(i))
        return out

    op.name = f"demote({i})"
    return op


def promote(i: int) -> PermutationOp:
    """Move the document at rank ``i`` to the front of the context."""

    def op(order: List[int]) -> List[int]:
        out = list(order)
        if i < len(out):
            out.insert(0, out.pop(i))
        return out

    op.name = f"promote({i})"
    return op


def shuffle(seed: int) -> PermutationOp:
    """Permute the order deterministically."""

    def op(order: List[int]) -> List[int]:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(order), generator=generator).tolist()
        return [order[p] for p in perm]

    op.name = f"shuffle({seed})"
    return op


@dataclass
class Selection:
    """What the boundary handed to the reader.

    Attributes
    ----------
    order:
        Candidate indices in the order they appear in the context.
    scores:
        Ranker score per candidate, indexed by candidate position (not rank).
    op_name:
        Which permutation produced this order.
    """

    order: List[int]
    scores: torch.Tensor
    op_name: str = "identity"

    def ranks(self) -> dict:
        """Map candidate index to its position in the context."""
        return {candidate: rank for rank, candidate in enumerate(self.order)}


class HardPermutation:
    """Exact, interventional boundary over a fixed candidate set.

    Parameters
    ----------
    k:
        How many of the top-scoring candidates reach the reader. ``None``
        passes all of them, which is what the position-bias probes want --
        otherwise demoting a document can push it out of the context and
        conflate "moved" with "removed".
    """

    def __init__(self, k: Optional[int] = None) -> None:
        if k is not None and k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k

    def select(
        self,
        scores: torch.Tensor,
        op: Optional[PermutationOp] = None,
    ) -> Selection:
        """Rank by score, truncate to ``k``, then apply ``op``.

        The permutation is applied *after* truncation so that an operation
        only ever reorders what the reader would actually have seen.
        """
        if scores.ndim != 1:
            raise ValueError(f"expected one score per candidate, got {scores.shape}")

        order = torch.argsort(scores, descending=True).tolist()
        if self.k is not None:
            order = order[: self.k]

        op = op or identity()
        return Selection(
            order=op(order),
            scores=scores.detach(),
            op_name=getattr(op, "name", "custom"),
        )

    def sweep(self, n_candidates: int) -> List[PermutationOp]:
        """Standard probe set for position sensitivity.

        Pairs every rank with rank 0, plus whole-order baselines. Enough to
        expose a primacy or recency bias without enumerating all n!
        permutations.
        """
        width = n_candidates if self.k is None else min(self.k, n_candidates)
        ops: List[PermutationOp] = [identity()]
        ops.extend(swap(0, j) for j in range(1, width))
        if width > 2:
            ops.append(reverse())
            ops.append(demote(0))
        return ops


__all__ = [
    "HardPermutation",
    "Selection",
    "PermutationOp",
    "identity",
    "swap",
    "reverse",
    "demote",
    "promote",
    "shuffle",
]
