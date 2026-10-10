"""Ranker-and-reader pipelines with hooks on both sides."""

from .boundary import (
    HardPermutation,
    PermutationOp,
    Selection,
    demote,
    identity,
    promote,
    reverse,
    shuffle,
    swap,
)
from .context import Context, ContextBuilder, DocumentSpan
from .metrics import answer_logprob, attention_to_documents, groundedness
from .pipeline import Pipeline, PipelineRun

__all__ = [
    "Pipeline",
    "PipelineRun",
    "HardPermutation",
    "Selection",
    "PermutationOp",
    "identity",
    "swap",
    "reverse",
    "demote",
    "promote",
    "shuffle",
    "Context",
    "ContextBuilder",
    "DocumentSpan",
    "answer_logprob",
    "attention_to_documents",
    "groundedness",
]
