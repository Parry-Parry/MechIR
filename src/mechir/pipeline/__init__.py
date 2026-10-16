"""Mechanistic interpretability for retrieval-augmented pipelines.

Holds a ranker and a reader with hooks on both sides, so an intervention on
one can be measured on the other -- the metric is read off the reader's
generation rather than the ranker's score.

Optional. Install the benchmark loaders it is usually driven from with:

    pip install "mechir[rag]"

Research code rather than settled library surface: the API will change. See
docs/research-notes.md for what has actually been measured, and for four
ways to produce a result here that looks like a finding and is not.
"""

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
from .ablation import (
    ablated_metric,
    causal_groundedness,
    document_necessity,
    document_sufficiency,
    resample_metric,
    resample_necessity,
)
from .context import Context, ContextBuilder, DocumentSpan, Span, TokenSpan
from .metrics import answer_logprob, attention_to_documents, groundedness
from .pipeline import Pipeline, PipelineRun
from .rollout import Turn, generate_turn

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
    "Span",
    "TokenSpan",
    "Turn",
    "generate_turn",
    "answer_logprob",
    "attention_to_documents",
    "groundedness",
    "ablated_metric",
    "causal_groundedness",
    "document_necessity",
    "document_sufficiency",
    "resample_metric",
    "resample_necessity",
]
