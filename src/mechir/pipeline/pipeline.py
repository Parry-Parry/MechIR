"""Ranker and reader wired together with hooks on both sides.

The point of holding both is that interventions on one can be measured on
the other. Patching a ranker head changes its scores, which changes the
document order, which changes what the reader sees -- and the metric is
read off the reader's generation rather than off the ranker's score.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

import torch
import transformer_lens.utils as utils

from mechir.pipeline.ablation import (
    causal_groundedness,
    document_necessity,
    document_sufficiency,
    resample_necessity,
)
from mechir.pipeline.boundary import HardPermutation, PermutationOp, Selection
from mechir.pipeline.context import Context, ContextBuilder
from mechir.pipeline.metrics import answer_logprob, attention_to_documents

logger = logging.getLogger(__name__)


@dataclass
class HeadPatchResult:
    """Cross-boundary patching results, with the diagnostics to read them.

    ``metric_delta`` alone is misleading. The reader metric is *piecewise
    constant* in the ranker's scores: the context only changes if an
    intervention reorders documents. So a head can shift a score
    substantially and still produce a delta of exactly zero.

    ``rank_flipped`` and ``score_delta`` separate the two cases -- "this
    head does not affect the ranker" from "this head affects the ranker but
    not enough to flip a rank".
    """

    metric_delta: torch.Tensor
    rank_flipped: torch.Tensor
    score_delta: torch.Tensor
    baseline_metric: float
    layers: List[int]
    heads: List[int]

    @property
    def flip_rate(self) -> float:
        """Share of patched heads that reordered the documents."""
        return float(self.rank_flipped.float().mean())

    def summary(self) -> str:
        return (
            f"{int(self.rank_flipped.sum())}/{self.rank_flipped.numel()} heads "
            f"flipped a rank (flip rate {self.flip_rate:.1%}); "
            f"max |score delta| {self.score_delta.max():.4f}; "
            f"max |metric delta| {self.metric_delta.abs().max():.4f}"
        )


@dataclass
class PipelineRun:
    """One pass through ranker, boundary and reader."""

    selection: Selection
    context: Context
    metric: float

    @property
    def order(self) -> List[int]:
        return self.selection.order


class Pipeline:
    """A hooked ranker feeding a hooked reader across an explicit boundary.

    Parameters
    ----------
    ranker:
        Any mechir ranker exposing ``score`` -- Cat, GenerativeRanker, or a
        callable taking (query, documents) and returning one score per
        document.
    reader:
        A TransformerLens ``HookedTransformer``.
    boundary:
        Defaults to ``HardPermutation()``, which keeps every candidate so
        that demotions do not silently become removals.
    metric:
        Reader-side quantity to report. Defaults to answer log-probability.
    """

    def __init__(
        self,
        ranker,
        reader,
        boundary: Optional[HardPermutation] = None,
        context_builder: Optional[ContextBuilder] = None,
        metric: Callable[..., torch.Tensor] = answer_logprob,
    ) -> None:
        self.ranker = ranker
        self.reader = reader
        self.boundary = boundary or HardPermutation()
        self.metric = metric
        self.context_builder = context_builder or ContextBuilder(
            tokenizer=getattr(reader, "tokenizer", None)
        )
        if self.context_builder.tokenizer is None:
            raise ValueError("reader has no tokenizer; pass context_builder explicitly")
        self._device = next(reader.parameters()).device

    # -- ranking ----------------------------------------------------------

    def rank(self, query: str, documents: Sequence[str]) -> torch.Tensor:
        """One ranker score per document."""
        if callable(self.ranker) and not hasattr(self.ranker, "score"):
            return self.ranker(query, documents)

        batch = self._encode_pairs(query, documents)
        scores, _ = self.ranker.score(batch)
        return scores.detach().flatten()

    def _encode_pairs(
        self, query: str, documents: Sequence[str], pad_to: Optional[int] = None
    ) -> dict:
        """Tokenise (query, document) pairs for the ranker.

        ``pad_to`` forces a fixed length. Patching needs it: the clean and
        corrupted batches must agree on sequence length or the cached
        activations will not line up with the hooked ones.
        """
        tokenizer = self.ranker.tokenizer
        encoded = tokenizer(
            [query] * len(documents),
            list(documents),
            return_tensors="pt",
            padding="max_length" if pad_to else True,
            truncation=True,
            max_length=pad_to or 256,
        )
        batch = {
            "input_ids": encoded.input_ids.to(self._device),
            "attention_mask": encoded.attention_mask.to(self._device),
        }
        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded.token_type_ids.to(self._device)
        return batch

    # -- running ----------------------------------------------------------

    def run(
        self,
        query: str,
        documents: Sequence[str],
        answer: str,
        op: Optional[PermutationOp] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> PipelineRun:
        """Rank, order, assemble, read."""
        if scores is None:
            scores = self.rank(query, documents)
        selection = self.boundary.select(scores, op=op)
        context = self.context_builder.build(
            query, documents, selection.order, answer=answer, device=self._device
        )
        return PipelineRun(
            selection=selection,
            context=context,
            metric=float(self.metric(self.reader, context)),
        )

    # -- boundary interventions ------------------------------------------

    def position_sensitivity(
        self,
        query: str,
        documents: Sequence[str],
        answer: str,
        ops: Optional[Sequence[PermutationOp]] = None,
    ) -> Dict[str, float]:
        """Metric under each permutation of a fixed candidate set.

        Because the set is fixed, every difference here is attributable to
        *order* alone -- which is what makes position bias measurable rather
        than confounded with retrieval quality.
        """
        scores = self.rank(query, documents)
        ops = ops or self.boundary.sweep(len(documents))
        return {
            getattr(op, "name", f"op{i}"): self.run(
                query, documents, answer, op=op, scores=scores
            ).metric
            for i, op in enumerate(ops)
        }

    def rank_vs_attention(
        self,
        query: str,
        documents: Sequence[str],
        answer: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> List[dict]:
        """Ranker score against reader attention, per document.

        Where the reader's attention ordering disagrees with the ranker's
        score ordering, the two components disagree about relevance.
        """
        scores = self.rank(query, documents)
        selection = self.boundary.select(scores)
        context = self.context_builder.build(
            query, documents, selection.order, answer=answer, device=self._device
        )
        attention = attention_to_documents(self.reader, context, layer=layer)

        return [
            {
                "rank": rank,
                "candidate": candidate,
                "ranker_score": float(scores[candidate]),
                "reader_attention": attention[rank],
            }
            for rank, candidate in enumerate(selection.order)
        ]

    def position_controlled_attention(
        self,
        query: str,
        documents: Sequence[str],
        answer: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> List[dict]:
        """Reader attention per document, measured at a fixed position.

        ``rank_vs_attention`` compares documents sitting at different ranks,
        so in a causal reader it is confounded by recency. Here each
        candidate is promoted to rank 0 in turn and read there, so every
        measurement is taken at the same position and differences are
        attributable to content.

        Costs one reader forward per candidate, which is what makes the
        interventional boundary worth having.
        """
        scores = self.rank(query, documents)
        selection = self.boundary.select(scores)
        rows = []

        for candidate in selection.order:
            order = [candidate] + [c for c in selection.order if c != candidate]
            context = self.context_builder.build(
                query, documents, order, answer=answer, device=self._device
            )
            attention = attention_to_documents(self.reader, context, layer=layer)
            rows.append(
                {
                    "candidate": candidate,
                    "ranker_rank": selection.order.index(candidate),
                    "ranker_score": float(scores[candidate]),
                    "attention_at_rank0": attention[0],
                }
            )

        return rows

    def causal_report(
        self,
        query: str,
        documents: Sequence[str],
        answer: str,
        layer: Optional[int] = None,
    ) -> dict:
        """Ranker score, reader attention and causal effect, per document.

        The attention and causal columns routinely disagree, and the causal
        ones are the defensible basis for a faithfulness claim. On a
        four-document example with pythia-70m, attention ranked a wholly
        irrelevant distractor first while its leave-one-out effect was
        *negative* -- removing it improved the answer. Reporting both makes
        that visible instead of inviting attention to be read as
        explanation.

        Returns
        -------
        dict
            ``groundedness`` holds the all-documents ablation, and
            ``documents`` one row per rank with ``necessity`` (leave-one-out)
            and ``sufficiency`` (keep-one-only) in nats.
        """
        scores = self.rank(query, documents)
        selection = self.boundary.select(scores)
        context = self.context_builder.build(
            query, documents, selection.order, answer=answer, device=self._device
        )

        overall = causal_groundedness(self.reader, context)
        necessity = document_necessity(self.reader, context)
        sufficiency = document_sufficiency(self.reader, context)
        attention = attention_to_documents(self.reader, context, layer=layer)
        resampled = resample_necessity(
            self.reader,
            self.context_builder,
            query,
            documents,
            selection.order,
            answer,
            device=self._device,
        )

        rows = [
            {
                "rank": rank,
                "candidate": candidate,
                "ranker_score": float(scores[candidate]),
                "reader_attention": attention[rank],
                "necessity": necessity[rank],
                "sufficiency": sufficiency[rank],
                "resample_effect": resampled[rank]["effect"],
                "resample_length_delta": resampled[rank]["length_delta"],
            }
            for rank, candidate in enumerate(selection.order)
        ]
        return {"groundedness": overall, "documents": rows}

    # -- cross-boundary patching -----------------------------------------

    def patch_ranker_heads(
        self,
        query: str,
        documents: Sequence[str],
        answer: str,
        documents_p: Optional[Sequence[str]] = None,
        layers: Optional[Sequence[int]] = None,
        heads: Optional[Sequence[int]] = None,
        pad_to: int = 256,
    ) -> torch.Tensor:
        """Effect of each ranker attention head on the reader's output.

        For every (layer, head), the head's output is replaced with its
        value from the perturbed run, the documents are re-ranked with the
        patched scores, and the reader is re-run. The returned value is the
        change in the reader metric.

        This is the interventional route across the boundary: no gradient
        has to cross the discrete selection step.

        Expect many exact zeros. The reader metric is piecewise constant in
        the ranker's scores, so a head only registers if it reorders
        documents. Measured on a 4-document set with a MiniLM cross-encoder,
        all 72 heads changed scores -- one by 1.5 -- and **none** flipped a
        rank, so every metric delta was exactly 0. Use ``flip_rate`` on the
        result to tell that apart from a head having no effect, and prefer
        candidate sets whose scores are close together, where flips are
        reachable. Where the hard boundary is too coarse, a differentiable
        boundary is the alternative.

        Returns
        -------
        HeadPatchResult
            Metric deltas, per-head rank-flip flags and score deltas, each
            shaped ``[len(layers), len(heads)]``.
        """
        model = self.ranker._model
        layers = list(range(model.cfg.n_layers)) if layers is None else list(layers)
        heads = list(range(model.cfg.n_heads)) if heads is None else list(heads)

        baseline = self.run(query, documents, answer)

        # Both batches are padded to the same length, or the cached
        # activations will not align with the hooked ones.
        clean_batch = self._encode_pairs(query, documents_p or documents, pad_to=pad_to)
        _, clean_cache = self.ranker.score(clean_batch, cache=True)
        corrupted = self._encode_pairs(query, documents, pad_to=pad_to)

        metric_delta = torch.zeros(len(layers), len(heads), dtype=torch.float32)
        rank_flipped = torch.zeros(len(layers), len(heads), dtype=torch.bool)
        score_delta = torch.zeros(len(layers), len(heads), dtype=torch.float32)
        baseline_scores = baseline.selection.scores
        baseline_order = baseline.order

        for li, layer in enumerate(layers):
            for hi, head in enumerate(heads):
                hook_name = "_model." + utils.get_act_name("z", layer)
                hook_fn = partial(
                    self.ranker._patch_head_vector,
                    head_index=head,
                    clean_cache=clean_cache,
                )
                with torch.no_grad():
                    patched_scores = self.ranker.run_with_hooks(
                        corrupted["input_ids"],
                        attention_mask=corrupted["attention_mask"],
                        **self.ranker._extra_forward_kwargs(corrupted),
                        fwd_hooks=[(hook_name, hook_fn)],
                    )
                self.ranker.reset_hooks()

                patched = patched_scores.detach().flatten()
                run = self.run(query, documents, answer, scores=patched)

                metric_delta[li, hi] = run.metric - baseline.metric
                rank_flipped[li, hi] = run.order != baseline_order
                score_delta[li, hi] = (patched - baseline_scores).abs().max()

        result = HeadPatchResult(
            metric_delta=metric_delta,
            rank_flipped=rank_flipped,
            score_delta=score_delta,
            baseline_metric=baseline.metric,
            layers=layers,
            heads=heads,
        )
        if not rank_flipped.any():
            logger.warning(
                "No ranker head flipped a rank, so every metric delta is "
                "exactly 0 (max |score delta| %.4f). The reader metric is "
                "piecewise constant in the ranker's scores. Try candidate "
                "documents with closer scores, or a differentiable boundary.",
                float(score_delta.max()),
            )
        return result


__all__ = ["Pipeline", "PipelineRun"]
