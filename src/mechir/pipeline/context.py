"""Context assembly with per-document token spans.

The reader sees one flat token sequence, but almost every question worth
asking is per-document: how much attention did rank 1 get, did the answer
come from document 3, what happens when two documents swap places. So
assembly records where each document landed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class DocumentSpan:
    """Where one document sits in the assembled context."""

    candidate: int
    rank: int
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass
class Context:
    """An assembled reader prompt plus the spans that make it up."""

    input_ids: torch.Tensor
    spans: List[DocumentSpan] = field(default_factory=list)
    answer_start: Optional[int] = None
    text: str = ""

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[-1])

    def span_for_rank(self, rank: int) -> DocumentSpan:
        for span in self.spans:
            if span.rank == rank:
                return span
        raise KeyError(f"no document at rank {rank}")

    def span_for_candidate(self, candidate: int) -> DocumentSpan:
        for span in self.spans:
            if span.candidate == candidate:
                return span
        raise KeyError(f"candidate {candidate} is not in this context")


class ContextBuilder:
    """Assembles a reader prompt and tracks document token offsets.

    Spans are computed by tokenising incrementally rather than by searching
    the final sequence for substrings, so they stay exact under tokenisers
    that merge across boundaries.
    """

    def __init__(
        self,
        tokenizer,
        instruction: str = "Answer the question using the documents.",
        document_template: str = "Document {rank}: {text}\n",
        question_template: str = "\nQuestion: {query}\nAnswer:",
        max_document_tokens: Optional[int] = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.document_template = document_template
        self.question_template = question_template
        self.max_document_tokens = max_document_tokens

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def document_token_length(self, text: str) -> int:
        """Token count ``text`` will occupy, after this builder's truncation.

        Exposed so a resample ablation can length-match a replacement and
        avoid shifting the documents that follow it.
        """
        return len(self._encode(self._truncate(text)))

    def _truncate(self, text: str) -> str:
        if self.max_document_tokens is None:
            return text
        ids = self._encode(text)
        if len(ids) <= self.max_document_tokens:
            return text
        return self.tokenizer.decode(ids[: self.max_document_tokens])

    def build(
        self,
        query: str,
        documents: Sequence[str],
        order: Sequence[int],
        answer: Optional[str] = None,
        device=None,
    ) -> Context:
        """Assemble the prompt for ``documents`` taken in ``order``."""
        ids: List[int] = list(self._encode(self.instruction + "\n"))
        spans: List[DocumentSpan] = []

        for rank, candidate in enumerate(order):
            chunk = self.document_template.format(
                rank=rank + 1, text=self._truncate(documents[candidate])
            )
            chunk_ids = self._encode(chunk)
            spans.append(
                DocumentSpan(
                    candidate=candidate,
                    rank=rank,
                    start=len(ids),
                    end=len(ids) + len(chunk_ids),
                )
            )
            ids.extend(chunk_ids)

        ids.extend(self._encode(self.question_template.format(query=query)))
        answer_start = len(ids)

        if answer is not None:
            ids.extend(self._encode(" " + answer.strip()))

        tensor = torch.tensor([ids], dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)

        return Context(
            input_ids=tensor,
            spans=spans,
            answer_start=answer_start if answer is not None else None,
        )


__all__ = ["Context", "ContextBuilder", "DocumentSpan"]
