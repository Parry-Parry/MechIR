"""Benchmark loaders that yield candidate sets with labels.

``Pipeline`` needs a query, a fixed set of candidate documents, and -- for
reader-side metrics -- a gold answer. These loaders produce exactly that,
plus per-document relevance labels so a causal measure can be scored
against ground truth rather than only inspected.

Two benchmarks, chosen because they cover the two halves of the pipeline:

TREC Deep Learning 2019/2020 has graded judgements (0-3) over MS MARCO
passages and no answers, so it serves ranker-side work. Note the corpus is
8.8M passages and ir_datasets will download it on first use.

HotpotQA comes from the HuggingFace ``hotpot_qa`` distractor config rather
than ir_datasets. ``beir/hotpotqa`` exposes only ``(query_id, text)``, with
no answer field, so reader metrics cannot be computed from it. The
distractor config instead gives ten paragraphs per question -- two
supporting and eight distractors -- with the answer string, which is a
ready-made labelled candidate set needing no retrieval and no 5.2M-document
corpus.

For the record, the BEIR splits do join back to HotpotQA by query id if you
need the retrieval view: ``beir/hotpotqa/dev`` matches HotpotQA *train*
(5447/5447) and ``beir/hotpotqa/test`` matches *validation* (7405/7405).
BEIR re-split the data, which is why dev does not map to validation.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class RagExample:
    """One query with a fixed, labelled candidate set.

    Attributes
    ----------
    relevance:
        Per-document label, parallel to ``documents``. Graded 0-3 for TREC
        DL; 1 for a HotpotQA supporting paragraph and 0 for a distractor.
    answer:
        Gold answer string, or ``None`` for benchmarks that have none.
    """

    query_id: str
    query: str
    documents: List[str]
    relevance: List[int]
    doc_ids: List[str] = field(default_factory=list)
    answer: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.documents) != len(self.relevance):
            raise ValueError(
                f"{len(self.documents)} documents but "
                f"{len(self.relevance)} relevance labels"
            )

    @property
    def relevant(self) -> List[int]:
        """Indices of documents judged relevant (label > 0)."""
        return [i for i, label in enumerate(self.relevance) if label > 0]

    @property
    def irrelevant(self) -> List[int]:
        return [i for i, label in enumerate(self.relevance) if label <= 0]


def load_trec_dl(
    year: int = 2019,
    max_queries: Optional[int] = None,
    max_documents: int = 10,
    min_relevant: int = 1,
    seed: int = 0,
) -> List[RagExample]:
    """Judged candidate sets from TREC Deep Learning 2019 or 2020.

    Each example holds the highest-graded judged passages for one query plus
    judged non-relevant ones, capped at ``max_documents``. Keeping the pool
    small and fixed is deliberate: the interventions in ``mechir.pipeline``
    reorder and ablate a *fixed* candidate set, so a full ranked list is
    neither needed nor wanted.

    There are no answers in TREC DL, so ``answer`` is ``None`` and
    reader-side metrics do not apply.
    """
    import ir_datasets

    if year not in (2019, 2020):
        raise ValueError(f"TREC DL year must be 2019 or 2020, got {year}")

    dataset = ir_datasets.load(f"msmarco-passage/trec-dl-{year}/judged")
    queries = {q.query_id: q.text for q in dataset.queries_iter()}

    by_query: Dict[str, List[tuple]] = {}
    for qrel in dataset.qrels_iter():
        by_query.setdefault(qrel.query_id, []).append((qrel.doc_id, qrel.relevance))

    store = dataset.docs_store()
    rng = random.Random(seed)
    examples: List[RagExample] = []

    for query_id in sorted(by_query):
        judged = by_query[query_id]
        positives = sorted((j for j in judged if j[1] > 0), key=lambda j: -j[1])
        negatives = [j for j in judged if j[1] <= 0]
        if len(positives) < min_relevant:
            continue

        rng.shuffle(negatives)
        half = max(1, max_documents // 2)
        chosen = positives[:half] + negatives[: max_documents - len(positives[:half])]
        if not chosen:
            continue
        rng.shuffle(chosen)

        doc_ids = [doc_id for doc_id, _ in chosen]
        texts = [store.get(doc_id).text for doc_id in doc_ids]
        examples.append(
            RagExample(
                query_id=query_id,
                query=queries[query_id],
                documents=texts,
                relevance=[label for _, label in chosen],
                doc_ids=doc_ids,
            )
        )
        if max_queries is not None and len(examples) >= max_queries:
            break

    return examples


def load_hotpotqa(
    split: str = "validation",
    max_queries: Optional[int] = None,
    only_supported: bool = True,
) -> List[RagExample]:
    """Labelled candidate sets from the HotpotQA distractor config.

    Each example is one question with its ten candidate paragraphs, labelled
    1 for a supporting paragraph and 0 for a distractor, plus the gold
    answer. That makes it the natural benchmark for the causal ablation
    measures: ``document_necessity`` can be scored directly against which
    paragraphs actually support the answer.

    Parameters
    ----------
    only_supported:
        Skip questions where no candidate paragraph matches a supporting
        title, which would otherwise look like a total ablation failure when
        it is really a data problem.
    """
    from datasets import load_dataset

    data = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)

    examples: List[RagExample] = []
    skipped = 0

    for row in data:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        supporting = set(row["supporting_facts"]["title"])

        relevance = [1 if title in supporting else 0 for title in titles]
        if only_supported and sum(relevance) == 0:
            skipped += 1
            continue

        examples.append(
            RagExample(
                query_id=row["id"],
                query=row["question"],
                documents=[f"{t}. {''.join(s)}" for t, s in zip(titles, sentences)],
                relevance=relevance,
                doc_ids=list(titles),
                answer=row["answer"],
            )
        )
        if max_queries is not None and len(examples) >= max_queries:
            break

    if skipped:
        logger.warning(
            "skipped %d HotpotQA questions whose supporting titles were "
            "absent from their candidate paragraphs",
            skipped,
        )
    return examples


def sample_filler(
    examples: Sequence[RagExample],
    exclude: int,
    count: int,
    seed: int = 0,
) -> List[str]:
    """Neutral replacement documents drawn from *other* queries.

    Resample ablation needs replacements that carry no evidence for the
    query under test. Documents from a different query satisfy that while
    staying in-distribution, which rotating the candidate set does not --
    see ``mechir.pipeline.ablation``.
    """
    pool = [
        document
        for index, example in enumerate(examples)
        if index != exclude
        for document in example.documents
    ]
    if not pool:
        raise ValueError("need at least two examples to sample filler from")

    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(count)]


def iter_examples(examples: Sequence[RagExample]) -> Iterator[RagExample]:
    """Yield examples that carry an answer, for reader-side work."""
    for example in examples:
        if example.answer:
            yield example


__all__ = [
    "RagExample",
    "load_trec_dl",
    "load_hotpotqa",
    "sample_filler",
    "iter_examples",
]
