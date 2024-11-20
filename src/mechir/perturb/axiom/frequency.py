from pathlib import Path
from functools import partial
from typing import Any
import random
from ..index import IndexPerturbation


class FrequencyPerturbation(IndexPerturbation):
    """
    A perturbation that adds terms to a document based on their frequency in the document or query. TFI, IDF, and TFIDF are supported.

    params:
        index_location: The location of the index to use for term frequency calculations. should be a PyTerrier index or a path to a PyTerrier index.
        mode: The method to use for selecting terms to add. Options are 'random', 'top_k', 'max', and 'min'.
        target: The target to use for term frequency calculations. Options are 'query' and 'document'.
        loc: The location to insert the terms. Options are 'start' and 'end'.
        frequency: The frequency metric to use for term selection. Options are 'tf', 'idf', and 'tfidf'.
        num_additions: The number of terms to add to the document.
        tokeniser: The tokeniser to use for tokenising the text. If None, the default tokeniser is used.
        stem: Whether or not to apply porter stemming for matching and lookup
        stopwords: Whether or not to filter valid terms with a stopword list
        exact_match: Forces returned terms to be present in both texts
    """

    def __init__(
        self,
        index_location: Any | Path | str,
        mode: str = "max",
        target: str = "query",
        loc="end",
        frequency: str = "tf",
        num_additions: int = 1,
        tokeniser: Any | None = None,
        stem: bool = False,
        stopwords: bool = False,
        exact_match: bool = False,
    ) -> None:
        super().__init__(index_location, tokeniser, stem, stopwords, exact_match)

        self.get_freq_terms = {
            "random": self._get_random_terms,
            "top_k": self._get_top_k_freq_terms,
            "max": self._get_max_freq_terms,
            "min": self._get_min_freq_terms,
        }[mode]
        self.get_freq_text = {
            "tf": self.get_tf_text,
            "idf": self.get_idf_text,
            "tfidf": self.get_tfidf_text,
        }[frequency]

        self._insert_terms = {
            "end": lambda text, terms: f"{text} {' '.join(terms)}",
            "start": lambda text, terms: f"{' '.join(terms)} {text}",
        }[loc]
        self.target = target
        self.num_additions = num_additions
        self.loc = loc

    def _get_random_terms(self, text: str, terms: list) -> list:
        return random.choices(
            list(self.get_freq_text(text, terms).keys()), k=self.num_additions
        )

    def _get_top_k_freq_terms(self, text: str, terms: list) -> dict:
        freq = self.get_freq_text(text, terms)
        # Get the top num_additions terms with the highest term frequency
        return sorted(freq.items(), key=lambda x: x[1], reverse=True).keys()[
            : self.num_additions
        ]

    def _get_max_freq_terms(self, text: str, terms: list) -> str:
        freq = self.get_freq_text(text, terms)
        term = max(freq, key=freq.get)
        return [term] * self.num_additions

    def _get_min_freq_terms(self, text: str, terms: list) -> str:
        freq = self.get_freq_text(text, terms)
        term = min(freq, key=freq.get)
        return [term] * self.num_additions

    def apply(self, document: str, query: str) -> str:
        terms = []
        if self.exact_match:
            # find stemmed terms that are in the document and query
            query_terms = self.get_terms(query)
            document_terms = self.get_terms(document)
            terms = [
                term for term in query_terms.values() if term in document_terms.values()
            ]
        terms = self.get_freq_terms(
            query if self.target == "query" else document, terms
        )
        return self._insert_terms(document, terms)


TFPerturbation = partial(FrequencyPerturbation, frequency="tf")
IDFPerturbation = partial(FrequencyPerturbation, frequency="idf")
TFIDFPerturbation = partial(FrequencyPerturbation, frequency="tfidf")

TFC1 = partial(TFPerturbation, num_additions=1, loc="end", mode="random")
TDC = partial(IDFPerturbation, num_additions=1, loc="end", mode="max")

__all__ = [
    "FrequencyPerturbation",
    "TFPerturbation",
    "IDFPerturbation",
    "TFIDFPerturbation",
    "TFC1",
    "TDC",
]
