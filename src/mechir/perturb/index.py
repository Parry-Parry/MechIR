from . import AbstractPerturbation
from typing import Union, Optional, NamedTuple, Callable, Sequence, Dict, Any
from pathlib import Path
from collections import Counter
from ir_datasets import Dataset
from ir_axioms.backend.pyterrier.util import (
    Index,
    IndexRef,
    Tokeniser,
    )
from ir_axioms.model import TextDocument
from ir_axioms.backend.pyterrier import TerrierIndexContext


ContentsAccessor = Union[str, Callable[[NamedTuple], str]]

class IndexPerturbation(AbstractPerturbation):
    def __init__(self,
                 index_location: Union[Index, IndexRef, Path, str],
                 dataset: Optional[Union[Dataset, str, Any]] = None,
                 contents_accessor: Optional[ContentsAccessor] = "text",
                 tokeniser: Optional[Tokeniser] = None,
                 cache_dir: Optional[Path] = None
                 ) -> None:
        super().__init__()
        self.context = TerrierIndexContext(
            index_location=index_location,
            dataset=dataset,
            contents_accessor=contents_accessor,
            tokeniser=tokeniser,
            cache_dir=cache_dir
        )
    
    def get_terms(self, text : str) -> Sequence[str]:
        return self.context.terms(TextDocument(text))
    
    def get_counts(self, text : str) -> Dict[str, int]:
        return Counter(self.get_terms(text))
    
    def get_tf(self, term : str, text : str) -> int:
        return self.context.term_frequency(TextDocument(text), term)

    def get_tf_text(self, text : str) -> Dict[str, int]:
        return {term : self.get_tf(term, text) for term in self.get_terms(text)}

    def get_idf(self, term : str, text : str) -> float:
        return self.context.inverse_document_frequency(term)

    def get_idf_text(self, text : str) -> Dict[str, float]:
        return {term : self.get_idf(term, text) for term in self.get_terms(text)}

    def get_tfidf(self, term : str, text : str) -> float:
        return self.get_tf(term, text) * self.get_idf(term, text)
    
    def get_tfidf_text(self, text : str) -> Dict[str, float]:
        return {term : self.get_tfidf(term, text) for term in self.get_terms(text)}
    
__all__ = ["IndexPerturbation"]