from . import AbstractPerturbation, IRDSDataset
from typing import Union, Optional, NamedTuple, Callable, Sequence, Dict
from pathlib import Path
from collections import Counter
import numpy as np
from ir_axioms.backends.pyterrier.util import (
    Index,
    IndexRef,
    Dataset,
    Tokeniser,
    )
from ir_axioms.model import TextDocument
from ir_axioms.backends.pyterrier import TerrierIndexContext


ContentsAccessor = Union[str, Callable[[NamedTuple], str]]

class IndexPerturbation(AbstractPerturbation):
    def __init__(self,
                 index_location: Union[Index, IndexRef, Path, str],
                 dataset: Optional[Union[Dataset, str, IRDSDataset]] = None,
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
    
    def get_counts(self, text : str) -> Dict[int]:
        return Counter(self.get_terms(text))
    
    def get_tf(self, term : str, text : str):
        return self.context.term_frequency(TextDocument(text), term)
    
    def get_idf(self, term : str, text : str):
        return self.context.inverse_document_frequency(term)

    def get_tfidf(self, term : str, text : str):
        return self.get_tf(term, text) * self.get_idf(term, text)