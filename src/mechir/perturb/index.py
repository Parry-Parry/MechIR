from . import AbstractPerturbation
from typing import Union, Optional, Sequence, Dict
from pathlib import Path
from collections import Counter
import pyterrier as pt 
if not pt.started():
    pt.init()
from functools import lru_cache
from collections import defaultdict
import math
from nltk import word_tokenize

StringReader = pt.autoclass("java.io.StringReader")
Index = pt.autoclass("org.terrier.structures.Index")
IndexRef = pt.autoclass('org.terrier.querying.IndexRef')
IndexFactory = pt.autoclass('org.terrier.structures.IndexFactory')
Tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser")

'''
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
        return self.context.terms(TextDocument(id=random.randint(1, 10000000), contents=text))
    
    def get_counts(self, text : str) -> Dict[str, int]:
        return Counter(self.get_terms(text))
    
    def get_tf(self, term : str, text : str) -> int:
        return self.context.term_frequency(TextDocument(id=random.randint(1, 10000000), contents=text), term)

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
'''

def get_index(index_location) -> Index:
        if isinstance(index_location, Index):
            return index_location
        elif isinstance(index_location, IndexRef):
            return IndexFactory.of(index_location)
        elif isinstance(index_location, str):
            return IndexFactory.of(index_location)
        elif isinstance(index_location, Path):
            return IndexFactory.of(str(index_location.absolute()))
        else:
            raise ValueError(
                f"Cannot load index from location {index_location}."
            )

class IndexPerturbation(AbstractPerturbation):
    def __init__(self,
                 index_location: Union[Index, IndexRef, Path, str],
                 tokeniser : Optional[callable] = None,
                 stem : bool = False,
                 ) -> None:
        
        self.index_location = index_location
        self.index = get_index(index_location)
        collection = self.index.getCollectionStatistics()
        lexicon = self.index.getLexicon()
        self.num_docs = collection.getNumberOfDocuments()
        self.avg_doc_len = collection.getAverageDocumentLength()
        self._tokeniser = tokeniser if tokeniser is not None else word_tokenize 
        self._stem = pt.autoclass("org.terrier.terms.PorterStemmer")().stem if stem else lambda x: x

        self.tf = defaultdict(float)
        self.df = defaultdict(float)

        for term, obj in lexicon:
            self.tf[term] = obj.getFrequency()
            self.df[term] = obj.getDocumentFrequency()


    @lru_cache(None)
    def _terms(self, text: str) -> Dict[str, str]:
        terms = {term : self._stem(str(term)) for term in self._tokeniser(text) if term is not None}
        return terms

    @property
    def document_count(self) -> int:
        return self.num_docs
    
    @lru_cache(None)
    def term_frequency(self, text : str, term : str):
        return self.tf.get(term, 0.0) 
    
    @lru_cache(None)
    def document_frequency(self, term : str):
        return self.df.get(term, 0.0)
    
    def inverse_document_frequency(self, term : str):
        return math.log(self.document_count / (1 + self.document_frequency(term)))
    
    def get_terms(self, text : str) -> Sequence[str]:
        return self._terms(text)
    
    def get_counts(self, text : str) -> Dict[str, int]:
        return Counter(self.get_terms(text).values())
    
    def get_tf(self, term : str, text : str) -> int:
        return self.get_counts(text)[term]
    
    def get_tf_text(self, text : str) -> Dict[str, int]:
        return {term : self.get_tf(stemmed, text) for term, stemmed in self.get_terms(text).items()}
    
    def get_idf(self, term : str, text : str) -> float:
        return self.inverse_document_frequency(term)
    
    def get_idf_text(self, text : str) -> Dict[str, float]:
        return {term : self.get_idf(stemmed, text) for term, stemmed in self.get_terms(text).items()}
    
    def get_tfidf(self, term : str, text : str):
        return self.get_tf(term, text) * self.get_idf(term, text)
    
    def get_tfidf_text(self, text : str) -> Dict[str, float]:
        return {term : self.get_tfidf(stemmed, text) for term, stemmed in self.get_terms(text).items()}
    
__all__ = ["IndexPerturbation"]