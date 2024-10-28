from . import AbstractPerturbation
from typing import Union, Optional, Sequence, Dict
from pathlib import Path
from collections import Counter
import pyterrier as pt 
if not pt.started():
    pt.init()
from functools import lru_cache
from collections import defaultdict

StringReader = pt.autoclass("java.io.StringReader")
RequestContextMatching = pt.autoclass("org.terrier.python.RequestContextMatching")
Index = pt.autoclass("org.terrier.structures.Index")
IndexRef = pt.autoclass('org.terrier.querying.IndexRef')
IndexFactory = pt.autoclass('org.terrier.structures.IndexFactory')
PropertiesIndex = pt.autoclass("org.terrier.structures.PropertiesIndex")
MetaIndex = pt.autoclass("org.terrier.structures.MetaIndex")
Lexicon = pt.autoclass("org.terrier.structures.Lexicon")
CollectionStatistics = pt.autoclass("org.terrier.structures.CollectionStatistics")
Tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser")
EnglishTokeniser = pt.autoclass(
    "org.terrier.indexing.tokenisation.EnglishTokeniser"
)
WeightingModel = pt.autoclass("org.terrier.matching.models.WeightingModel")
TfModel = pt.autoclass("org.terrier.matching.models.Tf")
TfIdfModel = pt.autoclass("org.terrier.matching.models.TF_IDF")
BM25Model = pt.autoclass("org.terrier.matching.models.BM25")
PL2Model = pt.autoclass("org.terrier.matching.models.PL2")
DirichletLMModel = pt.autoclass("org.terrier.matching.models.DirichletLM")
TermPipelineAccessor = pt.autoclass("org.terrier.terms.TermPipelineAccessor")
BaseTermPipelineAccessor = pt.autoclass(
    "org.terrier.terms.BaseTermPipelineAccessor"
)
SearchRequest = pt.autoclass('org.terrier.querying.SearchRequest')
ScoredDoc = pt.autoclass('org.terrier.querying.ScoredDoc')
ScoredDocList = pt.autoclass('org.terrier.querying.ScoredDocList')
Manager = pt.autoclass('org.terrier.querying.Manager')
ManagerFactory = pt.autoclass('org.terrier.querying.ManagerFactory')
ApplicationSetup = pt.autoclass('org.terrier.utility.ApplicationSetup')

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
                 tokeniser : Optional[Tokeniser] = None,
                 stem : bool = False,
                 ) -> None:
        
        self.index_location = index_location
        index = get_index(index_location)
        collection = self.index.getCollectionStatistics()
        lexicon = index.getLexicon()
        self.num_docs = collection.getNumberOfDocuments()
        self.avg_doc_len = collection.getAverageDocumentLength()
        self._tokeniser = tokeniser if tokeniser is not None else pt.autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")()
        self._stem = pt.autoclass("org.terrier.terms.PorterStemmer")().stem if stem else lambda x: x

        self.tf = defaultdict(float)
        self.df = defaultdict(float)

        for term, obj in lexicon:
            self.tf[term] = obj.getFrequency()
            self.df[term] = obj.getDocumentFrequency()


    @lru_cache(None)
    def _terms(self, text: str) -> Sequence[str]:
        reader = StringReader(text)
        terms = tuple(
            self.stem(str(term))
            for term in self._tokeniser.tokenise(reader)
            if term is not None
        )
        del reader
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
        return Counter(self.get_terms(text))
    
    def get_tf(self, term : str, text : str) -> int:
        return self.get_counts(text)[term]
    
    def get_tf_text(self, text : str) -> Dict[str, int]:
        return {term : self.get_tf(term, text) for term in self.get_terms(text)}
    
    def get_idf(self, term : str, text : str) -> float:
        return self.inverse_document_frequency(term)
    
    def get_idf_text(self, text : str) -> Dict[str, float]:
        return {term : self.get_idf(term, text) for term in self.get_terms(text)}
    
    def get_tfidf(self, term : str, text : str):
        return self.get_tf(term, text) * self.get_idf(term, text)
    
    def get_tfidf_text(self, text : str) -> Dict[str, float]:
        return {term : self.get_tfidf(term, text) for term in self.get_terms(text)}
    
__all__ = ["IndexPerturbation"]