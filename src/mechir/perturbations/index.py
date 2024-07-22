from collections import defaultdict
import re
from . import AbstractPerturbation


class IndexPerturbation(AbstractPerturbation):
    DEFAULT_TF = 0.
    DEFAULT_IDF = 1.
    def __init__(self, 
                 index_name_or_path : str,
                 default_tf : float = None,
                 default_idf : float = None,
                 stemming : bool = False,
                 preprocessing_function : callable = None,
                 ) -> None:
        import pyterrier as pt
        index = pt.IndexFactory.of(index_name_or_path)
        collection = index.getCollectionStatistics()
        lexicon = index.getLexicon()

        self.num_docs = collection.getNumberOfDocuments()
        self.avg_doc_len = collection.getAverageDocumentLength()

        self.tf = defaultdict(float)
        self.df = defaultdict(float)

        for term, obj in lexicon:
            self.tf[term] = obj.getFrequency()
            self.df[term] = obj.getDocumentFrequency()
        
        stemmer = pt.autoclass("org.terrier.terms.PorterStemmer")().stem 
        stem = lambda term : stemmer(re.sub(r'[^a-zA-Z0-9\s]', '', term).lower()) if stemming else lambda term : term
        if preprocessing_function is not None:
            self.stem = lambda term : preprocessing_function(stem(term))

        self.default_tf = default_tf if default_tf is not None else self.DEFAULT_TF
        self.default_idf = default_idf if default_idf is not None else self.DEFAULT_IDF

    def get_tf(self, term : str) -> float:
        return self.tf.get(term, self.default_tf)

    def get_df(self, term : str) -> float:
        return self.df.get(term, self.default_idf)

    def get_idf(self, term : str) -> float:
        df = self._df(term)
        if df == 0.0: return 0.0
        num = self.num_docs - df + 0.5
        denom = df + 0.5
        return np.log(num / denom)

    def get_tfidf(self, term : str) -> float:
        tf = self.get_tf(term)
        df = self.get_df(term)
        if tf == 0.0 or df == 0.0: return 0.0
        return self.get_tf(term) * self.get_idf(term)


import numpy as np

from utility.funcs import timer

class Lexicon(object):
    DEFAULT_PARAMS = {
        'k1' : 1.2,
        'k3' : 8,
        'b' : 0.75
    }
    def __init__(self, 
                 index, 
                 mode : str = 'BM25',
                 params : dict = {'k1' : 1.2, 'b' : 0.75},
                 default_idf : float = 1.0,
                 default_tf : float = 1.0,
                 **kwargs
                 ) -> None:
        self.index = index
        collection = self.index.getCollectionStatistics()
        lexicon = index.getLexicon()

        self.num_docs = collection.getNumberOfDocuments()
        self.avg_doc_len = collection.getAverageDocumentLength()

        self.tf = defaultdict(float)
        self.df = defaultdict(float)

        for term, obj in lexicon:
            self.tf[term] = obj.getFrequency()
            self.df[term] = obj.getDocumentFrequency()

        self.mode = mode
        self.params = self.DEFAULT_PARAMS 
        self.params.update(params)
        self.default_idf = default_idf
        self.default_tf = default_tf

        self._get_weight = {
            'TF' : self._tf,
            'IDF' : self._idf,
            'TFIDF' : self._tfidf,
            'BM25' : self._bm25
        }[mode]
    
    def _tf(self, term : str, *args) -> float:
        return self.tf[term]
    
    def _df(self, term : str, *args) -> float:
        return self.df[term]
        
    def _idf(self, term : str, *args) -> float:
        df = self._df(term)
        if df == 0.0: return 0.0
        num = self.num_docs - df + 0.5
        denom = df + 0.5
        return np.log(num / denom)

    def _tfidf(self, term : str, *args) -> float:
        tf = self._tf(term)
        df = self._df(term)
        if tf == 0.0 or df == 0.0: return 0.0
        return self._tf(term) * self._idf(term)

    def _bm25(self, 
              term : str,
              doc_len : int) -> float:
        '''
        Java implementation of BM25:
        double K = k_1 * ((1 - b) + b * docLength / averageDocumentLength) + tf;
	    return (tf * (k_3 + 1d) * keyFrequency / ((k_3 + keyFrequency) * K))
	            * WeightingModelLibrary.log((numberOfDocuments - documentFrequency + 0.5d) / (documentFrequency + 0.5d));

        '''
        k1 = self.params['k1']
        k3 = self.params['k3']
        b = self.params['b']
        tf = self._tf(term)
        df = self._df(term)

        if tf==0. or df==0.:
            return 0.0

        K = k1 * ((1 - b) + b * doc_len / self.avg_doc_len) + tf
        return (tf * (k3 + 1) * df / ((k3 + df) * K)) * np.log((self.num_docs - df + 0.5) / (df + 0.5))
    
    def get(self, term : str, doc_len : int) -> float:
        return self._get_weight(term, doc_len)

    def __get__(self, term : str, doc_len : int) -> float:
        return self._get_weight(term, doc_len)

