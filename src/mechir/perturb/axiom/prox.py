from functools import partial
from .. import AbstractPerturbation

class ProximityPerturbation(AbstractPerturbation):
    def __init__(self) -> None:
        super().__init__()
    def apply(self, document : str, query : str) -> str:
        terms = self.get_freq_terms(query if self.target=='query' else document)
        return self._insert_terms(document, terms)