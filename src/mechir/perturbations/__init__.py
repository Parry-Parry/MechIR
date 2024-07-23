from abc import ABC, abstractmethod
from pyterrier.datasets import IRDSDataset as _IRDSDataset
IRDSDataset = _IRDSDataset

class AbstractPerturbation(object, ABC):
    @abstractmethod
    def apply(self, document : str, query : str) -> str:
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def __call__(self, document : str, query : str = None) -> str:
        return self.apply(document, query)

class IdentityPerturbation(AbstractPerturbation):
    def apply(self, document : str, query : str = None) -> str:
        return document
    
from .index import IndexPerturbation