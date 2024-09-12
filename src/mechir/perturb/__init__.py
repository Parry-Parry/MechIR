from abc import ABC, abstractmethod
from functools import wraps
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
    
def perturbation(f):
    # check how many arguments the function has
    argcount = f.__code__.co_argcount

    class CustomPerturbation(AbstractPerturbation):
        def apply(self, document: str, query: str = None) -> str:
            return f(document, query) if argcount > 1 else f(document)

    instance = CustomPerturbation()
    return instance
    
from .index import IndexPerturbation