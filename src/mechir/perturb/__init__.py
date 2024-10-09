# __init__.py

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from ..util import is_ir_axioms_availible
from transformers.utils import _LazyModule, OptionalDependencyNotAvailable

class AbstractPerturbation(ABC):
    @abstractmethod
    def apply(self, document: str, query: str) -> str:
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def __call__(self, document: str, query: str = None) -> str:
        return self.apply(document, query)

class IdentityPerturbation(AbstractPerturbation):
    def apply(self, document: str, query: str = None) -> str:
        return document

def perturbation(f):
    argcount = f.__code__.co_argcount
    class CustomPerturbation(AbstractPerturbation):
        def apply(self, document: str, query: str = None) -> str:
            return f(document, query) if argcount > 1 else f(document)
    return CustomPerturbation()

# Define these classes and function in the current namespace
__all__ = ["AbstractPerturbation", "IdentityPerturbation", "perturbation"]

_import_structure = {
    "IRDSDataset": ["IRDSDataset"],
}

try:
    if not is_ir_axioms_availible():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from .index import __all__ as _index_all
    from .axiom import __all__ as _axiom_all
    _import_structure["index"] = _index_all
    _import_structure["axiom"] = _axiom_all

if TYPE_CHECKING:
    try:
        if not is_ir_axioms_availible():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .index import *
        from .axiom import *
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

# Update __all__ to include both directly defined and lazily loaded items
__all__ += list(_import_structure.keys())