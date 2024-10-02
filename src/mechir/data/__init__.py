from ..util import is_ir_datasets_availible
from transformers.utils import _LazyModule, OptionalDependencyNotAvailable
from .dataset import __all__ as _core_all
from .loader import __all__ as _loader_all
from typing import TYPE_CHECKING

_import_structure = {
    'dataset' : [*_core_all],
    'loader' : [*_loader_all],
}
try:
    if not is_ir_datasets_availible():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from .ir_dataset import __all__ as _ir_dataset_all
    _import_structure["dataset"] += _ir_dataset_all


if TYPE_CHECKING:
    from .dataset import MechDataset
    from .loader import *

    try:
        if not is_ir_datasets_availible():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .ir_dataset import MechIRDataset
else:
    import sys 
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)