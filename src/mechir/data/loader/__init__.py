def pad(a : str, b : str, tok : str):
    delta = len(b) - len(a)
    if delta > 0:
        a += f'{ tok}'*delta
    elif delta < 0:
        b += f'{ tok}'*abs(delta)
    return a, b

from .cat import __all__ as cat_all
from .cat import * 
from .dot import __all__ as dot_all
from .dot import *
from .t5 import __all__ as t5_all
from .t5 import *

__all__ = cat_all + dot_all + t5_all