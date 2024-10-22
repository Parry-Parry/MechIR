def pad(a : list, b : list, tok : str):
    assert type(a) == type(b) == list, "Both a and b must be lists"

    padded = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            padded.append(a[i])
            i += 1
            j += 1
        else:
            padded.append(tok)
            j += 1
    
    while j < len(b):
        padded.append(tok)
        j += 1
    
    return padded

from .cat import __all__ as cat_all
from .cat import * 
from .dot import __all__ as dot_all
from .dot import *
from .t5 import __all__ as t5_all
from .t5 import *

__all__ = cat_all + dot_all + t5_all + ["pad"]