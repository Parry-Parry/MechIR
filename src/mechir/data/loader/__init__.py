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

class BaseCollator(object):
    tokenizer = None
    transformation_func : callable = None
    q_max_length : int = 30
    d_max_length : int = 200
    special_token : int = "X"

    def __init__(self, tokenizer, transformation_func, q_max_length=30, d_max_length=200, special_token="X") -> None:
        self.tokenizer = tokenizer
        self.transformation_func = transformation_func
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_token = special_token
        self.special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)

    def pad(self, a : str, b : str):
        # turn both sequences into list of tokenized elements 
        a = self.tokenizer.tokenize(a)
        b = self.tokenizer.tokenize(b)  

        return self.tokenizer.decode(self.tokenizer.tokens_to_ids(pad(a, b, self.special_token)))

from .cat import __all__ as cat_all
from .cat import * 
from .dot import __all__ as dot_all
from .dot import *
from .t5 import __all__ as t5_all
from .t5 import *

__all__ = cat_all + dot_all + t5_all + ["pad"]