__version__ = "0.0.1"

CONFIG = {
    'ignore-official' : False,
}

from .modelling import (
    Cat,
    Dot,
    MonoT5,
    PatchedModel
)