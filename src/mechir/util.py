import torch
from torch import Tensor
from types import SimpleNamespace

class PatchingOutput(SimpleNamespace):
    result: Tensor
    scores : Tensor
    scores_p : Tensor

def batched_dot_product(a: Tensor, b: Tensor):
    """
    Calculating the dot product between two tensors a and b.

    Parameters
    ----------
    a: torch.Tensor
        size: batch_size x 1 x vector_dim
    b: torch.Tensor
        size: batch_size x group_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size x group_size)
        dot product for each group of vectors
    """
    if len(b.shape) == 2:
        return torch.matmul(a, b.transpose(0, 1))
    return torch.bmm(a,torch.permute(b,[0,2,1])).squeeze(1)

def linear_rank_function(patch_score : Tensor, score : Tensor, score_p : Tensor):
    return (patch_score - score) / (score_p - score)

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def is_pyterrier_availible():
    try:
        import pyterrier as pt
        return True
    except ImportError:
        return False

def is_ir_axioms_availible():
    try:
        import ir_axioms
        return True
    except ImportError:
        return False

def is_ir_datasets_availible():
    try:
        import ir_datasets
        return True
    except ImportError:
        return False