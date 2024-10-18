import torch
from torch import Tensor
from types import SimpleNamespace
import json

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
    
def activation_cache_to_disk(activation_cache, path):
    cache_dict = activation_cache.cache_dict
    has_batch_dim = activation_cache.has_batch_dim

    cache_dict = {k: v.cpu().numpy().tolist() for k,v in cache_dict.items()}
    out = {
        "cache_dict": cache_dict,
        "has_batch_dim": has_batch_dim,
    }
    with open(path, "w") as f:
        json.dump(out, f)

def disk_to_activation_cache(path, model):
    from transformer_lens import ActivationCache
    with open(path, "r") as f:
        data = json.load(f)
    cache_dict = {k: torch.tensor(v) for k,v in data["cache_dict"].items()}
    has_batch_dim = data["has_batch_dim"]
    return ActivationCache(cache_dict, model, has_batch_dim)