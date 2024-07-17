import re
import shutil 
import os
from time import time
from functools import wraps
import torch
from torch import Tensor
import itertools
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
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

def sample(dataset : str, out_file : str, subset : int = 100000):
    import pandas as pd
    import ir_datasets as irds
    dataset = irds.load(dataset)
    assert dataset.has_docpairs(), "Dataset must have docpairs! Make sure you're not using a test collection"
    df = pd.DataFrame(dataset.docpairs_iter())
    assert len(df) > subset, "Subset must be smaller than the dataset!"
    df = df.sample(n=subset) 
    df.to_csv(out_file, sep='\t', index=False)
    return f"Successfully took subset of {dataset} of size {subset} and saved to {out_file}"

def index_from_pt(index : str, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    return pt.IterDictIndexer(index, **kwargs)

def index_from_pisa(index : str, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyterrier_pisa import PisaIndex
    return PisaIndex(index, text_field='text', **kwargs)

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def concatenate(*lists) -> list:
    return itertools.chain.from_iterable(lists)

def copy_path(path : str, root : str = '/tmp') -> str:
    base = os.path.basename(path)
    new_dir = os.path.join(root, base)
    if not os.path.isdir(new_dir):
        new_dir = shutil.copytree(path, os.path.join(root, base))
    return new_dir

def load_yaml(path : str) -> dict:
    return load(open(path), Loader=Loader)

def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap