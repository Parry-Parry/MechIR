import re
import shutil 
import os
from time import time
from functools import wraps
import itertools
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

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

def load_monot5(checkpoint : str ='castorini/monot5-base-msmarco', batch_size : int = 64, **kwargs):
    from pyterrier_t5 import MonoT5ReRanker 

    return MonoT5ReRanker(model=checkpoint, batch_size=batch_size)

def bm25(index_dir : str, k1 : float = 1.2, b : float = 0.75, threads : int = 1, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyterrier_pisa import PisaIndex
    return PisaIndex(index_dir, threads=threads, **kwargs).bm25(k1=k1, b=b)

def pt_index(index : str, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    return pt.IterDictIndexer(index, **kwargs)

def pisa_index(index : str, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyterrier_pisa import PisaIndex
    return PisaIndex(index, text_field='text', **kwargs)

def load_dot_rerank(checkpoint : str ='sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size : int = 64, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from transformers import AutoModel, AutoTokenizer
    from pyterrier_dr import HgfBiEncoder

    model = AutoModel.from_pretrained(checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device, batch_size=batch_size)
    return backbone

def load_dot_dense(index_path : str, checkpoint : str ='sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size : int = 64, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyterrier_dr import NumpyIndex

    backbone = load_dot_rerank(checkpoint, batch_size)
    index = NumpyIndex(index_path)
    return backbone >> index

def load_electra(checkpoint : str ='crystina-z/monoELECTRA_LCE_nneg31', batch_size : int = 64, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyterrier_dr import ElectraScorer
    return ElectraScorer(model_name=checkpoint, batch_size=batch_size)

def load_splade(index_path : str, checkpoint : str = 'naver/splade-cocondenser-ensembledistil', batch_size : int = 128, index : str = 'msmarco_passage', **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyt_splade import SpladeFactory
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index_path, **kwargs).quantized()
    splade = SpladeFactory(model=checkpoint)
    return splade.query_encoder(batch_size=batch_size) >> index

def load_splade_indexer(index_path : str, checkpoint : str = 'naver/splade-cocondenser-ensembledistil', batch_size : int = 128, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyt_splade import SpladeFactory
    from pyterrier_pisa import PisaIndex

    print(checkpoint)

    index = PisaIndex(index_path, stemmer='none', **kwargs).toks_indexer()
    splade = SpladeFactory(model=checkpoint)    
    return splade.doc_encoder(batch_size=batch_size) >> index

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def cut_prompt(output : str, input : str) -> str:
    return output[len(input):]

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