import torch
from torch import Tensor


def batched_dot_product(a: Tensor, b: Tensor):
    """
    Calculating the dot product between two tensors a and b.

    Parameters
    ----------
    a: torch.Tensor
        size: batch_size x vector_dim
    b: torch.Tensor
        size: batch_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size)
        dot product
    """
    return (a * b).sum(dim=-1)


def linear_rank_function(patch_score: Tensor, score: Tensor, score_p: Tensor):
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


def is_sae_lens_availible():
    try:
        import sae_lens

        return True
    except ImportError:
        return False


def load_json(file: str):
    import json
    import gzip

    """
    Load a JSON or JSONL (optionally compressed with gzip) file.

    Parameters:
    file (str): The path to the file to load.

    Returns:
    dict or list: The loaded JSON content. Returns a list for JSONL files, 
                  and a dict for JSON files.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "r") as f:
            return json.load(f)
    elif file.endswith(".jsonl"):
        with open(file, "r") as f:
            return [json.loads(line) for line in f]
    elif file.endswith(".json.gz"):
        with gzip.open(file, "rt") as f:
            return json.load(f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "rt") as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown file type for {file}")


def save_json(data, file: str):
    import json
    import gzip

    """
    Save data to a JSON or JSONL file (optionally compressed with gzip).

    Parameters:
    data (dict or list): The data to save. Must be a list for JSONL files.
    file (str): The path to the file to save.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "w") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl"):
        with open(file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif file.endswith(".json.gz"):
        with gzip.open(file, "wt") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "wt") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError(f"Unknown file type for {file}")


def activation_cache_to_disk(activation_cache, path):
    cache_dict = activation_cache.cache_dict
    has_batch_dim = activation_cache.has_batch_dim

    cache_dict = {k: v.cpu().numpy().tolist() for k, v in cache_dict.items()}
    out = {
        "cache_dict": cache_dict,
        "has_batch_dim": has_batch_dim,
    }
    save_json(out, path)


def disk_to_activation_cache(path, model):
    from transformer_lens import ActivationCache

    data = load_json(path)
    cache_dict = {k: torch.tensor(v) for k, v in data["cache_dict"].items()}
    has_batch_dim = data["has_batch_dim"]
    return ActivationCache(cache_dict, model, has_batch_dim)
