"""Activation-patch a cross-encoder over an ir-datasets collection.

Requires the examples extra:

    pip install "mechir[examples]"   # from a checkout: uv sync --extra examples

Example:

    python examples/cat.py \
        --model_name_or_path cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --dataset vaswani \
        --out_path patch.json \
        --patch_type head_all
"""

import json
from typing import List, Optional, Sequence, Tuple

from fire import Fire
from torch.utils.data import DataLoader

from mechir import Cat
from mechir.data import CatDataCollator, MechIRDataset
from mechir.perturb import perturbation


@perturbation
def append_term(text: str, query: str = None) -> str:
    """Append a term to the document, the simplest useful perturbation."""
    return text + " Apple"


def patch_cat(
    model_name_or_path: str,
    out_path: str,
    dataset: str,
    num_labels: int = 2,
    batch_size: int = 1,
    patch_type: str = "block_all",
    layer_head_list: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    """Patch every component of a cross-encoder and dump the resulting map.

    Args:
        model_name_or_path: A ``*ForSequenceClassification`` checkpoint.
        out_path: JSON file to write the scores and patch maps to.
        dataset: An ir-datasets identifier, e.g. ``vaswani``.
        num_labels: Number of classification labels on the ranking head.
        batch_size: Query/document pairs per forward pass.
        patch_type: One of ``block_all``, ``head_all`` or ``head_by_pos``.
        layer_head_list: ``(layer, head)`` pairs, required by ``head_by_pos``.
    """
    if patch_type == "head_by_pos" and not layer_head_list:
        raise ValueError(
            "head_by_pos patching needs layer_head_list, e.g. --layer_head_list "
            "'[(0, 0)]'"
        )
    pairs: List[Tuple[int, int]] = list(layer_head_list or [])

    model = Cat(model_name_or_path, num_labels=num_labels)
    collator = CatDataCollator(model.tokenizer, transformation_func=append_term)
    loader = DataLoader(
        MechIRDataset(dataset), batch_size=batch_size, collate_fn=collator
    )

    result = {"scores": [], "scores_p": [], "patch_output": []}

    for batch in loader:
        # patch() scores both inputs internally but does not return the
        # scores, so they are taken separately to keep them in the dump.
        scores, _ = model.score(batch["sequences"])
        scores_p, _ = model.score(batch["perturbed_sequences"])
        # patch() returns (result, cache); cache is None unless the model was
        # built with return_cache=True.
        patched, _ = model.patch(**batch, patch_type=patch_type, layer_head_list=pairs)

        result["scores"].append(scores.cpu().tolist())
        result["scores_p"].append(scores_p.cpu().tolist())
        result["patch_output"].append(patched.cpu().tolist())

    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    Fire(patch_cat)
