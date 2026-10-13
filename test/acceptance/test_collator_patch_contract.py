"""The collators and `patch` must compose.

`model.patch(**batch)` is the workflow the README documents, and it raised
TypeError for every model: the collators emit "perturbed_documents" /
"perturbed_sequences" while patch declared "documents_p" / "sequences_p".
"""

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from mechir import Cat, Dot
from mechir.data import CatDataCollator, DotDataCollator, MechIRDataset
from mechir.perturb import perturbation

BI_ENCODER = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@perturbation
def append_marker(document, query=None):
    return f"{document} MECHIR"


@pytest.fixture(scope="module")
def dataset():
    import ir_datasets

    pairs = pd.DataFrame(ir_datasets.load("vaswani").qrels_iter()).head(2)
    return MechIRDataset("vaswani", pairs=pairs)


@pytest.fixture(scope="module")
def dot_model():
    return Dot(BI_ENCODER, pooling_type="cls")


@pytest.fixture(scope="module")
def cat_model():
    return Cat(CROSS_ENCODER)


def batch_for(model, dataset, collator_cls, **kwargs):
    collate = collator_cls(
        model.tokenizer,
        transformation_func=append_marker,
        q_max_length=16,
        d_max_length=48,
        **kwargs,
    )
    loader = DataLoader(dataset, collate_fn=collate, batch_size=2)
    return next(iter(loader))


# -- the contract ----------------------------------------------------------


def test_dot_collator_batch_unpacks_into_patch(dot_model, dataset):
    """Regression: this raised
    `Dot.patch() got an unexpected keyword argument 'perturbed_documents'`."""
    batch = batch_for(dot_model, dataset, DotDataCollator)

    result, _ = dot_model.patch(**batch, patch_type="head_all")

    assert torch.isfinite(result).all()


def test_cat_collator_batch_unpacks_into_patch(cat_model, dataset):
    batch = batch_for(cat_model, dataset, CatDataCollator)

    result, _ = cat_model.patch(**batch, patch_type="head_all")

    assert torch.isfinite(result).all()


# -- the old names must keep working --------------------------------------


def test_dot_positional_documents_p_still_works(dot_model, dataset):
    batch = batch_for(dot_model, dataset, DotDataCollator)

    result, _ = dot_model.patch(
        batch["queries"],
        batch["documents"],
        batch["perturbed_documents"],
        patch_type="head_all",
    )

    assert torch.isfinite(result).all()


def test_dot_documents_p_keyword_still_works(dot_model, dataset):
    batch = batch_for(dot_model, dataset, DotDataCollator)

    result, _ = dot_model.patch(
        queries=batch["queries"],
        documents=batch["documents"],
        documents_p=batch["perturbed_documents"],
        patch_type="head_all",
    )

    assert torch.isfinite(result).all()


def test_both_spellings_agree(dot_model, dataset):
    batch = batch_for(dot_model, dataset, DotDataCollator)

    via_alias, _ = dot_model.patch(**batch, patch_type="head_all")
    via_canonical, _ = dot_model.patch(
        batch["queries"],
        batch["documents"],
        documents_p=batch["perturbed_documents"],
        patch_type="head_all",
    )

    assert torch.equal(via_alias, via_canonical)


def test_cat_positional_sequences_p_still_works(cat_model, dataset):
    batch = batch_for(cat_model, dataset, CatDataCollator)

    result, _ = cat_model.patch(
        batch["sequences"], batch["perturbed_sequences"], patch_type="head_all"
    )

    assert torch.isfinite(result).all()


# -- omitting it must fail clearly ----------------------------------------


def test_dot_missing_perturbed_documents_raises_clearly(dot_model, dataset):
    batch = batch_for(dot_model, dataset, DotDataCollator)

    with pytest.raises(TypeError, match="documents_p"):
        dot_model.patch(batch["queries"], batch["documents"], patch_type="head_all")


def test_cat_missing_perturbed_sequences_raises_clearly(cat_model, dataset):
    batch = batch_for(cat_model, dataset, CatDataCollator)

    with pytest.raises(TypeError, match="sequences_p"):
        cat_model.patch(batch["sequences"], patch_type="head_all")
