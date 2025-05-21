import pytest
import torch
from mechir import Cat
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def cat_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return Cat(
        model_name_or_path="bert-base-uncased",
        tokenizer=tokenizer,
        softmax_output=True,
        return_cache=True,
    )


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.mark.parametrize("text,label", [("Good day", 0), ("Bad day", 1)])
def test_forward_and_softmax(cat_model, tokenizer, text, label):
    enc = tokenizer(text, return_tensors="pt", padding=True)
    logits = cat_model.forward(enc.input_ids, enc.attention_mask)
    # softmax_output=False returns raw logits
    assert logits.dim() == 1


@pytest.mark.parametrize("patch_type", ["block_all", "head_all", "head_by_pos"])
def test_patch_methods_shapes(cat_model, tokenizer, patch_type):
    text = "Patch test"
    enc = tokenizer(text, return_tensors="pt", padding=True)
    seqs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
    # Prepare positive and perturbed
    seqs_p = seqs
    scores, cache = cat_model.score(seqs, cache=True)
    # call patch
    if patch_type == "head_by_pos":
        layer_head_list = [(0, 0)]
        out = cat_model.patch(
            seqs, seqs_p, patch_type=patch_type, layer_head_list=layer_head_list
        )
        assert out.shape[0] == 2  # components
    else:
        out = cat_model.patch(seqs, seqs_p, patch_type=patch_type)
        assert out.ndim >= 2


def test_score_without_cache(cat_model, tokenizer):
    text = "Simple test"
    enc = tokenizer(text, return_tensors="pt", padding=True)
    logits, cache = cat_model.score(
        {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}, cache=False
    )
    assert cache is None
