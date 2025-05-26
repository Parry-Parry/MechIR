import pytest
import torch
from mechir import Dot
from transformers import AutoTokenizer
from transformer_lens.ActivationCache import ActivationCache


@pytest.fixture(scope="module")
def dot_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return Dot(
        model_name_or_path="bert-base-uncased", pooling_type="mean", return_cache=True
    )


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_forward_pooling(pooling):
    model = Dot("bert-base-uncased", pooling_type=pooling, return_cache=False)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tok("Forward test", return_tensors="pt", padding=True)
    out = model.forward(enc.input_ids, enc.attention_mask)
    assert out.dim() == 2  # (batch, d_model)


@pytest.mark.parametrize("cache_flag", [False, True])
def test_score_cache_flags(dot_model, tokenizer, cache_flag):
    texts = ["A quick test"]
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    q = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
    d = q
    scores, reps_q, reps_d, cache = dot_model.score(q, d, cache=cache_flag)
    assert isinstance(scores, torch.Tensor)
    if cache_flag:
        assert isinstance(cache, ActivationCache)
    else:
        assert cache is None


@pytest.mark.parametrize("patch_type", ["block_all", "head_all", "head_by_pos"])
def test_patch_methods_shapes(dot_model, tokenizer, patch_type):
    tok = tokenizer(["Patch"], return_tensors="pt", padding=True)
    q = {"input_ids": tok.input_ids, "attention_mask": tok.attention_mask}
    d = q
    d_p = q
    if patch_type == "head_by_pos":
        out, _ = dot_model.patch(
            q, d, d_p, patch_type=patch_type, layer_head_list=[(0, 0)]
        )
        assert out.shape[0] == 2
    else:
        out, _ = dot_model.patch(q, d, d_p, patch_type=patch_type)
        assert out.dim() >= 2
