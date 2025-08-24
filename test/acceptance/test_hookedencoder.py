from typing import List

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch.testing import assert_close
from transformers import AutoTokenizer, BertForPreTraining

from mechir.modelling.architectures import HookedEncoder

MODEL_NAME = "bert-base-cased"

def get_embeddings(model):
    try:
        return model.bert.embeddings
    except AttributeError:
        return model.embeddings

@pytest.fixture(scope="module")
def our_bert():
    return HookedEncoder.from_pretrained(MODEL_NAME, device="cpu", hf_model=BertForPreTraining.from_pretrained(MODEL_NAME))


@pytest.fixture(scope="module")
def huggingface_bert():
    return BertForPreTraining.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def tokens(tokenizer):
    return tokenizer("The [MASK] sat on the mat", return_tensors="pt")["input_ids"]


def test_full_model(our_bert, huggingface_bert, tokenizer):
    sequences = [
        "Hello, my [MASK] is Bert.",
        "I went to the [MASK] to buy some groceries.",
    ]
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    huggingface_bert_logits = huggingface_bert(
        input_ids, attention_mask=attention_mask, output_hidden_states=True
    ).hidden_states[-1]
    our_bert_logits = our_bert(input_ids, attention_mask=attention_mask)
    assert_close(huggingface_bert_logits, our_bert_logits, rtol=1.3e-6, atol=4e-5)


def test_embed_one_prediction(our_bert, huggingface_bert, tokens):
    huggingface_embed = get_embeddings(huggingface_bert)
    our_embed = our_bert.embed

    huggingface_embed_out = huggingface_embed(tokens)[0]
    our_embed_out = our_embed(tokens).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_embed_two_predictions(our_bert, huggingface_bert, tokenizer):
    encoding = tokenizer(
        "Hello, my [MASK] is Bert.",
        "I went to the [MASK] to buy some groceries.",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]

    huggingface_embed_out = get_embeddings(huggingface_bert)(
        input_ids, token_type_ids=token_type_ids
    )[0]
    our_embed_out = our_bert.embed(input_ids, token_type_ids=token_type_ids).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_attention(our_bert, huggingface_bert, tokens):
    huggingface_embed = get_embeddings(huggingface_bert)
    huggingface_attn = huggingface_bert.bert.encoder.layer[0].attention

    embed_out = huggingface_embed(tokens)

    our_attn = our_bert.blocks[0].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    huggingface_self_attn_out = huggingface_attn.self(embed_out)[0]
    huggingface_attn_out = huggingface_attn.output.dense(huggingface_self_attn_out)
    assert_close(our_attn_out, huggingface_attn_out)


def test_bert_block(our_bert, huggingface_bert, tokens):
    huggingface_embed = get_embeddings(huggingface_bert)
    huggingface_block = huggingface_bert.bert.encoder.layer[0]

    embed_out = huggingface_embed(tokens)

    our_block = our_bert.blocks[0]

    our_block_out = our_block(embed_out)
    huggingface_block_out = huggingface_block(embed_out)[0]
    assert_close(our_block_out, huggingface_block_out)


def test_run_with_cache(our_bert, tokens):
    _, cache = our_bert.run_with_cache(tokens)

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.7.hook_resid_post" in cache


def test_from_pretrained_revision():
    """
    Check that the from_pretrained parameter `revision` (= git version) works
    """

    _ = HookedEncoder.from_pretrained(MODEL_NAME, revision="main")

    try:
        _ = HookedEncoder.from_pretrained(MODEL_NAME, revision="inexistent_branch_name")
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


@pytest.mark.skipif(
    torch.backends.mps.is_available() or not torch.cuda.is_available(),
    reason="bfloat16 unsupported by MPS: https://github.com/pytorch/pytorch/issues/78168 or no GPU",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision(dtype):
    """Check the 16 bits loading and inferences."""
    model = HookedEncoder.from_pretrained(MODEL_NAME, torch_dtype=dtype, hf_model=BertForPreTraining.from_pretrained(MODEL_NAME))
    assert model.W_K.dtype == dtype

    _ = model(model.tokenizer("Hello, world", return_tensors="pt")["input_ids"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_cuda(mlm_tokens):
    model = HookedEncoder.from_pretrained(MODEL_NAME)
    model(mlm_tokens)
