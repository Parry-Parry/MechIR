from typing import List

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch.testing import assert_close
from transformers import AutoTokenizer, AutoModel, ElectraModel

from mechir.modelling.architectures import HookedEncoder

MODEL_NAME = "google/electra-base-discriminator"

def get_embeddings(model):
    try:
        return model.electra.embeddings
    except AttributeError:
        return model.embeddings

@pytest.fixture(scope="module")
def our_electra():
    return HookedEncoder.from_pretrained(MODEL_NAME, device="cpu", hf_model=ElectraModel.from_pretrained(MODEL_NAME))


@pytest.fixture(scope="module")
def huggingface_electra():
    return ElectraModel.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def tokens(tokenizer):
    return tokenizer("The [MASK] sat on the mat", return_tensors="pt")["input_ids"]


def test_full_model(our_electra, huggingface_electra, tokenizer):
    sequences = [
        "Hello, my [MASK] is electra.",
        "I went to the [MASK] to buy some groceries.",
    ]
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    huggingface_electra_logits = huggingface_electra(
        input_ids, attention_mask=attention_mask, output_hidden_states=True
    ).hidden_states[-1]
    our_electra_logits = our_electra(input_ids, attention_mask=attention_mask)
    assert_close(huggingface_electra_logits, our_electra_logits, rtol=1.3e-6, atol=4e-5)


def test_embed_one_prediction(our_electra, huggingface_electra, tokens):
    huggingface_embed = get_embeddings(huggingface_electra)
    our_embed = our_electra.embed

    huggingface_embed_out = huggingface_embed(tokens)[0]
    our_embed_out = our_embed(tokens).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_embed_two_predictions(our_electra, huggingface_electra, tokenizer):
    encoding = tokenizer(
        "Hello, my [MASK] is electra.",
        "I went to the [MASK] to buy some groceries.",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]

    huggingface_embed_out = get_embeddings(huggingface_electra)(
        input_ids, token_type_ids=token_type_ids
    )[0]
    our_embed_out = our_electra.embed(input_ids, token_type_ids=token_type_ids).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_attention(our_electra, huggingface_electra, tokens):
    huggingface_embed = get_embeddings(huggingface_electra)
    huggingface_attn = huggingface_electra.encoder.layer[0].attention

    embed_out = huggingface_embed(tokens)

    our_attn = our_electra.blocks[0].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    huggingface_self_attn_out = huggingface_attn.self(embed_out)[0]
    huggingface_attn_out = huggingface_attn.output.dense(huggingface_self_attn_out)
    assert_close(our_attn_out, huggingface_attn_out)


def test_electra_block(our_electra, huggingface_electra, tokens):
    huggingface_embed = get_embeddings(huggingface_electra)
    huggingface_block = huggingface_electra.encoder.layer[0]

    embed_out = huggingface_embed(tokens)

    our_block = our_electra.blocks[0]

    our_block_out = our_block(embed_out)
    huggingface_block_out = huggingface_block(embed_out)[0]
    assert_close(our_block_out, huggingface_block_out)


def test_run_with_cache(our_electra, tokens):
    _, cache = our_electra.run_with_cache(tokens)

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.7.hook_resid_post" in cache
