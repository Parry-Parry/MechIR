import pandas as pd
import torch
from torch.testing import assert_close

from transformers import AutoTokenizer, AutoModel

from transformer_lens import HookedEncoder

def test_files(fname1, fname2):

    # Test corpus and query embeddings
    df1 = pd.read_csv(fname1)
    df2 = pd.read_csv(fname2)

    t1 = torch.tensor(df1.values, dtype=torch.float32)
    t2 = torch.tensor(df2.values, dtype=torch.float32)

    assert_close(t1, t2, rtol=1.3e-6, atol=4e-5)


def test_embed(tl_model, hf_model, test_string, tokenizer):
    encoding = tokenizer(test_string, return_tensors="pt")
    input_ids = encoding["input_ids"]

    hf_embed_out = hf_model.embeddings(input_ids)[0]
    tl_embed_out = tl_model.embed(input_ids).squeeze(0)

    assert_close(hf_embed_out, tl_embed_out, rtol=1.3e-6, atol=4e-5)


def test_full_model(tl_model, hf_model, tokenizer, test_strings):

    tokenized = tokenizer(test_strings, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    hf_out = hf_model(input_ids, attention_mask=attention_mask)[0]
    print("hf out shape:", hf_out.shape)

    tl_out = tl_model(input_ids, return_type="embeddings", one_zero_attention_mask=attention_mask)

    assert_close(hf_out, tl_out, rtol=1.3e-6, atol=4e-5)


def test_run_with_cache(hf_model, tl_model, tokenizer, test_string):
    input_tokens = tokenizer(test_string, return_tensors="pt", padding=True)
    attn_mask = input_tokens["attention_mask"]

    tl_embeddings, cache = tl_model.run_with_cache(
        input_tokens["input_ids"], 
        return_type="embeddings",
        one_zero_attention_mask=attn_mask
    )

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.5.hook_resid_post" in cache

    # check embeddings match HF implementation
    hf_embeddings = hf_model(input_tokens["input_ids"], attention_mask=attn_mask)[0]
    assert_close(hf_embeddings, tl_embeddings, rtol=1.3e-6, atol=4e-5)



if __name__ == "__main__":
    torch.set_grad_enabled(False)

    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    hf_model = AutoModel.from_pretrained(pre_trained_model_name)
    tl_model = HookedEncoder.from_pretrained(pre_trained_model_name, hf_model=hf_model)

    test_embed(tl_model, hf_model, "what is the weather in providence, ri?", tokenizer)

    test_sequences = [
        "Hello, world!",
        # "this is another sequence of tokens",
    ]
    test_full_model(tl_model, hf_model, tokenizer, test_sequences)

    # test corpus embeddings
    # hf_corpus_fname = "scifact_hf_tasb_corpus_embeddings.csv"
    # tl_corpus_fname = "scifact_tl_tasb_corpus_embeddings.csv"
    # test_files(hf_corpus_fname, tl_corpus_fname)

    # # test query embeddings
    # hf_query_fname = "scifact_hf_tasb_query_embeddings.csv"
    # tl_query_fname = "scifact_tl_tasb_query_embeddings.csv"
    # test_files(hf_query_fname, tl_query_fname)

    # test run with cache
    test_run_with_cache(hf_model, tl_model, tokenizer, "hello world!")
