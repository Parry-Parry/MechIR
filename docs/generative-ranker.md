# `GenerativeRanker`

`mechir.GenerativeRanker` -- a decoder-backbone reranker, scored by
contrasting two label-token logits at the last non-padding position. Source:
`src/mechir/modelling/generative.py`.

```python
from mechir import GenerativeRanker

model = GenerativeRanker(
    "Qwen/Qwen3-Reranker-0.6B",
    pos_token="yes",
    neg_token="no",
    tokenizer=None,
    softmax_output=False,
    return_cache=False,
)
```

The checkpoint must be a decoder (one of `backbones.DECODER_ARCHITECTURES`);
anything else raises `ValueError` pointing at `Cat` or `MonoT5`. Both label
tokens must encode to exactly one vocabulary id.

There is no dedicated collator: build the prompt yourself and pass
`{"input_ids", "attention_mask"}` dicts.

## `forward`

```python
scores = model.forward(input_ids, attention_mask, token_type_ids=None)
```

Returns `(batch,)`. The label is read at the last non-padding position --
`attention_mask.sum(dim=1) - 1`, which with right padding differs per row --
and reduced to `logit(pos) - logit(neg)`, or to
`softmax([pos, neg])[:, 0]` in `[0, 1]` when `softmax_output=True`.
`token_type_ids` is accepted for signature compatibility and ignored.

## `score`

```python
scores, cache = model.score(sequences, cache=False, cache_as_dict=False)
```

Element | Shape / type
--- | ---
`scores` | `(batch,)`
`cache` | `ActivationCache` when `cache=True` (activation `dict` if `cache_as_dict=True`), otherwise `None`

## `patch`

```python
result, cache = model.patch(
    sequences,
    sequences_p,                 # required positionally; no alias kwarg
    patch_type="block_all",      # "block_all" | "head_all" | "head_by_pos"
    layer_head_list=[],          # required by "head_by_pos"
    patching_metric=mechir.util.linear_rank_function,
)
```

Always returns a 2-tuple. `cache` is the perturbed run's `ActivationCache`
when the model was built with `return_cache=True`, and `None` otherwise.
Unlike `Cat` and `Dot`, there is no `perturbed_sequences` alias.

`patch_type` | `result` shape | Axis 0
--- | --- | ---
`"block_all"` | `(3, n_layers, seq)` | `resid_pre`, `attn_out`, `mlp_out`
`"head_all"` | `(n_layers, n_heads)` | --
`"head_by_pos"` | `(2, len(layer_head_list), seq)` | `z`, `pattern`

`seq` is the padded prompt length, `sequences["input_ids"].shape[1]`. The
`"head_by_pos"` shape is the one verified directly by
`test/acceptance/test_qwen_rankers.py`, which asserts `(2, 1, 24)` for one
`(layer, head)` pair and 24 tokens.

Each entry is the patching metric averaged over the batch -- see
[the shape conventions](README.md#what-the-entries-mean).

Note on cost: these backbones are large and the loops are not. `"head_all"`
costs `n_layers * n_heads` forward passes per batch, which on a decoder
backbone runs into the hundreds.
