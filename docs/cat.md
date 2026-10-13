# `Cat`

`mechir.Cat` -- a cross-encoder: query and document are encoded jointly and
scored by a classification head. Source: `src/mechir/modelling/cat.py`.

```python
from mechir import Cat

model = Cat(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    num_labels=2,
    tokenizer=None,         # defaults to AutoTokenizer for the checkpoint
    softmax_output=False,
    return_cache=False,
)
```

The checkpoint must be a `*ForSequenceClassification` model. If it is not,
`AutoModelForSequenceClassification` attaches a randomly initialised head and
the scores are meaningless; the constructor logs a warning saying so rather
than raising.

Pair it with `mechir.data.CatDataCollator`, which emits
`{"sequences", "perturbed_sequences"}`. Both carry `token_type_ids`, which
`Cat` passes through every forward pass -- dropping them would score a
different input.

## `forward`

```python
scores = model.forward(input_ids, attention_mask, token_type_ids=None)
```

Returns `(batch,)`: the classification logits reduced to their **first**
column (`logits[:, 0]`), or `log_softmax(logits, dim=-1)[:, 0]` when
`softmax_output=True`. Note that `softmax_output` gives a log-probability,
not a probability, and that index 0 is used regardless of `num_labels`.

## `score`

```python
scores, cache = model.score(sequences, cache=False, cache_as_dict=False)
```

Element | Shape / type
--- | ---
`scores` | `(batch,)`
`cache` | `ActivationCache` when `cache=True`, otherwise `None`

## `patch`

```python
result, cache = model.patch(
    sequences,
    sequences_p=None,            # alias: perturbed_sequences
    patch_type="block_all",      # "block_all" | "head_all" | "head_by_pos"
    layer_head_list=[],          # required by "head_by_pos"
    patching_metric=mechir.util.linear_rank_function,
)
```

Always returns a 2-tuple. `cache` is the `ActivationCache` of the perturbed
run when the model was built with `return_cache=True`, and `None` otherwise.
`sequences_p` may be passed as `perturbed_sequences`, so
`model.patch(**batch, patch_type=...)` works directly on a `CatDataCollator`
batch.

`patch_type` | `result` shape | Axis 0
--- | --- | ---
`"block_all"` | `(3, n_layers, seq)` | `resid_pre`, `attn_out`, `mlp_out`
`"head_all"` | `(n_layers, n_heads)` | --
`"head_by_pos"` | `(2, len(layer_head_list), seq)` | `z`, `pattern`

`seq` is the padded joint sequence length,
`sequences["input_ids"].shape[1]` (330 with `CatDataCollator` defaults:
`q_max_length=30` plus `d_max_length=300`). Positions therefore cover the
query *and* the document, so a position-resolved map includes query tokens.

Each entry is the patching metric averaged over the batch -- see
[the shape conventions](README.md#what-the-entries-mean).
