# `Dot`

`mechir.Dot` -- a bi-encoder: query and document are encoded separately and
scored by a dot product of their pooled representations. Source:
`src/mechir/modelling/dot.py`.

```python
from mechir import Dot

model = Dot(
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    pooling_type="cls",     # "cls" | "mean" | "last"
    tokenizer=None,         # defaults to AutoTokenizer for the checkpoint
    return_cache=False,
)
```

Encoder checkpoints (BERT, DistilBERT, ELECTRA) and decoder checkpoints
(Qwen2/Qwen3, Llama, Mistral, ...) are both accepted; the backbone is chosen
from the checkpoint's declared architecture. Decoder bi-encoders such as
Qwen3-Embedding want `pooling_type="last"`.

Pair it with `mechir.data.DotDataCollator`, which emits
`{"queries", "documents", "perturbed_documents"}`.

## `forward`

```python
reps = model.forward(input_ids, attention_mask, token_type_ids=None)
```

Returns the pooled representation, `(batch, d_model)`.

## `score`

```python
scores, reps_q, reps_d, cache = model.score(
    queries, documents, reps_q=None, cache=False, cache_as_dict=False
)
```

`queries` and `documents` are the tokenised dicts from the collator.

Element | Shape / type
--- | ---
`scores` | `(batch,)` -- dot product of the pooled representations
`reps_q` | `(batch, d_model)`
`reps_d` | `(batch, d_model)`
`cache` | `ActivationCache` when `cache=True`, otherwise `None`

## `patch`

```python
result, cache = model.patch(
    queries,
    documents,
    documents_p=None,            # alias: perturbed_documents
    patch_type="block_all",      # "block_all" | "head_all" | "head_by_pos"
    layer_head_list=[],          # required by "head_by_pos"
    patching_metric=mechir.util.linear_rank_function,
)
```

Always returns a 2-tuple. `cache` is the `ActivationCache` of the perturbed
document run when the model was built with `return_cache=True`, and `None`
otherwise. `documents_p` may be passed as `perturbed_documents`, so
`model.patch(**batch, patch_type=...)` works directly on a
`DotDataCollator` batch.

`patch_type` | `result` shape | Axis 0
--- | --- | ---
`"block_all"` | `(3, n_layers, seq)` | `resid_pre`, `attn_out`, `mlp_out`
`"head_all"` | `(n_layers, n_heads)` | --
`"head_by_pos"` | `(2, len(layer_head_list), seq)` | `z`, `pattern`

`seq` is the padded *document* length, `documents["input_ids"].shape[1]`
(300 with `DotDataCollator` defaults). The query is never patched; it is
encoded once and its representation is reused for every patched run.

Each entry is the patching metric averaged over the batch -- see
[the shape conventions](README.md#what-the-entries-mean). For `Dot` the
patched document representation is turned back into a score by
`batched_dot_product(reps_q, patched_reps)` before the metric is applied.
