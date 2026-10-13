# `MonoT5`

`mechir.MonoT5` -- an encoder-decoder reranker, scored by contrasting the
`true` and `false` token logits at the first decoder position. Source:
`src/mechir/modelling/t5.py`.

```python
from mechir import MonoT5

model = MonoT5(
    "castorini/monot5-small-msmarco-10k",
    pos_token="true",
    neg_token="false",
    softmax_output=False,
    return_cache=False,
)
```

Both label tokens must encode to exactly one vocabulary id; anything else
raises `ValueError`.

## `forward`

```python
scores = model.forward(input_ids, attention_mask)
```

Returns `(batch,)`. The decoder is run for a single step from the
decoder-start token; the logits at that position are reduced to the `true`
logit, or to `softmax([true, false])[:, 0]` when `softmax_output=True`.

## `score`

```python
scores, cache = model.score(sequences, cache=False)
```

Element | Shape / type
--- | ---
`scores` | `(batch,)`
`cache` | activation `dict` when `cache=True`, otherwise `None`

Unlike `Dot` and `Cat`, `MonoT5` does not override `run_with_cache`, so the
cache is the plain `{hook_name: tensor}` dict returned by
`HookedRootModule.run_with_cache`, not an `ActivationCache`.

## `patch`

```python
result = model.patch(
    sequences,
    sequences_p=None,            # alias: perturbed_sequences
    patch_type="block_all",      # "block_all" | "head_all" | "head_by_pos"
    layer_head_list=[],
    patching_metric=mechir.util.linear_rank_function,
)
```

The return type is inconsistent with the other three classes: `MonoT5.patch`
returns a bare tensor when `return_cache=False` and a `(result, cache)`
tuple when `return_cache=True`. Write `result = model.patch(...)` for the
former and `result, cache = model.patch(...)` for the latter.

The shapes are the shared ones: `(3, n_layers, seq)` for `"block_all"`,
`(n_layers, n_heads)` for `"head_all"`, `(2, len(layer_head_list), seq)` for
`"head_by_pos"`.

### Which stack is patched

Patching runs over the **encoder** stack. The positions being swept are the
encoder's input tokens, and the decoder is only ever run for the single step
that carries the relevance label.

`MonoT5` selects this by overriding `PatchedMixin._hook_names` with
`mechir.modelling.ENCODER_HOOK_NAMES`. `HookedEncoderDecoder` names its hooks
`encoder.{layer}.…` and `decoder.{layer}.…` rather than the single-stack
`blocks.{layer}.…` that `transformer_lens.utils.get_act_name` emits, so
before hook naming became an architecture-level strategy every patching call
died with:

```
KeyError: '_model.blocks.0.hook_resid_pre'
```

To patch the decoder stack instead, override `_hook_names` with
`mechir.modelling.DECODER_HOOK_NAMES`. Note that the decoder is one position
long, so the position axis of the result collapses accordingly.
