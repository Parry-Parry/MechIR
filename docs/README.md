# mechir documentation

Reference for the model classes and what their patching methods return.
Everything here was read off the implementations in `src/mechir/modelling/`;
where the code's own type annotations disagree with what it allocates, the
allocated shape is what is documented and the disagreement is called out.

- [Dot](dot.md) -- bi-encoder, scored by dot product
- [Cat](cat.md) -- cross-encoder with a classification head
- [MonoT5](monot5.md) -- encoder-decoder, scored by `true`/`false` logits
- [GenerativeRanker](generative-ranker.md) -- decoder backbone, scored by two
  label-token logits
- [Worked example](worked-example.md) -- end to end, from dataset to heatmap
- [Research notes](research-notes.md) -- measured results from
  `mechir.pipeline`

## Shape conventions

Symbol | Meaning
--- | ---
`batch` | Items per forward pass, i.e. `input_ids.shape[0]`
`seq` / `pos` | Padded sequence length, i.e. `input_ids.shape[1]`
`n_layers` | `model._model.cfg.n_layers`
`n_heads` | `model._model.cfg.n_heads`
`d_model` | `model._model.cfg.d_model`

Which sequence `seq` refers to differs by model: for `Dot` it is the padded
*document* length (the collator tokenises queries and documents separately);
for `Cat`, `MonoT5` and `GenerativeRanker` it is the length of the joint
query-document sequence.

With the default collators, `padding="max_length"` makes `seq` a constant:
`d_max_length` (300) for `Dot`, and `q_max_length + d_max_length` (30 + 300 =
330) for `Cat` and `MonoT5`.

Every patch map comes back as `torch.float32` on `model._device`.

## What `patch_type` returns

All four classes share the three patching loops in
`mechir.modelling.patched.PatchedMixin`, so the shapes below hold for every
model (subject to the per-model caveats on each page).

`patch_type` | Shape | Axis 0
--- | --- | ---
`"block_all"` (default) | `(3, n_layers, seq)` | `resid_pre`, `attn_out`, `mlp_out`
`"head_all"` | `(n_layers, n_heads)` | --
`"head_by_pos"` | `(2, len(layer_head_list), seq)` | `z`, `pattern`

Notes:

- `"head_by_pos"` needs `layer_head_list=[(layer, head), ...]`; its second
  axis indexes that list, not layers. The method's annotation says
  `"layer pos head"`, which is wrong -- `torch.zeros(2, len(layer_head_list),
  seq_len)` is what it allocates. Verified by
  `test/acceptance/test_qwen_rankers.py`, which asserts `(2, 1, 24)` for one
  `(layer, head)` pair and 24 tokens.
- Cost scales with the loop: `"head_all"` is `n_layers * n_heads` forward
  passes per batch, `"block_all"` is `3 * n_layers * seq`, and
  `"head_by_pos"` is `2 * len(layer_head_list) * seq`.

## What the entries mean

Each entry is a scalar: the patching metric applied to the patched score and
then averaged over the batch. The default metric is
`mechir.util.linear_rank_function`, the normalised recovery of the
perturbation's effect -- 0 when the patched component recovers none of it, 1
when it recovers all of it. Entries where the perturbation did not move the
score at all are 0 rather than NaN.

A custom `patching_metric` is called as
`patching_metric(patched_score, scores, scores_p)`, where all three are
`(batch,)`, and its result must survive `.mean()` into a scalar slot.

## The direction of patching

`patch()` runs the model twice before patching:

1. the unperturbed input, giving `scores`;
2. the perturbed input, giving `scores_p` *and* the cached activations.

The patching loops then re-run the **unperturbed** input while splicing in
one component at a time from the **perturbed** run's cache. So the cache
named `clean_cache` in the code holds the perturbed activations, and the
`corrupted_tokens` are the unperturbed ones.
