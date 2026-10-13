# Worked example: patching a bi-encoder end to end

One runnable path from a checkpoint to a patch map, using `Dot` on
`vaswani` -- the smallest ir-datasets collection, so it is the cheapest way
to see the whole loop work. The same shape applies to `Cat` with
`CatDataCollator` and `sequences`/`perturbed_sequences` in place of
`queries`/`documents`/`perturbed_documents`.

Install the plotting extra for the last step:

```bash
pip install "mechir[plotting]"   # from a checkout: uv sync --extra plotting
```

## 1. Load a model

```python
from mechir import Dot

model = Dot("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
n_layers = model._model.cfg.n_layers
n_heads = model._model.cfg.n_heads
print(n_layers, n_heads)
```

`n_layers` and `n_heads` come from the checkpoint's hooked config and set
the shape of everything below. Print them rather than assuming: this page
does not hardcode them because the example was not executed while writing
it.

## 2. Define a perturbation

```python
from mechir.perturb import perturbation


@perturbation
def append_term(text: str, query: str = None) -> str:
    return text + " relevant"
```

The decorator returns an `AbstractPerturbation` instance (not a function)
carrying `perturb_type="append"`, which is what the collator uses to align
the original and perturbed documents token by token. Use
`@perturbation(perturb_type="prepend")` or `"replace"` for the other kinds;
the default is `"append"`.

## 3. Build a dataloader

```python
from torch.utils.data import DataLoader
from mechir.data import MechIRDataset, DotDataCollator

dataset = MechIRDataset("vaswani")
collator = DotDataCollator(model.tokenizer, transformation_func=append_term)
loader = DataLoader(dataset, batch_size=8, collate_fn=collator)
```

`MechIRDataset` yields `(query, document)` pairs from the collection's
qrels. Each batch from `DotDataCollator` is:

Key | Contents
--- | ---
`queries` | tokenised queries, `input_ids` of shape `(batch, 30)`
`documents` | tokenised documents, `input_ids` of shape `(batch, 300)`
`perturbed_documents` | the same documents after `append_term`, `(batch, 300)`

The `30` and `300` are the collator's `q_max_length` and `d_max_length`
defaults, and they are exact because the collator pads to
`padding="max_length"`.

## 4. Score, to check the perturbation did something

```python
batch = next(iter(loader))

scores, reps_q, reps_d, cache = model.score(batch["queries"], batch["documents"])
scores_p, _, _, _ = model.score(
    batch["queries"], batch["perturbed_documents"], reps_q=reps_q
)

print(scores.shape, reps_q.shape)   # (batch,) (batch, d_model)
print((scores_p - scores).mean())   # how far the perturbation moved the score
```

`cache` is `None` here because `cache=False` is the default. A perturbation
that leaves the score untouched also leaves nothing for patching to
recover: `linear_rank_function` reports those items as 0.

## 5. Patch every attention head

```python
import torch

patch_outputs = []
for batch in loader:
    # patch() returns (result, cache); cache is None unless the model was
    # built with return_cache=True.
    result, _ = model.patch(**batch, patch_type="head_all")
    patch_outputs.append(result)

mean_heads = torch.mean(torch.stack(patch_outputs), dim=0)
print(mean_heads.shape)   # (n_layers, n_heads)
```

`model.patch(**batch, ...)` works because `patch()` accepts the collator's
`perturbed_documents` key as an alias for `documents_p`.

`result` is `(n_layers, n_heads)`, float32, on `model._device`. Each entry
is the share of the perturbation's effect on the score that this head
recovers, averaged over the batch: 0 recovers nothing, 1 recovers all of it.

This is `n_layers * n_heads` forward passes **per batch**, so start with a
handful of batches before running the full collection.

## 6. Plot it

```python
from mechir.plotting import plot_components

plot_components(mean_heads.detach().cpu().numpy(), save_path="heads.png")
```

`plot_components` expects a numpy array of shape `(n_layers, n_heads)`, or
`(n_layers, n_heads + 1)` with `include_mlp=True`. Its colour scale is fixed
to `[-1, 1]`. Omit `save_path` to show the figure instead of writing it.

## Position-resolved variants

`"head_all"` says *which* heads matter but not *where*. The other two
`patch_type`s answer that, at proportionally higher cost:

```python
# (3, n_layers, 300) -- resid_pre, attn_out, mlp_out, per layer per position
blocks, _ = model.patch(**batch, patch_type="block_all")

# (2, 1, 300) -- z then pattern, for the single (layer, head) pair given
by_pos, _ = model.patch(
    **batch, patch_type="head_by_pos", layer_head_list=[(0, 0)]
)
```

For `Dot` the position axis runs over the padded *document*, so `300` with
the collator defaults. `mechir.plotting.plot_blocks(data, labels, save_path)`
takes the `block_all` map directly.

Full reference for the return values: [docs/README.md](README.md).
