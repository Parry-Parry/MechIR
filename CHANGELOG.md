# Changelog

## 0.1.0

First release that can actually be installed. `0.0.4` on PyPI cannot be
imported at all.

**This release changes numbers.** If you have results from `0.0.4`, read
"Results that change" below before comparing anything.

### Installation is fixed

`0.0.4` declared no upper bound on `transformer_lens`, so a fresh resolve
picked up 4.x — which removed `loading_from_pretrained`,
`pretrained.weight_conversions` and `HookedEncoder`, all of which mechir
imports. `import mechir` raised `ImportError` immediately.

There was a second trap underneath: TransformerLens 2.x caps `numpy<2`,
while `pyarrow>=21` requires `numpy>=2` at runtime without declaring it, so
dependency resolution succeeded and then failed at import with
`pyarrow requires NumPy 2.0 or newer`.

Dependencies are now bounded to a mutually satisfiable set
(`transformer_lens>=2.15,<3`, `sae_lens>=6`, `numpy>=1.26,<2`,
`datasets<4`, `pyarrow<21`), and `jaxtyping`, `pandas` and `plotly` — all
imported but previously undeclared — are listed.

### Models that did not work, now do

- **`Cat` could not load any trained cross-encoder.** Four stacked defects:
  the classifier weight was never transposed (so loading raised a size
  mismatch), the pooler was instantiated but its weights were never
  converted, the classification path bypassed the pooler entirely, and
  `score` dropped `token_type_ids`. Scores now match HuggingFace to 1e-06.
- **`MonoT5` could not be constructed.** `__init__` read
  `self.model_name_or_path` before assigning it, never called
  `super().__init__()`, and built its label tokens from whole encoded
  sequences rather than single ids. `forward` also passed a parameter name
  that does not exist on `HookedEncoderDecoder` and never supplied
  `decoder_input`. No monoT5 checkpoint was registered either.

### New

- **Decoder backbones.** Qwen2/Qwen3, Llama, Mistral and friends, dispatched
  from the checkpoint's architecture. This needed no architecture porting:
  the Qwen IR fine-tunes are architecturally unmodified Qwen3, so
  TransformerLens's existing conversion handles them.
- **`GenerativeRanker`** — decoder reranker scored by contrasting two
  label-token logits. Qwen3-Reranker contrasts `yes`/`no` at the final
  position where MonoT5 contrasts `true`/`false` at the first decoder
  position, so they share one scorer.
- **`"last"` pooling** for decoder bi-encoders such as Qwen3-Embedding,
  which matches `last_hidden_state` pooling to 4.5e-05.
- Registered checkpoints: monoT5 small/base/large, BERT-family
  cross-encoders, Qwen3-Embedding/Reranker, gte-Qwen2.
- **Autograd is no longer disabled process-wide.** Constructing a model used
  to call `torch.set_grad_enabled(False)`, switching autograd off for the
  whole interpreter. `score` and `patch` now carry `@torch.no_grad()` while
  `forward` stays differentiable, which is what gradient-based
  interpretability methods need.

### Results that change

- **Mean pooling no longer averages over padding.** The same document used
  to score differently depending on how far its batch was padded (measured
  up to 0.0715 on embeddings, and up to 11.62 on a raw dot-product score).
  Masked mean is what TAS-B and similar bi-encoders actually compute; the
  new values agree with HuggingFace to 9.5e-07. **Any mean-pooled number
  from 0.0.4 will move.** CLS pooling is unaffected.
- **`Cat` scores change from wrong to correct**, including the
  `softmax_output=True` path, which previously returned identically zero
  because the classification head came out width-1.
- **`linear_rank_function` returns 0 instead of NaN** when a perturbation
  leaves the score unchanged. Previously a single unaffected batch item
  turned an entire patch map into NaN via the `.mean()` in the patching
  loops.

### The documented workflow now runs

`model.patch(**batch)` -- the usage the README has always shown -- raised
`TypeError` for every model. The collators emit `perturbed_documents` /
`perturbed_sequences` while `patch` declared `documents_p` / `sequences_p`,
so the two halves never composed. `patch` now accepts both spellings;
existing positional and `documents_p=` calls are unaffected.

The collators also passed `truncation=False` alongside
`padding="max_length"`, which is incoherent: any document longer than
`d_max_length` raised `ValueError: expected sequence of length 57 at dim 1
(got 48)` instead of being truncated to the bound the caller asked for.
They now truncate. `MonoT5DataCollator` never passed `truncation` at all and
had the same failure.

Both README code samples were wrong and are corrected: `MechIRDataset` is
imported from `mechir.data`, not `mechir`, and `patch` returns
`(result, cache)` rather than a bare tensor, so the old example stacked a
tuple.

### Removed

- **`mechir.modelling.architectures.modernbert`.** It never ran:
  `components.py` did not parse, `ModernBertAttention` was defined twice,
  and `modernbert.py` imported `BertBlock` from TransformerLens while
  referencing its own components zero times — so it would have built a plain
  BERT. The work is preserved on the `modernbert` branch.

### Fixed

- `steering.py` raised `NameError` on import (missing `HookedRootModule`
  import). It loads now; three remaining defects are documented in its
  module docstring rather than guessed at.
- `MANIFEST.in` pointed at the wrong path for `stopwords.txt`.
- `setup.py` removed — it duplicated the `pyproject.toml` metadata while
  declaring author "Anon A. Mous" and description "A default python
  template", with no `install_requires`.
- `__version__` now derives from installed metadata instead of being
  hardcoded in a third place.

### Testing and CI

CI previously installed dependencies and stopped — no lint step, no test
step — so the one run on record was green without executing a test. It now
runs a lint gate on syntax errors and undefined names, `black --check`, and
the test suite. The conda workflow, whose first action referenced a
non-existent `environment.yml`, is deleted.

Three unit tests had never passed on any platform (they compared a
`torch.device` against a string). The suite grew from 39 passing with 3
failures to 85 passing, plus a Vaswani integration suite that exercises
every model against a real collection.
