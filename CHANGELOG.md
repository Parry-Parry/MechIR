# Changelog

## 0.3.0

### Removed: `SAEMixin` and the `sae_lens` dependency

**Breaking.** `mechir.SAEMixin`, `mechir.modelling.SAEMixin` and
`mechir.modelling.sae` are gone, and `Dot`, `Cat`, `MonoT5` and
`GenerativeRanker` no longer inherit from the mixin. Nothing else in the
public API changes.

`sae_lens` declares `transformer-lens<4.0.0`, so while it was a hard
dependency, TransformerLens 4 could not be installed alongside it at all.
Removing it was a precondition for evaluating that upgrade. It also cost more
than it provided: the mixin had no tests, no SAEs have ever been released
here, nothing in the package called any of its methods, and TransformerLens 4
has dropped `HookedSAETransformer`, `HookedSAE` and `HookedSAEConfig`
entirely, so there was no supported upstream to port it to.

### The TransformerLens pin stays, and now says why

`docs/upgrading.md` records the result of actually booting the ranker
checkpoints against TransformerLens 4.2 rather than reading its release notes.

Removing `sae_lens` was necessary and not sufficient: 4.2's
`TransformerBridge` loads `MonoT5` but cannot load either of the other two
ranker families. A `BertForSequenceClassification` cross-encoder fails looking
for an MLM `cls` head, and DistilBERT is not a supported architecture at all —
nor is ELECTRA. That is the same gap `modelling/architectures/` was written to
fill, so porting would mean rewriting that directory against a new API rather
than deleting it, while also forcing transformers 5.

The cheap half does port: every component mechir patches resolves to a hook
name that exists on the 4.2 bridge, and `get_act_name` keeps its semantics and
only changes module. The page records the measured hook-name table and the one
stack-naming change T5 needs, so the next attempt starts from facts.

## 0.2.0

### MonoT5 patching works

`PatchedMixin` built `blocks.{n}` hook names while T5 names its stacks
`encoder.{n}` and `decoder.{n}`, so every patching call against MonoT5 raised
`KeyError: blocks.0`. Two tests asserted that error, recording the defect
rather than a contract.

Hook naming is now a property of the architecture, so the encoder-decoder
case is just another strategy, and the three `get_act_patch_*` methods live
once on the mixin instead of being copied per ranker family -- 336
near-identical lines across twelve methods replaced by 47 lines of overrides.
Adding a ranker family is now one scoring strategy; adding a patching method
is one driver.

The pre-refactor loops were re-implemented verbatim and diffed against the
new public methods on `bert-base-uncased` and `ms-marco-MiniLM-L-6-v2`:
bit-exact across every patch type. `patch()` is unchanged, and so is every
documented return shape.

**Breaking** if you call `Dot.get_act_patch_*` directly: `reps_q` now arrives
by keyword. It was positional before -- third in two methods and second in
`head_by_pos`, so they disagreed with each other. `patch()` is unaffected.

### Activations and masks stay on one device

`tokens.device.type != self.cfg.device` compared a `str` to a `torch.device`,
so it was always unequal and re-ran `.to()` on every forward. The wasted copy
hid a real failure: the model moved its own copy of the inputs, but the
pooling functions received the caller's attention mask, and `token_type_ids`
was never moved at all. On an accelerator the activations and the mask ended
up on different devices.

CPU-only machines never saw this, which is why CI stayed green; on Apple
Silicon it failed twelve tests. `cfg.device`'s annotation said `Optional[str]`
while holding a `torch.device`, which is what made the broken comparisons look
right.

### Steering is no longer dead code

`mechir.modelling.steering` could not run: `increase` and `decrease` had their
signs swapped, the steering direction was computed in a `__post_init__` on a
class that is an `nn.Module` rather than a dataclass -- so the hook never
fired and the attribute never existed -- and `forward` returned `None`.

The direction is now the top *right* singular vector. For an embedding stored
`[d_vocab, d_model]`, the left vector is vocab-length and cannot be added to a
`d_model` activation at all.

### Checkpoints that declare no architectures load

`convert_hf_model_config` read `architectures[0]` unconditionally, so any
config omitting that field -- `prajjwal1/bert-tiny`, and anything built
directly as `BertConfig()` -- died with a `TypeError`. It falls back to
`model_type`, and names the checkpoint and the fix when that cannot be mapped.

### Examples that run, and docs for what `patch` returns

`examples/cat.py` imported a `PairDataset` that exists nowhere in the package,
took two more names from the wrong modules, and imported `DataLoader` from
`torch.data`. `scripts/generate_data.py` referenced the removed `TFC1` and
`TDC` axioms. Three of the four notebooks -- the ones the README links as the
primary demos, Colab included -- imported `MechIRDataset` from `mechir` rather
than `mechir.data`, so they failed on the first cell.

All are fixed. The notebooks' saved outputs still predate the current API.
There is now a page per model class stating what `patch()` returns and the
shape of it, plus a worked end-to-end example; previously the ECIR paper was
the de facto documentation. The runnable scripts' third-party dependencies
(`fire`, `pyterrier`, `pyterrier_dr`) were undeclared and are now an
`examples` extra.

### Packaging and CI

`transformers` was the last unpinned dependency and now has an upper bound.
TransformerLens 2.x reaches into internals that transformers 5.0 reorganised,
so unpinned, a fresh environment resolves to 5.x and every HookedTransformer
test dies with `'GPTNeoXConfig' object has no attribute 'rotary_pct'`, while
CI happens to resolve 4.57.6 and stays green.

Four `requirements*.txt` files duplicated `pyproject.toml` and are gone, along
with the Dockerfile's pip-compile step. black and flake8 are replaced by ruff.
`test/integration` now runs nightly and on dispatch, selected by marker rather
than by directory so `make test`, the Docker image and CI cannot drift apart.

### New: `mechir.pipeline`, behind the `rag` extra

Ranker-and-reader interpretability: holds both with hooks on each side, so an
intervention on one can be measured on the other, with the metric read off
the reader's generation rather than the ranker's score.

    pip install "mechir[rag]"

Optional and opt-in -- `import mechir` does not pull it, and the publish
workflow asserts that. Research code rather than settled library surface;
the API will change.

  `boundary`    HardPermutation: exact interventional reordering of a fixed
                candidate set, so no relaxation has to be defended
  `context`     prompt assembly recording per-document token spans
  `metrics`     answer_logprob, attention_to_documents
  `ablation`    causal measures -- masking and resampling, necessity and
                sufficiency
  `pipeline`    position_sensitivity, position_controlled_attention,
                causal_report, cross-boundary head patching
  `benchmarks`  TREC DL 2019/2020 and HotpotQA as labelled candidate sets
                (in `mechir.data`)

`docs/research-notes.md` records what has been measured and four ways to
produce a result here that looks like a finding and is not -- attention
anti-correlated with causation, attention mass confounded by span length and
recency, ablation style changing per-document conclusions, and a
piecewise-constant ranker/reader boundary where ~40% of real TREC DL queries
yield no signal. Everything there is preliminary: 25 questions, small
readers, no error bars.

### Plotting is an extra

`pip install mechir` used to pull seaborn, matplotlib, plotly **and
streamlit** -- a web framework -- for four helper functions. They now live
behind `pip install "mechir[plotting]"`.

**Breaking** if you relied on `mechir.plotting` working from a bare install.

Imports also resolve lazily per function instead of eagerly for the whole
package, which fixes a real bug: `visualize_attn_patterns` needs streamlit
while `plotting_funcs` does not, and the old star-import of both meant a
missing streamlit broke `plot_components` too, even though it only needs
seaborn. A missing dependency now raises an error naming the extra rather
than a bare `ModuleNotFoundError: streamlit`.

### The publish workflow no longer re-runs the test suite

Releasing re-ran the full suite, which downloads several models -- roughly
five minutes and hundreds of megabytes re-fetched per release, duplicating
what `python-package.yml` had already run on the same commit.

It now verifies the **built artefact** instead, which is what a release can
uniquely break and what the source tree cannot check: the wheel installs
into a clean environment, the public API imports from it, `__version__`
matches the installed metadata, `stopwords.txt` is present as package data,
and the `plotting` extra resolves. The artefact is built once and published
unchanged, where before `deploy` rebuilt it and uploaded files nothing had
verified.

### Fixed

- `black` is pinned (`==26.10.1`). It was unpinned, so CI installed a
  version two years newer than local and rejected correctly-formatted code,
  which failed the 0.1.0 release on its first attempt. An unpinned formatter
  in a gating step lets any upstream release break the build with no commit.

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
