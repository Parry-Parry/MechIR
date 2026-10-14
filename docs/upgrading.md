# Upgrading TransformerLens

`pyproject.toml` pins `transformer_lens>=2.15,<3`. This page records why, what
breaks above that bound, and what has to happen before it can move. Everything
here was checked against the published 4.2.0 wheel and the PyPI metadata of the
packages involved, on 14 October 2026.

## The pin is currently not a choice

`sae_lens` 6.54.5 — the latest release — declares:

    transformer-lens<4.0.0,>=2.16.1

So while `sae_lens` is a hard dependency of `mechir`, TransformerLens 4 cannot
be installed alongside it at all. This is a resolver conflict, not a porting
cost: no amount of work inside `mechir` makes the two coexist.

That makes the first decision about `sae_lens`, not about TransformerLens:

1. **Wait** for `sae_lens` to support TransformerLens 4.
2. **Move `sae_lens` behind an extra**, so the core package can move to 4.x
   while `mechir[sae]` stays on the 2.x line. `SAEMixin` is currently
   untested, which makes it the cheapest thing in the package to isolate.
3. **Drop** `sae_lens`.

Option 2 is the only one that unblocks the port without discarding
functionality, and it is a prerequisite for everything below.

## What 4.0 removed

The names still appear in `transformer_lens/__init__.py`, but only as stubs
that raise `ImportError` with a migration pointer — so the failure is a clear
message rather than a bare `AttributeError`. The ones `mechir` imports:

| Removed | TransformerLens' own replacement |
|---|---|
| `HookedTransformer` | `TransformerBridge.boot_transformers(name)`, then `enable_compatibility_mode()` for HookedTransformer-equivalent numerics |
| `HookedEncoder` | `TransformerBridge.boot_transformers(name)` on a BERT model |
| `HookedEncoderDecoder` | `TransformerBridge.boot_transformers(name)` on a T5 model |
| `HookedTransformerConfig` | `TransformerBridgeConfig` |
| `loading_from_pretrained` | names and aliases moved to `transformer_lens.supported_models`; config derivation is internal to the bridge's adapters |
| `utils` | `transformer_lens.utilities`, same names |
| `components` | gone entirely — the component tree no longer exists |

`HookedRootModule` and `HookPoint` are explicitly **kept**, and remain the
supported way to hook an arbitrary `nn.Module`. `ActivationCache` and
`FactoredMatrix` are also still top-level.

`HookedSAETransformer`, `HookedSAE` and `HookedSAEConfig` are not in 4.2 at
all, which is the other half of the `sae_lens` problem.

## What that costs here

The coupling is about 3000 lines:

| | lines |
|---|---|
| `modelling/hooked/loading_from_pretrained.py` | 1044 |
| `modelling/architectures/**` | 1000 |
| `modelling/hooked/{config,states,conversion}.py` | 960 |

Most of it exists *because* 2.x's `HookedEncoder` could not express the
encoder rankers mechir needs — the custom BERT, DistilBERT and ELECTRA
implementations, the config derivation, and the weight conversion. If
`TransformerBridge` covers those architectures, most of this directory stops
being necessary rather than needing a port. That is the upside, and it is
large; it is also why this is a rewrite rather than an API swap.

## It is a whole-stack migration, not one dependency

TransformerLens 4.2 requires:

    transformers>=5.9.0
    torch>=2.6

`mechir` currently pins `transformers>=4.40,<5`, because 2.x reaches into
internals that transformers 5.0 reorganised — against transformers 5, every
`HookedTransformer` test fails with `'GPTNeoXConfig' object has no attribute
'rotary_pct'`. The two bounds are therefore a matched pair: TransformerLens 4
*forces* transformers 5, and TransformerLens 2 *forbids* it. There is no
intermediate state, so the port has to land as one change.

Two further pins fall out of it. `numpy<2` and `pyarrow<21` exist only because
2.x caps numpy below 2 (and pyarrow>=21 needs numpy>=2 at runtime without
declaring it). TransformerLens 4 does not cap numpy, so both pins can be
dropped — but only as part of the same migration.

## Order of work

1. Move `sae_lens` behind an extra, keeping that extra on `transformer_lens<4`.
   Add tests for `SAEMixin` first; it has none, and it is about to be the one
   component left on the old line.
2. Spike `TransformerBridge` against one encoder ranker — `Dot` on TAS-B is
   the smallest useful target — in a throwaway environment, and check two
   things: that the bridge exposes hook names patching can address, and that
   scores match the current implementation. Everything else depends on the
   answer and neither is worth guessing at.
3. Only then decide how much of `architectures/` and `hooked/` survives.

Until step 1 is done, the pin is load-bearing and should not be touched.
