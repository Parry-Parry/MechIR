# Upgrading TransformerLens

`pyproject.toml` pins `transformer_lens>=2.15,<3`. This page records why, what
breaks above that bound, and what has to happen before it can move.

Everything here was measured against TransformerLens 4.2.0 in a throwaway
environment on 14 October 2026 — booting the real ranker checkpoints, not
reading release notes. Re-measure before acting on it; upstream coverage is
the thing most likely to have changed.

## Status: blocked upstream, and not by what we thought

The first suspected blocker is gone. `sae_lens` pinned `transformer-lens<4.0.0`,
which made the upgrade impossible while it was a hard dependency; `SAEMixin`
and that dependency were removed in 0.3.0. That was necessary and it was not
sufficient.

The real blocker is that **TransformerLens 4 cannot load two of the three
ranker families this package exists to study.**

| Ranker | Checkpoint | TransformerBridge 4.2 |
|---|---|---|
| `MonoT5` | `castorini/monot5-base-msmarco` | **boots** — 1435 hook points |
| `Cat` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `AttributeError: 'BertForSequenceClassification' object has no attribute 'cls'` |
| `Dot` | `sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco` | `ValueError: Could not determine supported architecture from config` |

`TransformerBridge.list_supported_models()` returns 15734 entries, but for the
architectures here the real coverage is narrow:

* **BERT** — only the `google-bert/bert-*` base and large checkpoints, and the
  adapter expects an MLM head. A sequence-classification cross-encoder fails
  looking for `cls`.
* **DistilBERT** — nothing. The only substring match is
  `distilbert/distilgpt2`, which is GPT-2.
* **ELECTRA** — nothing. `HookedElectra` has no counterpart.
* **T5** — works.

This is the same gap that caused `modelling/architectures/` to be written in
the first place: 2.x's `HookedEncoder` could not express these rankers, so
mechir implemented them. 4.x still cannot. Porting therefore does **not**
delete that directory — it means rewriting all of it as `TransformerBridge`
adapters for DistilBERT, ELECTRA and BERT-with-a-classification-head, against
a less familiar API, while also forcing the transformers 5 upgrade. Same
surface area, new API, no reduction in maintained code.

So the pin stays. The upside the handover note hoped for — deleting most of
`architectures/` and `loading_from_pretrained.py` — is not available at 4.2.

## What *would* port cleanly

Worth recording, because it is the cheap part and it is already done
structurally:

**Patching is fully compatible.** Every component mechir patches resolves to a
hook name that exists on the 4.2 bridge, and `get_act_name` keeps its
semantics — it only moves module:

    transformer_lens.utils  ->  transformer_lens.utilities   (same names)

| mechir component | 4.2 hook name | present |
|---|---|---|
| `resid_pre` | `blocks.0.hook_resid_pre` | yes |
| `attn_out` | `blocks.0.hook_attn_out` | yes |
| `mlp_out` | `blocks.0.hook_mlp_out` | yes |
| `z` | `blocks.0.attn.hook_z` | yes |
| `pattern` | `blocks.0.attn.hook_pattern` | yes |
| `q`, `k`, `v` | `blocks.0.attn.hook_{q,k,v}` | yes |

The bridge also exposes `run_with_cache`, `run_with_hooks`, `hook_dict`,
`add_hook`, `encoder_output` and `to_sentence_pair_tokens` — the surface
`PatchedMixin` and the hooked encoder actually use.

**One naming change.** Single-stack models keep `blocks.{n}`, so
`BLOCK_HOOK_NAMES` is unaffected. The T5 bridge names its stacks
`encoder_blocks` and `decoder_blocks`, not `encoder` and `decoder`, so
`ENCODER_HOOK_NAMES` and `DECODER_HOOK_NAMES` need new stack constants. That
is a two-line change, because hook naming is already a strategy seam rather
than something the patching loops hardcode.

## What 4.0 removed

The names still appear in `transformer_lens/__init__.py` as stubs that raise
`ImportError` with a migration pointer, so the failure is a clear message
rather than a bare `AttributeError`. The ones mechir imports:

| Removed | Replacement |
|---|---|
| `HookedTransformer` | `TransformerBridge.boot_transformers(name)`, then `enable_compatibility_mode()` for equivalent numerics |
| `HookedEncoder` | `TransformerBridge.boot_transformers(name)` on a BERT model |
| `HookedEncoderDecoder` | `TransformerBridge.boot_transformers(name)` on a T5 model |
| `HookedTransformerConfig` | `TransformerBridgeConfig` |
| `loading_from_pretrained` | names and aliases moved to `transformer_lens.supported_models`; config derivation is internal to the bridge adapters |
| `utils` | `transformer_lens.utilities`, same names |
| `components` | gone entirely |

`HookedRootModule` and `HookPoint` are explicitly **kept** and remain the
supported way to hook an arbitrary `nn.Module`, which is what every mechir
ranker is built on. `ActivationCache` and `FactoredMatrix` are still
top-level. `HookedSAETransformer`, `HookedSAE` and `HookedSAEConfig` are gone.

## It is a whole-stack migration

TransformerLens 4.2 requires:

    transformers>=5.9.0
    torch>=2.6

mechir pins `transformers>=4.40,<5`, because 2.x reaches into internals that
transformers 5.0 reorganised — against transformers 5 every
`HookedTransformer` test fails with `'GPTNeoXConfig' object has no attribute
'rotary_pct'`. The two bounds are a matched pair: 4 *forces* transformers 5
and 2 *forbids* it, so there is no intermediate state and the port has to land
as one change.

`numpy<2` and `pyarrow<21` exist only because 2.x caps numpy below 2 (and
pyarrow>=21 needs numpy>=2 at runtime without declaring it). 4.x does not cap
numpy, so both pins can be dropped — as part of the same change.

## When to revisit

The decision point is upstream architecture coverage, not effort here. Revisit
when `TransformerBridge` can boot a `BertForSequenceClassification`
cross-encoder and a DistilBERT bi-encoder — re-run the table at the top of
this page, which is three `boot_transformers` calls.

If that does not arrive, the alternative is to contribute those adapters
upstream rather than maintain a fork of them here. The spike that produced
this page is cheap to repeat and is the right first step either way.
