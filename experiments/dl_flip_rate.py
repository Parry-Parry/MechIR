"""How often does patching a ranker head reorder a TREC DL candidate set?

This is the quantity that decides whether hard-boundary cross-boundary
patching has any signal: the reader metric is piecewise constant in the
ranker's scores, so a head only registers if it flips a rank.

Reproduces RESEARCH_NOTES.md section 5.

    python experiments/dl_flip_rate.py 2019 6
"""

import statistics
import sys
from functools import partial

import torch
import transformer_lens.utils as utils
from mechir import Cat

from mechir.data import load_trec_dl

RANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 6

ranker = Cat(RANKER)
n_layers = ranker._model.cfg.n_layers
n_heads = ranker._model.cfg.n_heads
examples = load_trec_dl(year=year, max_queries=n_queries, max_documents=8)

print(
    f"RUN  DL{year}  {len(examples)} queries  "
    f"{n_layers}x{n_heads}={n_layers * n_heads} head patches each",
    flush=True,
)


def encode(query, documents):
    encoded = ranker.tokenizer(
        [query] * len(documents),
        list(documents),
        return_tensors="pt",
        padding="max_length",
        max_length=160,
        truncation=True,
    )
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "token_type_ids": encoded.token_type_ids,
    }


flip_rates, max_deltas, separated, gaps = [], [], 0, []

for example in examples:
    corrupted = encode(example.query, example.documents)
    # Perturbed run: append the query, a crude relevance boost.
    clean = encode(example.query, [f"{d} {example.query}" for d in example.documents])

    baseline, _ = ranker.score(corrupted)
    baseline = baseline.detach().flatten()
    baseline_order = baseline.argsort(descending=True).tolist()
    _, cache = ranker.score(clean, cache=True)

    labels = torch.tensor(example.relevance, dtype=torch.float32)
    relevant, nonrelevant = baseline[labels > 0], baseline[labels <= 0]
    if len(relevant) and len(nonrelevant):
        separated += int(relevant.mean() > nonrelevant.mean())
        gaps.append((relevant.mean() - nonrelevant.mean()).item())

    flips = total = 0
    worst = 0.0
    for layer in range(n_layers):
        for head in range(n_heads):
            hook = partial(
                ranker._patch_head_vector, head_index=head, clean_cache=cache
            )
            with torch.no_grad():
                patched = ranker.run_with_hooks(
                    corrupted["input_ids"],
                    attention_mask=corrupted["attention_mask"],
                    **ranker._extra_forward_kwargs(corrupted),
                    fwd_hooks=[("_model." + utils.get_act_name("z", layer), hook)],
                )
            ranker.reset_hooks()
            patched = patched.detach().flatten()

            total += 1
            flips += int(patched.argsort(descending=True).tolist() != baseline_order)
            worst = max(worst, (patched - baseline).abs().max().item())

    flip_rates.append(flips / total)
    max_deltas.append(worst)
    sorted_scores = baseline.sort(descending=True).values
    gaps_within = sorted_scores[:-1] - sorted_scores[1:]
    print(
        f"   {example.query_id}  flip_rate={flips / total:.1%}  "
        f"min_adjacent_gap={gaps_within.min().item():.4f}",
        flush=True,
    )

print(
    "RESULT relevant outscores non-relevant on %d/%d queries, mean gap %+.3f"
    % (separated, len(gaps), statistics.mean(gaps))
)
print(
    "RESULT mean flip rate %.1f%%  (min %.1f%%, max %.1f%%)"
    % (
        100 * statistics.mean(flip_rates),
        100 * min(flip_rates),
        100 * max(flip_rates),
    )
)
print(
    "RESULT queries with ZERO flips: %d/%d"
    % (sum(1 for r in flip_rates if r == 0), len(flip_rates))
)
print("RESULT mean max |score delta| per query: %.3f" % statistics.mean(max_deltas))
print("EXPERIMENT_COMPLETE")
