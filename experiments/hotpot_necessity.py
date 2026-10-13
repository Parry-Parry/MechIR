"""Does causal necessity recover HotpotQA's supporting paragraphs?

Reproduces RESEARCH_NOTES.md section 4.

    python experiments/hotpot_necessity.py EleutherAI/pythia-70m 25
    python experiments/hotpot_necessity.py Qwen/Qwen3-0.6B 25
"""

import statistics
import sys

import torch
from transformer_lens import HookedTransformer

from mechir.data import load_hotpotqa
from mechir.pipeline import (
    ContextBuilder,
    attention_to_documents,
    causal_groundedness,
    document_necessity,
)

name = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-70m"
n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 25

kwargs = {}
if "Qwen" in name or "qwen" in name:
    kwargs = dict(fold_ln=False, center_writing_weights=False, center_unembed=False)
reader = HookedTransformer.from_pretrained(name, device="cpu", **kwargs)
builder = ContextBuilder(reader.tokenizer, max_document_tokens=80)
examples = load_hotpotqa(split="validation", max_queries=n_questions)
print(f"RUN  reader={name} questions={len(examples)}", flush=True)

nec_hit = att_hit = n = 0
gold_nec, dist_nec, gold_att, dist_att, ground = [], [], [], [], []

for idx, example in enumerate(examples):
    order = list(range(len(example.documents)))
    context = builder.build(
        example.query, example.documents, order, answer=example.answer
    )
    if context.length > 1800:
        continue

    necessity = document_necessity(reader, context)
    attention = attention_to_documents(reader, context)
    gold = set(example.relevant)
    k = len(gold)

    nec_hit += len(set(sorted(necessity, key=necessity.get, reverse=True)[:k]) & gold)
    att_hit += len(set(sorted(attention, key=attention.get, reverse=True)[:k]) & gold)
    n += k

    for rank in order:
        (gold_nec if rank in gold else dist_nec).append(necessity[rank])
        (gold_att if rank in gold else dist_att).append(attention[rank])
    ground.append(causal_groundedness(reader, context)["effect"])
    print(f"   {idx + 1}/{len(examples)}", flush=True)

print(
    "RESULT precision@k  necessity=%.1f%%  attention=%.1f%%  random=20.0%%"
    % (100 * nec_hit / n, 100 * att_hit / n),
    flush=True,
)
print(
    "RESULT necessity  gold=%+.4f  distractor=%+.4f"
    % (statistics.mean(gold_nec), statistics.mean(dist_nec)),
    flush=True,
)
print(
    "RESULT attention  gold=%.5f  distractor=%.5f"
    % (statistics.mean(gold_att), statistics.mean(dist_att)),
    flush=True,
)
print(
    "RESULT groundedness mean=%+.3f nats over %d questions"
    % (statistics.mean(ground), len(ground)),
    flush=True,
)
print("EXPERIMENT_COMPLETE", flush=True)
