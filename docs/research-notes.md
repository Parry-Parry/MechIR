# Research notes

Measured results and methodological traps from building `mechir.pipeline`. Everything
here was run, not estimated. Numbers are reproducible from the scripts in
`experiments/`.

**Status: preliminary.** Every result below is on 25 questions or 6–15
queries, with small readers, on CPU. Treat them as direction-finding, not as
findings. The caveats at the end are load-bearing.

---

## 1. Attention is not causation, and can be anti-correlated with it

The motivating result. One query, four documents, all retrieved together.
Reader: `pythia-70m`. Ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Effects in nats.

| document | necessity | sufficiency | attention |
|---|---|---|---|
| Paris/France (gold) | **+2.7311** | **+6.4672** | 0.00311 |
| Berlin/Germany | +0.0010 | +2.4265 | 0.00176 |
| Eiffel Tower/Paris | −0.0487 | +2.3780 | 0.00313 |
| Bananas (unrelated) | **−0.7027** | +1.5125 | **0.00526** |

- Ranked by attention: `[3, 2, 0, 1]` — the irrelevant document **first**.
- Ranked by causal necessity: `[0, 1, 2, 3]` — exactly the gold ordering.

The distractor drew the most attention while *costing* 0.70 nats. Overall
groundedness effect was +6.3987 nats, so the evidence genuinely was
load-bearing — attention simply could not say which document carried it.

A faithfulness claim built on attention here would have been wrong, not
merely noisy.

## 2. Attention mass is confounded twice over

Two separate confounds, and neither is obvious from the metric's name.

**Span length.** Summed attention grows with token count, so a long document
outscores a short one for free. Fixed by normalising per token
(`normalize="mean"`, the default).

**Position.** In a causal reader, later spans sit nearer the answer positions
and attract more attention regardless of content. No normalisation removes
this. Measured: the irrelevant document above had the highest attention
(0.00526) purely by sitting last; promoting each candidate to rank 0 in turn
dropped it to 0.00408, and it was no longer the maximum.

The control is to hold the candidate set fixed and compare the *same*
document across ranks — which is what `HardPermutation` and
`position_controlled_attention` exist for. Comparing raw attention across
different ranks in one ordering mostly measures recency.

## 3. Ablation style changes the per-document conclusion

Masking removes a document; resampling replaces it with other real text of
matched token length. They disagree, and the disagreement is informative.

| document | mask | resample (rotation) | resample (neutral filler) |
|---|---|---|---|
| Paris (gold) | +2.7311 | +1.6418 | **+2.9850** |
| Berlin | +0.0010 | +0.3952 | +0.3188 |
| Eiffel/Paris | −0.0487 | −0.1806 | **+0.5773** |
| Bananas (unrelated) | **−0.7027** | **−2.2157** | **+0.0287** |

The gold document agrees across mask and neutral filler (+2.73 vs +2.99) —
its effect is robust to ablation style, which is the check worth running.

The unrelated document does **not** agree, and that is the useful part.
Masking made it look actively harmful (−0.70); rotation made that worse
(−2.22); against neutral filler its effect is ≈0, which is the honest
answer. Both of the first two were artifacts of *what replaced it*.

**Trap:** rotating the candidate set as the replacement source is confounded
whenever the pool contains relevant documents, because it conflates removing
the document under test with duplicating another. That is exactly what
inflated −2.22 (bananas was replaced by the gold Paris document). Use
replacements sampled from unrelated queries — `sample_filler` does this.

## 4. Causal necessity recovers HotpotQA supporting paragraphs

HotpotQA distractor setting: 10 candidate paragraphs per question, 2
supporting and 8 distractors, so precision@2 against a 20% random baseline.
25 questions.

| | pythia-70m (70M) | Qwen3-0.6B (600M) |
|---|---|---|
| **necessity** precision@2 | 40.0% | **50.0%** |
| attention precision@2 | 36.0% | 42.0% |
| random baseline | 20.0% | 20.0% |
| necessity, mean gold | +2.0760 | **+3.2718** |
| necessity, mean distractor | −0.0101 | +0.0834 |
| attention, gold:distractor ratio | 1.5× | 2.2× |
| mean groundedness | +6.675 nats | +6.844 nats |

Three observations:

1. **Precision is only modestly better than attention** (40→36, 50→42), but
   **effect size is not close**: necessity separates gold from distractor by
   roughly 200× in magnitude, attention by 1.5–2.2×.
2. **Distractor mean is ≈0 under necessity**, which is the correct value. No
   measure that reports a large distractor effect is measuring what it
   claims to.
3. **Both measures improve with reader capacity, and necessity stays
   ahead** — the gap widens from 4pp to 8pp. That is the reassuring
   direction: the measure tracks something real about the reader's
   computation rather than an artifact of a weak model.

Open question: does precision@2 keep climbing with an instruction-tuned
reader? The trend rests on two models, so it is suggestive only.

## 5. The ranker/reader boundary is piecewise constant

The reader metric only moves if an intervention **reorders** documents. A
ranker head can shift a score substantially and produce a metric delta of
*exactly* zero.

Synthetic four-document example, MiniLM cross-encoder:

| candidate set | score gaps | heads flipping a rank | max \|metric delta\| |
|---|---|---|---|
| wide | 10.3, 1.1, 7.4 | **0/72** (0.0%) | 0.0000 |
| tight | 1.32, 0.025, 0.217 | **19/72** (26.4%) | 0.1954 |

Max score delta was ~1.16 in *both* cases. A head moved one document's score
by 1.5 and still produced zero signal. The difference is entirely
candidate-set tightness.

**This is why `HeadPatchResult` reports `rank_flipped` and `score_delta`
alongside `metric_delta`.** Without those, a user gets a grid of zeros and
concludes the tool is broken.

### On real data it is milder than the synthetic case suggested

TREC DL judged pools, 8 documents per query:

| | relevant > non-relevant | mean score gap | median **min adjacent-rank gap** |
|---|---|---|---|
| DL2019 | 14/15 queries | +10.16 | **0.0533** |
| DL2020 | 15/15 queries | +11.16 | **0.1456** |

Real adjacent-rank gaps are tiny — far below the ~3.3 score deltas
single-head patching produces — so flips are reachable. Measured flip rates
over all 72 heads, 6 queries each:

| | mean flip rate | range | queries with **zero** flips |
|---|---|---|---|
| DL2019 | 11.6% | 0–34.7% | 2/6 |
| DL2020 | 19.7% | 0–59.7% | 3/6 |

Flip rate tracks the minimum adjacent gap inversely, which is the mechanism
rather than a coincidence. Two DL2019 queries:

| query | min adjacent gap | flip rate |
|---|---|---|
| 1037798 | 0.3229 | 4.2% |
| 104861 | 0.0271 | **34.7%** |

So the hard boundary has signal on real benchmarks, but it is strongly
query-dependent and roughly 40% of queries yield nothing at all. Two
consequences: cross-boundary patching needs query selection (filter to
near-tied pools), and a differentiable boundary remains worth building for
the queries where no head flips a rank — it would see precisely the score
movement the hard boundary discards.

## 6. Data notes

**TREC DL 2019/2020** (`ir_datasets`) — graded judgements 0–3, no answers, so
ranker-side only. Corpus is 8.8M MS MARCO passages, downloaded on first use.

**HotpotQA** — *not* usable from `ir_datasets` for reader work.
`beir/hotpotqa` exposes only `(query_id, text)`; there is **no answer
field**, so `answer_logprob` has nothing to score. Use the HuggingFace
`hotpot_qa` distractor config instead: 10 paragraphs per question with gold
labels and the answer string, needing no retrieval and no 5.2M-document
corpus.

The BEIR splits *do* join back to HotpotQA by query id, but not the obvious
way — BEIR re-split the data:

| BEIR split | joins against | rate |
|---|---|---|
| `beir/hotpotqa/dev` | HotpotQA **train** | 5447/5447 |
| `beir/hotpotqa/test` | HotpotQA **validation** | 7405/7405 |

A dev↔validation join returns 0/5447. Question text is identical on every
joined pair checked.

---

## Caveats

These matter more than any number above.

- **Scale.** 25 questions for the HotpotQA results; 6–15 queries for the DL
  results. No error bars anywhere. Nothing here supports a significance
  claim.
- **Readers.** `pythia-70m` and `Qwen3-0.6B`, neither instruction-tuned nor
  RAG-tuned, both run on CPU. A real reader may behave differently in kind,
  not just degree.
- **Groundedness is attention-free but still single-style.** The causal
  measures use attention-masking ablation. Resample ablation is implemented
  and agrees on the gold document, which is the robustness check — but both
  are ablations of the *context*, not of reader internals.
- **`position_controlled_attention` costs one reader forward per
  candidate**, so it does not scale to long candidate lists as written.
- **The DL flip-rate numbers use one ranker** (6-layer MiniLM
  cross-encoder). Deeper rankers will have different per-head effect sizes.

## Next

1. Instruction-tuned reader on HotpotQA — does precision@2 keep climbing?
2. Scale to hundreds of questions for error bars.
3. Differentiable (soft) boundary — continuous attention gating over
   document spans, validated against the hard-boundary results in §5.
4. Ablate reader internals rather than context: which reader heads carry
   the gold document's contribution.
5. Axiomatic perturbations propagated end-to-end, reusing `mechir.perturb`.
