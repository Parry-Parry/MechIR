"""A hooked generation turn on a real causal LM with a real tokenizer.

TinyStories-1M is a genuine autoregressive checkpoint small enough (8
layers, d_model 64) to decode a turn in milliseconds, so the unit tests'
random weights and stub tokenizer are backed up here by real BPE spans and
real text.
"""

import pytest
import torch
from transformer_lens import HookedTransformer

from mechir.pipeline import (
    ContextBuilder,
    answer_logprob,
    attention_to_documents,
    generate_turn,
)

READER = "roneneldan/TinyStories-1M"

QUERY = "what is the capital of france"
DOCS = [
    "Paris is the capital of France.",
    "Bananas grow in warm climates.",
]
# Token-length-matched against DOCS[0], so a prefix patch stays aligned.
CORRUPTED_DOC = "Berlin is the capital of Germany."


@pytest.fixture(scope="module")
def reader():
    return HookedTransformer.from_pretrained(READER, device="cpu")


@pytest.fixture(scope="module")
def builder(reader):
    return ContextBuilder(reader.tokenizer, max_document_tokens=32)


@pytest.fixture
def context(builder):
    return builder.build(QUERY, DOCS, [0, 1])


def zero_hook(activation, hook):
    return torch.zeros_like(activation)


# -- the turn itself -------------------------------------------------------


def test_a_turn_produces_real_text(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=8)

    assert len(turn) == 8
    assert turn.text.strip() != ""
    assert turn.text == reader.tokenizer.decode(turn.token_ids.tolist())


def test_greedy_decoding_is_deterministic(reader, context):
    first = generate_turn(reader, context, max_new_tokens=8)
    second = generate_turn(reader, context, max_new_tokens=8)

    assert first.text == second.text
    assert torch.equal(first.token_ids, second.token_ids)


@pytest.mark.parametrize("budget", [1, 4])
def test_max_new_tokens_is_respected(reader, context, budget):
    turn = generate_turn(reader, context, max_new_tokens=budget)

    assert len(turn) == budget
    assert turn.input_ids.shape[-1] == context.length + budget


# -- spans under a real tokenizer -----------------------------------------


def test_the_span_indexes_the_generated_region_of_the_full_sequence(reader, context):
    """Checked against the tokenizer, not just against the loop's own ids."""
    turn = generate_turn(reader, context, max_new_tokens=8)
    full = turn.input_ids[0]

    assert turn.span.start == context.length
    assert turn.span.end == full.shape[0]
    assert torch.equal(full[: turn.span.start], context.input_ids[0])
    assert reader.tokenizer.decode(full[turn.span.slice()].tolist()) == turn.text


def test_document_spans_survive_generation(reader, context):
    """The prompt is a prefix, so document text still decodes where it was."""
    turn = generate_turn(reader, context, max_new_tokens=6)
    span = turn.context.span_for_rank(0)

    decoded = reader.tokenizer.decode(turn.input_ids[0, span.slice()].tolist())
    assert "Paris" in decoded
    assert span.end <= turn.span.start


# -- hook composition ------------------------------------------------------


def test_zeroing_a_layer_changes_the_turn(reader, context):
    clean = generate_turn(reader, context, max_new_tokens=8)
    patched = generate_turn(
        reader,
        context,
        max_new_tokens=8,
        fwd_hooks=[("blocks.4.hook_resid_post", zero_hook)],
    )

    assert patched.text != clean.text
    assert clean.divergence(patched) is not None


def test_zeroing_one_documents_positions_changes_the_turn(reader, context):
    """The intended use: intervene on a document, watch the turn move.

    This is the generative counterpart of ``ablation.document_necessity`` --
    there the evidence is removed and a teacher-forced log-probability
    moves, here it is removed and the sampled turn itself changes.
    """
    span = context.span_for_rank(0)

    def zero_document(activation, hook):
        activation = activation.clone()
        activation[:, span.slice()] = 0.0
        return activation

    clean = generate_turn(reader, context, max_new_tokens=8)
    patched = generate_turn(
        reader,
        context,
        max_new_tokens=8,
        fwd_hooks=[("blocks.2.hook_resid_post", zero_document)],
    )

    assert patched.text != clean.text


def test_the_prefix_cache_can_source_a_patch(reader, builder):
    """A clean/corrupted comparison, which is what the cache is for.

    Pasting the clean prefix activations into a corrupted rollout restores
    the clean turn *exactly*, which is a sharp check on the recompute
    design: if only the first forward were hooked, or if a KV cache carried
    corrupted keys forward, restoration would hold for one token and then
    drift back to the corrupted turn.
    """
    clean_context = builder.build(QUERY, DOCS, [0, 1])
    # Length-matched on purpose: clean and corrupted must agree on sequence
    # length or the cached activations will not line up with the hooked run.
    corrupted_context = builder.build(QUERY, [CORRUPTED_DOC, DOCS[1]], [0, 1])
    assert clean_context.length == corrupted_context.length, "patching needs alignment"

    clean = generate_turn(reader, clean_context, max_new_tokens=6)
    prefix = clean.cache["hook_embed"]

    def paste_clean_prefix(activation, hook):
        activation = activation.clone()
        activation[:, : prefix.shape[1]] = prefix
        return activation

    corrupted = generate_turn(reader, corrupted_context, max_new_tokens=6)
    restored = generate_turn(
        reader,
        corrupted_context,
        max_new_tokens=6,
        fwd_hooks=[("hook_embed", paste_clean_prefix)],
    )

    assert corrupted.text != clean.text, "the two prompts must differ to begin with"
    assert restored.text == clean.text, "pasting the whole prefix must restore the turn"
    assert corrupted.divergence(restored) is not None


# -- composing with the teacher-forced metrics -----------------------------


def test_as_context_scores_the_generated_turn(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=6)
    scored = turn.as_context()

    assert scored.answer_start == turn.span.start
    assert torch.isfinite(answer_logprob(reader, scored))
    # Greedy tokens are each the argmax, so per-token log-probability is
    # high for this model's vocabulary.
    assert float(answer_logprob(reader, scored, reduction="mean")) > -5.0


def test_attention_from_the_turn_onto_the_documents(reader, context):
    """Attribution over the generated region, via the existing metric."""
    turn = generate_turn(reader, context, max_new_tokens=6)

    attention = attention_to_documents(reader, turn.as_context())

    assert set(attention) == {0, 1}
    assert all(value >= 0 for value in attention.values())


# -- stopping and sampling -------------------------------------------------


def test_eos_truncates_the_turn_to_a_prefix(reader, context):
    full = generate_turn(reader, context, max_new_tokens=8)
    second = int(full.token_ids[1])

    stopped = generate_turn(reader, context, max_new_tokens=8, eos_token_id=second)

    assert stopped.stopped_at_eos
    assert len(stopped) == 1
    assert full.text.startswith(stopped.text)


def test_the_real_eos_token_is_the_default(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=8)

    assert reader.tokenizer.eos_token_id is not None
    assert reader.tokenizer.eos_token_id not in turn.token_ids.tolist()


def test_a_seeded_sample_is_reproducible_and_can_leave_greedy(reader, context):
    greedy = generate_turn(reader, context, max_new_tokens=8)
    first = generate_turn(
        reader, context, max_new_tokens=8, do_sample=True, temperature=1.5, seed=0
    )
    second = generate_turn(
        reader, context, max_new_tokens=8, do_sample=True, temperature=1.5, seed=0
    )

    assert first.text == second.text
    assert first.text != greedy.text
