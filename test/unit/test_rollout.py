"""A single hooked generation turn. No downloads: random tiny weights.

The reader is a two-layer ``HookedTransformer`` built straight from a config
and given a stub tokenizer, so these tests pin the decoding loop, the span
arithmetic and the hook plumbing without touching a checkpoint. Real
tokenisers and real text live in ``test/acceptance/test_rollout.py``.
"""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from mechir.pipeline import (
    Context,
    DocumentSpan,
    TokenSpan,
    answer_logprob,
    generate_turn,
)

D_VOCAB = 37
EOS = 36
PROMPT = [3, 5, 7, 11, 13]


class StubTokenizer:
    """Just enough tokenizer for decoding and EOS lookup."""

    eos_token_id = EOS
    # TransformerLens reads this off the tokenizer on every forward.
    padding_side = "right"

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture(scope="module")
def reader():
    torch.manual_seed(1234)
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        n_ctx=64,
        d_head=8,
        n_heads=4,
        d_vocab=D_VOCAB,
        act_fn="gelu_new",
        device="cpu",
    )
    model = HookedTransformer(cfg)
    model.tokenizer = StubTokenizer()
    return model


@pytest.fixture
def context():
    return Context(input_ids=torch.tensor([PROMPT]), spans=[DocumentSpan(0, 0, 1, 4)])


def zero_hook(activation, hook):
    return torch.zeros_like(activation)


# -- span machinery --------------------------------------------------------


def test_token_span_is_a_half_open_range():
    span = TokenSpan(start=4, end=9)

    assert len(span) == 5
    assert span.slice() == slice(4, 9)
    assert list(range(10))[span.slice()] == [4, 5, 6, 7, 8]


def test_document_span_keeps_its_positional_signature():
    """Sharing span behaviour must not reorder ``DocumentSpan``'s fields."""
    span = DocumentSpan(2, 1, 5, 8)

    assert (span.candidate, span.rank, span.start, span.end) == (2, 1, 5, 8)
    assert len(span) == 3


# -- determinism -----------------------------------------------------------


def test_greedy_decoding_is_deterministic(reader, context):
    first = generate_turn(reader, context, max_new_tokens=5)
    second = generate_turn(reader, context, max_new_tokens=5)

    assert torch.equal(first.token_ids, second.token_ids)
    assert first.text == second.text
    assert first.span == second.span


def test_greedy_ignores_sampling_controls(reader, context):
    plain = generate_turn(reader, context, max_new_tokens=4)
    noisy = generate_turn(reader, context, max_new_tokens=4, temperature=0.1, top_k=3)

    assert torch.equal(plain.token_ids, noisy.token_ids)


# -- the generated span ----------------------------------------------------


def test_span_indexes_exactly_the_generated_tokens(reader, context):
    """The pin that everything downstream depends on.

    An off-by-one here would silently attribute activations to the wrong
    region, so the span is checked three independent ways: against the
    prompt it must start after, against the sequence it must end with, and
    against the ids the decoding loop collected for itself.
    """
    turn = generate_turn(reader, context, max_new_tokens=4)

    # Starts immediately after the prompt, ends at the end of the sequence.
    assert turn.span.start == context.length
    assert turn.span.end == turn.input_ids.shape[-1]
    assert len(turn.span) == 4

    # The prompt is untouched, and the token before the span is its last.
    assert torch.equal(turn.input_ids[0, : turn.span.start], context.input_ids[0])
    assert int(turn.input_ids[0, turn.span.start - 1]) == PROMPT[-1]

    # Reading the span out of the sequence recovers the loop's own record.
    assert torch.equal(turn.generated_ids, turn.token_ids)
    assert turn.generated_ids.tolist() == turn.token_ids.tolist()


def test_span_start_is_the_first_token_the_model_chose(reader, context):
    """Independent check on ``span.start``: decode one token and compare.

    A one-token turn can only contain the first sampled token, so its id
    must equal whatever sits at ``span.start`` in a longer turn. This fails
    if the span is shifted by one in either direction.
    """
    long_turn = generate_turn(reader, context, max_new_tokens=4)
    one_token = generate_turn(reader, context, max_new_tokens=1)

    first = int(one_token.token_ids[0])
    assert int(long_turn.input_ids[0, long_turn.span.start]) == first
    assert int(long_turn.token_ids[0]) == first


def test_document_spans_still_index_the_prompt_after_generation(reader, context):
    """The prompt stays a strict prefix, so document spans survive."""
    turn = generate_turn(reader, context, max_new_tokens=3)
    span = turn.context.spans[0]

    assert torch.equal(
        turn.input_ids[0, span.slice()], context.input_ids[0, span.slice()]
    )
    assert span.end <= turn.span.start


def test_text_decodes_the_generated_tokens_only(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=3)

    assert turn.text == "".join(f"<{i}>" for i in turn.token_ids.tolist())
    assert f"<{PROMPT[-1]}>" not in turn.text or PROMPT[-1] in turn.token_ids.tolist()


# -- length bounds ---------------------------------------------------------


@pytest.mark.parametrize("budget", [1, 2, 3, 8])
def test_max_new_tokens_is_respected(reader, context, budget):
    turn = generate_turn(reader, context, max_new_tokens=budget, stop_at_eos=False)

    assert len(turn) == budget
    assert turn.token_ids.shape[0] == budget
    assert turn.input_ids.shape[-1] == context.length + budget


def test_one_token_is_a_valid_turn(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=1)

    assert len(turn) == 1
    assert turn.span == TokenSpan(start=context.length, end=context.length + 1)
    assert not turn.stopped_at_eos


def test_max_new_tokens_must_be_positive(reader, context):
    with pytest.raises(ValueError, match="at least 1"):
        generate_turn(reader, context, max_new_tokens=0)


def test_a_turn_that_would_overflow_the_window_is_refused(reader, context):
    with pytest.raises(ValueError, match="context window"):
        generate_turn(reader, context, max_new_tokens=reader.cfg.n_ctx)


# -- hook composition ------------------------------------------------------


def test_a_zeroing_hook_changes_the_generated_turn(reader, context):
    """Proof the rollout runs through the hook machinery.

    If decoding bypassed hooks -- by calling the underlying module, or by
    hooking only the prefix pass -- this would come back identical.
    """
    clean = generate_turn(reader, context, max_new_tokens=4, stop_at_eos=False)
    patched = generate_turn(
        reader,
        context,
        max_new_tokens=4,
        stop_at_eos=False,
        fwd_hooks=[("blocks.1.hook_resid_post", zero_hook)],
    )

    assert not torch.equal(clean.token_ids, patched.token_ids)
    assert clean.divergence(patched) == 0


def test_hooks_fire_on_every_step_over_the_whole_sequence(reader, context):
    """Each step must be hooked, and each must see the full prefix.

    The recorded lengths growing by one per step is the recompute design
    being enforced: with a KV cache every step after the first would see a
    length-1 sequence and a position-indexed patch would misfire.
    """
    seen = []

    def record(activation, hook):
        seen.append(activation.shape[1])
        return activation

    generate_turn(
        reader,
        context,
        max_new_tokens=3,
        stop_at_eos=False,
        fwd_hooks=[("blocks.0.hook_resid_post", record)],
    )

    assert seen == [context.length, context.length + 1, context.length + 2]


def test_hooks_are_removed_after_the_turn(reader, context):
    generate_turn(
        reader,
        context,
        max_new_tokens=2,
        fwd_hooks=[("blocks.1.hook_resid_post", zero_hook)],
    )
    after = generate_turn(reader, context, max_new_tokens=2)
    clean = generate_turn(reader, context, max_new_tokens=2)

    assert torch.equal(after.token_ids, clean.token_ids)


def test_a_position_restricted_patch_only_bites_after_the_prompt(reader, context):
    """A hook keyed on absolute position is meaningful here, as designed."""

    def patch_generated_positions(activation, hook):
        activation = activation.clone()
        activation[:, context.length :] = 0.0
        return activation

    clean = generate_turn(reader, context, max_new_tokens=4, stop_at_eos=False)
    patched = generate_turn(
        reader,
        context,
        max_new_tokens=4,
        stop_at_eos=False,
        fwd_hooks=[("blocks.1.hook_resid_post", patch_generated_positions)],
    )

    # The first token is produced from the prompt alone, so it cannot move;
    # everything after it depends on a patched position.
    assert int(clean.token_ids[0]) == int(patched.token_ids[0])
    assert clean.divergence(patched) is not None
    assert clean.divergence(patched) >= 1


# -- the activation cache --------------------------------------------------


def test_cache_covers_the_prompt_prefix(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=4)

    assert turn.cache is not None
    resid = turn.cache["blocks.0.hook_resid_post"]
    assert resid.shape[1] == context.length, "cache is the prefix forward"


def test_cache_can_be_filtered_or_switched_off(reader, context):
    filtered = generate_turn(
        reader,
        context,
        max_new_tokens=2,
        names_filter=lambda name: name == "blocks.1.hook_resid_post",
    )
    uncached = generate_turn(reader, context, max_new_tokens=2, return_cache=False)

    assert list(filtered.cache.keys()) == ["blocks.1.hook_resid_post"]
    assert uncached.cache is None
    assert torch.equal(filtered.token_ids, uncached.token_ids)


def test_cache_holds_post_patch_values(reader, context):
    """A patched run's cache must show the patch, not the clean activation."""
    turn = generate_turn(
        reader,
        context,
        max_new_tokens=2,
        fwd_hooks=[("blocks.1.hook_resid_post", zero_hook)],
    )

    assert float(turn.cache["blocks.1.hook_resid_post"].abs().max()) == 0.0


# -- stopping --------------------------------------------------------------


def test_eos_ends_the_turn_and_is_excluded_from_the_span(reader, context):
    """Stop on the token that greedy decoding would have produced second."""
    unstopped = generate_turn(reader, context, max_new_tokens=4, stop_at_eos=False)
    second = int(unstopped.token_ids[1])

    turn = generate_turn(reader, context, max_new_tokens=4, eos_token_id=second)

    assert turn.stopped_at_eos
    assert len(turn) == 1
    assert turn.span.end == turn.input_ids.shape[-1]
    assert second not in turn.token_ids.tolist()
    assert int(turn.token_ids[0]) == int(unstopped.token_ids[0])


def test_an_immediate_eos_gives_an_empty_turn(reader, context):
    greedy = generate_turn(reader, context, max_new_tokens=1)

    turn = generate_turn(
        reader, context, max_new_tokens=4, eos_token_id=int(greedy.token_ids[0])
    )

    assert turn.stopped_at_eos
    assert len(turn) == 0
    assert turn.text == ""
    assert torch.equal(turn.input_ids, context.input_ids)
    with pytest.raises(ValueError, match="no tokens"):
        turn.as_context()


def test_stop_at_eos_false_decodes_through_it(reader, context):
    greedy = generate_turn(reader, context, max_new_tokens=1)
    forced = int(greedy.token_ids[0])

    turn = generate_turn(
        reader, context, max_new_tokens=3, eos_token_id=forced, stop_at_eos=False
    )

    assert len(turn) == 3
    assert not turn.stopped_at_eos


def test_the_tokenizers_eos_is_the_default(reader, context):
    """``stop_at_eos`` with no explicit id reads the reader's tokenizer."""
    assert reader.tokenizer.eos_token_id == EOS

    turn = generate_turn(reader, context, max_new_tokens=4)

    assert EOS not in turn.token_ids.tolist()


# -- sampling --------------------------------------------------------------


def test_a_seeded_sample_is_reproducible(reader, context):
    first = generate_turn(reader, context, max_new_tokens=5, do_sample=True, seed=7)
    second = generate_turn(reader, context, max_new_tokens=5, do_sample=True, seed=7)

    assert torch.equal(first.token_ids, second.token_ids)


def test_sampling_can_leave_the_greedy_path(reader, context):
    greedy = generate_turn(reader, context, max_new_tokens=5, stop_at_eos=False)
    sampled = [
        generate_turn(
            reader,
            context,
            max_new_tokens=5,
            stop_at_eos=False,
            do_sample=True,
            seed=seed,
        )
        for seed in range(5)
    ]

    assert any(not torch.equal(turn.token_ids, greedy.token_ids) for turn in sampled)


def test_top_k_one_reproduces_greedy(reader, context):
    greedy = generate_turn(reader, context, max_new_tokens=4, stop_at_eos=False)
    sampled = generate_turn(
        reader,
        context,
        max_new_tokens=4,
        stop_at_eos=False,
        do_sample=True,
        top_k=1,
        seed=3,
    )

    assert torch.equal(sampled.token_ids, greedy.token_ids)


def test_sampling_does_not_disturb_global_rng(reader, context):
    torch.manual_seed(99)
    expected = torch.rand(3)

    torch.manual_seed(99)
    generate_turn(reader, context, max_new_tokens=4, do_sample=True, seed=11)

    assert torch.equal(torch.rand(3), expected)


def test_temperature_must_be_positive(reader, context):
    with pytest.raises(ValueError, match="temperature must be positive"):
        generate_turn(reader, context, max_new_tokens=2, do_sample=True, temperature=0)


def test_top_k_must_be_positive(reader, context):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        generate_turn(reader, context, max_new_tokens=2, do_sample=True, top_k=0)


# -- composing with the rest of the pipeline -------------------------------


def test_as_context_points_the_metrics_at_the_generated_region(reader, context):
    turn = generate_turn(reader, context, max_new_tokens=3, stop_at_eos=False)
    scored = turn.as_context()

    assert scored.answer_start == turn.span.start
    assert scored.length == turn.input_ids.shape[-1]
    assert scored.spans == turn.context.spans
    assert torch.isfinite(answer_logprob(reader, scored))


def test_a_greedy_turn_is_the_most_likely_continuation(reader, context):
    """Sanity on ``as_context``: greedy beats a perturbed alternative.

    Each greedy token is the argmax of its position, so swapping any of them
    for another token cannot raise the sequence log-probability.
    """
    turn = generate_turn(reader, context, max_new_tokens=3, stop_at_eos=False)
    greedy_score = answer_logprob(reader, turn.as_context())

    altered = turn.as_context()
    altered.input_ids = altered.input_ids.clone()
    altered.input_ids[0, turn.span.start] = (
        int(turn.token_ids[0]) + 1
    ) % D_VOCAB  # any other token

    assert greedy_score >= answer_logprob(reader, altered)


def test_divergence_is_none_when_two_turns_agree(reader, context):
    first = generate_turn(reader, context, max_new_tokens=3)
    second = generate_turn(reader, context, max_new_tokens=3)

    assert first.divergence(second) is None


def test_divergence_reports_a_length_difference(reader, context):
    long_turn = generate_turn(reader, context, max_new_tokens=3, stop_at_eos=False)
    short_turn = generate_turn(reader, context, max_new_tokens=2, stop_at_eos=False)

    assert long_turn.divergence(short_turn) == 2


# -- input contract --------------------------------------------------------


def test_a_batched_context_is_refused(reader):
    batched = Context(input_ids=torch.tensor([PROMPT, PROMPT]))

    with pytest.raises(ValueError, match="one sequence at a time"):
        generate_turn(reader, batched, max_new_tokens=2)


def test_a_teacher_forced_context_is_refused(reader):
    """An answer in the prompt would make the "generated" turn meaningless."""
    forced = Context(input_ids=torch.tensor([PROMPT]), answer_start=3)

    with pytest.raises(ValueError, match="teacher-forces"):
        generate_turn(reader, forced, max_new_tokens=2)
