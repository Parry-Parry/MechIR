"""MaxSim, and the orientation of its reduction.

`maxsim` is `similarity.max(-1).values.sum(-1)` over a
`(batch, q_len, d_len)` similarity matrix: max over DOCUMENT tokens, sum
over QUERY tokens. The previous implementation reduced the other way round
-- `max(1).values.sum(-1)`, max over query tokens and sum over document
tokens -- which is not a relevance score at all: it grows with the document
length regardless of content, so it ranks long documents first.

The bug is invisible on a square, symmetric similarity matrix, which is
exactly what a self-scored batch padded to one length produces. So these
tests use non-square shapes, compare against an explicit nested-loop
reference, and assert the orientation directly rather than only checking a
number.

No model is loaded here; the end-to-end behaviour lives in
test/acceptance/test_late_interaction_model.py.
"""

import pytest
import torch

from mechir.modelling.hook_names import BLOCK_HOOK_NAMES
from mechir.modelling.late_interaction import (
    MASKED_SIMILARITY,
    LateInteraction,
    maxsim,
)
from mechir.modelling.patched import PatchedMixin

PATCH_METHODS = (
    "get_act_patch_block_every",
    "get_act_patch_attn_head_out_all_pos",
    "get_act_patch_attn_head_by_pos",
)


# -- references ------------------------------------------------------------


def reference_maxsim(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """MaxSim spelled out: per query token, the best document token."""
    scores = []
    for b in range(q.shape[0]):
        total = 0.0
        for i in range(q.shape[1]):  # query tokens: summed over
            total += max(
                float(q[b, i] @ d[b, j])
                for j in range(d.shape[1])  # doc: maxed over
            )
        scores.append(total)
    return torch.tensor(scores)


def transposed_maxsim(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The bug, kept as a test fixture so its behaviour can be contrasted."""
    s = q @ d.permute(0, 2, 1)
    return s.max(1).values.sum(-1)


def unit_vectors() -> torch.Tensor:
    """Two orthogonal query tokens, `(1, 2, 2)`."""
    return torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])


# -- the reduction's orientation ------------------------------------------


def test_maxsim_matches_the_nested_loop_reference():
    """Pins the orientation against a spelled-out MaxSim, on non-square input."""
    q = torch.randn(3, 4, 8)
    d = torch.randn(3, 7, 8)

    assert torch.allclose(maxsim(q, d), reference_maxsim(q, d), atol=1e-5)


def test_the_reference_and_the_transposed_version_actually_differ():
    """Guards the test above: on symmetric input both orientations agree."""
    q = torch.randn(3, 4, 8)
    d = torch.randn(3, 7, 8)

    assert not torch.allclose(
        reference_maxsim(q, d), transposed_maxsim(q, d), atol=1e-5
    )


def test_max_is_over_documents_and_sum_is_over_queries():
    """A hand-computed case where the two orientations give different numbers.

    `q0` matches `d0` exactly, `q1` matches `d1`, and `d2` half-matches both.

        similarity = [[1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.5]]

    MaxSim takes the row maxima (1.0, 1.0) and sums them: 2.0. Maxing over
    query tokens instead takes the column maxima (1.0, 1.0, 0.5): 2.5.
    """
    q = unit_vectors()
    d = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])

    assert torch.allclose(maxsim(q, d), torch.tensor([2.0]))
    assert torch.allclose(transposed_maxsim(q, d), torch.tensor([2.5]))


def test_per_query_token_maxima_are_the_summands():
    """The summands are exactly the per-query-token maxima, individually."""
    q = torch.randn(1, 5, 6)
    d = torch.randn(1, 9, 6)

    per_query_token = (q @ d.transpose(-1, -2)).max(dim=-1).values

    assert per_query_token.shape == (1, 5)  # one summand per QUERY token
    assert torch.allclose(maxsim(q, d), per_query_token.sum(-1), atol=1e-6)
    # Each summand is a real query/document token similarity, i.e. the max
    # was taken over documents and nothing has been averaged.
    for i in range(5):
        best = max(float(q[0, i] @ d[0, j]) for j in range(9))
        assert float(per_query_token[0, i]) == pytest.approx(best, abs=1e-5)


# -- the regression: scores must not grow with document length -------------

#: Document lengths. With the `unit_vectors()` construction below the old
#: implementation scores exactly `d_len` at each -- see
#: `test_score_is_invariant_to_repeated_document_content`.
DOC_LENGTHS = (7, 20, 100)


@pytest.mark.parametrize("d_len", DOC_LENGTHS)
def test_score_is_invariant_to_repeated_document_content(d_len):
    """The same content repeated `d_len` times scores the same as once.

    `q0` matches every document token and `q1` matches none, so MaxSim is
    1.0 + 0.0 however long the document is: extra document tokens are extra
    *candidates* for a fixed number of maxima, not extra summands. The
    transposed reduction instead scores `d_len`, one summand per document
    token.
    """
    q = unit_vectors()
    d = torch.tensor([[1.0, 0.0]]).expand(d_len, 2).unsqueeze(0)

    assert float(maxsim(q, d)) == pytest.approx(1.0, abs=1e-6)
    assert float(transposed_maxsim(q, d)) == pytest.approx(float(d_len), abs=1e-6)


def test_score_does_not_grow_with_document_length():
    """The headline regression, across the measured lengths at once."""
    q = unit_vectors()
    docs = [
        torch.tensor([[1.0, 0.0]]).expand(d_len, 2).unsqueeze(0)
        for d_len in DOC_LENGTHS
    ]

    scores = [float(maxsim(q, d)) for d in docs]
    old_scores = [float(transposed_maxsim(q, d)) for d in docs]

    assert scores == pytest.approx([scores[0]] * len(DOC_LENGTHS), abs=1e-6)
    # The failure mode being pinned: monotone growth in the document length.
    assert old_scores[0] < old_scores[1] < old_scores[2]


@pytest.mark.parametrize("pad_len", [0, 1, 13, 93])
def test_score_is_invariant_to_padding_length(pad_len):
    """One short document padded to several lengths scores identically.

    Padding tokens are not zero vectors -- a real encoder emits ordinary
    activations at padding positions -- so excluding them from the max is
    what makes the score independent of how far the batch was padded.
    """
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    document = torch.randn(1, 7, 8)
    mask_q = torch.ones(1, 4)

    unpadded = maxsim(q, document, mask_q, torch.ones(1, 7))

    padded = torch.cat([document, torch.randn(1, pad_len, 8)], dim=1)
    mask_d = torch.cat([torch.ones(1, 7), torch.zeros(1, pad_len)], dim=1)

    assert float(maxsim(q, padded, mask_q, mask_d)) == pytest.approx(
        float(unpadded), abs=1e-5
    )


def test_padding_that_would_win_the_max_is_excluded():
    """A padding token identical to the query must not contribute."""
    q = unit_vectors()
    # d0 half-matches q0; d1 is a perfect match for q1 but is padding.
    d = torch.tensor([[[0.5, 0.0], [0.0, 1.0]]])
    mask_d = torch.tensor([[1.0, 0.0]])

    assert float(maxsim(q, d, mask_d=mask_d)) == pytest.approx(0.5 + 0.0, abs=1e-6)
    assert float(maxsim(q, d)) == pytest.approx(0.5 + 1.0, abs=1e-6)


def test_score_is_invariant_to_query_padding_length():
    """Padding query tokens add no summands."""
    q = torch.randn(1, 3, 8)
    d = torch.randn(1, 5, 8)

    padded_q = torch.cat([q, torch.randn(1, 11, 8)], dim=1)
    mask_q = torch.cat([torch.ones(1, 3), torch.zeros(1, 11)], dim=1)

    assert float(maxsim(padded_q, d, mask_q=mask_q)) == pytest.approx(
        float(maxsim(q, d)), abs=1e-5
    )


def test_an_all_padding_document_scores_zero():
    """Not `q_len * MASKED_SIMILARITY`, which would swamp a batch mean."""
    q = torch.randn(1, 4, 8)
    d = torch.randn(1, 6, 8)

    score = float(maxsim(q, d, mask_d=torch.zeros(1, 6)))

    assert score == pytest.approx(0.0, abs=1e-6)
    assert score > MASKED_SIMILARITY


# -- shapes, batching and devices -----------------------------------------


def test_returns_one_score_per_batch_item():
    assert maxsim(torch.randn(5, 3, 8), torch.randn(5, 11, 8)).shape == (5,)


def test_batch_items_are_scored_independently():
    q = torch.randn(4, 3, 8)
    d = torch.randn(4, 6, 8)

    batched = maxsim(q, d)

    for b in range(4):
        alone = maxsim(q[b : b + 1], d[b : b + 1])
        assert float(batched[b]) == pytest.approx(float(alone), abs=1e-5)


@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 4, 8)])
def test_a_pooled_document_is_rejected(shape):
    """Passing pooled vectors is a shape error, not a silent broadcast."""
    with pytest.raises(ValueError, match="reps_d"):
        maxsim(torch.randn(2, 3, 8), torch.randn(*shape))


def test_a_pooled_query_is_rejected():
    with pytest.raises(ValueError, match="reps_q"):
        maxsim(torch.randn(2, 8), torch.randn(2, 3, 8))


@pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="needs an accelerator to have two devices to mix",
)
def test_masks_follow_the_activations_onto_the_accelerator():
    """Masks index the activations, so they are moved to the activations' device.

    The caller's masks stay on the CPU while the model's activations live on
    MPS or CUDA, so a mask used as-is raises a device mismatch.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    q = torch.randn(1, 3, 8, device=device)
    d = torch.randn(1, 6, 8, device=device)

    score = maxsim(q, d, torch.ones(1, 3), torch.ones(1, 6))

    assert score.shape == (1,)
    assert score.device.type == device


# -- the scoring seam ------------------------------------------------------


def test_patch_score_is_maxsim_against_the_query_tokens():
    q = torch.randn(2, 3, 8)
    d = torch.randn(2, 7, 8)

    scored = LateInteraction._patch_score(object(), d, reps_q=q)

    assert torch.allclose(scored, reference_maxsim(q, d), atol=1e-5)


def test_patch_score_applies_the_masks_it_is_handed():
    q = unit_vectors()
    d = torch.tensor([[[0.5, 0.0], [0.0, 1.0]]])

    scored = LateInteraction._patch_score(
        object(), d, reps_q=q, mask_d=torch.tensor([[1.0, 0.0]])
    )

    assert float(scored) == pytest.approx(0.5, abs=1e-6)


def test_patch_score_tolerates_surplus_context():
    """`patch` passes `layer_head_list` through to every driver."""
    q = unit_vectors()

    scored = LateInteraction._patch_score(
        object(), torch.ones(1, 3, 2), reps_q=q, layer_head_list=[(0, 0)]
    )

    assert scored.shape == (1,)


def test_patch_score_without_reps_q_fails_clearly():
    with pytest.raises(TypeError, match="reps_q"):
        LateInteraction._patch_score(object(), torch.ones(1, 3, 2))


# -- architecture seam -----------------------------------------------------


def test_late_interaction_is_a_single_stack_ranker():
    """One encoder stack, so the default `blocks.{layer}.…` naming applies."""
    assert LateInteraction._hook_names.fget(object()) is BLOCK_HOOK_NAMES
    assert "_hook_names" not in vars(LateInteraction)


@pytest.mark.parametrize("method", PATCH_METHODS)
def test_patching_drivers_are_inherited(method):
    assert method not in vars(LateInteraction)
    assert getattr(LateInteraction, method) is getattr(PatchedMixin, method)


def test_patch_type_names_are_stable():
    funcs = PatchedMixin._patch_funcs.fget(LateInteraction)

    assert set(funcs) == {"block_all", "head_all", "head_by_pos"}
