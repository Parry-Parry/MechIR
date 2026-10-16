"""RankT5 against real T5 weights.

RankT5 shares MonoT5's architecture and differs only in its scoring rule, so
what needs real weights is that the shared machinery still resolves for it:
the hook names exist on the model, patching moves the score, and the score
itself is the decoder's relevance logit rather than a `true`/`false`
contrast.

The fast tests run on the small MonoT5 checkpoint -- the scoring rule is a
property of the class, not of the weights, and that checkpoint is the one
test/acceptance/test_monot5.py already downloads. The real RankT5 weights
are exercised by the `integration`-marked test at the bottom, which is
excluded from the fast suite because it pulls ~1GB.
"""

import pytest
import torch
from transformers import AutoModelForSeq2SeqLM

from mechir import MonoT5
from mechir.modelling.rankt5 import RELEVANCE_TOKEN, RankT5

#: Small T5 ranker, shared with test_monot5.py so the fast suite downloads
#: nothing new.
MODEL = "castorini/monot5-small-msmarco-10k"

#: Real RankT5 weights, for the integration-marked test only.
RANKT5_MODEL = "Soyoung97/RankT5-base"

QUERY = "what is a cat"
RELEVANT = "a cat is a small domesticated carnivorous mammal"
IRRELEVANT = "the eiffel tower is in paris, france"


@pytest.fixture(scope="module")
def rankt5():
    return RankT5(MODEL, return_cache=True)


def encode(model, documents) -> dict:
    enc = model.tokenizer(
        [f"Query: {QUERY} Document: {document}" for document in documents],
        return_tensors="pt",
        padding=True,
    )
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


@pytest.fixture(scope="module")
def batch(rankt5):
    return encode(rankt5, [RELEVANT])


def test_relevance_token_is_a_single_id(rankt5):
    assert isinstance(rankt5.relevance_token, int)
    assert rankt5.tokenizer.convert_ids_to_tokens([rankt5.relevance_token]) == [
        RELEVANCE_TOKEN
    ]


def test_rejects_a_multi_token_relevance_label():
    with pytest.raises(ValueError, match="single token"):
        RankT5(MODEL, relevance_token="definitely relevant")


def test_score_returns_one_scalar_per_item(rankt5, batch):
    scores, cache = rankt5.score(batch)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (batch["input_ids"].shape[0],)
    assert torch.isfinite(scores).all()
    assert cache is None


def test_score_with_cache(rankt5, batch):
    scores, cache = rankt5.score(batch, cache=True)

    assert scores.shape == (batch["input_ids"].shape[0],)
    assert cache is not None


def test_score_is_the_decoder_relevance_logit(rankt5, batch):
    """The scoring rule, against the HuggingFace model it was loaded from."""
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    decoder_input = torch.zeros((batch["input_ids"].shape[0], 1), dtype=torch.long)
    with torch.no_grad():
        reference = hf_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=decoder_input,
        ).logits[:, 0, rankt5.relevance_token]

    scores, _ = rankt5.score(batch)

    assert torch.allclose(scores.cpu(), reference, atol=1e-2)


def test_scoring_rule_differs_from_monot5(rankt5, batch):
    """Same weights, same batch, different rule -- so different scores.

    A RankT5 that had quietly inherited MonoT5's `true`/`false` contrast
    would pass every shape test above.
    """
    monot5 = MonoT5(MODEL)

    scores, _ = rankt5.score(batch)
    monot5_scores, _ = monot5.score(batch)

    assert not torch.allclose(scores, monot5_scores)


def test_hook_names_target_the_encoder_stack(rankt5):
    """The MonoT5 bug this family inherits the fix for: hook names were
    built as `blocks.{layer}.…` against an encoder-decoder whose stacks are
    `encoder.{layer}.…` / `decoder.{layer}.…`, so patching died with
    `KeyError: '_model.blocks.0.hook_resid_pre'`."""
    for component in ("resid_pre", "attn_out", "mlp_out", "z", "pattern"):
        name = rankt5._hook_names(component, 0)
        assert name.startswith("_model.encoder.0."), name
        assert name in rankt5.hook_dict, name


@pytest.mark.parametrize(
    "patch_type, shape",
    [
        ("block_all", lambda m, seq: (3, m._model.cfg.n_layers, seq)),
        ("head_all", lambda m, seq: (m._model.cfg.n_layers, m._model.cfg.n_heads)),
        ("head_by_pos", lambda m, seq: (2, 1, seq)),
    ],
)
def test_patch_returns_the_shared_shapes(rankt5, batch, patch_type, shape):
    """Patching is inherited, so the shapes are MonoT5's, Dot's and Cat's."""
    seq_len = batch["input_ids"].shape[1]

    result, _ = rankt5.patch(
        batch, batch, patch_type=patch_type, layer_head_list=[(0, 0)]
    )

    assert result.shape == shape(rankt5, seq_len)
    assert torch.isfinite(result).all()


def test_patch_responds_to_a_perturbation(rankt5):
    """A patch map over a real perturbation is not identically zero.

    The shape tests above would also pass for a loop that patched nothing --
    which is what MonoT5 did before hook naming became an architecture-level
    strategy -- so pin that the hooks actually move RankT5's score.
    """
    sequences = encode(rankt5, [RELEVANT, f"{RELEVANT} cat"])
    clean = {key: value[:1] for key, value in sequences.items()}
    perturbed = {key: value[1:] for key, value in sequences.items()}

    result, _ = rankt5.patch(clean, perturbed, patch_type="head_all")

    assert torch.isfinite(result).all()
    assert (result != 0).any()


def test_patch_does_not_mutate_the_caller_batch(rankt5, batch):
    before = batch["input_ids"].clone()

    rankt5.patch(batch, batch, patch_type="head_all")

    assert torch.equal(batch["input_ids"], before)
    assert batch["input_ids"].device == before.device


@pytest.mark.integration
def test_real_checkpoint_ranks_a_relevant_document_higher():
    """The released RankT5 weights, which the fast suite does not download.

    Also the only coverage of the checkpoint-name registration: without it
    TransformerLens rejects `Soyoung97/RankT5-base` outright.
    """
    model = RankT5(RANKT5_MODEL)
    sequences = encode(model, [RELEVANT, IRRELEVANT])

    scores, _ = model.score(sequences)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert scores[0] > scores[1]
