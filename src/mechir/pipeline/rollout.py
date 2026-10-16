"""One hooked generation turn.

Every other reader-side measurement in this package is teacher-forced: the
answer tokens are supplied and scored. That is the right design for a
metric, but it cannot express a turn the model chose for itself -- a ReAct
step, a listwise permutation the ranker emits as text, a refusal. This
module adds exactly one primitive: generate a single turn, hook the forward
passes that produce it, and hand back the sampled tokens together with the
span they occupy and the activation cache of the prefix forward.

**One turn, not a loop.** ``generate_turn`` deliberately stops after one
turn. It does not execute tools, retrieve, or feed its own output back in.
Composition is the caller's business, because the moment a rollout becomes
causal the loop stops being a convenience and starts being a confound.

**No KV cache, by choice.** Decoding re-runs the full prefix for every new
token. That is quadratic and we accept it:

* A patch applied at position ``n`` changes the keys and values written at
  that position. A KV cache built by the unpatched prefix still holds the
  clean K/V, so a cached rollout under a patch silently measures a mixture
  of the two runs. Recomputing is not a shortcut avoided -- it is the only
  way the patched rollout is the thing you think you measured.
* Hooks would otherwise see different inputs at different steps. With a KV
  cache, every forward after the first is over a length-1 sequence, so a
  hook that indexes absolute positions either misfires or quietly becomes a
  no-op. Here every step sees the whole sequence, so one hook definition is
  valid for the baseline run and the patched run alike.
* Baseline and patched runs therefore travel identical code paths, which is
  what makes their outputs comparable at all.

``HookedTransformer.generate`` is not used for the same reasons: it returns
no activations, defaults to ``use_past_kv_cache=True`` and ``do_sample=True``,
and gives no token-level bookkeeping to attribute activations with.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from jaxtyping import Int
from transformer_lens.ActivationCache import ActivationCache

from mechir.pipeline.context import Context, TokenSpan

HookList = Sequence[Tuple[str, Callable]]


@dataclass
class Turn:
    """One generated turn, with everything needed to re-run and compare it.

    Attributes
    ----------
    context:
        The prompt this turn was generated from, unchanged. Re-running it
        under a hook is how a causal comparison is made.
    input_ids:
        Prompt followed by the generated tokens, shaped ``[1, pos]``. The
        prompt is a strict prefix, so ``context.spans`` still index it.
    span:
        Where the generated tokens sit in ``input_ids``. ``span.start`` is
        always ``context.length``.
    token_ids:
        The generated tokens alone, as collected by the decoding loop.
        Identical to ``generated_ids``, which reads them back out through
        ``span`` -- the two agreeing is what pins the span.
    text:
        ``token_ids`` decoded. Empty when the reader has no tokenizer.
    cache:
        Activations from the hooked forward over the prompt prefix, or
        ``None`` when caching was switched off. These are the values to
        source a patch from; note that they cover prompt positions only,
        because the generated tokens did not exist yet. For activations
        *over* the generated region, run ``as_context()`` back through the
        reader.
    stopped_at_eos:
        Whether decoding ended on the end-of-sequence token rather than on
        ``max_new_tokens``. The EOS token itself is not part of the turn.
    """

    context: Context
    input_ids: Int[torch.Tensor, "batch pos"]
    span: TokenSpan
    token_ids: Int[torch.Tensor, "generated"]  # noqa: F821
    text: str
    cache: Optional[ActivationCache] = None
    stopped_at_eos: bool = False

    def __len__(self) -> int:
        return len(self.span)

    @property
    def generated_ids(self) -> Int[torch.Tensor, "generated"]:  # noqa: F821
        """The generated tokens, read out of the full sequence via ``span``."""
        return self.input_ids[0, self.span.slice()]

    def as_context(self) -> Context:
        """This turn teacher-forced, as a ``Context``.

        ``answer_start`` is set to the start of the generated region, so the
        existing reader metrics apply unchanged: ``answer_logprob`` scores
        the turn the model actually produced, and ``attention_to_documents``
        reads attention from the generated positions onto each document.
        Running this context through ``run_with_cache`` is also how to get
        activations *at* the generated positions, which the prefix cache
        cannot contain.
        """
        if len(self.span) == 0:
            raise ValueError("turn generated no tokens, so there is nothing to score")
        return Context(
            input_ids=self.input_ids,
            spans=list(self.context.spans),
            answer_start=self.span.start,
            text=self.context.text + self.text,
        )

    def divergence(self, other: "Turn") -> Optional[int]:
        """First generated position where this turn differs from ``other``.

        ``None`` when the two turns generated the same tokens, which is the
        expected outcome for an intervention with no effect. An index of 0
        means the patch changed the very first sampled token.
        """
        mine, theirs = self.generated_ids, other.generated_ids
        shared = min(mine.shape[0], theirs.shape[0])
        for position in range(shared):
            if int(mine[position]) != int(theirs[position]):
                return position
        if mine.shape[0] != theirs.shape[0]:
            return shared
        return None


@torch.no_grad()
def generate_turn(
    reader,
    context: Context,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    stop_at_eos: bool = True,
    eos_token_id: Optional[int] = None,
    fwd_hooks: Optional[HookList] = None,
    names_filter: Optional[Callable[[str], bool]] = None,
    return_cache: bool = True,
) -> Turn:
    """Generate one turn from ``context``, with ``fwd_hooks`` on every step.

    Greedy by default, because every downstream causal comparison is a
    difference between two rollouts: if decoding is stochastic, the
    difference is noise plus effect and the effect is unrecoverable at this
    scale. Sampling is opt-in and seedable.

    Parameters
    ----------
    reader:
        A TransformerLens ``HookedTransformer``. Batch size must be 1.
    context:
        The prompt, assembled by ``ContextBuilder``. Must carry no answer;
        an answer would be teacher forcing and the point here is not to.
    max_new_tokens:
        Hard bound on the turn length. Costs one full-prefix forward each,
        so keep it small.
    do_sample, temperature, top_k:
        Sampling controls. ``do_sample=False`` ignores all of them and
        takes the argmax.
    seed:
        Seeds a private generator used for sampling, leaving global RNG
        untouched. With ``do_sample=True`` and no seed, the turn is not
        reproducible.
    stop_at_eos, eos_token_id:
        End the turn on EOS. The id defaults to the reader tokenizer's. The
        EOS token is *excluded* from the returned span, ids and text, so the
        span always indexes exactly the tokens the turn consists of;
        ``stopped_at_eos`` records that it happened.
    fwd_hooks:
        ``(name, hook)`` pairs applied to every forward in the turn, the
        prefix pass included. Passing them here rather than wrapping the
        call is what guarantees the later steps are hooked too.
    names_filter, return_cache:
        What to cache on the prefix forward, and whether to cache at all.

    Returns
    -------
    Turn
        Generated ids, decoded text, the generated token span, and the
        prefix activation cache.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
    if context.input_ids.shape[0] != 1:
        raise ValueError(
            "generate_turn decodes one sequence at a time; got a batch of "
            f"{context.input_ids.shape[0]}"
        )
    if context.answer_start is not None:
        raise ValueError(
            "context was built with an answer, which teacher-forces the turn; "
            "build it with answer=None and let the reader generate"
        )
    _check_fits(reader, context.length, max_new_tokens)

    hooks = list(fwd_hooks or ())
    eos = _resolve_eos(reader, eos_token_id) if stop_at_eos else None
    generator = _generator(seed) if do_sample else None

    input_ids = context.input_ids
    generated: List[int] = []
    cache: Optional[ActivationCache] = None
    stopped_at_eos = False

    for step in range(max_new_tokens):
        logits, step_cache = _hooked_forward(
            reader,
            input_ids,
            hooks,
            want_cache=return_cache and step == 0,
            names_filter=names_filter,
        )
        if step == 0:
            cache = step_cache

        next_id = _next_token(
            logits[0, -1],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        if eos is not None and next_id == eos:
            stopped_at_eos = True
            break

        generated.append(next_id)
        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor(
                    [[next_id]], dtype=input_ids.dtype, device=input_ids.device
                ),
            ],
            dim=-1,
        )

    span = TokenSpan(start=context.length, end=context.length + len(generated))
    token_ids = torch.tensor(generated, dtype=input_ids.dtype, device=input_ids.device)

    return Turn(
        context=context,
        input_ids=input_ids,
        span=span,
        token_ids=token_ids,
        text=_decode(reader, generated),
        cache=cache,
        stopped_at_eos=stopped_at_eos,
    )


def _hooked_forward(
    reader,
    input_ids: Int[torch.Tensor, "batch pos"],
    fwd_hooks: HookList,
    want_cache: bool,
    names_filter: Optional[Callable[[str], bool]],
):
    """One forward over the whole sequence, under ``fwd_hooks``.

    The patch hooks are installed before the caching hooks, so a returned
    cache holds post-patch values rather than the clean ones.
    """
    with reader.hooks(fwd_hooks=list(fwd_hooks)):
        if not want_cache:
            return reader(input_ids, return_type="logits"), None
        return reader.run_with_cache(
            input_ids, names_filter=names_filter, return_type="logits"
        )


def _next_token(
    logits: Int[torch.Tensor, "d_vocab"],  # noqa: F821
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    generator: Optional[torch.Generator],
) -> int:
    """Pick the next token id from the final-position logits."""
    if not do_sample:
        return int(logits.argmax(dim=-1))
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    scores = logits.float() / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        k = min(top_k, scores.shape[-1])
        cutoff = torch.topk(scores, k).values[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    # Sampled on CPU: MPS has no reliable seeded generator for multinomial,
    # and a seeded turn must reproduce regardless of where the model lives.
    probs = torch.softmax(scores, dim=-1).cpu()
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


def _generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """A private CPU generator, so sampling does not disturb global RNG."""
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _resolve_eos(reader, eos_token_id: Optional[int]) -> Optional[int]:
    if eos_token_id is not None:
        return eos_token_id
    tokenizer = getattr(reader, "tokenizer", None)
    return getattr(tokenizer, "eos_token_id", None)


def _check_fits(reader, prompt_length: int, max_new_tokens: int) -> None:
    """Fail before decoding rather than mid-turn on a context overflow."""
    n_ctx = getattr(getattr(reader, "cfg", None), "n_ctx", None)
    if n_ctx is None or prompt_length + max_new_tokens <= n_ctx:
        return
    raise ValueError(
        f"prompt of {prompt_length} tokens plus {max_new_tokens} new tokens "
        f"exceeds the reader's context window of {n_ctx}"
    )


def _decode(reader, token_ids: Sequence[int]) -> str:
    tokenizer = getattr(reader, "tokenizer", None)
    if tokenizer is None:
        return ""
    return tokenizer.decode(list(token_ids))


__all__ = ["Turn", "generate_turn"]
