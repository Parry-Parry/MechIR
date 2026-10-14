"""Hook naming: how a ``(component, layer)`` pair becomes a hook name.

:func:`transformer_lens.utils.get_act_name` names the hooks of a transformer
stack ``blocks.{layer}.…``. That is right for the encoder-only and
decoder-only backbones behind :class:`~mechir.Dot`, :class:`~mechir.Cat` and
:class:`~mechir.GenerativeRanker`, and wrong for encoder-decoders:
``HookedEncoderDecoder`` registers two stacks named ``encoder`` and
``decoder``, so :class:`~mechir.MonoT5`'s hooks are ``encoder.{layer}.…`` and
``decoder.{layer}.…`` and a ``blocks.…`` lookup raises ``KeyError``.

Naming is therefore a property of the architecture, not something the patching
loops can hardcode. Each ranker exposes one of these strategies through
``PatchedMixin._hook_names`` and the loops ask it for every hook they install.
"""

from typing import Protocol

import transformer_lens.utils as utils

# Every mechir ranker holds its hooked backbone on ``self._model``, so hook
# names as seen from the ranker are the backbone's own names under that
# attribute.
MODEL_PREFIX = "_model."

# The stack name TransformerLens itself emits, and so the one to rewrite.
BLOCKS = "blocks"


class HookNameStrategy(Protocol):
    """Maps a TransformerLens component and layer index to a hook name."""

    def __call__(self, component: str, layer: int) -> str:
        """Return the fully qualified hook name."""
        ...


class StackHookNames:
    """Hook names for one transformer stack of a hooked backbone.

    Parameters
    ----------
    stack: str
        Name of the stack holding the layers, as it appears in the hook
        name. ``"blocks"`` for a single-stack model, ``"encoder"`` or
        ``"decoder"`` for the two stacks of an encoder-decoder.
    prefix: str
        Attribute path from the ranker to its backbone.
    """

    def __init__(self, stack: str = BLOCKS, prefix: str = MODEL_PREFIX) -> None:
        self.stack = stack
        self.prefix = prefix

    def __call__(self, component: str, layer: int) -> str:
        """Return the fully qualified hook name for ``component`` at ``layer``."""
        name = utils.get_act_name(component, layer)
        if self.stack != BLOCKS:
            name = name.replace(BLOCKS, self.stack, 1)
        return self.prefix + name


#: Single-stack backbones: encoder-only and decoder-only rankers.
BLOCK_HOOK_NAMES = StackHookNames(BLOCKS)

#: The encoder stack of an encoder-decoder backbone. Patching runs here
#: because the positions being patched are the encoder's input tokens.
ENCODER_HOOK_NAMES = StackHookNames("encoder")

#: The decoder stack of an encoder-decoder backbone.
DECODER_HOOK_NAMES = StackHookNames("decoder")


__all__ = [
    "BLOCKS",
    "BLOCK_HOOK_NAMES",
    "DECODER_HOOK_NAMES",
    "ENCODER_HOOK_NAMES",
    "HookNameStrategy",
    "MODEL_PREFIX",
    "StackHookNames",
]
