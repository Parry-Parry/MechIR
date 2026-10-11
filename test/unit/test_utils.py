import torch
from transformer_lens import utils

# Devices mechir can legitimately resolve to. "mps" matters for local
# development on Apple silicon.
VALID_DEVICE_TYPES = {"cpu", "cuda", "mps"}


def test_get_device_returns_a_torch_device():
    """``get_device`` returns a ``torch.device``, not a string.

    ``HookedTransformerConfig.__post_init__`` assigns this straight onto
    ``cfg.device``, so anything comparing ``cfg.device`` to a string literal
    will silently never match.
    """
    device = utils.get_device()
    assert isinstance(device, torch.device)
    assert device.type in VALID_DEVICE_TYPES
