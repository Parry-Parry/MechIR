import pytest
import torch

from mechir.modelling.hooked.config import HookedTransformerConfig

# Devices mechir can legitimately resolve to; kept in step with test_utils.
VALID_DEVICE_TYPES = {"cpu", "cuda", "mps"}


def test_config_defaults():
    config = HookedTransformerConfig(
        d_model=64,
        d_head=8,
        n_layers=2,
        n_ctx=128,
        n_heads=8,
        d_vocab=100,
        # Required: __post_init__ asserts act_fn is set unless attn_only.
        act_fn="relu",
    )
    assert config.d_model == 64
    assert config.n_layers == 2
    assert config.n_heads == 8
    assert config.d_vocab == 100


def test_act_fn_required_for_non_attn_only():
    """A non-attn-only config without ``act_fn`` must be rejected."""
    with pytest.raises(AssertionError, match="act_fn must be specified"):
        HookedTransformerConfig(
            d_model=64, d_head=8, n_layers=2, n_ctx=128, n_heads=8, d_vocab=100
        )


def test_act_fn_optional_for_attn_only():
    config = HookedTransformerConfig(
        d_model=64,
        d_head=8,
        n_layers=2,
        n_ctx=128,
        n_heads=8,
        d_vocab=100,
        attn_only=True,
    )
    assert config.act_fn is None


def test_config_defaults_and_post_init():
    config = HookedTransformerConfig(
        d_model=64,
        d_head=8,
        n_layers=2,
        n_ctx=128,
        n_heads=-1,
        d_vocab=100,
        act_fn="relu",
        attn_only=False,
        device=None,
        d_mlp=None,
        initializer_range=-1.0,
        init_mode="gpt2",
        use_local_attn=False,
        window_size=None,
        attn_types=None,
        d_vocab_out=-1,
        rotary_dim=None,
        num_experts=None,
        experts_per_token=None,
        default_prepend_bos=True,
        n_devices=1,
        attn_scale=-1.0,
        use_attn_scale=True,
    )
    assert config.n_heads == 8
    assert config.d_mlp == 256
    assert config.initializer_range > 0
    assert config.d_vocab_out == 100
    assert config.act_fn == "relu"
    # TODO: `device` is declared Optional[str] upstream but __post_init__
    # assigns utils.get_device(), which returns a torch.device. Assert the
    # real behaviour rather than the annotation.
    assert isinstance(config.device, torch.device)
    assert config.device.type in VALID_DEVICE_TYPES
