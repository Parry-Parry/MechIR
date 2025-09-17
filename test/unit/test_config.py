from mechir.modelling.hooked.config import HookedTransformerConfig


def test_config_defaults():
    config = HookedTransformerConfig(
        d_model=64, d_head=8, n_layers=2, n_ctx=128, n_heads=8, d_vocab=100
    )
    assert config.d_model == 64
    assert config.n_layers == 2
    assert config.n_heads == 8
    assert config.d_vocab == 100


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
    assert config.device in ["cpu", "cuda"]
    assert config.act_fn == "relu"
