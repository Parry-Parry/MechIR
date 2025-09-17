import pytest
from mechir import config


def test_singleton_instance():
    from mechir import MechirConfig

    c1 = MechirConfig()
    c2 = MechirConfig()
    assert c1 is c2


def test_get_config_value():
    assert config("ignore-official") is True
    assert config["ignore-official"] is True


def test_set_config_value():
    config("ignore-official", False)
    assert config("ignore-official") is False
    config("ignore-official", True)  # restore


def test_key_error_on_invalid_key():
    with pytest.raises(KeyError):
        config("nonexistent-key")
    with pytest.raises(KeyError):
        _ = config["nonexistent-key"]
