from transformer_lens import utils


def test_get_device_real():
    device = utils.get_device()
    assert device == "cuda" or device == "cpu"
