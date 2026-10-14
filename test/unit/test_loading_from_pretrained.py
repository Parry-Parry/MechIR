"""Unit tests for architecture resolution in ``loading_from_pretrained``.

These never touch the network: the HuggingFace config is built directly as a
dict, which is all :func:`resolve_architecture` reads.
"""

import pytest

from mechir.modelling.hooked.loading_from_pretrained import (
    MODEL_TYPE_TO_ARCHITECTURE,
    resolve_architecture,
)


def test_declared_architecture_wins():
    config = {"architectures": ["BertForSequenceClassification"], "model_type": "bert"}
    expected = "BertForSequenceClassification"
    assert resolve_architecture(config, "some/checkpoint") == expected


def test_none_architectures_falls_back_to_model_type():
    """Regression test: ``architectures: None`` used to raise ``TypeError``.

    ``prajjwal1/bert-tiny`` is a real checkpoint whose config.json has no
    ``architectures`` key, so ``hf_config.architectures`` is ``None`` and
    ``architectures[0]`` blew up before reaching any architecture lookup.
    """
    config = {"architectures": None, "model_type": "bert"}
    assert resolve_architecture(config, "prajjwal1/bert-tiny") == "BertModel"


def test_empty_architectures_falls_back_to_model_type():
    config = {"architectures": [], "model_type": "distilbert"}
    assert resolve_architecture(config, "some/checkpoint") == "DistilBertModel"


@pytest.mark.parametrize("model_type", sorted(MODEL_TYPE_TO_ARCHITECTURE))
def test_every_mapped_model_type_resolves(model_type: str):
    config = {"architectures": None, "model_type": model_type}
    expected = MODEL_TYPE_TO_ARCHITECTURE[model_type]
    assert resolve_architecture(config, "some/checkpoint") == expected


def test_unmappable_model_type_raises_an_actionable_error():
    config = {"architectures": None, "model_type": "mamba"}
    with pytest.raises(ValueError) as excinfo:
        resolve_architecture(config, "some/checkpoint")
    message = str(excinfo.value)
    # The error must name the checkpoint and the model_type it could not map.
    assert "some/checkpoint" in message
    assert "mamba" in message


def test_missing_model_type_raises_rather_than_indexing_none():
    with pytest.raises(ValueError, match="declares no `architectures`"):
        resolve_architecture({}, "some/checkpoint")


def test_config_objects_are_supported():
    """Real callers pass a ``PretrainedConfig``, not a dict."""
    from transformers import BertConfig

    # BertConfig() leaves `architectures` unset, i.e. None -- the bug's shape.
    config = BertConfig()
    assert config.architectures is None
    assert resolve_architecture(config, "prajjwal1/bert-tiny") == "BertModel"
