import pytest
import torch
from transformers import AutoTokenizer
from mechir.modelling.hooked.HookedEncoderForSequenceClassification import HookedEncoderForSequenceClassification  # Adjust the import based on your module structure
from transformer_lens.ActivationCache import ActivationCache

# Constants
MODEL_NAME = "bert-base-uncased"  # Use the appropriate model name
BATCH_SIZE = 2
SEQ_LENGTH = 10

@pytest.fixture(scope="module")
def hooked_encoder_model():
    """Fixture to initialize HookedEncoderForSequenceClassification model."""
    model = HookedEncoderForSequenceClassification.from_pretrained(MODEL_NAME)
    yield model

@pytest.fixture
def sample_data():
    """Fixture to provide sample input data."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = tokenizer(["Hello world!"] * BATCH_SIZE, padding=True, truncation=True, return_tensors='pt', max_length=SEQ_LENGTH)["input_ids"]
    return input_ids

def test_model_initialization(hooked_encoder_model):
    """Test that the HookedEncoderForSequenceClassification model initializes correctly."""
    assert hooked_encoder_model is not None
    assert hooked_encoder_model.tokenizer is not None
    assert hasattr(hooked_encoder_model, 'classifier')

def test_forward_pass_logits(hooked_encoder_model, sample_data):
    """Test the forward pass of the model with logits output."""
    logits = hooked_encoder_model(sample_data, return_type='logits')
    assert logits.shape == (BATCH_SIZE, hooked_encoder_model.cfg.d_vocab)  # Ensure output shape matches

def test_forward_pass_none(hooked_encoder_model, sample_data):
    """Test the forward pass of the model with None return type."""
    output = hooked_encoder_model(sample_data, return_type=None)
    assert output is None  # Ensure no output is returned

def test_run_with_cache(hooked_encoder_model, sample_data):
    """Test the run_with_cache method."""
    output, cache = hooked_encoder_model.run_with_cache(sample_data)
    assert output.shape == (BATCH_SIZE, hooked_encoder_model.cfg.d_vocab)  # Ensure output shape matches
    assert isinstance(cache, ActivationCache)  # Ensure cache is an ActivationCache object

def test_run_with_cache_no_object(hooked_encoder_model, sample_data):
    """Test the run_with_cache method without returning a cache object."""
    output, cache_dict = hooked_encoder_model.run_with_cache(sample_data, return_cache_object=False)
    assert output.shape == (BATCH_SIZE, hooked_encoder_model.cfg.d_vocab)  # Ensure output shape matches
    assert isinstance(cache_dict, dict)  # Ensure cache is a dictionary

def test_from_pretrained(hooked_encoder_model):
    """Test loading a pretrained model."""
    assert hooked_encoder_model is not None
    assert hooked_encoder_model.tokenizer is not None
    assert hasattr(hooked_encoder_model, 'classifier')

# To run the tests, use the command: pytest test_hooked_encoder.py