import pytest
import torch
from transformers import AutoTokenizer
from your_module import MonoT5  # Adjust the import based on your module structure

# Constants
MODEL_NAME = "t5-base"  # Use the appropriate model name
BATCH_SIZE = 2
SEQ_LENGTH = 10

@pytest.fixture(scope="module")
def monot5_model():
    """Fixture to initialize MonoT5 model."""
    model = MonoT5(model_name_or_path=MODEL_NAME)
    yield model

@pytest.fixture
def sample_data():
    """Fixture to provide sample input data."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    queries = tokenizer(["Sample query"] * BATCH_SIZE, padding=True, truncation=True, return_tensors='pt', max_length=SEQ_LENGTH)
    documents = tokenizer(["Sample document"] * BATCH_SIZE, padding=True, truncation=True, return_tensors='pt', max_length=SEQ_LENGTH)
    return queries, documents

def test_model_initialization(monot5_model):
    """Test that the MonoT5 model initializes correctly."""
    assert monot5_model is not None
    assert monot5_model.tokenizer is not None
    assert monot5_model.pos_token.shape == (1,)
    assert monot5_model.neg_token.shape == (1,)

def test_forward_pass(monot5_model, sample_data):
    """Test the forward pass of the model."""
    queries, documents = sample_data
    outputs = monot5_model._forward(queries['input_ids'], queries['attention_mask'])
    assert outputs.shape == (BATCH_SIZE, 2, monot5_model.tokenizer.vocab_size)  # Adjust output shape as needed

def test_forward_cache(monot5_model, sample_data):
    """Test the forward pass with cache."""
    queries, documents = sample_data
    outputs, cache = monot5_model._forward_cache(queries['input_ids'], queries['attention_mask'])
    assert outputs.shape == (BATCH_SIZE, 2, monot5_model.tokenizer.vocab_size)  # Adjust as needed
    assert isinstance(cache, dict)  # Ensure cache is a dictionary

def test_score_function(monot5_model, sample_data):
    """Test the score function."""
    queries, documents = sample_data
    scores, logits = monot5_model.score(queries)
    assert scores.shape == (BATCH_SIZE, 2)  # Assuming two classes for positive and negative tokens
    assert logits.shape == (BATCH_SIZE, 2, monot5_model.tokenizer.vocab_size)

def test_call_function(monot5_model, sample_data):
    """Test the call function."""
    queries, documents = sample_data
    queries_p = queries  # For simplicity, use the same queries
    documents_p = documents  # For simplicity, use the same documents
    patching_output = monot5_model(queries, queries_p)

    assert patching_output is not None
    assert hasattr(patching_output, 'scores')
    assert hasattr(patching_output, 'scores_p')

def test_get_act_patch_block_every(monot5_model, sample_data):
    """Test the _get_act_patch_block_every method."""
    queries, documents = sample_data
    corrupted_tokens = documents
    clean_cache = monot5_model._model.run_with_cache(corrupted_tokens['input_ids'], one_zero_attention_mask=corrupted_tokens['attention_mask'])[1]  # Get cache

    results = monot5_model._get_act_patch_block_every(
        corrupted_tokens['input_ids'],
        clean_cache,
        monot5_model._model.run_with_cache,
        torch.randn(BATCH_SIZE, SEQ_LENGTH),  # Dummy scores
        torch.randn(BATCH_SIZE, SEQ_LENGTH)   # Dummy scores_p
    )

    assert results.shape == (3, monot5_model._model.cfg.n_layers, SEQ_LENGTH)

# To run the tests, use the command: pytest test_monot5.py
