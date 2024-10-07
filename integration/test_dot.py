import pytest
import torch
from transformers import AutoTokenizer
from your_module import Dot  # Adjust the import based on your module structure

# Constants
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 2
SEQ_LENGTH = 10

@pytest.fixture(scope="module")
def dot_model():
    """Fixture to initialize Dot model."""
    model = Dot(model_name_or_path=MODEL_NAME)
    yield model

@pytest.fixture
def sample_data():
    """Fixture to provide sample input data."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    queries = tokenizer(["Sample query"] * BATCH_SIZE, padding=True, truncation=True, return_tensors='pt', max_length=SEQ_LENGTH)
    documents = tokenizer(["Sample document"] * BATCH_SIZE, padding=True, truncation=True, return_tensors='pt', max_length=SEQ_LENGTH)
    return queries, documents

def test_model_initialization(dot_model):
    """Test that the Dot model initializes correctly."""
    assert dot_model is not None
    assert dot_model.tokenizer is not None
    assert dot_model._pooling_type == 'cls'  # Default pooling type

def test_forward_pass(dot_model, sample_data):
    """Test the forward pass of the model."""
    queries, documents = sample_data
    outputs = dot_model._forward(queries['input_ids'], queries['attention_mask'])
    assert outputs.shape == (BATCH_SIZE, dot_model._model.cfg.hidden_size)

def test_forward_cache(dot_model, sample_data):
    """Test the forward pass with cache."""
    queries, documents = sample_data
    outputs, cache = dot_model._forward_cache(queries['input_ids'], queries['attention_mask'])
    assert outputs.shape == (BATCH_SIZE, dot_model._model.cfg.hidden_size)
    assert isinstance(cache, dict)  # Ensure cache is a dictionary

def test_score_function(dot_model, sample_data):
    """Test the score function."""
    queries, documents = sample_data
    scores, reps_q, reps_d = dot_model.score(queries, documents)
    assert scores.shape == (BATCH_SIZE,)  # Assuming scores are scalar values for each input
    assert reps_q.shape == (BATCH_SIZE, dot_model._model.cfg.hidden_size)
    assert reps_d.shape == (BATCH_SIZE, dot_model._model.cfg.hidden_size)

def test_call_function(dot_model, sample_data):
    """Test the call function."""
    queries, documents = sample_data
    queries_p = queries  # For simplicity, use the same queries
    documents_p = documents  # For simplicity, use the same documents
    patching_output = dot_model(queries, documents, queries_p, documents_p)
    
    assert patching_output is not None
    assert hasattr(patching_output, 'scores')
    assert hasattr(patching_output, 'scores_p')

def test_get_act_patch_block_every(dot_model, sample_data):
    """Test the _get_act_patch_block_every method."""
    queries, documents = sample_data
    corrupted_tokens = documents
    clean_cache = dot_model._model.run_with_cache(corrupted_tokens['input_ids'], one_zero_attention_mask=corrupted_tokens['attention_mask'])[1]  # Get cache

    results = dot_model._get_act_patch_block_every(
        corrupted_tokens['input_ids'],
        clean_cache,
        dot_model._model.run_with_cache,
        torch.randn(BATCH_SIZE, SEQ_LENGTH),  # Dummy scores
        torch.randn(BATCH_SIZE, SEQ_LENGTH)   # Dummy scores_p
    )

    assert results.shape == (3, dot_model._model.cfg.n_layers, SEQ_LENGTH)

# To run the tests, use the command: pytest test_dot.py
