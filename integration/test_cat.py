import pytest
import torch
from mechir.modelling import Cat  # Replace with the actual path to your module

# Sample parameters for the model
MODEL_NAME = "distilbert-base-uncased"  # Replace with your desired model
NUM_LABELS = 2  # Adjust based on your specific use case

@pytest.fixture(scope='module')
def cat_model():
    """Fixture to initialize Cat model once per module."""
    model = Cat(model_name_or_path=MODEL_NAME, num_labels=NUM_LABELS)
    yield model
    del model

def test_model_initialization(cat_model):
    """Test that the model initializes correctly."""
    assert cat_model is not None
    assert cat_model.tokenizer is not None
    assert isinstance(cat_model._model, type(cat_model._model))

def test_forward_pass(cat_model):
    """Test the forward pass of the model."""
    input_ids = torch.tensor([[101, 2023, 2003, 1037, 3921, 102]])  # Example input (DistilBERT)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])  # Example attention mask
    outputs = cat_model._forward(input_ids, attention_mask)
    
    assert outputs.shape == (1, NUM_LABELS)  # Check the output shape

def test_score_function(cat_model):
    """Test the score function of the model."""
    sequences = {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3921, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    scores, logits = cat_model.score(sequences)
    
    assert scores.shape == (1, NUM_LABELS)  # Check the output shape
    assert logits.shape == (1, NUM_LABELS)  # Check logits shape

def test_patch_function(cat_model):
    """Test the patching function."""
    sequences = {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3921, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    sequences_p = {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3921, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    patching_metric = cat_linear_ranking_function
    layer_head_list = []  # Add any specific layers/heads to test
    
    output = cat_model(sequences=sequences, sequences_p=sequences_p,
                       patch_type='block_all', layer_head_list=layer_head_list,
                       patching_metric=patching_metric)

    assert output.scores.shape == (1, NUM_LABELS)  # Check scores shape
    assert output.scores_p.shape == (1, NUM_LABELS)  # Check scores_p shape

def test_get_act_patch_block_every(cat_model):
    """Test the method _get_act_patch_block_every."""
    corrupted_tokens = {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3921, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    clean_cache = None  # Initialize or mock clean_cache
    scores = torch.tensor([[0.9, 0.1]])
    scores_p = torch.tensor([[0.8, 0.2]])
    patching_metric = cat_linear_ranking_function
    
    result = cat_model._get_act_patch_block_every(corrupted_tokens, clean_cache, patching_metric, scores, scores_p)

    assert result.shape[0] == 3  # Ensure it returns results for the three components
    assert result.shape[1] == cat_model._model.cfg.n_layers  # Ensure the correct number of layers
    assert result.shape[2] == corrupted_tokens['input_ids'].size(1)  # Ensure the correct sequence length

# Add additional tests for other methods as needed