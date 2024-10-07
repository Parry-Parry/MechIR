import pytest
import torch
from mechir.modelling.hooked import Linear  

# Constants
BATCH_SIZE = 2
SEQ_LENGTH = 10
D_MODEL = 128  # Size of the model dimension
NUM_LABELS = 5  # Number of output labels
D_TYPE = torch.float32  # Data type

@pytest.fixture
def linear_layer():
    """Fixture to initialize the Linear layer."""
    cfg = {
        "d_model": D_MODEL,
        "num_labels": NUM_LABELS,
        "dtype": D_TYPE
    }
    layer = Linear(cfg)
    yield layer

def test_initialization(linear_layer):
    """Test that the Linear layer initializes correctly."""
    assert linear_layer is not None
    assert linear_layer.W_in.shape == (D_MODEL, NUM_LABELS)
    assert linear_layer.b_in.shape == (NUM_LABELS,)
    assert linear_layer.cfg.d_model == D_MODEL
    assert linear_layer.cfg.num_labels == NUM_LABELS
    assert linear_layer.cfg.dtype == D_TYPE

def test_forward_pass(linear_layer):
    """Test the forward pass of the Linear layer."""
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL, dtype=D_TYPE)  # Random input tensor
    output = linear_layer(x)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, NUM_LABELS)  # Check output shape

def test_forward_pass_empty_input(linear_layer):
    """Test the forward pass with an empty input."""
    x = torch.empty(BATCH_SIZE, 0, D_MODEL, dtype=D_TYPE)  # Empty input tensor
    output = linear_layer(x)
    assert output.shape == (BATCH_SIZE, 0, NUM_LABELS)  # Check output shap