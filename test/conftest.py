import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
