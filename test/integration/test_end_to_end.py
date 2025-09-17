import pytest
import torch
from mechir import Dot
from transformers import AutoTokenizer


@pytest.mark.integration
def test_dot_model_end_to_end():
    tokenizer
