# Transformer tests
import torch

from src.abilities.transformer_paper_code_implementation import (
    create_paper_code_implementation,
)


def test_model_creation():
    model = create_paper_code_implementation()
    assert model is not None

def test_forward_pass():
    model = create_paper_code_implementation()
    # Create appropriate test input based on detected concepts
    if "alita" in "transformer":
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids)
    else:
        x = torch.randn(2, 512)
        output = model(x)
    assert output is not None
