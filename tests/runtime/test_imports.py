import fastapi
import networkx as nx
import torch


def test_core_imports():
    assert fastapi.__version__
    assert nx.__version__
    assert torch.__version__
    assert not torch.cuda.is_available()
