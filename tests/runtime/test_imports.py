import fastapi
import networkx as nx


def test_fastapi_and_networkx_importable():
    assert fastapi.__version__
    assert nx.__version__
