"""
Compat shim for legacy ``python_graphml`` usages.

This module proxies read and write operations to NetworkX's GraphML
helpers backed by ``lxml``. It allows older imports such as
``import python_graphml`` to remain functional after the dependency was
removed.
"""

from __future__ import annotations

import networkx as nx
from typing import Any


def write_graph(graph: Any, path: str, **kwargs: Any) -> None:
    """Write ``graph`` to ``path`` in GraphML format.

    Parameters
    ----------
    graph:
        A NetworkX graph instance to serialise.
    path:
        Destination file path.
    **kwargs:
        Additional keyword arguments forwarded to ``nx.write_graphml``.
    """

    nx.write_graphml(graph, path, **kwargs)


def read_graph(path: str, **kwargs: Any) -> Any:
    """Read a GraphML file located at ``path`` into a NetworkX graph."""

    return nx.read_graphml(path, **kwargs)


# Older code may call ``write``/``read`` directly.
def write(graph: Any, path: str, **kwargs: Any) -> None:
    write_graph(graph, path, **kwargs)


def read(path: str, **kwargs: Any) -> Any:
    return read_graph(path, **kwargs)

