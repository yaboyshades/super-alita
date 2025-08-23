from __future__ import annotations

"""Simple navigation utilities for traversing the temporal graph."""

from dataclasses import dataclass
from enum import Enum
from typing import List

from .temporal_graph import TemporalGraph, NeuralAtom


class NavigationStrategy(Enum):
    SEMANTIC = "semantic"


@dataclass
class NavigationConfig:
    enable_telemetry: bool = True


class NeuralNavigator:
    def __init__(self, graph: TemporalGraph, config: NavigationConfig) -> None:
        self.graph = graph
        self.config = config

    async def navigate(
        self,
        prompt: str,
        strategy: NavigationStrategy,
        depth: int = 5,
        semantic_query: str | None = None,
    ) -> List[NeuralAtom]:
        return list(self.graph.atoms.values())
