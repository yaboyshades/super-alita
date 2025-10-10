"""Neural memory federation repository (placeholder)."""

from __future__ import annotations

from typing import Protocol

from .atoms import KnowledgeAtom


class VectorStoreProtocol(Protocol):
    def add(
        self,
        collection: str,
        *,
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None: ...

    def query(
        self,
        collection: str,
        *,
        embedding: list[float],
        top_k: int,
    ) -> list[dict]: ...


class NeuralFederationRepository:
    """Facade around the neural memory federation layer."""

    def __init__(
        self, *, vector_store: VectorStoreProtocol, collection: str
    ) -> None:
        self._store = vector_store
        self._collection = collection

    def store_atom(self, atom: KnowledgeAtom) -> None:
        raise NotImplementedError(
            "NeuralFederationRepository.store_atom pending implementation"
        )

    def search_similar(
        self, *, text: str, top_k: int = 5
    ) -> list[KnowledgeAtom]:
        raise NotImplementedError(
            "NeuralFederationRepository.search_similar pending implementation"
        )
