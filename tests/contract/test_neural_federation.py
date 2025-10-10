from dataclasses import dataclass
from typing import Any

import pytest

from src.memory.atoms import KnowledgeAtom
from src.memory.federation import NeuralFederationRepository


@dataclass
class FakeVectorStore:
    stored: list[dict[str, Any]]

    def add(
        self,
        collection: str,
        *,
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for embedding, metadata in zip(embeddings, metadatas, strict=True):
            record = metadata.copy()
            record["embedding"] = embedding
            record["collection"] = collection
            self.stored.append(record)

    def query(
        self, collection: str, *, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        # Return stored records sorted by dot product similarity (very rough for contract scope).
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self.stored:
            if record["collection"] != collection:
                continue
            score = sum(
                a * b
                for a, b in zip(record["embedding"], embedding, strict=False)
            )
            scored.append((score, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore(stored=[])


def test_store_atom_persists_metadata(fake_store: FakeVectorStore) -> None:
    repository = NeuralFederationRepository(
        vector_store=fake_store, collection="unified-atoms"
    )
    atom = KnowledgeAtom(
        atom_id="atom-001",
        text="Implement resilient event bus with retries",
        tags=["eventbus", "resiliency"],
        embedding=[0.1, 0.2, 0.3],
    )

    repository.store_atom(atom)

    assert (
        fake_store.stored
    ), "store_atom must push entries into the vector store"
    record = fake_store.stored[0]
    assert record["atom_id"] == "atom-001"
    assert record["tags"] == ["eventbus", "resiliency"]


def test_similarity_search_returns_ranked_atoms(
    fake_store: FakeVectorStore,
) -> None:
    repository = NeuralFederationRepository(
        vector_store=fake_store, collection="unified-atoms"
    )
    atom_a = KnowledgeAtom(
        atom_id="atom-a",
        text="Binary search implementation",
        tags=["algorithms"],
        embedding=[0.9, 0.1, 0.0],
    )
    atom_b = KnowledgeAtom(
        atom_id="atom-b",
        text="Merge sort implementation",
        tags=["algorithms"],
        embedding=[0.1, 0.9, 0.0],
    )

    repository.store_atom(atom_a)
    repository.store_atom(atom_b)

    results = repository.search_similar(text="binary search", top_k=1)
    assert results
    assert results[0].atom_id == "atom-a"
