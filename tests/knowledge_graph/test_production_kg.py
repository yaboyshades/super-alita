import json
from pathlib import Path

import pytest

from src.knowledge_graph.production_kg import ProductionKnowledgeGraph


class FakeCollection:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, object]] = {}

    def add(self, ids, embeddings=None, documents=None, metadatas=None):  # type: ignore[override]
        for index, item_id in enumerate(ids):
            self.store[item_id] = {
                "embedding": embeddings[index] if embeddings else None,
                "document": documents[index] if documents else "",
                "metadata": metadatas[index] if metadatas else {},
            }

    def get(self, ids=None, include=None, where=None):  # type: ignore[override]
        results: list[tuple[str, dict[str, object]]] = []
        if where:
            for item_id, record in self.store.items():
                if self._matches_where(record, where):
                    results.append((item_id, record))
        elif ids:
            for item_id in ids:
                if item_id in self.store:
                    results.append((item_id, self.store[item_id]))
        else:
            results = list(self.store.items())

        return self._build_response(results, include)

    def query(self, *_, **__):  # type: ignore[override]
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    def _matches_where(self, record, where):
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        for key, expected in (where or {}).items():
            value = metadata.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if value not in expected["$in"]:
                    return False
            elif value != expected:
                return False
        return True

    def _build_response(self, results, include):
        include = include or []
        response = {"ids": [item_id for item_id, _ in results]}
        if "embeddings" in include:
            response["embeddings"] = [record["embedding"] for _, record in results]
        if "documents" in include:
            response["documents"] = [record["document"] for _, record in results]
        if "metadatas" in include:
            response["metadatas"] = [record["metadata"] for _, record in results]
        return response


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name, metadata=None):  # type: ignore[override]
        return self.collections.setdefault(name, FakeCollection())

    def delete_collection(self, name):  # type: ignore[override]
        self.collections.pop(name, None)


class StubModel:
    def encode(self, items):  # type: ignore[override]
        return [[0.1, 0.2, 0.3] for _ in items]


@pytest.fixture()
def kg(monkeypatch):
    client = FakeClient()
    model = StubModel()
    graph = ProductionKnowledgeGraph(
        persist_directory=":memory:",
        client=client,
        embedding_model=model,
    )

    async def noop(*_args, **_kwargs):  # pragma: no cover - helper
        return None

    monkeypatch.setattr(graph, "_create_similarity_bonds", noop)
    return graph


@pytest.mark.asyncio
async def test_create_atom_persists_atom(kg):
    atom = await kg.create_atom("concept", "Test content", metadata={"source": "user"})

    assert atom["type"] == "concept"
    assert atom["metadata"]["source"] == "user"

    stored = await kg.get_atom(atom["id"])
    assert stored is not None
    assert stored["content"] == "Test content"


@pytest.mark.asyncio
async def test_create_atom_returns_existing_on_duplicate(monkeypatch, kg):
    existing_id = "atom_existing"
    kg.atoms_collection.add(
        ids=[existing_id],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Existing"],
        metadatas=[{"type": "concept", "metadata": json.dumps({"source": "seed"})}],
    )

    async def duplicate(*_args, **_kwargs):
        return existing_id

    monkeypatch.setattr(kg, "_find_semantic_duplicate", duplicate)

    result = await kg.create_atom("concept", "New content")
    assert result["id"] == existing_id
    assert result["content"] == "Existing"


@pytest.mark.asyncio
async def test_backup_writes_file(tmp_path: Path, kg):
    atom_one = await kg.create_atom("concept", "first")
    atom_two = await kg.create_atom("concept", "second")
    await kg.create_bond(atom_one["id"], atom_two["id"], "enables")

    backup_path = tmp_path / "kg_backup.json"
    success = await kg.backup(str(backup_path))

    assert success is True
    data = json.loads(backup_path.read_text())
    assert data["atoms_count"] >= 2
    assert data["bonds_count"] >= 1
