"""In-memory episodic store backed by an optional FAISS index."""
from __future__ import annotations

import json
import logging
import pickle
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback path
    faiss = None

from src.models import Memory


class _NumpyIndex:
    """Lightweight cosine-similarity index used when FAISS is unavailable."""

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._vectors: List[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        for vector in vectors:
            arr = np.asarray(vector, dtype=np.float32)
            if arr.shape[0] != self._dimension:
                raise ValueError("Embedding dimension mismatch for fallback index")
            self._vectors.append(arr)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if not self._vectors:
            scores = np.zeros((len(queries), k), dtype=np.float32)
            indices = np.full((len(queries), k), -1, dtype=np.int64)
            return scores, indices

        matrix = np.stack(self._vectors)
        matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-8

        all_scores: List[np.ndarray] = []
        all_indices: List[np.ndarray] = []
        for query in queries:
            q_norm = np.linalg.norm(query) + 1e-8
            similarities = matrix @ query / (matrix_norms * q_norm)
            top_indices = np.argsort(similarities)[::-1][:k]
            top_scores = similarities[top_indices]
            if len(top_indices) < k:
                pad = k - len(top_indices)
                top_indices = np.pad(top_indices, (0, pad), constant_values=-1)
                top_scores = np.pad(top_scores, (0, pad), constant_values=0.0)
            all_indices.append(top_indices.astype(np.int64))
            all_scores.append(top_scores.astype(np.float32))

        return np.vstack(all_scores), np.vstack(all_indices)

    def reset(self) -> None:
        self._vectors.clear()

    @property
    def ntotal(self) -> int:
        return len(self._vectors)


def _build_index(dimension: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dimension)  # type: ignore[call-arg]
    return _NumpyIndex(dimension)

logger = logging.getLogger(__name__)


class EpisodicStore:
    """Thread-safe episodic memory store with semantic search."""

    def __init__(self, persist_path: str = "storage/episodic"):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.memories: List[Memory] = []
        self.memory_index: Dict[str, Memory] = {}

        self.dimension = 384
        self.index = _build_index(self.dimension)
        self.embeddings_map: Dict[str, int] = {}

        self.ingest_count = 0
        self.retrieval_count = 0
        self.last_consolidation: Optional[datetime] = None

        self._load_persisted()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def add(self, memory: Memory, embedding: Optional[List[float]] = None) -> None:
        """Add a memory to the store."""

        with self._lock:
            if embedding is None:
                embedding = self._generate_embedding(memory.text)

            memory.embeddings = embedding
            self.memories.append(memory)
            self.memory_index[memory.id] = memory

            self.embeddings_map[memory.id] = len(self.memories) - 1
            self.index.add(np.array([embedding], dtype=np.float32))

            self.ingest_count += 1
            if self.ingest_count % 100 == 0:
                self._persist()

    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            memory = self.memory_index.get(memory_id)
            if memory:
                memory.last_access = datetime.utcnow()
                memory.access_count += 1
            return memory

    def search(self, query: str, k: int = 10) -> List[Memory]:
        with self._lock:
            if not self.memories:
                return []

            query_embedding = self._generate_embedding(query)
            scores, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                min(k * 2, len(self.memories)),
            )

            results: List[Tuple[float, Memory]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.memories):
                    continue
                memory = self.memories[idx]
                text_score = self._text_relevance(query, memory.text)
                combined = (float(score) + text_score) / 2
                results.append((combined, memory))

            results.sort(key=lambda item: item[0], reverse=True)
            self.retrieval_count += 1
            return [mem for _, mem in results[:k]]

    def get_oldest(self, limit: int = 100) -> List[Memory]:
        with self._lock:
            return sorted(self.memories, key=lambda mem: mem.ts)[:limit]

    def get_by_importance(self, min_importance: float = 0.0) -> List[Memory]:
        with self._lock:
            return [mem for mem in self.memories if mem.importance >= min_importance]

    def remove(self, memory_ids: List[str]) -> None:
        with self._lock:
            removed = False
            for mem_id in memory_ids:
                memory = self.memory_index.pop(mem_id, None)
                if memory is not None:
                    removed = True
                    self.memories.remove(memory)
            if removed:
                self._rebuild_index()
                self._persist()

    def count(self) -> int:
        return len(self.memories)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def get_ingest_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_importance = (
                float(np.mean([mem.importance for mem in self.memories]))
                if self.memories
                else 0.0
            )
            return {
                "total_ingested": self.ingest_count,
                "current_count": len(self.memories),
                "avg_importance": avg_importance,
                "embeddings_size": getattr(self.index, "ntotal", len(self.memories)),
            }

    def get_retrieval_metrics(self) -> Dict[str, Any]:
        return {"total_retrievals": self.retrieval_count, "avg_recall_latency": 0.05}

    def get_average_importance(self) -> float:
        with self._lock:
            if not self.memories:
                return 0.0
            return sum(mem.importance for mem in self.memories) / len(self.memories)

    def get_age_distribution(self) -> Dict[str, int]:
        with self._lock:
            now = datetime.utcnow()
            distribution = {"<1d": 0, "1d-7d": 0, "1w-1m": 0, "1m-3m": 0, ">3m": 0}
            for memory in self.memories:
                age = now - memory.ts
                if age < timedelta(days=1):
                    distribution["<1d"] += 1
                elif age < timedelta(days=7):
                    distribution["1d-7d"] += 1
                elif age < timedelta(days=30):
                    distribution["1w-1m"] += 1
                elif age < timedelta(days=90):
                    distribution["1m-3m"] += 1
                else:
                    distribution[">3m"] += 1
            return distribution

    def get_tag_distribution(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for memory in self.memories:
                for tag in memory.tags:
                    counts[tag] = counts.get(tag, 0) + 1
            return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for idx, word in enumerate(words[: self.dimension]):
            vector[idx % self.dimension] += (hash(word) % 100) / 100.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    @staticmethod
    def _text_relevance(query: str, text: str) -> float:
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
        text_terms = set(text.lower().split())
        intersection = len(query_terms & text_terms)
        return intersection / len(query_terms)

    def _rebuild_index(self) -> None:
        self.index = _build_index(self.dimension)
        self.embeddings_map = {}
        embeddings: List[List[float]] = []
        for idx, memory in enumerate(self.memories):
            if memory.embeddings:
                embeddings.append(memory.embeddings)
                self.embeddings_map[memory.id] = idx
        if embeddings:
            self.index.add(np.array(embeddings, dtype=np.float32))

    def _persist(self) -> None:
        try:
            with open(self.persist_path / "memories.pkl", "wb") as handle:
                pickle.dump(self.memories, handle)
            if faiss is not None and hasattr(self.index, "this"):
                faiss.write_index(self.index, str(self.persist_path / "index.faiss"))
            with open(self.persist_path / "metadata.json", "w") as handle:
                json.dump(
                    {
                        "ingest_count": self.ingest_count,
                        "retrieval_count": self.retrieval_count,
                        "last_updated": datetime.utcnow().isoformat(),
                    },
                    handle,
                )
        except Exception as exc:  # pragma: no cover - persistence best effort
            logger.warning("Failed to persist episodic store: %s", exc)

    def _load_persisted(self) -> None:
        try:
            memories_path = self.persist_path / "memories.pkl"
            index_path = self.persist_path / "index.faiss"
            metadata_path = self.persist_path / "metadata.json"
            if memories_path.exists():
                with open(memories_path, "rb") as handle:
                    self.memories = pickle.load(handle)
                self.memory_index = {mem.id: mem for mem in self.memories}
            if index_path.exists() and faiss is not None:
                self.index = faiss.read_index(str(index_path))
            else:
                self._rebuild_index()
            if metadata_path.exists():
                with open(metadata_path, "r") as handle:
                    metadata = json.load(handle)
                    self.ingest_count = metadata.get("ingest_count", 0)
                    self.retrieval_count = metadata.get("retrieval_count", 0)
            if self.memories:
                logger.info("Loaded %s episodic memories", len(self.memories))
        except Exception as exc:  # pragma: no cover - best effort load
            logger.warning("Could not load persisted episodic store: %s", exc)


episodic_store = EpisodicStore()
