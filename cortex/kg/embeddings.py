from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np


class EmbeddingProvider:
    """Abstract embedding provider: returns a float vector for a piece of text or node payload."""

    dim: int

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self.embed_text(t) for t in texts]


class HashingEmbedder(EmbeddingProvider):
    """
    Deterministic, dependency-free embedder (for tests and cold starts).
    Produces a pseudo-embedding by hashing tokens into a fixed-size vector.
    Replace with Sentence-Transformers or OpenAI embeddings in production.
    """

    def __init__(self, dim: int = 256, buckets: int = 2048):
        self.dim = dim
        self.buckets = buckets

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.lower().split() if tok.strip()]

    def _hash(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16) % self.buckets

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = self._tokenize(text)
        if not toks:
            return vec
        for t in toks:
            idx = self._hash(t) % self.dim
            vec[idx] += 1.0
        # Normalize
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
