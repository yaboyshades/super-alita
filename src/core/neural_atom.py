# src/core/neural_atom.py
"""
The cognitive substrate.

This module defines:
- NeuralAtom: a single neuron holding a symbolic payload + 1024-D semantic vector.
- NeuralStore: a sparse, differentiable graph that manages neurons and synapses.

Atoms are the fundamental units of memory, skill, and state.
The store supports forward propagation, attention, and Hebbian learning.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

T = TypeVar("T")


class NeuralAtom(Generic[T]):
    """
    A cognitive neuron.

    Holds:
    - key      : unique identifier (global address)
    - value    : symbolic payload (dict, str, MCP, etc.)
    - vector   : 1024-D semantic embedding (activation)
    - bias     : learnable activation bias
    - parent_keys/children_keys/birth_event/lineage_metadata
      for Darwin-Gödel genealogy
    """

    def __init__(
        self,
        key: str,
        default_value: T,
        vector: Optional[np.ndarray] = None,
        bias: float = 0.0,
        parent_keys: Optional[List[str]] = None,
        birth_event: Optional[str] = None,
        lineage_metadata: Optional[Dict[str, Any]] = None,
    ):
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        self.key = key
        self.value = default_value
        self.vector = (
            vector.astype(np.float32)
            if vector is not None
            else np.zeros(1024, dtype=np.float32)
        )
        if self.vector.shape != (1024,):
            raise ValueError("Vector must be 1024-D")
        self.bias = bias
        self.active = True  # Whether this atom is currently active

        # Genealogy bookkeeping
        self.parent_keys = parent_keys or []
        self.children_keys: List[str] = []
        self.birth_event = birth_event
        self.lineage_metadata = lineage_metadata or {}

    def add_child(self, child_key: str) -> None:
        """Register a child atom in the genealogical tree."""
        if child_key not in self.children_keys:
            self.children_keys.append(child_key)

    def __repr__(self) -> str:
        return f"NeuralAtom(key='{self.key}', value={self.value})"



class NeuralStore:
    """
    Differentiable cognitive graph.

    - Sparse adjacency matrix for synapses
    - Forward propagation of activations
    - Attention-based retrieval
    - Hebbian learning
    """

    def __init__(self, learning_rate: float = 0.01):
        self._atoms: Dict[str, NeuralAtom] = {}
        self._keys: List[str] = []  # stable ordering for matrix ops
        self._key_idx: Dict[str, int] = {}
        self._adjacency: Optional[sp.lil_matrix] = None
        self._lr = learning_rate
        logger.info("NeuralStore initialized with learning_rate=%s", learning_rate)

    # ---------- Registration ----------
    def register(self, atom: NeuralAtom) -> None:
        """Add or overwrite a neuron."""
        if atom.key not in self._key_idx:
            self._key_idx[atom.key] = len(self._keys)
            self._keys.append(atom.key)
        self._atoms[atom.key] = atom
        self._resize_adjacency()

    def register_with_lineage(
        self,
        atom: NeuralAtom,
        parents: List["NeuralAtom"],
        birth_event: str,
        lineage_metadata: Dict[str, Any],
    ) -> None:
        """
        Convenience wrapper for genealogy registration.

        Sets parent/child links and metadata, then registers the atom.
        """
        atom.parent_keys = [p.key for p in parents]
        atom.birth_event = birth_event
        atom.lineage_metadata = lineage_metadata
        for parent in parents:
            parent.add_child(atom.key)
        self.register(atom)

    def _resize_adjacency(self) -> None:
        """Grow sparse matrix to accommodate new neurons."""
        n = len(self._keys)
        new_adj = sp.lil_matrix((n, n), dtype=np.float32)
        if self._adjacency is not None:
            rows, cols = self._adjacency.nonzero()
            new_adj[rows, cols] = self._adjacency[rows, cols]
        self._adjacency = new_adj

    # ---------- Access ----------
    def get(self, key: str) -> Optional[NeuralAtom]:
        """Retrieve an atom by its key."""
        return self._atoms.get(key)

    def set_value(self, key: str, value: Any) -> None:
        """Update the symbolic payload of an atom."""
        if key not in self._atoms:
            raise KeyError(key)
        self._atoms[key].value = value

    def add_synapse(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add or update directed synapse."""
        if source not in self._key_idx or target not in self._key_idx:
            raise KeyError("Source or target neuron not registered")
        s, t = self._key_idx[source], self._key_idx[target]
        self._adjacency[s, t] = weight

    # ---------- Forward Pass ----------
    async def forward_pass(self, activation_fn=np.tanh) -> None:
        """One-shot vectorized activation propagation."""
        if self._adjacency is None:
            return
        n = len(self._keys)
        if n == 0:
            return
        vectors = np.array([self._atoms[k].vector for k in self._keys])
        biases = np.array([self._atoms[k].bias for k in self._keys])
        incoming = self._adjacency.T.dot(vectors) + biases
        new_vecs = activation_fn(incoming)
        for k, vec in zip(self._keys, new_vecs):
            self._atoms[k].vector = vec

    # ---------- Attention ----------
    async def attention(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> List[tuple[str, float]]:
        """Semantic lookup via cosine similarity."""
        if not self._atoms:
            return []
        vecs = np.array([self._atoms[k].vector for k in self._keys])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        sims = np.dot(vecs / norms, query_vec / (np.linalg.norm(query_vec) + 1e-9))
        indices = np.argsort(sims)[-top_k:][::-1]
        return [(self._keys[i], sims[i]) for i in indices]

    # ---------- Learning ----------
    def hebbian_update(self, active_keys: List[str]) -> None:
        """Strengthen synapses between co-active neurons (fire-together-wire-together)."""
        indices = [self._key_idx[k] for k in active_keys if k in self._key_idx]
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                w = self._adjacency[i, j]
                update = self._lr * (1 - abs(w))
                new_w = np.clip(w + update, -1.0, 1.0)
                self._adjacency[i, j] = new_w

    # ---------- Genealogy Export ----------
    def export_genealogy(self) -> Dict[str, Dict[str, Any]]:
        """Dump the entire genealogy DAG as a nested dict for GraphML."""
        return {
            key: {
                "value": atom.value,
                "parent_keys": atom.parent_keys,
                "children_keys": atom.children_keys,
                "birth_event": atom.birth_event,
                "lineage_metadata": atom.lineage_metadata,
            }
            for key, atom in self._atoms.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the neural store."""
        total_connections = 0
        if self._adjacency is not None:
            total_connections = self._adjacency.nnz if hasattr(self._adjacency, 'nnz') else 0
        
        return {
            "total_atoms": len(self._atoms),
            "active_atoms": sum(1 for atom in self._atoms.values() if atom.active),
            "total_connections": total_connections,
            "learning_rate": self._lr,
            "memory_usage_mb": len(self._atoms) * 1024 * 4 / (1024 * 1024)  # Rough estimate
        }


async def create_skill_atom(
    skill_name: str,
    skill_vector: List[float],
    parent_skills: List[str] = None,
    metadata: Dict[str, Any] = None
) -> NeuralAtom:
    """
    Create a neural atom representing a skill.
    
    Args:
        skill_name: Name of the skill
        skill_vector: Vector representation of the skill
        parent_skills: Parent skills that contributed to this skill
        metadata: Additional metadata for the skill
        
    Returns:
        NeuralAtom representing the skill
    """
    if parent_skills is None:
        parent_skills = []
    if metadata is None:
        metadata = {}
    
    # Create skill data structure
    skill_data = {
        "name": skill_name,
        "type": "skill",
        "created_at": datetime.utcnow().isoformat(),
        "metadata": metadata
    }
    
    # Create neural atom for the skill
    atom = NeuralAtom(
        key=f"skill:{skill_name}",
        default_value=skill_data,
        vector=skill_vector,
        parent_keys=[f"skill:{parent}" for parent in parent_skills],
        birth_event="skill_creation",
        lineage_metadata={
            "skill_type": metadata.get("skill_type", "unknown"),
            "complexity": metadata.get("complexity", 0.0),
            "generation": metadata.get("generation", 0)
        }
    )
    
    return atom
