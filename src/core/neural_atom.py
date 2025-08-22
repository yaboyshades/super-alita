# Version: 3.0.0
# Description: The core Neural Atom structure, the unifying element of the architecture.

from __future__ import annotations

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

import numpy as np
import scipy.sparse as sp

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

# Import Gemini utilities with safe error handling

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class NeuralAtomMetadata:
    """Metadata for Neural Atoms with full cognitive capabilities."""

    name: str
    description: str
    capabilities: list[str]
    version: str = "1.0.0"
    usage_count: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0
    tags: set[str] | None = None
    created_at: float | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.created_at is None:
            self.created_at = time.time()


class NeuralAtom(ABC):
    """
    Base class for all Neural Atoms with full cognitive capabilities.

    The unifying element of the architecture - semantically-rich, modular units
    of intelligence containing executable logic, metadata, and semantic embeddings.
    """

    def __init__(self, metadata: NeuralAtomMetadata):
        self.metadata = metadata
        self._execution_history: list[dict[str, Any]] = []
        self._semantic_embedding: list[float] | None = None

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """Execute the Neural Atom's core functionality."""

    @abstractmethod
    def get_embedding(self) -> list[float]:
        """Return semantic embedding for similarity search."""

    @abstractmethod
    def can_handle(self, task_description: str) -> float:
        """Return confidence score (0-1) for handling this task."""

    async def safe_execute(self, input_data: Any) -> dict[str, Any]:
        """Execute with comprehensive monitoring and performance tracking."""
        start_time = time.time()

        try:
            result = await self.execute(input_data)
            execution_time = time.time() - start_time
            success = True

            # Update performance metrics
            self._update_performance_metrics(execution_time, success)

            return {
                "result": result,
                "success": True,
                "execution_time": execution_time,
                "metadata": self.metadata,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, False)

            return {
                "result": None,
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "metadata": self.metadata,
            }

    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update metadata with performance information."""
        # Update usage count
        self.metadata.usage_count += 1

        # Update average execution time using exponential moving average
        if self.metadata.avg_execution_time == 0.0:
            self.metadata.avg_execution_time = execution_time
        else:
            alpha = 0.1  # Learning rate for exponential moving average
            self.metadata.avg_execution_time = (
                alpha * execution_time + (1 - alpha) * self.metadata.avg_execution_time
            )

        # Update success rate using exponential moving average
        if success:
            success_value = 1.0
        else:
            success_value = 0.0

        if self.metadata.usage_count == 1:
            self.metadata.success_rate = success_value
        else:
            alpha = 0.1  # Learning rate for exponential moving average
            self.metadata.success_rate = (
                alpha * success_value + (1 - alpha) * self.metadata.success_rate
            )

    @property
    def average_latency_ms(self) -> float:
        """Return average execution latency in milliseconds."""
        return self.metadata.avg_execution_time * 1000.0


# Legacy Neural Atom for backwards compatibility
class LegacyNeuralAtom(Generic[T]):
    """
    A single cognitive neuron, the fundamental unit of thought and memory.

    Each atom is a neuro-symbolic entity, holding:
    - key (str): A unique, global address for this thought/memory.
    - value (T): The symbolic, human-readable payload (e.g., a piece of text,
      a tool's state, a plan step).
    - vector (np.ndarray): A 1024-D semantic embedding representing the atom's
      meaning and its activation state in the cognitive graph.
    - bias (float): A learnable activation bias, allowing the neuron to be
      more or less easily activated.
    - Genealogy Fields: Attributes for tracking the atom's origin, parents,
      and children, forming the Darwin-Gödel evolutionary tree.
    - Neural Dynamics: Advanced features for realistic neural behavior.
    """

    """
    A single cognitive neuron, the fundamental unit of thought and memory.

    Each atom is a neuro-symbolic entity, holding:
    - key (str): A unique, global address for this thought/memory.
    - value (T): The symbolic, human-readable payload (e.g., a piece of text,
      a tool's state, a plan step).
    - vector (np.ndarray): A 1024-D semantic embedding representing the atom's
      meaning and its activation state in the cognitive graph.
    - bias (float): A learnable activation bias, allowing the neuron to be
      more or less easily activated.
    - Genealogy Fields: Attributes for tracking the atom's origin, parents,
      and children, forming the Darwin-Gödel evolutionary tree.
    - Neural Dynamics: Advanced features for realistic neural behavior.
    """

    def __init__(
        self,
        key: str,
        default_value: T,
        vector: np.ndarray | None = None,
        bias: float = 0.0,
        parent_keys: list[str] | None = None,
        birth_event: str | None = None,
        lineage_metadata: dict[str, Any] | None = None,
    ):
        if not key or not isinstance(key, str):
            raise ValueError("NeuralAtom key must be a non-empty string.")

        self.key = key
        self.value = default_value

        # Initialize 1024-D semantic vector
        if vector is not None:
            if vector.shape != (1024,):
                raise ValueError(
                    f"Vector for atom '{key}' must be 1024-D, but got shape {vector.shape}."
                )
            self.vector = vector.astype(np.float32)
        else:
            self.vector = np.zeros(1024, dtype=np.float32)

        self.bias = float(bias)
        self.active = True  # Whether this atom is currently active

        # Genealogy bookkeeping for the Darwin-Gödel model
        self.parent_keys: list[str] = parent_keys or []
        self.children_keys: list[str] = []
        self.birth_event = birth_event
        self.lineage_metadata: dict[str, Any] = lineage_metadata or {}

        # Enhanced genealogy features
        self.depth = len(self.parent_keys)  # Genealogical depth
        self.signature = self._generate_darwin_godel_signature()
        self.creation_time = datetime.now(UTC)
        self.last_activation_time = datetime.now(UTC)
        self.activation_count = 0
        self.fitness_score = 0.0  # For evolutionary selection

        # Neural dynamics
        self.activation_threshold = 0.5
        self.decay_rate = 0.95
        self.refractory_period = 0.1  # seconds
        self.last_spike_time = 0.0

    def _generate_darwin_godel_signature(self) -> str:
        """Generate unique Darwin-Gödel signature for genealogy tracking."""
        signature_data = f"{self.key}:{':'.join(sorted(self.parent_keys))}"
        return hashlib.md5(signature_data.encode()).hexdigest()[:16]

    def add_child(self, child_key: str) -> None:
        """Register a child atom in the genealogical tree."""
        if child_key not in self.children_keys:
            self.children_keys.append(child_key)

    def activate(self, input_activation: float = 1.0) -> float:
        """
        Activate the neuron with input and return output activation.

        Implements neural dynamics including threshold, bias, and refractory period.
        """
        import time

        current_time = time.time()

        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return 0.0

        # Calculate activation with bias
        total_activation = input_activation + self.bias

        # Apply threshold
        if total_activation >= self.activation_threshold:
            self.last_spike_time = current_time
            self.last_activation_time = datetime.now(UTC)
            self.activation_count += 1

            # Return sigmoid activation
            return 1.0 / (1.0 + np.exp(-total_activation))

        return 0.0

    def update_fitness(self, performance_score: float, weight: float = 0.1) -> None:
        """Update fitness score using exponential moving average."""
        self.fitness_score = (
            1 - weight
        ) * self.fitness_score + weight * performance_score

    def get_genealogy_summary(self) -> dict[str, Any]:
        """Get comprehensive genealogy information."""
        return {
            "key": self.key,
            "signature": self.signature,
            "depth": self.depth,
            "parent_count": len(self.parent_keys),
            "child_count": len(self.children_keys),
            "creation_time": self.creation_time.isoformat(),
            "activation_count": self.activation_count,
            "fitness_score": self.fitness_score,
            "lineage_metadata": self.lineage_metadata,
        }

    def __repr__(self) -> str:
        return f"NeuralAtom(key='{self.key}', value={self.value})"


class NeuralStore:
    """
    The differentiable cognitive graph, acting as the agent's main memory and
    reasoning substrate. It manages a collection of NeuralAtoms and the sparse
    synaptic connections between them.

    Enhanced with sophisticated attention mechanisms, neural dynamics,
    genealogy-aware operations, and biologically-inspired learning.
    """

    def __init__(self, learning_rate: float = 0.01):
        self._atoms: dict[str, NeuralAtom] = {}
        self._keys: list[str] = []  # Guarantees a stable order for matrix operations
        self._key_idx: dict[str, int] = {}
        self._adjacency: sp.lil_matrix | None = None
        self._lr = learning_rate
        logger.info(f"NeuralStore initialized with learning_rate={self._lr}")

    # --- Graph Structure Management ---

    def register(self, atom: NeuralAtom) -> None:
        """Adds a NeuralAtom to the store or overwrites an existing one."""
        if atom.key not in self._key_idx:
            # This is a new atom, add it to our index
            idx = len(self._keys)
            self._key_idx[atom.key] = idx
            self._keys.append(atom.key)
            self._resize_adjacency()
        self._atoms[atom.key] = atom
        logger.debug(f"Registered atom: {atom.key}")

    def register_with_lineage(
        self,
        atom: NeuralAtom,
        parents: list[NeuralAtom],
        birth_event: str,
        lineage_metadata: dict[str, Any],
    ) -> None:
        """
        A high-level API for registering a new atom while correctly setting up its
        genealogical links. This is the preferred method for creating new thoughts.
        """
        atom.parent_keys = [p.key for p in parents]
        atom.birth_event = birth_event
        atom.lineage_metadata = lineage_metadata
        atom.depth = max([p.depth for p in parents], default=0) + 1
        atom.signature = atom._generate_darwin_godel_signature()

        for parent in parents:
            parent.add_child(atom.key)
        self.register(atom)

    def _resize_adjacency(self) -> None:
        """Expands the sparse adjacency matrix to accommodate a new atom."""
        n = len(self._keys)
        new_adj = sp.lil_matrix((n, n), dtype=np.float32)
        if self._adjacency is not None:
            # Copy over the old connections to the new, larger matrix
            prev_n = self._adjacency.shape[0]
            new_adj[:prev_n, :prev_n] = self._adjacency
        self._adjacency = new_adj
        logger.debug(f"Adjacency matrix resized to ({n}, {n}).")

    def add_synapse(
        self, source_key: str, target_key: str, weight: float = 1.0
    ) -> None:
        """Creates or updates a directed, weighted connection (synapse) between two atoms."""
        if source_key not in self._key_idx or target_key not in self._key_idx:
            raise KeyError(
                f"Source '{source_key}' or target '{target_key}' atom not found in store."
            )

        source_idx = self._key_idx[source_key]
        target_idx = self._key_idx[target_key]
        self._adjacency[source_idx, target_idx] = weight

    # --- State Access ---

    def get(self, key: str) -> NeuralAtom | None:
        """Retrieves a single NeuralAtom by its unique key."""
        return self._atoms.get(key)

    def set_value(self, key: str, value: Any) -> None:
        """Update the symbolic payload of an atom."""
        if key not in self._atoms:
            raise KeyError(f"Atom '{key}' not found in store")
        self._atoms[key].value = value

    async def delete(self, key: str) -> None:
        """Remove an atom from the store. Raises KeyError if key not found."""
        if key not in self._atoms:
            raise KeyError(f"Atom '{key}' not found in store")

        # Remove from atoms dictionary
        del self._atoms[key]

        # Remove from key index and update indices
        old_idx = self._key_idx[key]
        del self._key_idx[key]
        self._keys.remove(key)

        # Update adjacency matrix by removing the row and column for this atom
        if self._adjacency is not None and self._adjacency.shape[0] > 0:
            n = self._adjacency.shape[0]
            if old_idx < n:
                # Create new smaller adjacency matrix
                new_size = n - 1
                if new_size > 0:
                    new_adj = sp.lil_matrix((new_size, new_size), dtype=np.float32)
                    # Copy rows and columns, skipping the deleted index
                    row_indices = list(range(old_idx)) + list(range(old_idx + 1, n))
                    col_indices = row_indices
                    for new_row, old_row in enumerate(row_indices):
                        for new_col, old_col in enumerate(col_indices):
                            new_adj[new_row, new_col] = self._adjacency[
                                old_row, old_col
                            ]
                    self._adjacency = new_adj
                else:
                    self._adjacency = None

        # Update key indices for remaining atoms
        for i, remaining_key in enumerate(self._keys):
            self._key_idx[remaining_key] = i

        logger.debug(f"Atom '{key}' deleted from NeuralStore")

    # --- Cognitive Functions ---

    async def forward_pass(self, activation_fn: Callable = np.tanh) -> None:
        """
        Performs a one-shot, vectorized propagation of activation across the entire graph.
        This is a fundamental "thinking" step, simulating how neurons influence each other.
        """
        if self._adjacency is None or len(self._keys) == 0:
            return

        len(self._keys)
        # 1. Gather all current activation vectors and biases
        vectors = np.array([self._atoms[k].vector for k in self._keys])
        biases = np.array([self._atoms[k].bias for k in self._keys]).reshape(-1, 1)

        # 2. Calculate incoming signals for all neurons at once
        # We use the transposed adjacency matrix for incoming connections
        incoming_signals = self._adjacency.T.dot(vectors)

        # 3. Apply biases and the non-linear activation function
        activated_signals = activation_fn(incoming_signals + biases.squeeze())

        # 4. Update the state of all atoms with their new activation vectors
        for i, key in enumerate(self._keys):
            atom = self._atoms[key]
            atom.vector = activated_signals[i]
            atom.last_activation_time = datetime.now(UTC)
            atom.activation_count += 1

    async def attention(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Performs semantic retrieval with enhanced attention mechanisms.

        Finds the `top_k` atoms most similar to a given query vector using cosine similarity,
        enhanced with fitness-based weighting and genealogy awareness.
        """
        if not self._atoms:
            return []

        n = len(self._keys)
        # Stack all atom vectors into a single matrix for efficient computation
        matrix = np.array([self._atoms[k].vector for k in self._keys])

        # Normalize vectors for cosine similarity calculation
        query_norm = np.linalg.norm(query_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        # Avoid division by zero for zero-vectors
        query_norm = query_norm if query_norm > 0 else 1e-9
        matrix_norms[matrix_norms == 0] = 1e-9

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(matrix, query_vec) / (matrix_norms * query_norm)

        # Apply fitness weighting - atoms with higher fitness get slight boost
        fitness_scores = np.array([self._atoms[k].fitness_score for k in self._keys])
        weighted_similarities = similarities * (
            1.0 + 0.1 * fitness_scores
        )  # 10% fitness boost

        # Get the indices of the top k most similar atoms
        if top_k >= n:
            top_indices = np.argsort(weighted_similarities)[::-1]
        else:
            top_indices = np.argpartition(weighted_similarities, -top_k)[-top_k:]
            top_indices = top_indices[
                np.argsort(weighted_similarities[top_indices])[::-1]
            ]

        return [(self._keys[i], float(weighted_similarities[i])) for i in top_indices]

    async def genealogy_attention(
        self, query_atom_key: str, generations: int = 3, top_k: int = 5
    ) -> list[tuple[str, float, str]]:
        """
        Attention mechanism based on genealogical relationships.

        Returns atoms related by lineage within specified generations.
        """
        if query_atom_key not in self._atoms:
            return []

        query_atom = self._atoms[query_atom_key]
        related_keys = set()

        # Collect ancestors and descendants
        def collect_relatives(key: str, depth: int, direction: str):
            if depth <= 0 or key not in self._atoms:
                return
            atom = self._atoms[key]
            related_keys.add(key)

            if direction in ["up", "both"]:
                for parent_key in atom.parent_keys:
                    collect_relatives(parent_key, depth - 1, "up")

            if direction in ["down", "both"]:
                for child_key in atom.children_keys:
                    collect_relatives(child_key, depth - 1, "down")

        collect_relatives(query_atom_key, generations, "both")

        # Calculate similarity scores
        results = []
        query_vec = query_atom.vector

        for key in related_keys:
            if key != query_atom_key:
                atom = self._atoms[key]
                sim = np.dot(query_vec, atom.vector) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(atom.vector) + 1e-9
                )
                relationship = (
                    "ancestor"
                    if key in self._get_ancestors(query_atom_key)
                    else "descendant"
                )
                results.append((key, sim, relationship))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def _get_ancestors(self, key: str) -> set:
        """Get all ancestor keys for a given atom."""
        ancestors = set()
        if key not in self._atoms:
            return ancestors

        def collect_ancestors(current_key: str):
            if current_key in self._atoms:
                atom = self._atoms[current_key]
                for parent_key in atom.parent_keys:
                    if parent_key not in ancestors:
                        ancestors.add(parent_key)
                        collect_ancestors(parent_key)

        collect_ancestors(key)
        return ancestors

    # ---------- Learning ----------
    def hebbian_update(self, active_keys: list[str]) -> None:
        """
        Performs a Hebbian learning step: "neurons that fire together, wire together."
        This strengthens the synaptic weights between co-activated atoms, forming
        associative links and habits over time.
        """
        indices = [self._key_idx[k] for k in active_keys if k in self._key_idx]

        # Iterate through all pairs of co-active neurons
        for i in indices:
            for j in indices:
                if i == j:
                    continue

                # Get the current weight
                current_weight = self._adjacency[i, j]

                # The learning rule: strengthen the connection based on the learning rate
                # and how far the weight is from its maximum absolute value (1.0).
                update_delta = self._lr * (1.0 - abs(current_weight))
                new_weight = np.clip(current_weight + update_delta, -1.0, 1.0)

                self._adjacency[i, j] = new_weight

    def update_fitness_scores(self, fitness_updates: dict[str, float]) -> None:
        """Update fitness scores for multiple atoms based on performance."""
        for key, fitness in fitness_updates.items():
            if key in self._atoms:
                atom = self._atoms[key]
                # Use exponential moving average for fitness updates
                weight = 0.1
                atom.fitness_score = (
                    1 - weight
                ) * atom.fitness_score + weight * fitness

    def decay_inactive_synapses(self, decay_factor: float = 0.99) -> None:
        """Decay synaptic weights that haven't been recently activated."""
        if self._adjacency is None:
            return

        # Apply decay to all synapses
        self._adjacency.data *= decay_factor

        # Remove very weak connections (synaptic pruning)
        self._adjacency.data[np.abs(self._adjacency.data) < 1e-4] = 0
        self._adjacency.eliminate_zeros()

    # ---------- Analysis and Debugging ----------

    def get_topology_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the neural graph topology."""
        if not self._atoms:
            return {"total_atoms": 0, "total_synapses": 0}

        synapse_count = self._adjacency.nnz if self._adjacency is not None else 0

        # Calculate connectivity statistics
        avg_degree = synapse_count / len(self._atoms) if self._atoms else 0

        # Fitness statistics
        fitness_scores = [atom.fitness_score for atom in self._atoms.values()]

        # Genealogy statistics
        depth_distribution = self._get_depth_distribution()
        max_depth = max(depth_distribution.keys()) if depth_distribution else 0

        return {
            "total_atoms": len(self._atoms),
            "active_atoms": sum(1 for atom in self._atoms.values() if atom.active),
            "total_synapses": synapse_count,
            "average_degree": avg_degree,
            "max_genealogy_depth": max_depth,
            "fitness_stats": {
                "mean": np.mean(fitness_scores) if fitness_scores else 0,
                "std": np.std(fitness_scores) if fitness_scores else 0,
                "min": np.min(fitness_scores) if fitness_scores else 0,
                "max": np.max(fitness_scores) if fitness_scores else 0,
            },
            "genealogy_depth_distribution": depth_distribution,
            "learning_rate": self._lr,
            "memory_usage_estimate_mb": len(self._atoms) * 1024 * 4 / (1024 * 1024),
        }

    def _get_depth_distribution(self) -> dict[int, int]:
        """Get distribution of genealogical depths."""
        depth_counts = {}
        for atom in self._atoms.values():
            depth = atom.depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        return depth_counts

    def export_genealogy(self) -> dict[str, dict[str, Any]]:
        """Export the complete genealogy DAG with enhanced metadata."""
        return {
            key: {
                "value": str(atom.value)[:200],  # Truncate for serialization
                "parent_keys": atom.parent_keys,
                "children_keys": atom.children_keys,
                "birth_event": atom.birth_event,
                "lineage_metadata": atom.lineage_metadata,
                "depth": atom.depth,
                "signature": atom.signature,
                "fitness_score": atom.fitness_score,
                "activation_count": atom.activation_count,
                "creation_time": atom.creation_time.isoformat(),
                "last_activation_time": atom.last_activation_time.isoformat(),
            }
            for key, atom in self._atoms.items()
        }

    def export_graph_data(self) -> dict[str, Any]:
        """Export the complete graph structure for analysis or persistence."""
        nodes = []
        edges = []

        # Export nodes (atoms)
        for atom in self._atoms.values():
            nodes.append(
                {
                    "key": atom.key,
                    "value": str(atom.value)[:100],  # Truncate for serialization
                    "vector": atom.vector.tolist(),
                    "fitness_score": atom.fitness_score,
                    "depth": atom.depth,
                    "signature": atom.signature,
                    "parent_keys": atom.parent_keys,
                    "children_keys": atom.children_keys,
                    "creation_time": atom.creation_time.isoformat(),
                    "activation_count": atom.activation_count,
                    "active": atom.active,
                }
            )

        # Export edges (synapses)
        if self._adjacency is not None:
            rows, cols = self._adjacency.nonzero()
            for i, j in zip(rows, cols, strict=False):
                weight = self._adjacency[i, j]
                edges.append(
                    {
                        "source": self._keys[i],
                        "target": self._keys[j],
                        "weight": float(weight),
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_atoms": len(self._atoms),
                "total_synapses": len(edges),
                "learning_rate": self._lr,
                "export_time": datetime.now(UTC).isoformat(),
                "topology_summary": self.get_topology_summary(),
            },
        }

    async def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        """Return 768-dim Gemini text-embedding-004 embeddings for the given texts."""
        import httpx

        from .config import EMBEDDING_DIM

        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": os.getenv("GEMINI_API_KEY"),
        }

        try:
            embeddings = []
            for text in texts:
                body = {"content": {"parts": [{"text": text}]}}
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    result = response.json()
                    embedding = np.array(
                        result["embedding"]["values"], dtype=np.float32
                    )
                    embeddings.append(embedding)

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return EMBEDDING_DIM-dimensional zero vectors as fallback
            return [np.zeros(EMBEDDING_DIM, dtype=np.float32) for _ in texts]

    def embed_query(self, text: str) -> np.ndarray:
        """Return EMBEDDING_DIM-dimensional fallback vector (replace with real embedding later)."""
        from .config import EMBEDDING_DIM

        return np.random.rand(EMBEDDING_DIM).astype(np.float32)

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get all available tools from the neural store."""
        try:
            # Search for tool entries in the tools hierarchy
            results = await self.search_by_content(
                query="tool", hierarchy_filter=["tools"], limit=100
            )

            # Extract tool metadata
            tools = []
            for result in results:
                if isinstance(result.get("content"), dict):
                    content = result["content"]
                    if content.get("type") == "tool" or "signature" in content:
                        tools.append(
                            {
                                "key": content.get("key", result.get("memory_id", "")),
                                "name": content.get("name", ""),
                                "description": content.get("description", ""),
                                "signature": content.get("signature", {}),
                                "enabled": content.get("enabled", True),
                            }
                        )

            # Add built-in tools if no tools found
            if not tools:
                tools = [
                    {
                        "key": "web_agent",
                        "name": "Web & Code Search",
                        "description": "Search web and GitHub for information",
                        "signature": {
                            "query": {"type": "string", "description": "Search query"},
                            "web_k": {"type": "integer", "default": 5},
                            "github_k": {"type": "integer", "default": 3},
                        },
                        "enabled": True,
                    }
                ]

            logger.info(f"Found {len(tools)} available tools")
            return tools

        except Exception as e:
            logger.error(f"Error getting available tools: {e}")
            return []

    async def register_tool(self, tool_key: str, tool_metadata: dict[str, Any]):
        """Register a new tool in the neural store."""
        try:
            tool_data = {
                "type": "tool",
                "key": tool_key,
                "registered_at": datetime.now().isoformat(),
                **tool_metadata,
            }

            await self.upsert(
                memory_id=f"tool_{tool_key}",
                content=tool_data,
                hierarchy_path=["tools", "registered"],
            )

            logger.info(f"Registered tool: {tool_key}")

        except Exception as e:
            logger.error(f"Error registering tool {tool_key}: {e}")

    async def delete_expired(self):
        """Delete expired memories based on TTL."""
        try:
            # This would need to be implemented based on your ChromaDB setup
            # For now, just log the intent
            logger.info("Expired memory cleanup requested (not yet implemented)")

        except Exception as e:
            logger.error(f"Error during expired memory cleanup: {e}")

    async def downvote_unused(self):
        """Downvote unused atoms based on usage telemetry."""
        try:
            # This would analyze usage patterns and adjust ratings
            logger.info("Unused atom downvoting requested (not yet implemented)")

        except Exception as e:
            logger.error(f"Error during unused atom downvoting: {e}")

    def get_all(self) -> list[NeuralAtom]:
        """
        Get all Neural Atoms in the store.

        Returns:
            List of all NeuralAtom objects
        """
        return list(self._atoms.values())

    def get_by_name(self, name: str) -> NeuralAtom | None:
        """
        Get a Neural Atom by its metadata name.

        Args:
            name: The name to search for

        Returns:
            The NeuralAtom with matching name, or None if not found
        """
        for atom in self._atoms.values():
            if hasattr(atom, "metadata") and atom.metadata.name == name:
                return atom
        return None


def create_skill_atom(
    skill_name: str,
    skill_code: str,
    skill_description: str,
    parent_atoms: list[NeuralAtom] | None = None,
    embedding_vector: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> NeuralAtom:
    """
    Factory function to create a skill atom with proper Darwin-Gödel genealogy.

    This is the preferred method for creating skill atoms as it ensures
    proper genealogical tracking and metadata consistency.
    """
    if metadata is None:
        metadata = {}

    skill_data = {
        "name": skill_name,
        "code": skill_code,
        "description": skill_description,
        "type": "skill",
        "active": True,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": metadata,
    }

    return NeuralAtom(
        key=f"skill:{skill_name}",
        default_value=skill_data,
        vector=embedding_vector,
        parent_keys=[atom.key for atom in parent_atoms] if parent_atoms else [],
        birth_event=f"skill_creation:{skill_name}",
        lineage_metadata={
            "creation_method": "skill_discovery",
            "skill_type": metadata.get("skill_type", "generated"),
            "parent_count": len(parent_atoms) if parent_atoms else 0,
            "complexity_score": metadata.get("complexity", 0.0),
            "generation": metadata.get("generation", 0),
        },
    )


def create_memory_atom(
    memory_key: str,
    content: Any,
    memory_type: str = "semantic",
    embedding_vector: np.ndarray | None = None,
    parent_atoms: list[NeuralAtom] | None = None,
    confidence: float = 1.0,
) -> NeuralAtom:
    """
    Factory function to create a memory atom with proper genealogy tracking.

    Memory atoms represent stored knowledge, experiences, or learned patterns.
    """
    memory_data = {
        "content": content,
        "type": memory_type,
        "confidence": confidence,
        "timestamp": datetime.now(UTC).isoformat(),
        "access_count": 0,
        "last_accessed": datetime.now(UTC).isoformat(),
    }

    return NeuralAtom(
        key=f"memory:{memory_type}:{memory_key}",
        default_value=memory_data,
        vector=embedding_vector,
        parent_keys=[atom.key for atom in parent_atoms] if parent_atoms else [],
        birth_event=f"memory_creation:{memory_type}",
        lineage_metadata={
            "memory_type": memory_type,
            "confidence": confidence,
            "creation_method": "direct_storage",
            "content_hash": str(hash(str(content)))[:16] if content else "",
        },
    )


def create_goal_atom(
    goal_id: str,
    goal_description: str,
    priority: float = 0.5,
    parent_atoms: list[NeuralAtom] | None = None,
    embedding_vector: np.ndarray | None = None,
) -> NeuralAtom:
    """
    Factory function to create a goal atom representing an agent objective.

    Goal atoms drive the agent's behavior and decision-making processes.
    """
    goal_data = {
        "goal_id": goal_id,
        "description": goal_description,
        "priority": priority,
        "status": "active",
        "created_at": datetime.now(UTC).isoformat(),
        "progress": 0.0,
        "sub_goals": [],
        "completion_criteria": [],
    }

    return NeuralAtom(
        key=f"goal:{goal_id}",
        default_value=goal_data,
        vector=embedding_vector,
        parent_keys=[atom.key for atom in parent_atoms] if parent_atoms else [],
        birth_event=f"goal_creation:{goal_id}",
        lineage_metadata={
            "goal_type": "primary",
            "priority": priority,
            "creation_method": "direct_assignment",
            "expected_difficulty": "unknown",
        },
    )


# --- Helper Functions ---


def create_memory_atom(
    key: str, content: dict[str, Any], hierarchy_path: list[str], vector: np.ndarray
) -> NeuralAtom:
    """
    Creates a NeuralAtom specifically for storing a memory chunk.
    This standardizes the structure of memory atoms for easier testing and retrieval.
    """
    return NeuralAtom(
        key=key,
        default_value={"content": content, "hierarchy_path": hierarchy_path},
        vector=vector,
        birth_event="memory_creation",
    )


class TextualMemoryAtom(NeuralAtom):
    """
    A concrete NeuralAtom for storing and retrieving a piece of text.

    This class solves the TypeError that occurs when trying to instantiate
    the abstract NeuralAtom class directly for memory storage.
    """

    def __init__(
        self, metadata: NeuralAtomMetadata, content: str, embedding_client: Any = None
    ):
        # Call parent constructor with just metadata (the abstract NeuralAtom(ABC) expects only this)
        super().__init__(metadata)
        # Store additional properties specific to textual memories
        self.content = content
        self.key = metadata.name  # Add key attribute for NeuralStore compatibility
        self._embedding_client = embedding_client
        self._semantic_embedding = None
        # Generate embedding on creation if client available
        if self._embedding_client:
            self._semantic_embedding = self.get_embedding()

    async def execute(self, input_data: Any = None) -> Any:
        """Executing a memory atom simply returns its content."""
        return {
            "content": self.content,
            "memory_type": "textual",
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "usage_count": self.metadata.usage_count,
            },
        }

    def store(self, data: dict[str, Any]) -> bool:
        """
        Store data in the memory atom.

        Args:
            data: Data to store

        Returns:
            True if storage succeeded, False otherwise
        """
        try:
            # Store data by appending to content
            if not hasattr(self, "_stored_data"):
                self._stored_data = []

            self._stored_data.append(data)

            # Update content with summary
            self.content = f"Memory contains {len(self._stored_data)} records"

            # Update usage count
            self.metadata.usage_count += 1

            return True

        except Exception as e:
            logger.error(f"Failed to store data in memory atom: {e}")
            return False

    def retrieve(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Retrieve data from the memory atom.

        Args:
            limit: Maximum number of records to retrieve

        Returns:
            List of stored data records
        """
        try:
            # Return stored data
            if not hasattr(self, "_stored_data"):
                return []

            # Update usage count
            self.metadata.usage_count += 1

            return self._stored_data[-limit:]

        except Exception as e:
            logger.error(f"Failed to retrieve data from memory atom: {e}")
            return []

    def get_embedding(self) -> list[float]:
        """Generates or retrieves the semantic embedding for the memory content."""
        if self._semantic_embedding is None:
            if self._embedding_client and hasattr(self._embedding_client, "embed_text"):
                try:
                    self._semantic_embedding = self._embedding_client.embed_text(
                        self.content
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for memory: {e}")
                    self._semantic_embedding = self._generate_simple_embedding(
                        self.content
                    )
            else:
                # Fallback to simple hash-based embedding
                self._semantic_embedding = self._generate_simple_embedding(self.content)

        return self._semantic_embedding

    def can_handle(self, task_description: str) -> float:
        """
        A memory atom can assist with recall tasks but cannot handle active execution.
        Returns a higher score for memory/recall related tasks.
        """
        task_lower = task_description.lower()

        # Check for memory/recall keywords
        memory_keywords = [
            "remember",
            "recall",
            "memory",
            "what did",
            "previous",
            "stored",
        ]
        if any(keyword in task_lower for keyword in memory_keywords):
            # Check if the task description relates to our content
            content_words = self.content.lower().split()
            task_words = task_lower.split()
            overlap = len(set(content_words) & set(task_words))
            return min(0.7, overlap * 0.1)  # Cap at 0.7, scale by word overlap

        return 0.0  # Memory atoms don't handle active tasks

    def _generate_simple_embedding(self, text: str) -> list[float]:
        """Generate simple hash-based embedding as fallback."""
        import hashlib
        import math

        # Create multiple hash variations for better distribution
        hashes = [
            hashlib.md5(text.encode()).hexdigest(),
            hashlib.sha1(text.encode()).hexdigest(),
        ]

        # Convert to 128-dimensional vector
        embedding = []
        for hash_str in hashes:
            for i in range(0, min(len(hash_str), 32), 2):
                hex_val = int(hash_str[i : i + 2], 16)
                # Use trigonometric functions for better distribution
                val = math.sin(hex_val * 0.1) * math.cos(hex_val * 0.05)
                embedding.append(val)

        # Ensure exactly 128 dimensions
        while len(embedding) < 128:
            embedding.append(0.0)

        return embedding[:128]

    def get_content_summary(self) -> str:
        """Get a brief summary of the stored content."""
        if len(self.content) <= 100:
            return self.content
        return self.content[:97] + "..."
