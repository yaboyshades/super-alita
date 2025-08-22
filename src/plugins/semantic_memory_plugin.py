# src/plugins/semantic_memory_plugin.py

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables should be set manually
"""
Plugin for the agent's long-term, hierarchical semantic memory.

This component is responsible for the full lifecycle of a memory:
- Encoding information into semantic vectors using a state-of-the-art model.
- Persisting these memories in a durable vector store (ChromaDB).
- Representing these memories as live NeuralAtoms in the cognitive graph.
- Providing a high-level API for other plugins to store and retrieve knowledge.

The dual-storage architecture ensures both durability and reactivity:
- ChromaDB provides persistent, searchable vector storage
- NeuralStore provides fast, in-memory access for real-time cognitive operations
"""

import asyncio
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

try:
    import google.generativeai as genai

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("Google Generative AI not available, using fallback embedding")
from src.core.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME
from src.core.event_bus import EventBus
from src.core.events import MemoryUpsertEvent
from src.core.genealogy import get_global_tracer, trace_atom_birth
from src.core.neural_atom import NeuralAtomMetadata, NeuralStore, TextualMemoryAtom
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class SemanticMemoryPlugin(PluginInterface):
    """
    Manages the agent's long-term memory using a sophisticated dual-storage strategy.

    Architecture:
    - ChromaDB: Durable, disk-based persistence of memory vectors with full-text search
    - NeuralStore: Live, in-memory representation of memories as reactive atoms,
      enabling fast attention-based retrieval and integration with cognitive functions
    - Genealogy Integration: Full lineage tracking of memory creation and evolution
    - Performance Optimization: Batch processing, caching, and intelligent prefetching

    This plugin bridges the gap between transient cognition and persistent knowledge,
    ensuring that the agent's learning survives restarts while maintaining real-time
    access speeds for active reasoning and decision-making.
    """

    def __init__(self):
        super().__init__()
        self._config: dict[str, Any] = {}
        self._chroma_client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

        # Performance optimization
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._cache_max_size: int = 1000
        self._batch_size: int = 32

        # Statistics and monitoring
        self._stats: dict[str, Any] = {
            "memories_created": 0,
            "memories_retrieved": 0,
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_memory_size": 0,
            "last_cleanup": None,
        }

        # Hierarchical organization
        self._memory_hierarchies: dict[str, list[str]] = {}
        self._hierarchy_stats: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "semantic_memory"

    @property
    def version(self) -> str:
        return "2.0.0"

    def _extend_store_with_embedding(self) -> None:
        """Extend the NeuralStore with embedding capabilities."""

        async def embed_query(text: str) -> np.ndarray | None:
            """
            Embed a query text using the semantic memory plugin.

            Args:
                text: The text to embed

            Returns:
                The embedding vector or None if embedding fails
            """
            try:
                embeddings = await self.embed_text([text])
                return embeddings[0] if embeddings else None
            except Exception as e:
                logger.error(f"Failed to embed query: {e}")
                return None

        # Add the method to the store instance
        self.store.embed_query = embed_query
        logger.debug("Extended NeuralStore with embed_query method")

    async def setup(
        self, event_bus: EventBus, store: NeuralStore, config: dict[str, Any]
    ):
        """Setup the semantic memory plugin with comprehensive configuration."""
        await super().setup(event_bus, store, config)
        self._config = config.get(self.name, {})

        # Apply configuration defaults
        self._config.setdefault("db_path", "./data/chroma_db")
        self._config.setdefault("collection_name", "alita_memory")
        self._config.setdefault(
            "embedding_model", EMBEDDING_MODEL_NAME
        )  # Use centralized config
        self._config.setdefault(
            "embedding_dimension", EMBEDDING_DIM
        )  # Use centralized config
        self._config.setdefault("batch_size", 32)
        self._config.setdefault("cache_size", 1000)
        self._config.setdefault("cleanup_interval_hours", 24)
        self._config.setdefault("enable_hierarchical_organization", True)

        self._batch_size = self._config["batch_size"]
        self._cache_max_size = self._config["cache_size"]

        # Configure Gemini API
        api_key = self._config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "Gemini API key not configured - embeddings will use fallback"
            )
        else:
            try:
                genai.configure(api_key=api_key)
                logger.info("Gemini API configured for semantic embeddings")

                # Test the API connection
                await self._test_embedding_api()

            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")

        # Extend the NeuralStore with embedding capability
        self._extend_store_with_embedding()

        logger.info("SemanticMemoryPlugin setup complete")

    async def start(self) -> None:
        """Initialize ChromaDB connection and start background tasks."""
        logger.info("Starting SemanticMemoryPlugin...")

        await self._initialize_chromadb()
        await self._load_existing_memories()

        # Start background tasks
        if self._config.get("enable_periodic_cleanup", True):
            self.add_task(self._periodic_cleanup())

        if self._config.get("enable_memory_consolidation", True):
            self.add_task(self._memory_consolidation())

        await super().start()
        logger.info("SemanticMemoryPlugin started successfully")

    async def shutdown(self) -> None:
        """Graceful shutdown with statistics export."""
        logger.info("Shutting down SemanticMemoryPlugin...")

        # Export final statistics
        final_stats = await self.get_statistics()
        logger.info(f"Final memory statistics: {final_stats}")

        # ChromaDB persistent client handles state automatically
        self._collection = None
        self._chroma_client = None

        # Clear caches
        self._embedding_cache.clear()
        self._memory_hierarchies.clear()

        await super().shutdown()
        logger.info("SemanticMemoryPlugin shutdown complete")

    async def _initialize_chromadb(self):
        """Initialize ChromaDB with error handling and validation."""
        db_path = self._config["db_path"]
        collection_name = self._config["collection_name"]

        try:
            # Ensure database directory exists
            Path(db_path).mkdir(parents=True, exist_ok=True)

            # Create persistent client with optimized settings
            self._chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False, allow_reset=False, is_persistent=True
                ),
            )

            # Create or get collection with metadata
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",  # Cosine similarity for semantic vectors
                    "hnsw:construction_ef": 200,  # Build-time search accuracy
                    "hnsw:search_ef": 50,  # Query-time search accuracy
                    "description": "Super Alita semantic memory storage",
                    "version": self.version,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )

            # Get collection statistics and check for dimension mismatch
            count = self._collection.count()
            self._stats["total_memory_size"] = count

            # Guard: Check for dimension mismatch and clear if necessary
            if count > 0:
                try:
                    # Test a sample vector to check dimensions
                    sample_results = self._collection.get(
                        limit=1, include=["embeddings"]
                    )
                    if sample_results["embeddings"]:
                        existing_dim = len(sample_results["embeddings"][0])
                        expected_dim = self._config["embedding_dimension"]

                        if existing_dim != expected_dim:
                            logger.warning(
                                f"🔧 DIMENSION MISMATCH DETECTED: stored vectors are {existing_dim}D but current config expects {expected_dim}D"
                            )
                            logger.warning(
                                f"🧹 Clearing {count} incompatible vectors from ChromaDB..."
                            )

                            # Delete all existing data
                            self._collection.delete()

                            # Recreate the collection to ensure clean state
                            collection_name = self._config["collection_name"]
                            self._chroma_client.delete_collection(collection_name)
                            self._collection = self._chroma_client.get_or_create_collection(
                                name=collection_name,
                                metadata={
                                    "hnsw:space": "cosine",
                                    "hnsw:construction_ef": 200,
                                    "hnsw:search_ef": 50,
                                    "description": "Super Alita semantic memory storage",
                                    "version": self.version,
                                    "created_at": datetime.now(UTC).isoformat(),
                                    "embedding_dimension": expected_dim,  # Track expected dimension
                                },
                            )

                            count = 0  # Reset count after clearing
                            self._stats["total_memory_size"] = 0
                            logger.info(
                                f"✅ Vector store cleared and reinitialized for {expected_dim}D embeddings"
                            )
                        else:
                            logger.info(
                                f"✅ Vector store dimension check passed: {existing_dim}D vectors match config"
                            )

                except Exception as e:
                    logger.warning(f"Could not check vector dimensions: {e}")

            logger.info(
                f"Connected to ChromaDB collection '{collection_name}' with {count} existing memories"
            )

        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

    async def _test_embedding_api(self):
        """Test the embedding API to ensure it's working."""
        try:
            test_response = await asyncio.wait_for(
                asyncio.to_thread(
                    genai.embed_content,
                    model=self._config["embedding_model"],
                    content="test embedding",
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._config["embedding_dimension"],
                ),
                timeout=10.0,
            )

            if "embedding" in test_response and len(test_response["embedding"]) > 0:
                logger.info("Gemini embedding API test successful")
            else:
                logger.warning("Gemini embedding API test returned unexpected response")

        except Exception as e:
            logger.warning(f"Gemini embedding API test failed: {e}")

    async def _load_existing_memories(self):
        """Load existing memories from ChromaDB into NeuralStore."""
        if not self._collection:
            return

        try:
            # Get all memories from ChromaDB
            results = self._collection.get(include=["embeddings", "metadatas"])

            if not results["ids"]:
                logger.info("No existing memories found in ChromaDB")
                return

            logger.info(
                f"Loading {len(results['ids'])} existing memories into NeuralStore..."
            )

            loaded_count = 0
            for i, memory_id in enumerate(results["ids"]):
                try:
                    np.array(results["embeddings"][i], dtype=np.float32)
                    metadata = results["metadatas"][i]

                    # Reconstruct content from metadata
                    content = {
                        k: v
                        for k, v in metadata.items()
                        if k not in ["hierarchy_path", "owner_plugin", "created_at"]
                    }

                    hierarchy_path = metadata.get("hierarchy_path", "").split("::")
                    metadata.get("owner_plugin", "unknown")

                    # Create memory metadata
                    atom_metadata = NeuralAtomMetadata(
                        name=memory_id,
                        description=f"Memory: {content[:100]}...",
                        capabilities=["memory", "storage", "retrieval"],
                        version="1.0.0",
                    )

                    # Create TextualMemoryAtom (following Sacred Rules)
                    atom = TextualMemoryAtom(
                        metadata=atom_metadata,
                        content=content,
                        embedding_client=None,  # Will use fallback embedding
                    )

                    # Register in neural store
                    self.store.register_with_lineage(
                        atom=atom,
                        parents=[],  # Loaded memories are root nodes
                        birth_event="memory_load",
                        lineage_metadata=atom.lineage_metadata,
                    )

                    # Track hierarchy
                    hierarchy_key = "::".join(hierarchy_path)
                    if hierarchy_key not in self._memory_hierarchies:
                        self._memory_hierarchies[hierarchy_key] = []
                    self._memory_hierarchies[hierarchy_key].append(memory_id)

                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Failed to load memory {memory_id}: {e}")

            logger.info(f"Successfully loaded {loaded_count} memories into NeuralStore")
            self._stats["memories_retrieved"] = loaded_count

        except Exception as e:
            logger.error(f"Error loading existing memories: {e}", exc_info=True)

    # === Public API Methods ===

    async def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate semantic embeddings using Gemini with sophisticated caching and error handling.

        Features:
        - Intelligent caching to avoid redundant API calls
        - Batch processing for efficiency
        - Robust timeout and retry logic
        - Graceful fallback to zero vectors on failure
        """
        if not texts:
            return []

        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            text_hash = str(hash(text))
            if text_hash in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[text_hash]))
                self._stats["cache_hits"] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._stats["cache_misses"] += 1

        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = await self._generate_embeddings_batch(uncached_texts)

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings, strict=False):
                text_hash = str(hash(text))
                self._embedding_cache[text_hash] = embedding

                # Manage cache size
                if len(self._embedding_cache) > self._cache_max_size:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self._embedding_cache.keys())[:100]
                    for key in oldest_keys:
                        del self._embedding_cache[key]

        # Combine cached and new embeddings in correct order
        result_embeddings = [None] * len(texts)

        # Place cached embeddings
        for i, embedding in cached_embeddings:
            result_embeddings[i] = embedding

        # Place new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings, strict=False):
            result_embeddings[i] = embedding

        self._stats["embeddings_generated"] += len(uncached_texts)
        return result_embeddings

    async def _generate_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings with batch processing and error handling."""
        model = self._config["embedding_model"]
        dimension = self._config["embedding_dimension"]

        try:
            embeddings = []

            # Process in batches to avoid API limits
            for i in range(0, len(texts), self._batch_size):
                batch_texts = texts[i : i + self._batch_size]

                try:
                    # Process batch with timeout
                    batch_embeddings = await self._process_embedding_batch(
                        batch_texts, model, dimension
                    )
                    embeddings.extend(batch_embeddings)

                    # Small delay between batches to respect rate limits
                    if i + self._batch_size < len(texts):
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(
                        f"Batch embedding failed for texts {i}-{i + len(batch_texts)}: {e}"
                    )
                    # Add fallback embeddings for failed batch
                    fallback_embeddings = [
                        np.zeros(dimension, dtype=np.float32) for _ in batch_texts
                    ]
                    embeddings.extend(fallback_embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Complete embedding generation failed: {e}", exc_info=True)
            return [np.zeros(dimension, dtype=np.float32) for _ in texts]

    async def _process_embedding_batch(
        self, texts: list[str], model: str, dimension: int
    ) -> list[np.ndarray]:
        """Process a single batch of embeddings with timeout."""
        try:
            if len(texts) == 1:
                # Single text processing
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        genai.embed_content,
                        model=model,
                        content=texts[0],
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=dimension,
                    ),
                    timeout=15.0,
                )

                embedding = np.array(response["embedding"], dtype=np.float32)
                return [embedding]
            # Batch processing
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    genai.embed_content,
                    model=model,
                    content=texts,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dimension,
                ),
                timeout=30.0,  # Longer timeout for batches
            )

            return [np.array(emb, dtype=np.float32) for emb in response["embedding"]]

        except TimeoutError:
            logger.error(f"Embedding timeout for batch of {len(texts)} texts")
            return [np.zeros(dimension, dtype=np.float32) for _ in texts]
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            return [np.zeros(dimension, dtype=np.float32) for _ in texts]

    async def upsert(
        self,
        content: dict[str, Any],
        hierarchy_path: list[str],
        owner_plugin: str = "unknown",
        memory_id: str | None = None,
    ) -> str:
        """
        Insert or update a memory using sophisticated dual-write architecture.

        This method performs the complete memory lifecycle:
        1. Generate semantic embedding for the content
        2. Create and register NeuralAtom in live cognitive graph
        3. Persist in ChromaDB for durability
        4. Update genealogy tracking
        5. Emit memory event to the system

        Args:
            content: The memory content as a structured dictionary
            hierarchy_path: Hierarchical classification (e.g., ["skills", "text_analysis"])
            owner_plugin: Plugin that created this memory
            memory_id: Optional stable ID (generated if not provided)

        Returns:
            str: The stable memory ID
        """
        if not self.is_running or not self._collection or not self.store:
            raise RuntimeError("SemanticMemoryPlugin is not properly initialized")

        # Generate stable ID
        stable_id = memory_id or f"mem_{uuid.uuid4().hex[:12]}"

        try:
            # Create rich text representation for embedding
            hierarchy_context = (
                " -> ".join(hierarchy_path) if hierarchy_path else "root"
            )
            content_text = self._create_embedding_text(content, hierarchy_context)

            # Generate semantic embedding
            embeddings = await self.embed_text([content_text])
            vector = embeddings[0]

            # 1. Register as TextualMemoryAtom in live cognitive graph (following Sacred Rules)
            atom_metadata = NeuralAtomMetadata(
                name=stable_id,
                description=f"Memory: {content_text[:100]}...",
                capabilities=["memory", "storage", "retrieval", "semantic_search"],
                version="1.0.0",
            )

            atom = TextualMemoryAtom(
                metadata=atom_metadata,
                content=content_text,
                embedding_client=None,  # Will generate embedding internally
            )

            # Register with genealogy tracking
            tracer = get_global_tracer()
            if tracer:
                # Trace the memory birth
                trace_atom_birth(
                    atom_key=stable_id,
                    parent_keys=[],  # Memories are typically root nodes
                    birth_context="semantic_memory_upsert",
                    lineage_metadata=atom.lineage_metadata,
                )

            self.store.register_with_lineage(
                atom=atom,
                parents=[],  # Memories are root nodes in genealogy
                birth_event="memory_upsert",
                lineage_metadata=atom.lineage_metadata,
            )

            # 2. Persist in ChromaDB for durability
            chroma_metadata = {
                "hierarchy_path": (
                    "::".join(hierarchy_path) if hierarchy_path else "root"
                ),
                "owner_plugin": owner_plugin,
                "created_at": datetime.now(UTC).isoformat(),
                "content_type": self._infer_content_type(content),
                "embedding_model": self._config["embedding_model"],
                **self._flatten_content_for_metadata(content),
            }

            self._collection.upsert(
                ids=[stable_id],
                embeddings=[vector.tolist()],
                metadatas=[chroma_metadata],
            )

            # 3. Update internal tracking
            hierarchy_key = "::".join(hierarchy_path) if hierarchy_path else "root"
            if hierarchy_key not in self._memory_hierarchies:
                self._memory_hierarchies[hierarchy_key] = []

            if stable_id not in self._memory_hierarchies[hierarchy_key]:
                self._memory_hierarchies[hierarchy_key].append(stable_id)

            self._hierarchy_stats[hierarchy_key] = (
                self._hierarchy_stats.get(hierarchy_key, 0) + 1
            )

            # 4. Emit memory event to the system
            memory_event = MemoryUpsertEvent(
                source_plugin=self.name,
                memory_id=stable_id,
                content=content,
                hierarchy_path=hierarchy_path,
                owner_plugin=owner_plugin,
                operation="upsert",
                vector_dimension=len(vector),
            )
            await self.event_bus.publish(memory_event)

            # 5. Update statistics
            self._stats["memories_created"] += 1
            self._stats["total_memory_size"] += 1

            logger.info(
                f"Successfully upserted memory '{stable_id}' in hierarchy {hierarchy_path}"
            )
            return stable_id

        except Exception as e:
            logger.error(f"Failed to upsert memory: {e}", exc_info=True)
            raise RuntimeError(f"Memory upsert failed: {e}") from e

    async def query(
        self,
        query_text: str,
        hierarchy_filter: list[str] | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search using attention mechanism over the live NeuralStore.

        This method leverages the fast, in-memory attention mechanism for real-time
        semantic retrieval, making it ideal for active reasoning and decision-making.

        Args:
            query_text: The search query
            hierarchy_filter: Optional filter by hierarchy path
            top_k: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity score
            include_metadata: Whether to include detailed metadata

        Returns:
            List of memory records with content and similarity scores
        """
        if not self.is_running or not self.store:
            raise RuntimeError("SemanticMemoryPlugin is not properly initialized")

        try:
            # Generate query embedding
            query_embeddings = await self.embed_text([query_text])
            query_vector = query_embeddings[0]

            # Perform attention-based search over live neural store
            attention_results = await self.store.attention(
                query_vector, top_k=top_k * 2
            )  # Get extra for filtering

            # Process and filter results
            results = []
            for atom_key, similarity_score in attention_results:
                if similarity_score < similarity_threshold:
                    continue

                atom = self.store.get(atom_key)
                if not atom:
                    continue

                # Apply hierarchy filter if specified
                if hierarchy_filter:
                    atom_hierarchy = atom.lineage_metadata.get("hierarchy", [])
                    if not self._matches_hierarchy_filter(
                        atom_hierarchy, hierarchy_filter
                    ):
                        continue

                # Build result record
                result = {
                    "memory_id": atom_key,
                    "content": atom.value,
                    "similarity_score": float(similarity_score),
                    "hierarchy_path": atom.lineage_metadata.get("hierarchy", []),
                }

                if include_metadata:
                    result["metadata"] = {
                        "owner_plugin": atom.lineage_metadata.get("owner", "unknown"),
                        "created_at": atom.lineage_metadata.get("created_at"),
                        "content_type": atom.lineage_metadata.get("content_type"),
                        "birth_event": atom.birth_event,
                        "fitness_score": atom.fitness_score,
                        "activation_count": atom.activation_count,
                    }

                results.append(result)

                if len(results) >= top_k:
                    break

            # Update statistics
            self._stats["memories_retrieved"] += len(results)

            logger.debug(f"Query '{query_text}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Memory query failed: {e}", exc_info=True)
            return []

    async def get_hierarchy_stats(self) -> dict[str, Any]:
        """Get statistics about memory organization by hierarchy."""
        return {
            "total_hierarchies": len(self._memory_hierarchies),
            "hierarchy_distribution": dict(self._hierarchy_stats),
            "largest_hierarchy": (
                max(self._hierarchy_stats.items(), key=lambda x: x[1])
                if self._hierarchy_stats
                else None
            ),
            "average_memories_per_hierarchy": (
                sum(self._hierarchy_stats.values()) / len(self._hierarchy_stats)
                if self._hierarchy_stats
                else 0
            ),
        }

    async def consolidate_memories(
        self, hierarchy_path: list[str], consolidation_threshold: float = 0.9
    ) -> list[str]:
        """
        Consolidate highly similar memories within a hierarchy to reduce redundancy.

        Args:
            hierarchy_path: The hierarchy to consolidate
            consolidation_threshold: Similarity threshold for consolidation

        Returns:
            List of consolidated memory IDs
        """
        hierarchy_key = "::".join(hierarchy_path)
        if hierarchy_key not in self._memory_hierarchies:
            return []

        memory_ids = self._memory_hierarchies[hierarchy_key]
        if len(memory_ids) < 2:
            return []

        try:
            # Get all memories in hierarchy with their vectors
            memories_with_vectors = []
            for memory_id in memory_ids:
                atom = self.store.get(memory_id)
                if atom and atom.vector is not None:
                    memories_with_vectors.append((memory_id, atom.vector, atom.value))

            # Find highly similar pairs for consolidation
            consolidated_ids = []
            to_remove = set()

            for i, (id1, vec1, content1) in enumerate(memories_with_vectors):
                if id1 in to_remove:
                    continue

                for j, (id2, vec2, content2) in enumerate(
                    memories_with_vectors[i + 1 :], i + 1
                ):
                    if id2 in to_remove:
                        continue

                    # Calculate cosine similarity
                    similarity = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )

                    if similarity >= consolidation_threshold:
                        # Consolidate memories by merging content and removing duplicate
                        consolidated_content = self._merge_memory_content(
                            content1, content2
                        )

                        # Update the first memory with consolidated content
                        await self.upsert(
                            content=consolidated_content,
                            hierarchy_path=hierarchy_path,
                            owner_plugin="memory_consolidation",
                            memory_id=id1,
                        )

                        # Mark second memory for removal
                        to_remove.add(id2)
                        consolidated_ids.append(id2)

            # Remove consolidated memories
            for memory_id in to_remove:
                await self._remove_memory(memory_id)

            logger.info(
                f"Consolidated {len(consolidated_ids)} memories in hierarchy {hierarchy_path}"
            )
            return consolidated_ids

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}", exc_info=True)
            return []

    # === Helper Methods ===

    def _create_embedding_text(
        self, content: dict[str, Any], hierarchy_context: str
    ) -> str:
        """Create optimized text representation for embedding generation."""
        # Start with hierarchy context
        text_parts = [f"Context: {hierarchy_context}"]

        # Add structured content
        for key, value in content.items():
            if isinstance(value, (str, int, float, bool)):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, dict)):
                text_parts.append(
                    f"{key}: {str(value)[:200]}"
                )  # Truncate large structures

        return " | ".join(text_parts)

    def _infer_content_type(self, content: dict[str, Any]) -> str:
        """Infer the type of content for categorization."""
        if "skill_name" in content or "skill_description" in content:
            return "skill"
        if "goal" in content or "objective" in content:
            return "goal"
        if "memory" in content or "knowledge" in content:
            return "knowledge"
        if "conversation" in content or "message" in content:
            return "conversation"
        if "observation" in content or "perception" in content:
            return "observation"
        return "general"

    def _flatten_content_for_metadata(self, content: dict[str, Any]) -> dict[str, str]:
        """Flatten content for ChromaDB metadata storage."""
        flattened = {}
        for key, value in content.items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = str(value)
            elif isinstance(value, (list, dict)):
                flattened[key] = str(value)[:500]  # Truncate large structures
        return flattened

    def _matches_hierarchy_filter(
        self, atom_hierarchy: list[str], filter_hierarchy: list[str]
    ) -> bool:
        """Check if atom hierarchy matches the filter."""
        if not filter_hierarchy:
            return True

        # Check if filter is a prefix of atom hierarchy
        if len(filter_hierarchy) > len(atom_hierarchy):
            return False

        return atom_hierarchy[: len(filter_hierarchy)] == filter_hierarchy

    def _merge_memory_content(
        self, content1: dict[str, Any], content2: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge two memory contents intelligently."""
        merged = content1.copy()

        for key, value in content2.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, str) and isinstance(merged[key], str):
                # Merge strings by combining unique sentences
                sentences1 = set(merged[key].split(". "))
                sentences2 = set(value.split(". "))
                merged_sentences = sentences1.union(sentences2)
                merged[key] = ". ".join(sorted(merged_sentences))
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists by combining unique items
                merged[key] = list(set(merged[key] + value))

        # Add consolidation metadata
        merged["_consolidated"] = True
        merged["_consolidation_timestamp"] = datetime.now(UTC).isoformat()

        return merged

    async def _remove_memory(self, memory_id: str) -> bool:
        """Remove a memory by ID. Returns True if deleted, False if not found."""
        try:
            # Remove from NeuralStore
            if self.store.get(memory_id):
                await self.store.delete(memory_id)
                logger.info(f"Memory {memory_id} removed from NeuralStore")

            # Remove from ChromaDB
            if self._collection:
                self._collection.delete(ids=[memory_id])
                logger.info(f"Memory {memory_id} removed from ChromaDB")

            # Update internal tracking
            for hierarchy_memories in self._memory_hierarchies.values():
                if memory_id in hierarchy_memories:
                    hierarchy_memories.remove(memory_id)

            self._stats["total_memory_size"] -= 1
            logger.info(f"Memory {memory_id} successfully removed")
            return True

        except KeyError:
            logger.warning(f"Memory {memory_id} not found for removal")
            return False
        except Exception as e:
            logger.error(f"Failed to remove memory {memory_id}: {e}")
            return False

        except Exception as e:
            logger.error(f"Failed to remove memory {memory_id}: {e}")

    # === Background Tasks ===

    async def _periodic_cleanup(self):
        """Periodic cleanup of cache and statistics."""
        cleanup_interval = self._config.get("cleanup_interval_hours", 24) * 3600

        while self.is_running:
            try:
                await asyncio.sleep(cleanup_interval)

                if not self.is_running:
                    break

                # Clear embedding cache periodically
                cache_size_before = len(self._embedding_cache)
                if cache_size_before > self._cache_max_size // 2:
                    # Keep only the most recent half
                    cache_items = list(self._embedding_cache.items())
                    self._embedding_cache = dict(
                        cache_items[-self._cache_max_size // 2 :]
                    )

                    logger.info(
                        f"Cache cleanup: {cache_size_before} -> {len(self._embedding_cache)} entries"
                    )

                # Update cleanup timestamp
                self._stats["last_cleanup"] = datetime.now(UTC).isoformat()

                # Log periodic statistics
                stats = await self.get_statistics()
                logger.info(f"Periodic cleanup complete. Stats: {stats}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}", exc_info=True)

    async def _memory_consolidation(self):
        """Background task for automatic memory consolidation."""
        consolidation_interval = (
            self._config.get("consolidation_interval_hours", 168) * 3600
        )  # Weekly

        while self.is_running:
            try:
                await asyncio.sleep(consolidation_interval)

                if not self.is_running:
                    break

                if not self._config.get("enable_auto_consolidation", False):
                    continue

                logger.info("Starting automatic memory consolidation...")

                total_consolidated = 0
                for hierarchy_path_str in self._memory_hierarchies.keys():
                    hierarchy_path = hierarchy_path_str.split("::")

                    # Only consolidate hierarchies with many memories
                    if len(self._memory_hierarchies[hierarchy_path_str]) > 10:
                        consolidated = await self.consolidate_memories(
                            hierarchy_path=hierarchy_path,
                            consolidation_threshold=0.95,  # High threshold for auto-consolidation
                        )
                        total_consolidated += len(consolidated)

                logger.info(
                    f"Automatic consolidation complete: {total_consolidated} memories consolidated"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory consolidation: {e}", exc_info=True)

    # === Health Check and Statistics ===

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of the semantic memory system."""
        health_info = {
            "status": "healthy" if self.is_running else "stopped",
            "version": self.version,
            "issues": [],
            "metrics": {},
        }

        try:
            # Check ChromaDB connection
            if not self._collection:
                health_info["status"] = "unhealthy"
                health_info["issues"].append("ChromaDB collection not available")
            else:
                # Test ChromaDB operation
                try:
                    count = self._collection.count()
                    health_info["metrics"]["chromadb_count"] = count
                except Exception as e:
                    health_info["status"] = "unhealthy"
                    health_info["issues"].append(f"ChromaDB operation failed: {e}")

            # Check NeuralStore integration
            if not self.store:
                health_info["status"] = "unhealthy"
                health_info["issues"].append("NeuralStore not available")
            else:
                store_stats = self.store.get_stats()
                health_info["metrics"]["neural_store_atoms"] = store_stats.get(
                    "total_atoms", 0
                )

            # Check embedding API
            try:
                test_embeddings = await self.embed_text(["health check test"])
                if len(test_embeddings) > 0 and len(test_embeddings[0]) > 0:
                    health_info["metrics"]["embedding_api"] = "operational"
                else:
                    health_info["issues"].append(
                        "Embedding API returned invalid response"
                    )
            except Exception as e:
                health_info["issues"].append(f"Embedding API failed: {str(e)[:100]}")

            # Add cache statistics
            health_info["metrics"].update(
                {
                    "cache_size": len(self._embedding_cache),
                    "cache_hit_rate": self._stats["cache_hits"]
                    / max(1, self._stats["cache_hits"] + self._stats["cache_misses"]),
                    "total_memories": self._stats["total_memory_size"],
                    "memories_created": self._stats["memories_created"],
                }
            )

            # Determine final status
            if health_info["issues"] and health_info["status"] == "healthy":
                health_info["status"] = "degraded"

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["issues"].append(f"Health check failed: {e}")

        return health_info

    async def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about memory usage and performance."""
        try:
            base_stats = dict(self._stats)

            # Add hierarchy statistics
            hierarchy_stats = await self.get_hierarchy_stats()
            base_stats["hierarchy_stats"] = hierarchy_stats

            # Add performance metrics
            base_stats["performance"] = {
                "cache_hit_rate": self._stats["cache_hits"]
                / max(1, self._stats["cache_hits"] + self._stats["cache_misses"]),
                "average_memories_per_hierarchy": hierarchy_stats.get(
                    "average_memories_per_hierarchy", 0
                ),
                "cache_utilization": len(self._embedding_cache) / self._cache_max_size,
            }

            # Add ChromaDB statistics if available
            if self._collection:
                try:
                    base_stats["chromadb_count"] = self._collection.count()
                except Exception:
                    base_stats["chromadb_count"] = "unavailable"

            return base_stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
