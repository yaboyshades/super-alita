# Version: 3.0.0
# Description: Implements memory management for the unified Neural Atoms architecture.

import hashlib
import logging
import time
from typing import Any

from src.core.global_workspace import AttentionLevel, GlobalWorkspace, WorkspaceEvent
from src.core.neural_atom import NeuralStore
from src.core.plugin_interface import PluginInterface
from src.core.schemas import (
    MemoryRequest,
    MemoryResult,
    MemoryType,
    WorkingMemoryUpdate,
)

# Try to import ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Try to import Google embeddings
try:
    import google.generativeai as genai

    HAS_GEMINI_EMBEDDINGS = True
except ImportError:
    HAS_GEMINI_EMBEDDINGS = False

logger = logging.getLogger(__name__)


class MemoryManagerPlugin(PluginInterface):
    """
    Manages different types of memory for the agent:
    - Working Memory: Short-term, active information
    - Episodic Memory: Experience and event sequences
    - Semantic Memory: Knowledge and concept relationships
    - Long-term Memory: Persistent knowledge storage
    """

    def __init__(self):
        super().__init__()
        self.workspace: GlobalWorkspace | None = None
        self.store: NeuralStore | None = None

        # Memory storage components
        self.chroma_client: Any | None = None
        self.semantic_collection: Any | None = None
        self.episodic_collection: Any | None = None

        # Memory state
        self.working_memory: dict[str, Any] = {}
        self.working_memory_capacity = 128  # Configurable capacity
        self.working_memory_access_times: dict[str, float] = {}

        # Episodic memory tracking
        self.current_episode: dict[str, Any] | None = None
        self.episode_buffer: list[dict[str, Any]] = []
        self.max_episode_length = 50

        # Performance tracking
        self.memory_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "working_memory_size": 0,
            "episodic_memories_stored": 0,
            "semantic_memories_stored": 0,
            "average_retrieval_time": 0.0,
        }

    async def setup(
        self, workspace: GlobalWorkspace, store: NeuralStore, config: dict[str, Any]
    ):
        """Initialize the Memory Manager Plugin with workspace and store."""
        await super().setup(workspace, store, config)

        self.workspace = workspace
        self.store = store

        # Configure memory settings
        self.working_memory_capacity = config.get("working_memory_capacity", 128)
        self.max_episode_length = config.get("max_episode_length", 50)

        # Initialize vector database if available
        if HAS_CHROMADB:
            await self._initialize_vector_store(config)
        else:
            logger.warning("ChromaDB not available - semantic memory limited")

        # Initialize embeddings if available
        if HAS_GEMINI_EMBEDDINGS:
            await self._initialize_embeddings(config)
        else:
            logger.warning("Gemini embeddings not available - using simple hashing")

        logger.info("Memory Manager Plugin initialized")

    async def _initialize_vector_store(self, config: dict[str, Any]):
        """Initialize ChromaDB for vector storage."""
        try:
            persist_dir = config.get("chroma_persist_dir", "./data/chroma_memory")

            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir, settings=Settings(anonymized_telemetry=False)
            )

            # Create collections for different memory types
            self.semantic_collection = self.chroma_client.get_or_create_collection(
                name="semantic_memory",
                metadata={"description": "Long-term semantic knowledge"},
            )

            self.episodic_collection = self.chroma_client.get_or_create_collection(
                name="episodic_memory",
                metadata={"description": "Experience and event sequences"},
            )

            logger.info("Vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.chroma_client = None

    async def _initialize_embeddings(self, config: dict[str, Any]):
        """Initialize embedding generation."""
        try:
            import os

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                logger.info("Embeddings initialized with Gemini")
            else:
                logger.warning("GEMINI_API_KEY not found for embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")

    async def start(self):
        """Start the Memory Manager Plugin and subscribe to workspace events."""
        await super().start()

        if self.workspace:
            self.workspace.subscribe("memory", self._handle_workspace_event)
            logger.info("Memory Manager Plugin subscribed to Global Workspace")

    async def shutdown(self):
        """Gracefully shutdown the Memory Manager Plugin."""
        # Save current episode if active
        if self.current_episode and self.episode_buffer:
            await self._store_current_episode()

        await super().shutdown()
        logger.info("Memory Manager Plugin shutdown complete")

    async def _handle_workspace_event(self, event: WorkspaceEvent):
        """Handle events from the Global Workspace."""
        try:
            if isinstance(event.data, dict):
                event_type = event.data.get("type")

                if event_type == "memory_request":
                    request = MemoryRequest(**event.data)
                    await self._handle_memory_request(request)
                elif event_type == "working_memory_update":
                    update = WorkingMemoryUpdate(**event.data)
                    await self._update_working_memory(update)
                elif event_type == "episodic_event":
                    await self._record_episodic_event(event.data)
                else:
                    # Track all events for episodic memory
                    await self._record_workspace_event(event)

        except Exception as e:
            logger.error(f"Error handling workspace event in Memory Manager: {e}")

    async def _handle_memory_request(self, request: MemoryRequest):
        """Handle memory query requests."""
        start_time = time.time()
        self.memory_stats["total_queries"] += 1

        logger.info(f"🧠 MEMORY: Processing {request.memory_type.value} request")

        try:
            if request.memory_type == MemoryType.WORKING:
                result = await self._query_working_memory(request)
            elif request.memory_type == MemoryType.EPISODIC:
                result = await self._query_episodic_memory(request)
            elif request.memory_type == MemoryType.SEMANTIC:
                result = await self._query_semantic_memory(request)
            else:
                result = await self._query_long_term_memory(request)

            retrieval_time = time.time() - start_time
            self._update_memory_stats(retrieval_time, True)

            # Send result back to workspace
            memory_result = MemoryResult(
                request_id=request.request_id,
                success=True,
                result=result,
                memory_type=request.memory_type,
                retrieval_time=retrieval_time,
            )

            await self.workspace.update(
                data={"type": "memory_result", **memory_result.model_dump()},
                source="memory",
                attention_level=AttentionLevel.MEDIUM,
            )

            self.memory_stats["successful_retrievals"] += 1

        except Exception as e:
            logger.error(f"Memory request failed: {e}")

            retrieval_time = time.time() - start_time
            self._update_memory_stats(retrieval_time, False)

            error_result = MemoryResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
                memory_type=request.memory_type,
                retrieval_time=retrieval_time,
            )

            await self.workspace.update(
                data={"type": "memory_result", **error_result.model_dump()},
                source="memory",
                attention_level=AttentionLevel.MEDIUM,
            )

            self.memory_stats["failed_retrievals"] += 1

    async def _query_working_memory(self, request: MemoryRequest) -> dict[str, Any]:
        """Query working memory."""
        query = request.query.lower() if request.query else ""

        # Direct key lookup
        if query in self.working_memory:
            self.working_memory_access_times[query] = time.time()
            return {"key": query, "value": self.working_memory[query]}

        # Pattern matching
        matches = {}
        for key, value in self.working_memory.items():
            if query in key.lower() or query in str(value).lower():
                matches[key] = value
                self.working_memory_access_times[key] = time.time()

        if matches:
            return {"matches": matches}

        return {"message": "No matches found in working memory"}

    async def _query_episodic_memory(self, request: MemoryRequest) -> dict[str, Any]:
        """Query episodic memory for experiences and events."""
        if not self.episodic_collection:
            return {"error": "Episodic memory not available"}

        try:
            # Query vector database
            query_embedding = await self._generate_embedding(request.query)

            results = self.episodic_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(request.limit or 5, 20),
            )

            episodes = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 0.0
                    )
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )

                    episodes.append(
                        {
                            "content": doc,
                            "similarity": 1.0
                            - distance,  # Convert distance to similarity
                            "metadata": metadata,
                            "episode_id": results["ids"][0][i],
                        }
                    )

            return {"episodes": episodes, "total_found": len(episodes)}

        except Exception as e:
            logger.error(f"Episodic memory query failed: {e}")
            return {"error": f"Query failed: {e}"}

    async def _query_semantic_memory(self, request: MemoryRequest) -> dict[str, Any]:
        """Query semantic memory for knowledge and concepts."""
        if not self.semantic_collection:
            return {"error": "Semantic memory not available"}

        try:
            # Query vector database
            query_embedding = await self._generate_embedding(request.query)

            results = self.semantic_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(request.limit or 10, 50),
            )

            knowledge = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 0.0
                    )
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )

                    knowledge.append(
                        {
                            "content": doc,
                            "relevance": 1.0
                            - distance,  # Convert distance to relevance
                            "metadata": metadata,
                            "knowledge_id": results["ids"][0][i],
                        }
                    )

            return {"knowledge": knowledge, "total_found": len(knowledge)}

        except Exception as e:
            logger.error(f"Semantic memory query failed: {e}")
            return {"error": f"Query failed: {e}"}

    async def _query_long_term_memory(self, request: MemoryRequest) -> dict[str, Any]:
        """Query long-term persistent memory."""
        # Combine semantic and episodic results for comprehensive long-term query
        semantic_result = await self._query_semantic_memory(request)
        episodic_result = await self._query_episodic_memory(request)

        return {
            "semantic": semantic_result,
            "episodic": episodic_result,
            "source": "long_term_memory_composite",
        }

    async def _update_working_memory(self, update: WorkingMemoryUpdate):
        """Update working memory with new information."""
        try:
            if update.operation == "store":
                # Check capacity and evict if necessary
                await self._ensure_working_memory_capacity()

                self.working_memory[update.key] = update.value
                self.working_memory_access_times[update.key] = time.time()

                logger.debug(f"Stored in working memory: {update.key}")

            elif update.operation == "remove":
                if update.key in self.working_memory:
                    del self.working_memory[update.key]
                    if update.key in self.working_memory_access_times:
                        del self.working_memory_access_times[update.key]

            elif update.operation == "clear":
                self.working_memory.clear()
                self.working_memory_access_times.clear()

            self.memory_stats["working_memory_size"] = len(self.working_memory)

        except Exception as e:
            logger.error(f"Failed to update working memory: {e}")

    async def _ensure_working_memory_capacity(self):
        """Ensure working memory doesn't exceed capacity."""
        while len(self.working_memory) >= self.working_memory_capacity:
            # Evict least recently used item
            oldest_key = min(
                self.working_memory_access_times.keys(),
                key=lambda k: self.working_memory_access_times[k],
            )

            del self.working_memory[oldest_key]
            del self.working_memory_access_times[oldest_key]

            logger.debug(f"Evicted from working memory: {oldest_key}")

    async def _record_episodic_event(self, event_data: dict[str, Any]):
        """Record an event in episodic memory."""
        try:
            # Add to current episode buffer
            self.episode_buffer.append(
                {
                    "timestamp": time.time(),
                    "event_type": event_data.get("type", "unknown"),
                    "data": event_data,
                    "source": event_data.get("source", "unknown"),
                }
            )

            # Start new episode if this is the first event
            if not self.current_episode:
                self.current_episode = {
                    "episode_id": f"episode_{int(time.time())}",
                    "start_time": time.time(),
                    "events": [],
                }

            # Check if episode should be stored
            if len(self.episode_buffer) >= self.max_episode_length:
                await self._store_current_episode()

        except Exception as e:
            logger.error(f"Failed to record episodic event: {e}")

    async def _record_workspace_event(self, event: WorkspaceEvent):
        """Record workspace events for episodic memory."""
        # Convert workspace event to episodic event
        episodic_data = {
            "type": "workspace_event",
            "source": event.source,
            "attention_level": event.attention_level.value,
            "data": event.data,
            "timestamp": event.timestamp,
        }

        await self._record_episodic_event(episodic_data)

    async def _store_current_episode(self):
        """Store the current episode in long-term episodic memory."""
        if not self.current_episode or not self.episode_buffer:
            return

        try:
            # Complete the episode
            self.current_episode["events"] = self.episode_buffer.copy()
            self.current_episode["end_time"] = time.time()
            self.current_episode["duration"] = (
                self.current_episode["end_time"] - self.current_episode["start_time"]
            )

            # Generate summary for storage
            episode_summary = self._generate_episode_summary(self.current_episode)

            if self.episodic_collection:
                # Store in vector database
                embedding = await self._generate_embedding(episode_summary)

                self.episodic_collection.add(
                    documents=[episode_summary],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "episode_id": self.current_episode["episode_id"],
                            "start_time": self.current_episode["start_time"],
                            "duration": self.current_episode["duration"],
                            "event_count": len(self.episode_buffer),
                        }
                    ],
                    ids=[self.current_episode["episode_id"]],
                )

                self.memory_stats["episodic_memories_stored"] += 1

            logger.info(
                f"Stored episode: {self.current_episode['episode_id']} "
                f"({len(self.episode_buffer)} events)"
            )

            # Reset for new episode
            self.current_episode = None
            self.episode_buffer = []

        except Exception as e:
            logger.error(f"Failed to store episode: {e}")

    def _generate_episode_summary(self, episode: dict[str, Any]) -> str:
        """Generate a text summary of an episode for embedding."""
        event_types = [event["event_type"] for event in episode["events"]]
        sources = [event["source"] for event in episode["events"]]

        summary = f"""
Episode {episode["episode_id"]} ({episode["duration"]:.1f}s):
- {len(episode["events"])} events
- Event types: {", ".join(set(event_types))}
- Sources: {", ".join(set(sources))}
- Duration: {episode["duration"]:.1f} seconds
"""

        return summary.strip()

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if HAS_GEMINI_EMBEDDINGS:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]
            except Exception as e:
                logger.warning(f"Gemini embedding failed, using fallback: {e}")

        # Fallback: simple hash-based embedding
        return self._generate_simple_embedding(text)

    def _generate_simple_embedding(self, text: str) -> list[float]:
        """Generate simple hash-based embedding."""
        import hashlib
        import math

        # Create multiple hash variations for better distribution
        hashes = [
            hashlib.md5(text.encode()).hexdigest(),
            hashlib.sha1(text.encode()).hexdigest(),
            hashlib.sha256(text.encode()).hexdigest(),
        ]

        # Convert to 384-dimensional vector (compatible with ChromaDB)
        embedding = []
        for hash_str in hashes:
            for i in range(0, min(len(hash_str), 32), 2):
                hex_val = int(hash_str[i : i + 2], 16)
                # Use trigonometric functions for better distribution
                val = math.sin(hex_val * 0.1) * math.cos(hex_val * 0.05)
                embedding.append(val)

        # Ensure exactly 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)

        return embedding[:384]

    def _update_memory_stats(self, retrieval_time: float, success: bool):
        """Update memory statistics."""
        # Update average retrieval time
        alpha = 0.1
        if self.memory_stats["average_retrieval_time"] == 0.0:
            self.memory_stats["average_retrieval_time"] = retrieval_time
        else:
            self.memory_stats["average_retrieval_time"] = (
                alpha * retrieval_time
                + (1 - alpha) * self.memory_stats["average_retrieval_time"]
            )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory statistics."""
        return {
            **self.memory_stats,
            "current_episode_events": len(self.episode_buffer),
            "chroma_available": self.chroma_client is not None,
            "embeddings_available": HAS_GEMINI_EMBEDDINGS,
        }

    async def store_knowledge(
        self, content: str, metadata: dict[str, Any] | None = None
    ):
        """Store knowledge in semantic memory."""
        if not self.semantic_collection:
            logger.warning("Semantic memory not available for knowledge storage")
            return

        try:
            embedding = await self._generate_embedding(content)
            knowledge_id = f"knowledge_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"

            self.semantic_collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                ids=[knowledge_id],
            )

            self.memory_stats["semantic_memories_stored"] += 1
            logger.info(f"Stored knowledge: {knowledge_id}")

        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
