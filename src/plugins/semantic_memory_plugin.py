# src/plugins/semantic_memory_plugin.py
"""
Plugin for long-term, hierarchical semantic memory.

- Embeds text via Gemini-embedding-004
- Stores memories as NeuralAtoms in NeuralStore
- Persists vectors in ChromaDB for durability
- Retrieves via attention/cosine similarity
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

from src.core.plugin_interface import PluginInterface
from src.core.neural_atom import NeuralStore, NeuralAtom
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class SemanticMemoryPlugin(PluginInterface):
    """
    Long-term memory plugin.

    - Embeds text with Gemini
    - Dual-write to ChromaDB and NeuralStore
    - Attention-based retrieval
    """

    def __init__(self):
        self._store: Optional[NeuralStore] = None
        self._event_bus: Optional[EventBus] = None
        self._config: Dict[str, Any] = {}
        self._chroma_client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._tasks = []  # Track background tasks for cleanup

    @property
    def name(self) -> str:
        return "semantic_memory"

    async def setup(self, event_bus: EventBus, store: NeuralStore, config: Dict[str, Any]) -> None:
        self._event_bus = event_bus
        self._store = store
        self._config = config
        
        # Configure Gemini API if key is provided
        import os
        gemini_api_key = self._config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            logger.info("Gemini API configured for real embeddings")
        else:
            logger.warning("No Gemini API key found - will use fallback embeddings")
            
        logger.info("SemanticMemoryPlugin setup complete.")

    async def start(self) -> None:
        """Connects to ChromaDB and creates collection."""
        await super().start()  # This sets is_running = True
        
        db_path = self._config.get("db_path", "./data/chroma_db")
        collection_name = self._config.get("collection_name", "alita_memory")
        
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=db_path, settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Connected to ChromaDB collection '{collection_name}' at '{db_path}'")
        except Exception as e:
            logger.critical(f"Failed to connect to ChromaDB: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop the plugin gracefully."""
        await super().stop()  # This sets is_running = False and calls shutdown
    
    async def shutdown(self) -> None:
        """Graceful shutdown. ChromaDB client handles persistence."""
        logger.info("SemanticMemoryPlugin shutting down.")

    # ---------- Public API ----------
    async def embed_text(self, texts: List[str]) -> List[np.ndarray]:
        """Async wrapper for Gemini embedding-004."""
        import asyncio
        
        model = self._config.get("embedding_model", "models/text-embedding-004")
        try:
            loop = asyncio.get_running_loop()
            
            # Handle single text or batch
            if len(texts) == 1:
                response = await loop.run_in_executor(
                    None,
                    lambda: genai.embed_content(
                        model=model,
                        content=texts[0],
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                embedding = np.array(response["embedding"], dtype=np.float32)
                return [embedding]
            else:
                # Batch processing
                response = await loop.run_in_executor(
                    None,
                    lambda: genai.embed_content(
                        model=model,
                        content=texts,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                return [np.array(vec, dtype=np.float32) for vec in response["embedding"]]
                
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            # Fallback to deterministic embeddings for development
            embeddings = []
            for text in texts:
                text_hash = hash(text)
                np.random.seed(abs(text_hash) % 2**32)
                embedding = np.random.normal(0, 1, 1024).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return embeddings

    async def upsert(
        self,
        content: Dict[str, Any],
        hierarchy_path: List[str],
        owner: str = "unknown",
    ) -> str:
        """
        Insert or update a memory.

        Dual-write:
        - ChromaDB for persistence
        - NeuralStore for reactive access
        """
        memory_id = content.get("id") or f"mem_{uuid.uuid4().hex[:8]}"
        text_repr = "::".join(hierarchy_path) + " " + str(content)

        embeddings = await self.embed_text([text_repr])
        vector = embeddings[0]

        # 1. Register as NeuralAtom (live graph)
        atom = NeuralAtom(
            key=memory_id,
            default_value=content,
            vector=vector,
            birth_event="memory_upsert",
            lineage_metadata={"owner": owner, "hierarchy": hierarchy_path},
        )
        self._store.register_with_lineage(
            atom,
            parents=[],
            birth_event="memory_upsert",
            lineage_metadata={"owner": owner, "hierarchy": hierarchy_path},
        )

        # 2. Persist in ChromaDB (disk)
        metadata = {
            "hierarchy_path": "::".join(hierarchy_path),
            "owner": owner,
            **content,
        }
        self._collection.upsert(
            ids=[memory_id],
            embeddings=[vector.tolist()],
            metadatas=[metadata],
        )

        # Emit event (using basic emit for now)
        await self._event_bus.emit(
            "memory_upsert",
            source_plugin=self.name,
            memory_id=memory_id,
            content=content,
            hierarchy_path=hierarchy_path,
            operation="INSERT",
            owner=owner,
        )
        logger.debug(f"Upserted memory: {memory_id}")
        return memory_id

    async def query(
        self,
        query_text: str,
        hierarchy_filter: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search via attention over NeuralStore.

        Uses cosine similarity on vectors.
        """
        query_vec = (await self.embed_text([query_text]))[0]
        matches = await self._store.attention(query_vec, top_k=top_k)

        results = []
        for key, score in matches:
            atom = self._store.get(key)
            if atom is None:
                continue
            results.append({"content": atom.value, "score": float(score)})
        return results
