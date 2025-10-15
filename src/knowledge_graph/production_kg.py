from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class Atom:
    """Atomic unit of knowledge persisted in the production graph."""

    id: str
    type: str
    content: str
    embedding: Optional[Sequence[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        embedding = payload.get("embedding")
        if isinstance(embedding, np.ndarray):  # pragma: no cover - defensive
            payload["embedding"] = embedding.tolist()
        return payload


@dataclass(slots=True)
class Bond:
    """Relationship between atoms inside the graph."""

    id: str
    source_id: str
    target_id: str
    bond_type: str
    strength: float
    metadata: Dict[str, Any]


class ProductionKnowledgeGraph:
    """ChromaDB-backed knowledge graph with semantic capabilities."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        *,
        client: Optional[ClientAPI] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.client = client or chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")

        self.atoms_collection = self.client.get_or_create_collection(
            name="atoms",
            metadata={"description": "Knowledge graph atoms with embeddings"},
        )
        self.bonds_collection = self.client.get_or_create_collection(
            name="bonds",
            metadata={"description": "Knowledge graph bonds"},
        )
        self.logger.info("ProductionKnowledgeGraph initialised at %s", persist_directory)

    @staticmethod
    def _to_vector(embedding: Sequence[float] | np.ndarray) -> List[float]:
        if hasattr(embedding, "tolist"):
            return list(embedding.tolist())
        return list(embedding)

    async def create_atom(
        self,
        atom_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        raw_embedding = self.embedding_model.encode([content_str])[0]
        embedding = self._to_vector(raw_embedding)

        duplicate_atom_id = await self._find_semantic_duplicate(content_str, embedding)
        if duplicate_atom_id:
            self.logger.info("Semantic duplicate detected: %s", duplicate_atom_id)
            existing = await self.get_atom(duplicate_atom_id)
            if existing:
                return existing

        atom_id = f"atom_{uuid.uuid4().hex[:12]}"
        atom_metadata = dict(metadata or {})
        atom_metadata.setdefault("confidence", 1.0)
        atom_metadata.setdefault("source", "system")
        atom_metadata.setdefault("tags", [])
        atom_metadata.setdefault("created_at", datetime.now(UTC).isoformat())

        atom = Atom(
            id=atom_id,
            type=atom_type,
            content=content_str,
            embedding=embedding,
            metadata=atom_metadata,
        )

        self.atoms_collection.add(
            ids=[atom_id],
            embeddings=[embedding],
            documents=[content_str],
            metadatas=[{
                "type": atom_type,
                "metadata": json.dumps(atom_metadata),
                "created_at": atom_metadata["created_at"],
            }],
        )

        await self._create_similarity_bonds(atom_id, embedding)
        self.logger.info("Created atom %s of type %s", atom_id, atom_type)
        return atom.to_dict()

    async def _find_semantic_duplicate(
        self,
        content: str,
        embedding: Sequence[float],
        similarity_threshold: float = 0.95,
    ) -> Optional[str]:
        try:
            results = self.atoms_collection.query(
                query_embeddings=[embedding],
                n_results=5,
                include=["distances", "ids"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Semantic duplicate search failed: %s", exc)
            return None

        ids = results.get("ids") or []
        distances = results.get("distances") or []
        if not ids or not ids[0]:
            return None

        for index, distance in enumerate(distances[0]):
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                return ids[0][index]
        return None

    async def _create_similarity_bonds(
        self,
        atom_id: str,
        embedding: Sequence[float],
        similarity_threshold: float = 0.3,
    ) -> None:
        try:
            results = self.atoms_collection.query(
                query_embeddings=[embedding],
                n_results=10,
                include=["distances", "ids"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Similarity bond creation failed: %s", exc)
            return

        ids = results.get("ids") or []
        distances = results.get("distances") or []
        if not ids or not ids[0]:
            return

        for index, target_id in enumerate(ids[0]):
            if target_id == atom_id:
                continue
            similarity = 1 - distances[0][index]
            if similarity < similarity_threshold:
                continue
            await self.create_bond(
                source_id=atom_id,
                target_id=target_id,
                bond_type="similar",
                strength=similarity,
                metadata={"auto_generated": True, "similarity_score": similarity},
            )

    async def create_bond(
        self,
        source_id: str,
        target_id: str,
        bond_type: str,
        *,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_atom = await self.get_atom(source_id)
        target_atom = await self.get_atom(target_id)
        if not source_atom or not target_atom:
            raise ValueError(f"Source or target atom missing: {source_id} -> {target_id}")

        bond_id = f"bond_{uuid.uuid4().hex[:12]}"
        bond_metadata = dict(metadata or {})
        bond_metadata.setdefault("created_at", datetime.now(UTC).isoformat())
        bond_metadata.setdefault("source_type", source_atom["type"])
        bond_metadata.setdefault("target_type", target_atom["type"])

        bond = Bond(
            id=bond_id,
            source_id=source_id,
            target_id=target_id,
            bond_type=bond_type,
            strength=strength,
            metadata=bond_metadata,
        )

        bond_content = f"{bond_type}:{source_id}->{target_id}"
        bond_embedding = self._to_vector(self.embedding_model.encode([bond_content])[0])
        self.bonds_collection.add(
            ids=[bond_id],
            embeddings=[bond_embedding],
            documents=[bond_content],
            metadatas=[{
                "source_id": source_id,
                "target_id": target_id,
                "bond_type": bond_type,
                "strength": strength,
                "metadata": json.dumps(bond_metadata),
            }],
        )

        self.logger.info("Created bond %s: %s -[%s]-> %s", bond_id, source_id, bond_type, target_id)
        return asdict(bond)

    async def get_atom(self, atom_id: str) -> Optional[Dict[str, Any]]:
        results = self.atoms_collection.get(
            ids=[atom_id],
            include=["embeddings", "metadatas", "documents"],
        )
        ids = results.get("ids") or []
        if not ids:
            return None

        metadata_raw = (results.get("metadatas") or [{}])[0]
        stored_metadata = metadata_raw.get("metadata")
        if isinstance(stored_metadata, str):
            metadata = json.loads(stored_metadata)
        else:
            metadata = stored_metadata or {}

        embedding = None
        if results.get("embeddings"):
            embedding = results["embeddings"][0]

        return {
            "id": atom_id,
            "type": metadata_raw.get("type", "unknown"),
            "content": (results.get("documents") or [""])[0],
            "embedding": embedding,
            "metadata": metadata,
        }

    async def semantic_search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self._to_vector(self.embedding_model.encode([query])[0])
        where_clause: Dict[str, Any] = {}
        if filters:
            if "type" in filters:
                where_clause["type"] = filters["type"]
            if "tags" in filters:
                where_clause["metadata.tags"] = {"$contains": filters["tags"]}
            if "confidence" in filters:
                where_clause["metadata.confidence"] = {"$gte": filters["confidence"]}
        where = where_clause or None

        results = self.atoms_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["ids", "distances", "metadatas", "documents"],
        )
        ids = results.get("ids") or []
        if not ids or not ids[0]:
            return []

        atoms: List[Dict[str, Any]] = []
        distances = results.get("distances") or [[]]
        for index, atom_id in enumerate(ids[0]):
            metadata_raw = results["metadatas"][0][index]
            stored_metadata = metadata_raw.get("metadata")
            if isinstance(stored_metadata, str):
                metadata = json.loads(stored_metadata)
            else:
                metadata = stored_metadata or {}

            similarity = 1 - distances[0][index]
            related = await self._get_related_atoms(atom_id)
            atoms.append(
                {
                    "id": atom_id,
                    "type": metadata_raw.get("type", "unknown"),
                    "content": results["documents"][0][index],
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "related_atoms": related,
                }
            )
        return atoms

    async def _get_related_atoms(
        self,
        atom_id: str,
        bond_types: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        types = list(bond_types) if bond_types else ["enables", "requires", "similar", "derives"]
        related: List[Dict[str, Any]] = []

        source_bonds = self.bonds_collection.get(
            where={"source_id": atom_id, "bond_type": {"$in": types}},
            include=["ids", "metadatas"],
        )
        target_bonds = self.bonds_collection.get(
            where={"target_id": atom_id, "bond_type": {"$in": types}},
            include=["ids", "metadatas"],
        )

        for bonds, relationship_prefix in ((source_bonds, "source"), (target_bonds, "target")):
            ids = bonds.get("ids") or []
            metadatas = bonds.get("metadatas") or []
            for index, _ in enumerate(ids):
                metadata_raw = metadatas[index]
                stored_metadata = metadata_raw.get("metadata")
                if isinstance(stored_metadata, str):
                    bond_metadata = json.loads(stored_metadata)
                else:
                    bond_metadata = stored_metadata or {}
                target_id = (
                    metadata_raw["target_id"]
                    if relationship_prefix == "source"
                    else metadata_raw["source_id"]
                )
                related_atom = await self.get_atom(target_id)
                if not related_atom:
                    continue
                related.append(
                    {
                        "atom": related_atom,
                        "relationship": f"{relationship_prefix}_{metadata_raw['bond_type']}",
                        "strength": metadata_raw.get("strength", 0.0),
                        "bond_metadata": bond_metadata,
                    }
                )
        return related

    async def traverse_graph(
        self,
        start_atom_id: str,
        *,
        max_hops: int = 3,
        relationship_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        visited: set[str] = set()
        hops: Dict[int, List[Dict[str, Any]]] = {}
        all_atoms: Dict[str, Dict[str, Any]] = {}

        async def traverse(current_id: str, depth: int) -> None:
            if depth > max_hops or current_id in visited:
                return
            visited.add(current_id)
            atom = await self.get_atom(current_id)
            if not atom:
                return
            all_atoms[current_id] = atom
            related = await self._get_related_atoms(current_id, relationship_types)
            hops.setdefault(depth, [])
            for relation in related:
                hops[depth].append(
                    {
                        "from": current_id,
                        "to": relation["atom"]["id"],
                        "relationship": relation["relationship"],
                        "strength": relation["strength"],
                    }
                )
                await traverse(relation["atom"]["id"], depth + 1)

        await traverse(start_atom_id, 1)
        return {"start_atom": await self.get_atom(start_atom_id), "hops": hops, "all_atoms": all_atoms}

    async def backup(self, backup_path: str) -> bool:
        try:
            atoms = self.atoms_collection.get(include=["ids", "embeddings", "documents", "metadatas"])
            bonds = self.bonds_collection.get(include=["ids", "embeddings", "documents", "metadatas"])
            payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "atoms": atoms,
                "bonds": bonds,
                "atoms_count": len(atoms.get("ids") or []),
                "bonds_count": len(bonds.get("ids") or []),
            }
            with open(backup_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.logger.info("Knowledge graph backup created at %s", backup_path)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Backup failed: %s", exc)
            return False

    async def restore(self, backup_path: str) -> bool:
        try:
            with open(backup_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            try:
                self.client.delete_collection("atoms")
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                self.client.delete_collection("bonds")
            except Exception:  # pragma: no cover - defensive
                pass

            self.atoms_collection = self.client.get_or_create_collection(name="atoms")
            self.bonds_collection = self.client.get_or_create_collection(name="bonds")

            atoms = payload.get("atoms") or {}
            if atoms.get("ids"):
                self.atoms_collection.add(
                    ids=atoms["ids"],
                    embeddings=atoms.get("embeddings"),
                    documents=atoms.get("documents"),
                    metadatas=atoms.get("metadatas"),
                )

            bonds = payload.get("bonds") or {}
            if bonds.get("ids"):
                self.bonds_collection.add(
                    ids=bonds["ids"],
                    embeddings=bonds.get("embeddings"),
                    documents=bonds.get("documents"),
                    metadatas=bonds.get("metadatas"),
                )

            self.logger.info("Knowledge graph restored from %s", backup_path)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Restore failed: %s", exc)
            return False


async def create_knowledge_graph(persist_directory: str = "./data/chroma_db") -> ProductionKnowledgeGraph:
    """Factory helper mirroring existing runtime patterns."""

    return ProductionKnowledgeGraph(persist_directory)
