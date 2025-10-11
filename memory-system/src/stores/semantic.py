"""SQLite-backed semantic memory store."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import ConsolidationBatch, Memory, MemoryType


class SemanticStore:
    """Persistence for consolidated semantic memories."""

    def __init__(self, db_path: str = "storage/semantic/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    tags TEXT,
                    source TEXT,
                    importance REAL,
                    confidence REAL,
                    evidence_ids TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_access TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consolidation_batches (
                    id TEXT PRIMARY KEY,
                    episodic_ids TEXT NOT NULL,
                    semantic_id TEXT,
                    summary TEXT,
                    confidence REAL,
                    evidence_count INTEGER,
                    ts TEXT,
                    FOREIGN KEY(semantic_id) REFERENCES memories(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT,
                    attributes TEXT,
                    first_seen TEXT,
                    last_seen TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    subject_id TEXT,
                    predicate TEXT,
                    object_id TEXT,
                    confidence REAL,
                    source TEXT,
                    created_at TEXT,
                    FOREIGN KEY(subject_id) REFERENCES entities(id),
                    FOREIGN KEY(object_id) REFERENCES entities(id)
                )
                """
            )
            conn.commit()

    def add(self, memory: Memory) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, text, embedding, tags, source, importance, confidence,
                 evidence_ids, first_seen, last_seen, access_count, last_access)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.text,
                    json.dumps(memory.embeddings) if memory.embeddings else None,
                    json.dumps(memory.tags),
                    memory.source,
                    memory.importance,
                    memory.confidence,
                    json.dumps(memory.meta.get("consolidated_from", [])),
                    memory.ts.isoformat(),
                    memory.last_access.isoformat(),
                    memory.access_count,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row:
            return self._row_to_memory(row)
        return None

    def search(self, query: str, k: int = 10) -> List[Memory]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE text LIKE ?
                ORDER BY importance DESC, last_seen DESC
                LIMIT ?
                """,
                (f"%{query}%", k),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def record_consolidation(self, batch: ConsolidationBatch) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO consolidation_batches
                (id, episodic_ids, semantic_id, summary, confidence, evidence_count, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch.id,
                    json.dumps(batch.episodic_ids),
                    batch.semantic_id,
                    batch.summary,
                    batch.confidence,
                    batch.evidence_count,
                    batch.ts.isoformat(),
                ),
            )
            conn.commit()

    def get_consolidation_metrics(self) -> Dict[str, Any]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            total_batches = (
                conn.execute("SELECT COUNT(*) FROM consolidation_batches").fetchone()[0]
            )
            avg_confidence = (
                conn.execute("SELECT AVG(confidence) FROM consolidation_batches").fetchone()[0]
                or 0.0
            )
        return {
            "total_semantic_memories": total_memories,
            "total_consolidation_batches": total_batches,
            "average_consolidation_confidence": round(avg_confidence, 3),
        }

    def count(self) -> int:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row[0],
            text=row[1],
            embeddings=json.loads(row[2]) if row[2] else None,
            tags=json.loads(row[3]) if row[3] else [],
            source=row[4],
            importance=row[5],
            confidence=row[6],
            kind=MemoryType.SEMANTIC,
            ts=datetime.fromisoformat(row[8]),
            last_access=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
            access_count=row[10] or 0,
            meta={"evidence_ids": json.loads(row[7]) if row[7] else []},
        )


semantic_store = SemanticStore()
