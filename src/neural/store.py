#!/usr/bin/env python3
"""
SQLite-based message store for Neural Atoms and Bonds
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class MessageStore:
    """Event sourcing store for atoms and bonds"""

    def __init__(self, db_path: str = "/tmp/neural_atoms.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database with proper indexing"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            # Add indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)"
            )
            conn.commit()
        logger.info(f"Message store initialized at {self.db_path}")

    def persist(self, event_type: str, payload: dict[str, Any]):
        """Persist event to store"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO events(type, payload, timestamp) VALUES(?, ?, ?)",
                    (event_type, json.dumps(payload), datetime.now().isoformat()),
                )
                conn.commit()
            logger.debug(f"Persisted event: {event_type}")
        except Exception as e:
            logger.error(f"Failed to persist event {event_type}: {e}")
            raise

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Retrieve events by type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT payload, timestamp FROM events WHERE type = ? ORDER BY timestamp",
                (event_type,),
            )
            return [
                {"payload": json.loads(row[0]), "timestamp": row[1]}
                for row in cursor.fetchall()
            ]

    def get_events_by_time_range(
        self, start_time: str, end_time: str
    ) -> list[dict[str, Any]]:
        """Retrieve events within time range"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT type, payload, timestamp FROM events WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (start_time, end_time),
            )
            return [
                {"type": row[0], "payload": json.loads(row[1]), "timestamp": row[2]}
                for row in cursor.fetchall()
            ]

    def count_events(self) -> int:
        """Count total events in store"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]

    def clear_events(self, event_type: str | None = None):
        """Clear events from store"""
        with sqlite3.connect(self.db_path) as conn:
            if event_type:
                conn.execute("DELETE FROM events WHERE type = ?", (event_type,))
                logger.info(f"Cleared events of type: {event_type}")
            else:
                conn.execute("DELETE FROM events")
                logger.info("Cleared all events")
            conn.commit()
