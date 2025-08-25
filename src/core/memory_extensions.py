"""
Memory extensions for TextualMemoryAtom.
"""

import logging
from typing import Any

from src.core.neural_atom import TextualMemoryAtom

logger = logging.getLogger(__name__)

# Extend TextualMemoryAtom with memory operations
# Original TextualMemoryAtom is designed for retrieval, but we need to add storage functionality
# This monkey patches the TextualMemoryAtom class with store and retrieve methods


async def store(self, data: dict[str, Any]) -> bool:
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


async def retrieve(self, limit: int = 50) -> list[dict[str, Any]]:
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


# Monkey patch the TextualMemoryAtom class
TextualMemoryAtom.store = store
TextualMemoryAtom.retrieve = retrieve
