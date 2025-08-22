"""
Neural Atom Framework for Super-Alita
====================================

This module provides a neural-symbolic knowledge representation layer
that integrates with the existing EventBus system to create persistent,
deterministic knowledge atoms and bonds.

Key Features:
- Deterministic UUIDv5 IDs for reproducibility
- Event sourcing with SQLite persistence
- MCP-style ability contracts
- Prometheus metrics integration
- Bridging with existing EventBus
"""

from .atom import Atom
from .bond import Bond
from .mcp_server import MCPServer
from .store import MessageStore

__all__ = [
    "Atom",
    "Bond",
    "MessageStore",
    "MCPServer",
]
