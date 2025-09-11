"""
Simplified Mangle Reasoning Ability for testing dependencies.
"""

from typing import Any


class MangleReasoningAbility:
    """Simplified version for dependency testing."""

    def __init__(self, workspace_root: str = "."):
        """Initialize with minimal dependencies for testing."""
        self.workspace_root = workspace_root

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return empty tool definitions for testing."""
        return []

    def enhance_user_input(self, user_input: str) -> dict[str, Any]:
        """Return basic enhancement for testing."""
        return {"original_input": user_input, "mangle_context": {"test": "working"}}
