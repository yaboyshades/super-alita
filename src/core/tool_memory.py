"""
Enhanced Memory Atom System for Dynamic Tools
Stores full provenance and enables tool library functionality
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DynamicToolAtom:
    """Memory atom for storing dynamic tool information"""

    memory_id: str
    tool_name: str
    code: str
    params: dict[str, Any]
    result: Any
    error: str | None
    invoked_by: str
    created_at: str
    executed_in_context: str
    execution_time: float = 0.0
    usage_count: int = 0
    success_rate: float = 1.0
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DynamicToolMemoryManager:
    """Enhanced memory manager for dynamic tools"""

    def __init__(self, store=None):
        self.store = store
        self.tool_atoms: dict[str, DynamicToolAtom] = {}
        self.tool_library: dict[str, str] = {}  # tool_name -> memory_id mapping

    def store_dynamic_tool_atom(
        self,
        tool_name: str,
        code: str,
        params: dict[str, Any],
        result: Any,
        error: str | None,
        user_id: str,
        context_id: str,
        execution_time: float = 0.0,
        tags: list[str] = None,
    ) -> str:
        """Store dynamic tool execution with full provenance"""

        memory_id = uuid.uuid4().hex

        # Create the tool atom
        tool_atom = DynamicToolAtom(
            memory_id=memory_id,
            tool_name=tool_name,
            code=code,
            params=params,
            result=result,
            error=error,
            invoked_by=user_id,
            created_at=datetime.now().isoformat(),
            executed_in_context=context_id,
            execution_time=execution_time,
            tags=tags or [],
        )

        # Store in local cache
        self.tool_atoms[memory_id] = tool_atom

        # Update tool library mapping
        if not error:  # Only add successful tools to library
            self.tool_library[tool_name] = memory_id

        # Store in persistent memory if available
        if self.store:
            try:
                atom_dict = asdict(tool_atom)
                atom_dict["type"] = "dynamic_tool"

                self.store.upsert(
                    memory_id=memory_id,
                    content=atom_dict,
                    hierarchy_path=["dynamic_tools", tool_name],
                    owner_plugin="planner",
                )

                logger.info(f"Stored dynamic tool atom: {tool_name} ({memory_id})")

            except Exception as e:
                logger.error(f"Failed to store tool atom in persistent memory: {e}")

        return memory_id

    def get_tool_from_library(self, tool_name: str) -> DynamicToolAtom | None:
        """Retrieve tool from library by name"""
        if tool_name in self.tool_library:
            memory_id = self.tool_library[tool_name]
            return self.tool_atoms.get(memory_id)

        # Try to load from persistent storage
        if self.store:
            try:
                memories = self.store.query(
                    query=f"tool_name:{tool_name}",
                    hierarchy_path=["dynamic_tools"],
                    limit=1,
                )

                if memories:
                    memory_data = memories[0]
                    if memory_data.get("type") == "dynamic_tool":
                        # Reconstruct the atom
                        atom_data = memory_data.copy()
                        atom_data.pop("type", None)
                        tool_atom = DynamicToolAtom(**atom_data)

                        # Cache it
                        self.tool_atoms[tool_atom.memory_id] = tool_atom
                        self.tool_library[tool_name] = tool_atom.memory_id

                        return tool_atom

            except Exception as e:
                logger.error(f"Failed to load tool from persistent memory: {e}")

        return None

    def update_tool_usage(self, tool_name: str, success: bool = True) -> None:
        """Update tool usage statistics"""
        tool_atom = self.get_tool_from_library(tool_name)
        if tool_atom:
            tool_atom.usage_count += 1

            # Update success rate
            if success:
                tool_atom.success_rate = (
                    tool_atom.success_rate * (tool_atom.usage_count - 1) + 1
                ) / tool_atom.usage_count
            else:
                tool_atom.success_rate = (
                    tool_atom.success_rate * (tool_atom.usage_count - 1)
                ) / tool_atom.usage_count

            # Update in persistent storage
            if self.store:
                try:
                    atom_dict = asdict(tool_atom)
                    atom_dict["type"] = "dynamic_tool"

                    self.store.upsert(
                        memory_id=tool_atom.memory_id,
                        content=atom_dict,
                        hierarchy_path=["dynamic_tools", tool_name],
                        owner_plugin="planner",
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to update tool usage in persistent memory: {e}"
                    )

    def search_similar_tools(
        self, description: str, limit: int = 5
    ) -> list[DynamicToolAtom]:
        """Search for similar tools based on description"""
        if not self.store:
            return []

        try:
            # Search in persistent memory
            memories = self.store.query(
                query=description, hierarchy_path=["dynamic_tools"], limit=limit
            )

            similar_tools = []
            for memory_data in memories:
                if memory_data.get("type") == "dynamic_tool":
                    atom_data = memory_data.copy()
                    atom_data.pop("type", None)
                    similar_tools.append(DynamicToolAtom(**atom_data))

            return similar_tools

        except Exception as e:
            logger.error(f"Failed to search similar tools: {e}")
            return []

    def get_popular_tools(self, limit: int = 10) -> list[DynamicToolAtom]:
        """Get most popular tools by usage"""
        tools = list(self.tool_atoms.values())
        tools.sort(key=lambda x: x.usage_count, reverse=True)
        return tools[:limit]

    def get_tools_by_user(self, user_id: str) -> list[DynamicToolAtom]:
        """Get tools created by a specific user"""
        return [atom for atom in self.tool_atoms.values() if atom.invoked_by == user_id]

    def get_tools_by_context(self, context_id: str) -> list[DynamicToolAtom]:
        """Get tools created in a specific context"""
        return [
            atom
            for atom in self.tool_atoms.values()
            if atom.executed_in_context == context_id
        ]

    def promote_tool_to_library(self, memory_id: str, tool_name: str = None) -> bool:
        """Promote a tool execution to the permanent library"""
        if memory_id in self.tool_atoms:
            tool_atom = self.tool_atoms[memory_id]
            library_name = tool_name or tool_atom.tool_name

            # Add to library
            self.tool_library[library_name] = memory_id

            # Add library tag
            if "library" not in tool_atom.tags:
                tool_atom.tags.append("library")

            logger.info(f"Promoted tool to library: {library_name}")
            return True

        return False

    def export_tool_library(self) -> dict[str, dict[str, Any]]:
        """Export tool library for backup or sharing"""
        library_export = {}

        for tool_name, memory_id in self.tool_library.items():
            if memory_id in self.tool_atoms:
                tool_atom = self.tool_atoms[memory_id]
                library_export[tool_name] = {
                    "code": tool_atom.code,
                    "params": tool_atom.params,
                    "description": f"Dynamic tool: {tool_name}",
                    "created_at": tool_atom.created_at,
                    "author": tool_atom.invoked_by,
                    "usage_count": tool_atom.usage_count,
                    "success_rate": tool_atom.success_rate,
                    "tags": tool_atom.tags,
                }

        return library_export

    def import_tool_library(
        self, library_data: dict[str, dict[str, Any]], user_id: str
    ) -> int:
        """Import tool library from export data"""
        imported_count = 0

        for tool_name, tool_data in library_data.items():
            try:
                memory_id = self.store_dynamic_tool_atom(
                    tool_name=tool_name,
                    code=tool_data["code"],
                    params=tool_data.get("params", {}),
                    result="Imported from library",
                    error=None,
                    user_id=user_id,
                    context_id="library_import",
                    tags=tool_data.get("tags", []) + ["imported"],
                )

                # Promote to library
                self.promote_tool_to_library(memory_id, tool_name)
                imported_count += 1

            except Exception as e:
                logger.error(f"Failed to import tool {tool_name}: {e}")

        logger.info(f"Imported {imported_count} tools to library")
        return imported_count


# Global instance
_memory_manager = None


def get_memory_manager(store=None) -> DynamicToolMemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = DynamicToolMemoryManager(store)
    elif store and not _memory_manager.store:
        _memory_manager.store = store
    return _memory_manager


def format_tool_response(
    result: Any,
    code: str,
    params: dict[str, Any],
    error: str | None = None,
    tool_name: str = "dynamic_tool",
) -> dict[str, Any]:
    """Format tool execution response for user display"""
    response = {
        "Tool Name": tool_name,
        "Result": result if not error else "Error occurred",
        "Parameters": params,
        "Tool Code": code,
        "Status": "Success" if not error else "Failed",
    }

    if error:
        response["Error"] = error
        response["Error Details"] = str(error)

    # Format result nicely if it's a complex object
    if result and hasattr(result, "__dict__"):
        response["Result Details"] = vars(result)

    return response


def prompt_save_tool(user_id: str, tool_name: str, success_rate: float = 1.0) -> str:
    """Generate prompt to ask user about saving tool to library"""
    if success_rate >= 0.8:
        recommendation = "This tool worked well and could be useful in the future."
    elif success_rate >= 0.5:
        recommendation = "This tool had mixed results but might be worth keeping."
    else:
        recommendation = (
            "This tool had issues but you might want to keep it for reference."
        )

    return f"""
🔧 Tool Created: '{tool_name}'
{recommendation}

Would you like to save this tool to your personal library for instant future use?
- Type 'yes' to save to library
- Type 'no' to skip
- Type 'rename:<new_name>' to save with a different name

Saving to library allows you to reuse this exact tool with a simple command.
    """.strip()
