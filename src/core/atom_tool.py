#!/usr/bin/env python3
"""
Atom-Tool Bridge: Makes NeuralAtoms behave like MCP tools

This module provides the thin adapter layer that turns any NeuralAtom into
a callable tool interface, enabling seamless integration with the planning
and execution system.

Key Benefits:
- Lightweight: No heavy MCP servers, just atoms
- Unified: Same storage, retrieval, and lifecycle as other atoms
- Discoverable: Tools are found via attention mechanism
- Serializable: Tools persist and transfer like any other atom
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class ToolSignature(BaseModel):
    """JSON schema for tool parameters."""

    type: str = "object"
    properties: dict[str, dict[str, Any]] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    def add_param(
        self, name: str, param_type: str, description: str, required: bool = False
    ):
        """Add a parameter to the tool signature."""
        self.properties[name] = {"type": param_type, "description": description}
        if required:
            self.required.append(name)


class ToolResult(BaseModel):
    """Standardized tool execution result."""

    success: bool
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AtomTool(BaseModel, ABC):
    """
    Abstract base class that makes any NeuralAtom behave like an MCP tool.

    Contract:
    1. Defines tool metadata (key, name, description, signature)
    2. Implements call() method for execution
    3. Returns structured ToolResult
    4. Can be serialized as atom.value

    Note: Event emission must be handled by the plugin that uses this tool.
    """

    key: str
    name: str
    description: str
    signature: ToolSignature
    category: str = "general"
    version: str = "1.0"

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def call(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Parameters matching the tool signature

        Returns:
            ToolResult with success/failure and output data
        """
        ...

    def validate_params(self, params: dict[str, Any]) -> bool:
        """
        Validate parameters against tool signature.

        Args:
            params: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required parameters
            for required_param in self.signature.required:
                if required_param not in params:
                    logger.error(f"Missing required parameter: {required_param}")
                    return False

            # Basic type checking could be added here
            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False

    def to_atom_value(self) -> dict[str, Any]:
        """
        Convert tool to NeuralAtom value format.

        Returns:
            Dictionary suitable for atom.value
        """
        return {
            "tool_type": "atom_tool",
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "signature": self.signature.model_dump(),
            "category": self.category,
            "version": self.version,
            "callable": True,
        }

    @classmethod
    @abstractmethod
    def from_atom_value(cls, value: dict[str, Any]) -> "AtomTool":
        """
        Reconstruct tool from NeuralAtom value.

        Args:
            value: The atom.value dictionary

        Returns:
            AtomTool instance (requires concrete implementation)
        """
        pass


class ShellTool(AtomTool):
    """
    Concrete example: Shell command execution tool.
    """

    def __init__(self):
        signature = ToolSignature()
        signature.add_param(
            "command", "string", "Shell command to execute", required=True
        )
        signature.add_param("timeout", "integer", "Timeout in seconds", required=False)

        super().__init__(
            key="shell_executor",
            name="Shell Command Executor",
            description="Execute shell commands and return output",
            signature=signature,
            category="system",
        )

    async def call(self, **kwargs) -> ToolResult:
        """Execute shell command."""
        import asyncio

        command = kwargs.get("command")
        timeout = kwargs.get("timeout", 30)

        if not self.validate_params(kwargs):
            return ToolResult(success=False, error="Invalid parameters")

        if not command:
            return ToolResult(success=False, error="Command cannot be empty or None")

        try:
            logger.info(f"Executing shell command: {command}")

            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return ToolResult(
                success=process.returncode == 0,
                result={
                    "stdout": stdout.decode("utf-8"),
                    "stderr": stderr.decode("utf-8"),
                    "return_code": process.returncode,
                },
                metadata={"command": command, "timeout": timeout},
            )

        except TimeoutError:
            return ToolResult(
                success=False, error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Execution failed: {e!s}")


class PythonMathTool(AtomTool):
    """
    Concrete example: Python math evaluation tool.
    """

    def __init__(self):
        signature = ToolSignature()
        signature.add_param(
            "expression", "string", "Python mathematical expression", required=True
        )
        signature.add_param(
            "imports", "array", "Additional modules to import", required=False
        )

        super().__init__(
            key="python_math",
            name="Python Math Evaluator",
            description="Evaluate mathematical expressions using Python",
            signature=signature,
            category="math",
        )

    async def call(self, **kwargs) -> ToolResult:
        """Evaluate Python math expression."""
        import math

        import numpy as np

        expression = kwargs.get("expression")
        imports = kwargs.get("imports", [])

        if not self.validate_params(kwargs):
            return ToolResult(success=False, error="Invalid parameters")

        if not expression:
            return ToolResult(success=False, error="Expression cannot be empty or None")

        try:
            # Safe execution environment
            safe_globals = {
                "__builtins__": {},
                "math": math,
                "np": np,
                "pi": math.pi,
                "e": math.e,
            }

            # Add requested imports (with basic safety)
            for module_name in imports:
                if module_name in ["sympy", "scipy", "matplotlib"]:
                    try:
                        module = __import__(module_name)
                        safe_globals[module_name] = module
                    except ImportError:
                        pass

            logger.info(f"Evaluating math expression: {expression}")
            result = eval(expression, safe_globals, {})

            return ToolResult(
                success=True,
                result={
                    "value": result,
                    "type": str(type(result).__name__),
                    "expression": expression,
                },
                metadata={"imports_used": list(safe_globals.keys())},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Math evaluation failed: {e!s}",
                metadata={"expression": expression},
            )


# Tool Registry for dynamic loading
BUILT_IN_TOOLS = {"shell_executor": ShellTool, "python_math": PythonMathTool}


def create_tool(tool_key: str) -> AtomTool | None:
    """
    Factory function to create tools by key.

    Args:
        tool_key: The tool identifier

    Returns:
        AtomTool instance or None if not found
    """
    tool_class = BUILT_IN_TOOLS.get(tool_key)
    if tool_class:
        return tool_class()
    return None


def register_tool_atoms(store, owner_plugin: str = "atom_tools"):
    """
    Register all built-in tools as NeuralAtoms.

    Args:
        store: NeuralStore instance
        owner_plugin: Plugin that owns these tools
    """
    import numpy as np

    from src.core.neural_atom import create_memory_atom

    for tool_key, tool_class in BUILT_IN_TOOLS.items():
        tool_instance = tool_class()

        # Create atom with tool data
        atom = create_memory_atom(
            key=f"tool_{tool_key}",
            content=tool_instance.to_atom_value(),
            hierarchy_path="tool",
            vector=np.random.rand(EMBEDDING_DIM).astype(
                np.float32
            ),  # Would use real embedding
        )

        # Register with store
        store.register(atom)
        logger.info(f"Registered tool atom: {tool_key}")


if __name__ == "__main__":
    print("AtomTool Bridge Implementation")
    print("=" * 50)

    # Demo the concept
    shell_tool = ShellTool()
    print(f"Tool: {shell_tool.name}")
    print(f"Key: {shell_tool.key}")
    print(f"Signature: {shell_tool.signature.model_dump()}")
    print("\nAtom Value:")
    print(shell_tool.to_atom_value())
