"""Shared result type helpers for MCP integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(slots=True)
class Result(Generic[T, E]):
    """Lightweight success/error container used across MCP flows."""

    ok: bool
    value: T | None = None
    error: E | None = None

    @classmethod
    def Ok(cls, value: T) -> Result[T, E]:
        return cls(ok=True, value=value)

    @classmethod
    def Err(cls, error: E) -> Result[T, E]:
        return cls(ok=False, error=error)

    def unwrap(self) -> T:
        if not self.ok or self.value is None:
            raise ValueError("Attempted to unwrap error result")
        return self.value

    def unwrap_error(self) -> E:
        if self.ok or self.error is None:
            raise ValueError("Attempted to unwrap value from ok result")
        return self.error


@dataclass(slots=True)
class MCPToolSuccess:
    """Successful MCP tool execution payload."""

    tool_name: str
    result: Any
    execution_time_ms: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class MCPToolFailure:
    """Failure metadata for MCP tool execution."""

    tool_name: str
    error: str
    retryable: bool = False
    execution_time_ms: float | None = None
    metadata: dict[str, Any] | None = None


MCPResult = Result[MCPToolSuccess, MCPToolFailure]

__all__ = ["Result", "MCPToolSuccess", "MCPToolFailure", "MCPResult"]
