"""
MCP Event Models for Super Alita Integration

Pydantic v2 models for event-driven MCP operations with deterministic UUIDs,
provenance tracking, and structured tool invocation.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MCPToolInvocation(BaseModel):
    """Model for MCP tool invocation requests"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    tool_name: str = Field(description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    session_id: str = Field(description="Session identifier for tracking")
    request_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique request ID"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MCPToolResult(BaseModel):
    """Model for MCP tool execution results"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(description="Request ID from invocation")
    tool_name: str = Field(description="Name of the executed tool")
    success: bool = Field(description="Whether execution succeeded")
    result: Any | None = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time_ms: float | None = Field(
        default=None, description="Execution time in milliseconds"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MCPCacheEvent(BaseModel):
    """Model for MCP cache operations"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cache_key: str = Field(description="Cache key")
    operation: str = Field(description="Cache operation: hit, miss, set, evict")
    tool_name: str = Field(description="Associated tool name")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MCPCircuitBreakerEvent(BaseModel):
    """Model for circuit breaker state changes"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str = Field(description="Tool name for breaker")
    state: str = Field(description="Breaker state: closed, open, half_open")
    failure_count: int = Field(description="Current failure count")
    last_failure_time: datetime | None = Field(
        default=None, description="Last failure timestamp"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MCPProvenanceEvent(BaseModel):
    """Model for provenance tracking of MCP operations"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(description="Request ID")
    tool_name: str = Field(description="Tool name")
    operation: str = Field(
        description="Operation type: invoke, cache_hit, cache_miss, circuit_open"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Operation metadata"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
